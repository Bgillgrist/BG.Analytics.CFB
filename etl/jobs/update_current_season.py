#!/usr/bin/env python3
"""
ETL Job: update_current_season

Uniform contract:
- Load config (PG_DSN, CFBD_API_KEY, SEASON, RUN_ID) via etl.common_config
- Fetch CFBD data using shared retrying session (etl.common_http)
- Normalize/validate payload
- Replace season partition in Postgres inside a single transaction (delete + copy insert)
- Return a StepResult with consistent row counts and status
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import os
import re
import sys
from typing import Any, Dict, List, Optional

import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import setup_logging, log_timing, format_step_prefix
from etl.common_types import StepResult
from etl.common_validate import require_columns, require_nonempty, require_no_dupes


STEP_NAME = "update_current_season"

# CFBD endpoint
API_PATH = "/games"

# Target columns in Postgres (must match your public.game_data table)
COLS = [
    "id", "season", "week", "seasontype", "startdate", "starttimetbd", "completed", "neutralsite",
    "conferencegame", "attendance", "venueid", "venue",
    "homeid", "hometeam", "homeclassification", "homeconference", "homepoints", "homelinescores",
    "homepostgamewinprobability", "homepregameelo", "homepostgameelo",
    "awayid", "awayteam", "awayclassification", "awayconference", "awaypoints", "awaylinescores",
    "awaypostgamewinprobability", "awaypregameelo", "awaypostgameelo",
    "excitementindex", "highlights", "notes",
]

# For validation / uniqueness
KEY_COLS = ["id"]

COPY_SQL = f"""
COPY public.game_data (
  {", ".join(COLS)}
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL (
    attendance, venueid,
    homeid, homepoints,
    awayid, awaypoints,
    starttimetbd, completed, neutralsite, conferencegame,
    homepregameelo, homepostgameelo,
    awaypregameelo, awaypostgameelo,
    homepostgamewinprobability, awaypostgamewinprobability,
    excitementindex
  )
)
"""


# -------------------------
# Normalization helpers
# -------------------------
def _to_snake(s: str) -> str:
    # camelCase/PascalCase -> snake_case; also collapse spaces
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).replace(" ", "_")
    return s.lower()


def _normalize_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {_to_snake(k): v for k, v in d.items()}


def _join_linescores(xs: Any) -> Optional[str]:
    if not xs:
        return None
    if isinstance(xs, (list, tuple)):
        return ",".join(str(x) for x in xs)
    return str(xs)


def _cfbd_to_row(g: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps a CFBD game object into a row dict matching COLS.
    Includes a few defensive fallbacks in case CFBD names drift.
    """
    g2 = _normalize_keys(g)

    # Robust fallbacks for a few fields that have drifted names
    startdate = g2.get("start_date") or g2.get("start_time")
    starttimetbd = g2.get("start_time_tbd") or g2.get("starttimetbd")
    homeclassification = g2.get("home_classification") or g2.get("home_division")
    awayclassification = g2.get("away_classification") or g2.get("away_division")

    home_pg_wp = (
        g2.get("home_postgame_win_prob")
        or g2.get("home_post_win_prob")
        or g2.get("home_postgame_win_probability")
    )
    away_pg_wp = (
        g2.get("away_postgame_win_prob")
        or g2.get("away_post_win_prob")
        or g2.get("away_postgame_win_probability")
    )

    return {
        # core
        "id": g2.get("id") or g2.get("game_id"),
        "season": g2.get("season"),
        "week": g2.get("week"),
        "seasontype": g2.get("season_type"),
        "startdate": startdate,
        "starttimetbd": starttimetbd,
        "completed": g2.get("completed"),
        "neutralsite": g2.get("neutral_site"),
        "conferencegame": g2.get("conference_game"),
        "attendance": g2.get("attendance"),
        "venueid": g2.get("venue_id"),
        "venue": g2.get("venue"),

        # home
        "homeid": g2.get("home_id"),
        "hometeam": g2.get("home_team"),
        "homeclassification": homeclassification,
        "homeconference": g2.get("home_conference"),
        "homepoints": g2.get("home_points"),
        "homelinescores": _join_linescores(g2.get("home_line_scores")),
        "homepostgamewinprobability": home_pg_wp,
        "homepregameelo": g2.get("home_pregame_elo"),
        "homepostgameelo": g2.get("home_postgame_elo"),

        # away
        "awayid": g2.get("away_id"),
        "awayteam": g2.get("away_team"),
        "awayclassification": awayclassification,
        "awayconference": g2.get("away_conference"),
        "awaypoints": g2.get("away_points"),
        "awaylinescores": _join_linescores(g2.get("away_line_scores")),
        "awaypostgamewinprobability": away_pg_wp,
        "awaypregameelo": g2.get("away_pregame_elo"),
        "awaypostgameelo": g2.get("away_postgame_elo"),

        # misc
        "excitementindex": g2.get("excitement_index"),
        "highlights": g2.get("highlights"),
        "notes": g2.get("notes"),
    }


# -------------------------
# CSV helpers
# -------------------------
def _to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """
    Convert list-of-dicts into a CSV bytes payload for COPY.
    We write empty string for None, and COPY is configured with NULL ''.
    """
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS})
    return sio.getvalue().encode("utf-8")


def _validate_rows(rows: List[Dict[str, Any]], allow_empty: bool) -> None:
    # basic shape checks
    if not rows:
        if allow_empty:
            return
        raise ValueError("CFBD returned 0 games; refusing to delete/replace season data.")

    # column presence (row dicts)
    for col in ("id", "season", "week", "hometeam", "awayteam", "startdate"):
        if col not in rows[0]:
            raise ValueError(f"Missing expected field in mapped rows: {col}")

    # required columns in output
    # (dict-based check: ensure keys exist)
    missing_any = [c for c in COLS if c not in rows[0]]
    if missing_any:
        # This is defensive; typically rows[0] contains all COLS keys.
        raise ValueError(f"Mapped rows missing keys for: {missing_any}")

    # duplicates on key
    seen = set()
    dupes = 0
    for r in rows:
        k = r.get("id")
        if k in seen:
            dupes += 1
        seen.add(k)
    if dupes:
        raise ValueError(f"Found {dupes} duplicate game ids in payload; aborting.")


# -------------------------
# Fetch + Load
# -------------------------
def _fetch_games(season: int, api_key: Optional[str], logger) -> List[Dict[str, Any]]:
    session = build_retry_session(api_key=api_key, timeout_seconds=60, total_retries=6)

    logger.info(f"Fetching CFBD games for season={season}")
    resp = cfbd_get(session, API_PATH, params={"year": season})
    # If CFBD returns an error body but status != 200, raise with context
    if resp.status_code >= 400:
        raise RuntimeError(f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CFBD response type for games: {type(data)}")

    rows = [_cfbd_to_row(g) for g in data]
    logger.info(f"Fetched {len(rows)} games from CFBD.")
    return rows


def _replace_season_partition(
    pg_dsn: str,
    season: int,
    rows: List[Dict[str, Any]],
    logger,
) -> tuple[int, int]:
    """
    Delete-and-reload pattern, but ALWAYS in one transaction.
    If COPY fails, the delete is rolled back automatically.
    """
    csv_bytes = _to_csv_bytes(rows)

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            # delete
            cur.execute("DELETE FROM public.game_data WHERE season = %s;", (season,))
            deleted = int(cur.rowcount or 0)
            logger.info(f"Deleted {deleted} existing rows from public.game_data for season={season}")

            # copy insert
            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            # rowcount isn't available for COPY reliably; count inserted by querying
            cur.execute("SELECT COUNT(*) FROM public.game_data WHERE season=%s;", (season,))
            inserted = int(cur.fetchone()[0])

            cur.execute("COMMIT;")
            return deleted, inserted


# -------------------------
# Public entrypoint
# -------------------------
def run() -> StepResult:
    cfg = load_config()
    logger = setup_logging()
    prefix = format_step_prefix(cfg.run_id, STEP_NAME)

    logger.info(f"{prefix} start (season={cfg.season})")

    try:
        with log_timing(logger, f"{prefix} fetch"):
            rows = _fetch_games(cfg.season, cfg.cfbd_api_key, logger)

        # Validation: for current season, empty is NOT okay.
        # If CFBD gives you 0 games, we refuse to delete season data.
        with log_timing(logger, f"{prefix} validate"):
            _validate_rows(rows, allow_empty=False)

        with log_timing(logger, f"{prefix} load"):
            deleted, inserted = _replace_season_partition(cfg.pg_dsn, cfg.season, rows, logger)

        msg = f"season={cfg.season} deleted={deleted} inserted={inserted}"
        logger.info(f"{prefix} success | {msg}")
        return StepResult(
            step_name=STEP_NAME,
            season=cfg.season,
            status="success",
            rows_fetched=len(rows),
            rows_deleted=deleted,
            rows_inserted=inserted,
            message=msg,
        )

    except Exception as e:
        logger.exception(f"{prefix} FAILED: {e}")
        return StepResult(
            step_name=STEP_NAME,
            season=cfg.season,
            status="failed",
            message="Job failed; see logs for details.",
            error=str(e),
        )


def main() -> None:
    res = run()
    if res.status != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
