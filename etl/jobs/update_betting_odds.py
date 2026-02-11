#!/usr/bin/env python3
"""
ETL Job: update_betting_odds

Uniform contract:
- Load config (PG_DSN, CFBD_API_KEY, SEASON, RUN_ID) via etl.common_config
- Fetch CFBD data using shared retrying session (etl.common_http)
- Normalize/validate payload
- Replace season partition in Postgres inside a single transaction (delete + copy insert)
- Return a StepResult with consistent row counts and status
"""

from __future__ import annotations

import csv
import io
import sys
from typing import Any, Dict, List, Optional

import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import setup_logging, log_timing, format_step_prefix
from etl.common_types import StepResult


STEP_NAME = "update_betting_odds"

# CFBD endpoint
API_PATH = "/lines"

# Target table
BETTING_TABLE = "public.betting_odds"

# COPY columns match your table EXACTLY (quoted to preserve case)
COLS = [
    "Id",
    "HomeTeam",
    "HomeScore",
    "AwayTeam",
    "AwayScore",
    "FormattedSpread",
    "LineProvider",
    "OverUnder",
    "Spread",
    "OpeningSpread",
    "OpeningOverUnder",
    "HomeMoneyline",
    "AwayMoneyline",
]

COPY_SQL = f"""
COPY {BETTING_TABLE} (
  "Id",
  "HomeTeam",
  "HomeScore",
  "AwayTeam",
  "AwayScore",
  "FormattedSpread",
  "LineProvider",
  "OverUnder",
  "Spread",
  "OpeningSpread",
  "OpeningOverUnder",
  "HomeMoneyline",
  "AwayMoneyline"
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL (
    "HomeScore",
    "AwayScore",
    "OverUnder",
    "Spread",
    "OpeningSpread",
    "OpeningOverUnder",
    "HomeMoneyline",
    "AwayMoneyline"
  )
)
"""

# Pull both season types, weeks 1..20
SEASON_TYPES = ("regular", "postseason")
WEEKS = range(1, 21)


# -------------------------
# Normalization helpers
# -------------------------
def _pick(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _validate_rows(rows: List[Dict[str, Any]], allow_empty: bool) -> None:
    if not rows:
        if allow_empty:
            return
        raise ValueError("CFBD returned 0 rows for betting odds; refusing to delete/replace data.")

    # Must have at least these
    for c in ("Id", "HomeTeam", "AwayTeam"):
        if c not in rows[0]:
            raise ValueError(f"Missing required field in rows: {c}")


def _rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    """
    Convert rows to an in-memory CSV matching the DB table schema exactly.
    Postgres COPY will coerce '' (empty string) to NULL thanks to FORCE_NULL above.
    """
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS})
    return buf.getvalue().encode("utf-8")


# -------------------------
# Fetch + Transform
# -------------------------
def _fetch_lines_for_season(season: int, api_key: Optional[str], logger) -> List[Dict[str, Any]]:
    session = build_retry_session(api_key=api_key, timeout_seconds=60, total_retries=6)

    logger.info(f"Fetching CFBD betting lines for season={season}")

    out: List[Dict[str, Any]] = []

    for season_type in SEASON_TYPES:
        for week in WEEKS:
            resp = cfbd_get(session, API_PATH, params={"year": season, "seasonType": season_type, "week": week})
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}"
                )

            data = resp.json()
            if not data:
                continue

            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected CFBD response type for lines: {type(data)}")

            for game in data:
                game_id = _pick(game, "id", "gameId")
                home = _pick(game, "home_team", "homeTeam", "home")
                away = _pick(game, "away_team", "awayTeam", "away")
                home_pts = _pick(game, "home_points", "homePoints", "home_score", "homeScore")
                away_pts = _pick(game, "away_points", "awayPoints", "away_score", "awayScore")

                lines = game.get("lines") or []
                if not lines:
                    out.append(
                        {
                            "Id": game_id,
                            "HomeTeam": home,
                            "HomeScore": home_pts,
                            "AwayTeam": away,
                            "AwayScore": away_pts,
                            "FormattedSpread": None,
                            "LineProvider": None,
                            "OverUnder": None,
                            "Spread": None,
                            "OpeningSpread": None,
                            "OpeningOverUnder": None,
                            "HomeMoneyline": None,
                            "AwayMoneyline": None,
                        }
                    )
                    continue

                for ln in lines:
                    out.append(
                        {
                            "Id": game_id,
                            "HomeTeam": home,
                            "HomeScore": home_pts,
                            "AwayTeam": away,
                            "AwayScore": away_pts,
                            "FormattedSpread": ln.get("formattedSpread"),
                            "LineProvider": ln.get("provider"),
                            "OverUnder": ln.get("overUnder"),
                            "Spread": ln.get("spread"),
                            "OpeningSpread": ln.get("spreadOpen"),
                            "OpeningOverUnder": ln.get("overUnderOpen"),
                            "HomeMoneyline": ln.get("homeMoneyline"),
                            "AwayMoneyline": ln.get("awayMoneyline"),
                        }
                    )

    logger.info(f"Fetched {len(out)} line rows (provider×game).")
    return out


# -------------------------
# Load
# -------------------------
def _replace_rows_by_ids(
    pg_dsn: str,
    rows: List[Dict[str, Any]],
    logger,
) -> tuple[int, int]:
    """
    Betting odds are keyed by game Id (and provider). We replace rows for the specific Ids we fetched.
    Use a temp table + delete join, then COPY insert, all in one transaction.
    """
    csv_bytes = _rows_to_csv_bytes(rows)

    # Collect distinct Ids for deletion; ignore null/unparseable
    ids = []
    seen = set()
    for r in rows:
        v = r.get("Id")
        if v is None:
            continue
        try:
            v2 = int(v)
        except Exception:
            continue
        if v2 in seen:
            continue
        seen.add(v2)
        ids.append(v2)

    if not ids:
        raise ValueError("No valid Ids found in CFBD odds payload; aborting.")

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")

            # 1) Temp table of Ids to delete/replace
            cur.execute('CREATE TEMP TABLE temp_betting_ids ("Id" BIGINT) ON COMMIT DROP;')
            cur.executemany('INSERT INTO temp_betting_ids ("Id") VALUES (%s)', [(i,) for i in ids])

            # 2) Delete existing rows for those Ids
            cur.execute(
                f'''
                DELETE FROM {BETTING_TABLE} bo
                USING temp_betting_ids t
                WHERE bo."Id" = t."Id";
                '''
            )
            deleted = int(cur.rowcount or 0)
            logger.info(f"Deleted {deleted} existing rows from {BETTING_TABLE} for {len(ids)} Ids")

            # 3) COPY new rows
            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            # 4) Estimate inserted as count for these Ids after load
            cur.execute(
                f'''
                SELECT COUNT(*)
                FROM {BETTING_TABLE} bo
                JOIN temp_betting_ids t
                ON bo."Id" = t."Id";
                '''
            )
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
            rows = _fetch_lines_for_season(cfg.season, cfg.cfbd_api_key, logger)

        # For this job, empty is OK early season, but we treat it as SKIPPED (no DB changes).
        with log_timing(logger, f"{prefix} validate"):
            if not rows:
                msg = f"season={cfg.season} no rows returned; skipping without DB changes"
                logger.info(f"{prefix} skipped | {msg}")
                return StepResult(
                    step_name=STEP_NAME,
                    season=cfg.season,
                    status="skipped",
                    rows_fetched=0,
                    rows_deleted=0,
                    rows_inserted=0,
                    message=msg,
                )
            _validate_rows(rows, allow_empty=False)

        with log_timing(logger, f"{prefix} load"):
            deleted, inserted = _replace_rows_by_ids(cfg.pg_dsn, rows, logger)

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
    if res.status != "success" and res.status != "skipped":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
