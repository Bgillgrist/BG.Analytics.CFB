#!/usr/bin/env python3
"""
ETL Job: update_rankings

Uniform contract:
- Load config (PG_DSN, CFBD_API_KEY, SEASON, RUN_ID) via etl.common_config
- Fetch CFBD data using shared retrying session (etl.common_http)
- Normalize/validate payload
- Replace season partition in Postgres inside a single transaction (delete + copy insert)
- Return a StepResult with consistent row counts and status
"""

from __future__ import annotations

import io
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import setup_logging, log_timing, format_step_prefix
from etl.common_types import StepResult


STEP_NAME = "update_rankings"

# CFBD endpoint
API_PATH = "/rankings"

# Target columns in Postgres (match your table)
COLS = [
    "season",
    "season_type",
    "week",
    "poll",
    "rank",
    "school",
    "conference",
    "first_place_votes",
    "points",
]

# Everything except these text fields is numeric (NULLable where applicable)
TEXT_COLS = {"season_type", "poll", "school", "conference"}
NUMERIC_COLS = [c for c in COLS if c not in TEXT_COLS]
FORCE_NULL_SQL = ", ".join(NUMERIC_COLS)

COPY_SQL = f"""
COPY public.rankings (
  {", ".join(COLS)}
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL ({FORCE_NULL_SQL})
)
"""


# -------------------------
# Normalization helpers
# -------------------------
def _validate_df(df: pd.DataFrame, allow_empty: bool) -> None:
    if df.empty:
        if allow_empty:
            return
        raise ValueError("CFBD returned 0 rows for rankings; refusing to delete/replace season data.")

    # Must have at least these
    for c in ("season", "season_type", "week", "poll", "rank", "school"):
        if c not in df.columns:
            raise ValueError(f"Missing required column after normalization: {c}")

    # Duplicate check on natural key / table PK
    dupes = df.duplicated(subset=["season", "season_type", "week", "poll", "rank", "school"]).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate rows on key (season, week, poll, rank, school); aborting.")


# -------------------------
# Fetch + Transform
# -------------------------
def _fetch_rankings(season: int, api_key: Optional[str], logger) -> List[Dict[str, Any]]:
    session = build_retry_session(api_key=api_key, timeout_seconds=90, total_retries=6)

    logger.info(f"Fetching CFBD rankings for season={season}")
    resp = cfbd_get(session, API_PATH, params={"year": season})
    if resp.status_code >= 400:
        raise RuntimeError(f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CFBD response type for rankings: {type(data)}")

    logger.info(f"Retrieved {len(data)} weekly ranking payloads from CFBD.")
    return data


def _flatten_to_df(payload: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for entry in payload:
        season = entry.get("season")
        season_type = entry.get("seasonType")
        week = entry.get("week")

        for poll in entry.get("polls", []) or []:
            poll_name = poll.get("poll")
            for r in poll.get("ranks", []) or []:
                rows.append(
                    {
                        "season": season,
                        "season_type": season_type,
                        "week": week,
                        "poll": poll_name,
                        "rank": r.get("rank"),
                        "school": r.get("school"),
                        "conference": r.get("conference"),
                        "first_place_votes": r.get("firstPlaceVotes"),
                        "points": r.get("points"),
                    }
                )

    return pd.DataFrame(rows)


def _transform_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    # Ensure all expected columns exist (in case some fields missing from API)
    for c in COLS:
        if c not in df2.columns:
            df2[c] = None

    df2 = df2[COLS]

    # Coerce numeric fields
    df2["season"] = pd.to_numeric(df2["season"], errors="raise").astype(int)
    df2["week"] = pd.to_numeric(df2["week"], errors="raise").astype(int)
    df2["rank"] = pd.to_numeric(df2["rank"], errors="raise").astype(int)

    # Let first_place_votes / points be nullable
    df2["first_place_votes"] = pd.to_numeric(df2["first_place_votes"], errors="coerce")
    df2["points"] = pd.to_numeric(df2["points"], errors="coerce")

    return df2


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame into CSV bytes payload for COPY.
    Keep empty fields truly empty (not 'NaN'); COPY uses NULL '' + FORCE_NULL.
    """
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# -------------------------
# Load
# -------------------------
def _replace_season_partition(
    pg_dsn: str,
    season: int,
    df: pd.DataFrame,
    logger,
) -> tuple[int, int]:
    """
    Delete-and-reload pattern, but ALWAYS in one transaction.
    If COPY fails, the delete is rolled back automatically.
    """
    csv_bytes = _df_to_csv_bytes(df)

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")

            # delete
            cur.execute("DELETE FROM public.rankings WHERE season = %s;", (season,))
            deleted = int(cur.rowcount or 0)
            logger.info(f"Deleted {deleted} existing rows from public.rankings for season={season}")

            # copy insert
            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            # count inserted after copy
            cur.execute("SELECT COUNT(*) FROM public.rankings WHERE season=%s;", (season,))
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
            payload = _fetch_rankings(cfg.season, cfg.cfbd_api_key, logger)

        with log_timing(logger, f"{prefix} transform"):
            raw_df = _flatten_to_df(payload)
            df = _transform_df(raw_df)

        # For this job, empty is NOT okay. If CFBD gives 0 rows, refuse to delete.
        with log_timing(logger, f"{prefix} validate"):
            _validate_df(df, allow_empty=False)

        with log_timing(logger, f"{prefix} load"):
            deleted, inserted = _replace_season_partition(cfg.pg_dsn, cfg.season, df, logger)

        msg = f"season={cfg.season} deleted={deleted} inserted={inserted}"
        logger.info(f"{prefix} success | {msg}")
        return StepResult(
            step_name=STEP_NAME,
            season=cfg.season,
            status="success",
            rows_fetched=int(len(df)),
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
