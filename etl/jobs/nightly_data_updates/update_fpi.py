#!/usr/bin/env python3
"""
ETL Job: update_fpi

Fetch the latest CFBD Football Power Index ratings and append one daily pull
snapshot. CFBD exposes FPI by season only, so pull_date is part of the natural
key to preserve historical daily changes without duplicating same-day runs.
"""

from __future__ import annotations

import csv
import io
import sys
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import format_step_prefix, log_timing, setup_logging
from etl.common_types import StepResult


STEP_NAME = "update_fpi"
API_PATH = "/ratings/fpi"
TABLE = "public.fpi_ratings"
PULL_DATE_TIMEZONE = "America/New_York"

COLS = [
    "season",
    "pull_date",
    "team",
    "conference",
    "fpi",
    "resume_game_control_rank",
    "resume_remaining_strength_of_schedule_rank",
    "resume_strength_of_schedule_rank",
    "resume_average_win_probability_rank",
    "resume_fpi_rank",
    "resume_strength_of_record_rank",
    "efficiency_special_teams",
    "efficiency_defense",
    "efficiency_offense",
    "efficiency_overall",
]

TEXT_COLS = {"team", "conference"}
NUMERIC_OR_DATE_COLS = [c for c in COLS if c not in TEXT_COLS]
FORCE_NULL_SQL = ", ".join(NUMERIC_OR_DATE_COLS)

COPY_SQL = f"""
COPY temp_fpi_ratings (
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

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  season INTEGER NOT NULL,
  pull_date DATE NOT NULL,
  team TEXT NOT NULL,
  conference TEXT,
  fpi DOUBLE PRECISION,
  resume_game_control_rank INTEGER,
  resume_remaining_strength_of_schedule_rank INTEGER,
  resume_strength_of_schedule_rank INTEGER,
  resume_average_win_probability_rank INTEGER,
  resume_fpi_rank INTEGER,
  resume_strength_of_record_rank INTEGER,
  efficiency_special_teams DOUBLE PRECISION,
  efficiency_defense DOUBLE PRECISION,
  efficiency_offense DOUBLE PRECISION,
  efficiency_overall DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE {TABLE}
  ADD COLUMN IF NOT EXISTS season INTEGER,
  ADD COLUMN IF NOT EXISTS pull_date DATE,
  ADD COLUMN IF NOT EXISTS team TEXT,
  ADD COLUMN IF NOT EXISTS conference TEXT,
  ADD COLUMN IF NOT EXISTS fpi DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS resume_game_control_rank INTEGER,
  ADD COLUMN IF NOT EXISTS resume_remaining_strength_of_schedule_rank INTEGER,
  ADD COLUMN IF NOT EXISTS resume_strength_of_schedule_rank INTEGER,
  ADD COLUMN IF NOT EXISTS resume_average_win_probability_rank INTEGER,
  ADD COLUMN IF NOT EXISTS resume_fpi_rank INTEGER,
  ADD COLUMN IF NOT EXISTS resume_strength_of_record_rank INTEGER,
  ADD COLUMN IF NOT EXISTS efficiency_special_teams DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS efficiency_defense DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS efficiency_offense DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS efficiency_overall DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS idx_fpi_ratings_unique_pull
ON {TABLE} (season, pull_date, team);

CREATE INDEX IF NOT EXISTS idx_fpi_ratings_latest_lookup
ON {TABLE} (season, pull_date DESC);

CREATE INDEX IF NOT EXISTS idx_fpi_ratings_team_lookup
ON {TABLE} (team, season, pull_date DESC);
"""

CREATE_TEMP_TABLE_SQL = """
CREATE TEMP TABLE temp_fpi_ratings (
  season INTEGER NOT NULL,
  pull_date DATE NOT NULL,
  team TEXT NOT NULL,
  conference TEXT,
  fpi DOUBLE PRECISION,
  resume_game_control_rank INTEGER,
  resume_remaining_strength_of_schedule_rank INTEGER,
  resume_strength_of_schedule_rank INTEGER,
  resume_average_win_probability_rank INTEGER,
  resume_fpi_rank INTEGER,
  resume_strength_of_record_rank INTEGER,
  efficiency_special_teams DOUBLE PRECISION,
  efficiency_defense DOUBLE PRECISION,
  efficiency_offense DOUBLE PRECISION,
  efficiency_overall DOUBLE PRECISION
) ON COMMIT DROP;
"""

INSERT_FROM_TEMP_SQL = f"""
INSERT INTO {TABLE} (
  {", ".join(COLS)}
)
SELECT
  {", ".join(COLS)}
FROM temp_fpi_ratings
ON CONFLICT (season, pull_date, team) DO NOTHING;
"""


def _current_pull_date() -> date:
    return datetime.now(ZoneInfo(PULL_DATE_TIMEZONE)).date()


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _nested_pick(d: Any, *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    return _pick(d, *keys)


def _cfbd_to_row(rating: Dict[str, Any], pull_date: date) -> Dict[str, Any]:
    resume_ranks = _pick(rating, "resumeRanks", "resume_ranks") or {}
    efficiencies = _pick(rating, "efficiencies", "efficiency") or {}

    return {
        "season": _pick(rating, "year", "season"),
        "pull_date": pull_date.isoformat(),
        "team": _pick(rating, "team"),
        "conference": _pick(rating, "conference"),
        "fpi": _pick(rating, "fpi"),
        "resume_game_control_rank": _nested_pick(resume_ranks, "gameControl", "game_control"),
        "resume_remaining_strength_of_schedule_rank": _nested_pick(
            resume_ranks,
            "remainingStrengthOfSchedule",
            "remaining_strength_of_schedule",
        ),
        "resume_strength_of_schedule_rank": _nested_pick(resume_ranks, "strengthOfSchedule", "strength_of_schedule"),
        "resume_average_win_probability_rank": _nested_pick(
            resume_ranks,
            "averageWinProbability",
            "average_win_probability",
        ),
        "resume_fpi_rank": _nested_pick(resume_ranks, "fpi"),
        "resume_strength_of_record_rank": _nested_pick(resume_ranks, "strengthOfRecord", "strength_of_record"),
        "efficiency_special_teams": _nested_pick(efficiencies, "specialTeams", "special_teams"),
        "efficiency_defense": _nested_pick(efficiencies, "defense"),
        "efficiency_offense": _nested_pick(efficiencies, "offense"),
        "efficiency_overall": _nested_pick(efficiencies, "overall"),
    }


def _fetch_fpi_ratings(
    season: int,
    pull_date: date,
    api_key: Optional[str],
    logger,
) -> List[Dict[str, Any]]:
    session = build_retry_session(api_key=api_key, timeout_seconds=60, total_retries=6)

    logger.info(f"Fetching CFBD FPI ratings for season={season}")
    resp = cfbd_get(session, API_PATH, params={"year": season})
    if resp.status_code >= 400:
        raise RuntimeError(f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CFBD response type for FPI ratings: {type(data)}")

    rows = [_cfbd_to_row(rating, pull_date) for rating in data]
    logger.info(f"Fetched {len(rows)} FPI rating rows from CFBD.")
    return rows


def _validate_rows(rows: List[Dict[str, Any]], season: int, pull_date: date) -> None:
    for row in rows:
        missing = [c for c in COLS if c not in row]
        if missing:
            raise ValueError(f"Mapped FPI row missing keys for: {missing}")
        if row.get("season") in (None, ""):
            raise ValueError("Mapped FPI row is missing a season.")
        if row.get("team") in (None, ""):
            raise ValueError("Mapped FPI row is missing a team.")

    wrong_seasons = sorted(
        {
            int(row["season"])
            for row in rows
            if row.get("season") not in (None, "") and int(row["season"]) != season
        }
    )
    if wrong_seasons:
        raise ValueError(f"CFBD payload contained seasons other than {season}: {wrong_seasons}")

    wrong_pull_dates = sorted({str(row["pull_date"]) for row in rows if str(row["pull_date"]) != pull_date.isoformat()})
    if wrong_pull_dates:
        raise ValueError(f"Mapped FPI rows contained pull dates other than {pull_date.isoformat()}: {wrong_pull_dates}")

    seen = set()
    dupes = 0
    for row in rows:
        key = (row.get("season"), row.get("pull_date"), row.get("team"))
        if key in seen:
            dupes += 1
        seen.add(key)
    if dupes:
        raise ValueError(f"Found {dupes} duplicate FPI rows on key (season, pull_date, team); aborting.")


def _to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLS})
    return buf.getvalue().encode("utf-8")


def _ensure_table_and_count_existing_pull(
    pg_dsn: str,
    season: int,
    pull_date: date,
    logger,
) -> int:
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE season = %s AND pull_date = %s;", (season, pull_date))
            existing = int(cur.fetchone()[0])
            cur.execute("COMMIT;")

    if existing:
        logger.info(f"{TABLE} already has {existing} rows for season={season} pull_date={pull_date}; skipping")
    return existing


def _insert_new_pull(
    pg_dsn: str,
    season: int,
    pull_date: date,
    rows: List[Dict[str, Any]],
    logger,
) -> int:
    csv_bytes = _to_csv_bytes(rows)

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_TEMP_TABLE_SQL)

            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            cur.execute(INSERT_FROM_TEMP_SQL)
            inserted = int(cur.rowcount or 0)

            cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE season = %s AND pull_date = %s;", (season, pull_date))
            pull_count = int(cur.fetchone()[0])
            logger.info(
                f"Inserted {inserted} new rows into {TABLE} for season={season} pull_date={pull_date}; "
                f"pull now has {pull_count} rows"
            )

            cur.execute("COMMIT;")
            return inserted


def run() -> StepResult:
    cfg = load_config()
    logger = setup_logging()
    prefix = format_step_prefix(cfg.run_id, STEP_NAME)
    pull_date = _current_pull_date()

    logger.info(f"{prefix} start (season={cfg.season}, pull_date={pull_date})")

    try:
        with log_timing(logger, f"{prefix} preflight"):
            existing = _ensure_table_and_count_existing_pull(cfg.pg_dsn, cfg.season, pull_date, logger)
            if existing:
                msg = f"season={cfg.season} pull_date={pull_date} already exists with {existing} rows; skipping"
                logger.info(f"{prefix} skipped | {msg}")
                return StepResult(
                    step_name=STEP_NAME,
                    season=cfg.season,
                    status="skipped",
                    rows_fetched=0,
                    rows_deleted=0,
                    rows_inserted=0,
                    message=msg,
                    meta={"pull_date": pull_date.isoformat(), "existing_rows": existing},
                )

        with log_timing(logger, f"{prefix} fetch"):
            rows = _fetch_fpi_ratings(cfg.season, pull_date, cfg.cfbd_api_key, logger)

        with log_timing(logger, f"{prefix} validate"):
            if not rows:
                msg = f"season={cfg.season} no FPI ratings returned; skipping without DB changes"
                logger.info(f"{prefix} skipped | {msg}")
                return StepResult(
                    step_name=STEP_NAME,
                    season=cfg.season,
                    status="skipped",
                    rows_fetched=0,
                    rows_deleted=0,
                    rows_inserted=0,
                    message=msg,
                    meta={"pull_date": pull_date.isoformat()},
                )
            _validate_rows(rows, cfg.season, pull_date)

        with log_timing(logger, f"{prefix} load"):
            inserted = _insert_new_pull(cfg.pg_dsn, cfg.season, pull_date, rows, logger)

        msg = f"season={cfg.season} pull_date={pull_date} inserted={inserted}"
        logger.info(f"{prefix} success | {msg}")
        return StepResult(
            step_name=STEP_NAME,
            season=cfg.season,
            status="success",
            rows_fetched=len(rows),
            rows_deleted=0,
            rows_inserted=inserted,
            message=msg,
            meta={"pull_date": pull_date.isoformat()},
        )

    except Exception as e:
        logger.exception(f"{prefix} FAILED: {e}")
        return StepResult(
            step_name=STEP_NAME,
            season=cfg.season,
            status="failed",
            message="Job failed; see logs for details.",
            meta={"pull_date": pull_date.isoformat()},
            error=str(e),
        )


def main() -> None:
    res = run()
    if res.status != "success" and res.status != "skipped":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
