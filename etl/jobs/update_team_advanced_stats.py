#!/usr/bin/env python3
"""
ETL Job: update_team_advanced_stats

Uniform contract:
- Load config (PG_DSN, CFBD_API_KEY, SEASON, RUN_ID) via etl.common_config
- Fetch CFBD data using shared retrying session (etl.common_http)
- Normalize/validate payload
- Replace season partition in Postgres inside a single transaction (delete + copy insert)
- Return a StepResult with consistent row counts and status
"""

from __future__ import annotations

import datetime as dt
import io
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import setup_logging, log_timing, format_step_prefix
from etl.common_types import StepResult


STEP_NAME = "update_team_advanced_stats"

# CFBD endpoint
API_PATH = "/stats/season/advanced"

# Target columns in Postgres (match your table)
COLS = [
    "season","team","conference",
    "offense_plays","offense_drives","offense_ppa","offense_totalppa",
    "offense_successrate","offense_explosiveness","offense_powersuccess",
    "offense_stuffrate","offense_lineyards","offense_lineyardstotal",
    "offense_secondlevelyards","offense_secondlevelyardstotal",
    "offense_openfieldyards","offense_openfieldyardstotal",
    "offense_totalopportunies","offense_pointsperopportunity",
    "offense_fieldposition_averagestart","offense_fieldposition_averagepredictedpoints",
    "offense_havoc_total","offense_havoc_frontseven","offense_havoc_db",
    "offense_standarddowns_rate","offense_standarddowns_ppa",
    "offense_standarddowns_successrate","offense_standarddowns_explosiveness",
    "offense_passingdowns_rate","offense_passingdowns_ppa",
    "offense_passingdowns_successrate","offense_passingdowns_explosiveness",
    "offense_rushingplays_rate","offense_rushingplays_ppa",
    "offense_rushingplays_totalppa","offense_rushingplays_successrate",
    "offense_rushingplays_explosiveness",
    "offense_passingplays_rate","offense_passingplays_ppa",
    "offense_passingplays_totalppa","offense_passingplays_successrate",
    "offense_passingplays_explosiveness",
    "defense_plays","defense_drives","defense_ppa","defense_totalppa",
    "defense_successrate","defense_explosiveness","defense_powersuccess",
    "defense_stuffrate","defense_lineyards","defense_lineyardstotal",
    "defense_secondlevelyards","defense_secondlevelyardstotal",
    "defense_openfieldyards","defense_openfieldyardstotal",
    "defense_totalopportunies","defense_pointsperopportunity",
    "defense_fieldposition_averagestart","defense_fieldposition_averagepredictedpoints",
    "defense_havoc_total","defense_havoc_frontseven","defense_havoc_db",
    "defense_standarddowns_rate","defense_standarddowns_ppa",
    "defense_standarddowns_successrate","defense_standarddowns_explosiveness",
    "defense_passingdowns_rate","defense_passingdowns_ppa",
    "defense_passingdowns_totalppa","defense_passingdowns_successrate",
    "defense_passingdowns_explosiveness",
    "defense_rushingplays_rate","defense_rushingplays_ppa",
    "defense_rushingplays_totalppa","defense_rushingplays_successrate",
    "defense_rushingplays_explosiveness",
    "defense_passingplays_rate","defense_passingplays_ppa",
    "defense_passingplays_totalppa","defense_passingplays_successrate",
    "defense_passingplays_explosiveness",
]

# Everything except team, conference is numeric in your table
NUMERIC_COLS = [c for c in COLS if c not in ("team", "conference")]
FORCE_NULL_SQL = ", ".join(NUMERIC_COLS)

COPY_SQL = f"""
COPY public.team_advanced_season_stats (
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
def _normalize(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    # keep your table's misspelling
    s = s.replace("totalopportunities", "totalopportunies")
    return s


def _validate_df(df: pd.DataFrame, allow_empty: bool) -> None:
    if df.empty:
        if allow_empty:
            return
        raise ValueError("CFBD returned 0 rows for advanced season stats; refusing to delete/replace season data.")

    # Must have at least these after normalization
    for c in ("season", "team", "conference"):
        if c not in df.columns:
            raise ValueError(f"Missing required column after normalization: {c}")

    # Duplicate check on natural key
    dupes = df.duplicated(subset=["season", "team"]).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate rows on key (season, team); aborting.")


# -------------------------
# Fetch + Transform
# -------------------------
def _fetch_stats(season: int, api_key: Optional[str], logger) -> pd.DataFrame:
    session = build_retry_session(api_key=api_key, timeout_seconds=60, total_retries=6)

    logger.info(f"Fetching CFBD team advanced season stats for season={season}")
    resp = cfbd_get(session, API_PATH, params={"year": season})
    if resp.status_code >= 400:
        raise RuntimeError(f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CFBD response type for advanced season stats: {type(data)}")

    df = pd.json_normalize(data, sep="_")
    logger.info(f"Retrieved {len(df)} rows from CFBD.")
    return df


def _transform_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    # Normalize headers to match your SQL: lowercase + spaces -> underscores (+ misspelling fix)
    df2 = df.copy()
    df2.rename(columns=lambda c: _normalize(str(c)), inplace=True)

    # Ensure season exists and is set (defensive; CFBD typically provides it)
    if "season" not in df2.columns:
        df2["season"] = season

    # Ensure all expected columns exist; if missing, add empty
    for c in COLS:
        if c not in df2.columns:
            df2[c] = ""

    # Reorder and keep only expected columns
    df2 = df2[COLS]
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
            cur.execute("DELETE FROM public.team_advanced_season_stats WHERE season = %s;", (season,))
            deleted = int(cur.rowcount or 0)
            logger.info(f"Deleted {deleted} existing rows from public.team_advanced_season_stats for season={season}")

            # copy insert
            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            # count inserted after copy
            cur.execute("SELECT COUNT(*) FROM public.team_advanced_season_stats WHERE season=%s;", (season,))
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
            raw_df = _fetch_stats(cfg.season, cfg.cfbd_api_key, logger)

        with log_timing(logger, f"{prefix} transform"):
            df = _transform_df(raw_df, cfg.season)

        # Early season can legitimately have no advanced season stats yet.
        # Skip before load so we never delete existing data for an empty payload.
        with log_timing(logger, f"{prefix} validate"):
            if df.empty:
                msg = f"season={cfg.season} no advanced season stats returned; skipping without DB changes"
                logger.info(f"{prefix} skipped | {msg}")
                return StepResult(
                    step_name=STEP_NAME,
                    season=cfg.season,
                    status="skipped",
                    rows_fetched=int(len(raw_df)),
                    rows_deleted=0,
                    rows_inserted=0,
                    message=msg,
                )
            _validate_df(df, allow_empty=False)

        with log_timing(logger, f"{prefix} load"):
            deleted, inserted = _replace_season_partition(cfg.pg_dsn, cfg.season, df, logger)

        msg = f"season={cfg.season} deleted={deleted} inserted={inserted}"
        logger.info(f"{prefix} success | {msg}")
        return StepResult(
            step_name=STEP_NAME,
            season=cfg.season,
            status="success",
            rows_fetched=int(len(raw_df)),
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
    if res.status not in ("success", "skipped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
