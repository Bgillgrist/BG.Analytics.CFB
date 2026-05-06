#!/usr/bin/env python3
"""
ETL Job: update_team_advanced_game_stats

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

import numpy as np
import pandas as pd
import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import setup_logging, log_timing, format_step_prefix
from etl.common_types import StepResult


STEP_NAME = "update_team_advanced_game_stats"

# CFBD endpoint
API_PATH = "/stats/game/advanced"

# Target columns in Postgres (match your table)
FINAL_COLS = [
    "game_id","season","season_type","week","team","opponent",

    "offense_plays","offense_drives","offense_ppa","offense_totalppa",
    "offense_successrate","offense_explosiveness","offense_powersuccess",
    "offense_stuffrate","offense_lineyards","offense_lineyardstotal",
    "offense_secondlevelyards","offense_secondlevelyardstotal",
    "offense_openfieldyards","offense_openfieldyardstotal",

    "offense_standarddowns_ppa","offense_standarddowns_successrate","offense_standarddowns_explosiveness",
    "offense_passingdowns_ppa","offense_passingdowns_successrate","offense_passingdowns_explosiveness",
    "offense_rushingplays_ppa","offense_rushingplays_totalppa","offense_rushingplays_successrate","offense_rushingplays_explosiveness",
    "offense_passingplays_ppa","offense_passingplays_totalppa","offense_passingplays_successrate","offense_passingplays_explosiveness",

    "defense_plays","defense_drives","defense_ppa","defense_totalppa",
    "defense_successrate","defense_explosiveness","defense_powersuccess",
    "defense_stuffrate","defense_lineyards","defense_lineyardstotal",
    "defense_secondlevelyards","defense_secondlevelyardstotal",
    "defense_openfieldyards","defense_openfieldyardstotal",

    "defense_standarddowns_ppa","defense_standarddowns_successrate","defense_standarddowns_explosiveness",
    "defense_passingdowns_ppa","defense_passingdowns_successrate","defense_passingdowns_explosiveness",
    "defense_rushingplays_ppa","defense_rushingplays_totalppa","defense_rushingplays_successrate","defense_rushingplays_explosiveness",
    "defense_passingplays_ppa","defense_passingplays_totalppa","defense_passingplays_successrate","defense_passingplays_explosiveness",
]

# Text-only columns
TEXT_COLS = {"season_type", "team", "opponent"}

# Everything else should be numeric (NULLable)
NUMERIC_COLS = [c for c in FINAL_COLS if c not in TEXT_COLS]
FORCE_NULL_SQL = ", ".join(NUMERIC_COLS)

COPY_SQL = f"""
COPY public.team_advanced_game_stats (
  {", ".join(FINAL_COLS)}
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
        raise ValueError("CFBD returned 0 rows for advanced game stats; refusing to delete/replace season data.")

    # Must have at least these
    for c in ("game_id", "season", "season_type", "week", "team", "opponent"):
        if c not in df.columns:
            raise ValueError(f"Missing required column after normalization: {c}")

    # Drop rows without game_id, but require at least 1 row left
    if df["game_id"].isna().any():
        df.dropna(subset=["game_id"], inplace=True)

    if df.empty and not allow_empty:
        raise ValueError("All rows had null game_id; refusing to delete/replace season data.")

    # Duplicate check on natural key
    dupes = df.duplicated(subset=["game_id", "team"]).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate rows on key (game_id, team); aborting.")

    # Ensure metric columns are not all null (schema drift guard)
    metric_cols = [c for c in FINAL_COLS if c not in ("game_id","season","season_type","week","team","opponent")]
    non_null_metric_cells = df[metric_cols].notna().sum().sum()
    if non_null_metric_cells == 0:
        raise ValueError("All advanced metric fields are NULL; CFBD schema likely changed. Aborting load.")


# -------------------------
# Fetch + Transform
# -------------------------
def _fetch_stats(season: int, api_key: Optional[str], logger) -> pd.DataFrame:
    session = build_retry_session(api_key=api_key, timeout_seconds=120, total_retries=6)

    logger.info(f"Fetching CFBD team advanced *game* stats for season={season}")
    resp = cfbd_get(session, API_PATH, params={"year": season})
    if resp.status_code >= 400:
        raise RuntimeError(f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CFBD response type for advanced game stats: {type(data)}")

    df = pd.json_normalize(data, sep="_")
    logger.info(f"Retrieved {len(df)} rows from CFBD.")
    return df


def _transform_df(df: pd.DataFrame) -> pd.DataFrame:
    # Map top-level keys to our schema names
    df2 = df.copy()
    rename_map = {
        "gameId": "game_id",
        "seasonType": "season_type",
    }
    df2.rename(columns=rename_map, inplace=True)

    # Convert CFBD camelCase within flattened names to table snake_case
    to_table = {
        "offense_totalPPA":"offense_totalppa",
        "offense_successRate":"offense_successrate",
        "offense_powerSuccess":"offense_powersuccess",
        "offense_stuffRate":"offense_stuffrate",
        "offense_lineYards":"offense_lineyards",
        "offense_lineYardsTotal":"offense_lineyardstotal",
        "offense_secondLevelYards":"offense_secondlevelyards",
        "offense_secondLevelYardsTotal":"offense_secondlevelyardstotal",
        "offense_openFieldYards":"offense_openfieldyards",
        "offense_openFieldYardsTotal":"offense_openfieldyardstotal",
        "offense_standardDowns_ppa":"offense_standarddowns_ppa",
        "offense_standardDowns_successRate":"offense_standarddowns_successrate",
        "offense_standardDowns_explosiveness":"offense_standarddowns_explosiveness",
        "offense_passingDowns_ppa":"offense_passingdowns_ppa",
        "offense_passingDowns_successRate":"offense_passingdowns_successrate",
        "offense_passingDowns_explosiveness":"offense_passingdowns_explosiveness",
        "offense_rushingPlays_ppa":"offense_rushingplays_ppa",
        "offense_rushingPlays_totalPPA":"offense_rushingplays_totalppa",
        "offense_rushingPlays_successRate":"offense_rushingplays_successrate",
        "offense_rushingPlays_explosiveness":"offense_rushingplays_explosiveness",
        "offense_passingPlays_ppa":"offense_passingplays_ppa",
        "offense_passingPlays_totalPPA":"offense_passingplays_totalppa",
        "offense_passingPlays_successRate":"offense_passingplays_successrate",
        "offense_passingPlays_explosiveness":"offense_passingplays_explosiveness",

        "defense_totalPPA":"defense_totalppa",
        "defense_successRate":"defense_successrate",
        "defense_powerSuccess":"defense_powersuccess",
        "defense_stuffRate":"defense_stuffrate",
        "defense_lineYards":"defense_lineyards",
        "defense_lineYardsTotal":"defense_lineyardstotal",
        "defense_secondLevelYards":"defense_secondlevelyards",
        "defense_secondLevelYardsTotal":"defense_secondlevelyardstotal",
        "defense_openFieldYards":"defense_openfieldyards",
        "defense_openFieldYardsTotal":"defense_openfieldyardstotal",
        "defense_standardDowns_ppa":"defense_standarddowns_ppa",
        "defense_standardDowns_successRate":"defense_standarddowns_successrate",
        "defense_standardDowns_explosiveness":"defense_standarddowns_explosiveness",
        "defense_passingDowns_ppa":"defense_passingdowns_ppa",
        "defense_passingDowns_successRate":"defense_passingdowns_successrate",
        "defense_passingDowns_explosiveness":"defense_passingdowns_explosiveness",
        "defense_rushingPlays_ppa":"defense_rushingplays_ppa",
        "defense_rushingPlays_totalPPA":"defense_rushingplays_totalppa",
        "defense_rushingPlays_successRate":"defense_rushingplays_successrate",
        "defense_rushingPlays_explosiveness":"defense_rushingplays_explosiveness",
        "defense_passingPlays_ppa":"defense_passingplays_ppa",
        "defense_passingPlays_totalPPA":"defense_passingplays_totalppa",
        "defense_passingPlays_successRate":"defense_passingplays_successrate",
        "defense_passingPlays_explosiveness":"defense_passingplays_explosiveness",
    }
    df2.rename(columns=to_table, inplace=True)

    # Ensure every expected column exists; add missing as NaN
    for c in FINAL_COLS:
        if c not in df2.columns:
            df2[c] = np.nan

    # Keep only FINAL_COLS
    df2 = df2[FINAL_COLS]
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
            cur.execute("DELETE FROM public.team_advanced_game_stats WHERE season = %s;", (season,))
            deleted = int(cur.rowcount or 0)
            logger.info(f"Deleted {deleted} existing rows from public.team_advanced_game_stats for season={season}")

            # copy insert
            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            # count inserted after copy
            cur.execute("SELECT COUNT(*) FROM public.team_advanced_game_stats WHERE season=%s;", (season,))
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
            df = _transform_df(raw_df)

        # Early season can legitimately have no advanced game stats yet.
        # Skip before load so we never delete existing data for an empty payload.
        with log_timing(logger, f"{prefix} validate"):
            if df.empty:
                msg = f"season={cfg.season} no advanced game stats returned; skipping without DB changes"
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
