#!/usr/bin/env python3
"""
ETL Job: import_returning_production

Fetch team returning production from CFBD and replace one season in Neon.
"""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import format_step_prefix, log_timing, setup_logging
from etl.common_types import StepResult


STEP_NAME = "import_returning_production"
API_PATH = "/player/returning"
TABLE = "public.team_returning_production"

COLS = [
    "season",
    "team",
    "conference",
    "total_ppa",
    "total_passing_ppa",
    "total_receiving_ppa",
    "total_rushing_ppa",
    "percent_ppa",
    "percent_passing_ppa",
    "percent_receiving_ppa",
    "percent_rushing_ppa",
    "usage",
    "passing_usage",
    "receiving_usage",
    "rushing_usage",
]
TEXT_COLS = {"team", "conference"}
NUMERIC_COLS = [c for c in COLS if c not in TEXT_COLS]
FORCE_NULL_SQL = ", ".join(NUMERIC_COLS)

COPY_SQL = f"""
COPY {TABLE} (
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

RENAMES = {
    "totalPPA": "total_ppa",
    "totalPassingPPA": "total_passing_ppa",
    "totalReceivingPPA": "total_receiving_ppa",
    "totalRushingPPA": "total_rushing_ppa",
    "percentPPA": "percent_ppa",
    "percentPassingPPA": "percent_passing_ppa",
    "percentReceivingPPA": "percent_receiving_ppa",
    "percentRushingPPA": "percent_rushing_ppa",
    "passingUsage": "passing_usage",
    "receivingUsage": "receiving_usage",
    "rushingUsage": "rushing_usage",
}


def _fetch_returning_production(season: int, api_key: Optional[str], logger) -> pd.DataFrame:
    session = build_retry_session(api_key=api_key, timeout_seconds=60, total_retries=6)

    logger.info(f"Fetching CFBD returning production for season={season}")
    resp = cfbd_get(session, API_PATH, params={"year": season})
    if resp.status_code >= 400:
        raise RuntimeError(f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CFBD response type for returning production: {type(data)}")

    df = pd.json_normalize(data, sep="_")
    logger.info(f"Retrieved {len(df)} rows from CFBD.")
    return df


def _transform_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    df2 = df.copy()
    df2.rename(columns=RENAMES, inplace=True)

    if "season" not in df2.columns:
        df2["season"] = season

    for c in COLS:
        if c not in df2.columns:
            df2[c] = None

    df2 = df2[COLS]
    if df2.empty:
        return df2

    df2["season"] = pd.to_numeric(df2["season"], errors="raise").astype(int)
    for c in NUMERIC_COLS:
        if c == "season":
            continue
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2


def _validate_df(df: pd.DataFrame, season: int) -> None:
    if df.empty:
        raise ValueError("CFBD returned 0 rows for returning production; refusing to delete/replace season data.")

    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    wrong_seasons = sorted(set(df.loc[df["season"].ne(season), "season"].dropna().astype(int).tolist()))
    if wrong_seasons:
        raise ValueError(f"CFBD payload contained seasons other than {season}: {wrong_seasons}")

    dupes = df.duplicated(subset=["season", "team"]).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate rows on key (season, team); aborting.")

    if df["team"].isna().any() or df["team"].astype(str).str.strip().eq("").any():
        raise ValueError("Found returning production rows with missing team; aborting.")


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _replace_season_partition(pg_dsn: str, season: int, df: pd.DataFrame, logger) -> tuple[int, int]:
    csv_bytes = _df_to_csv_bytes(df)

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")

            cur.execute(f"DELETE FROM {TABLE} WHERE season = %s;", (season,))
            deleted = int(cur.rowcount or 0)
            logger.info(f"Deleted {deleted} existing rows from {TABLE} for season={season}")

            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE season = %s;", (season,))
            inserted = int(cur.fetchone()[0])

            cur.execute("COMMIT;")
            return deleted, inserted


def run() -> StepResult:
    cfg = load_config()
    logger = setup_logging()
    prefix = format_step_prefix(cfg.run_id, STEP_NAME)

    logger.info(f"{prefix} start (season={cfg.season})")

    try:
        with log_timing(logger, f"{prefix} fetch"):
            raw_df = _fetch_returning_production(cfg.season, cfg.cfbd_api_key, logger)

        with log_timing(logger, f"{prefix} transform"):
            df = _transform_df(raw_df, cfg.season)

        with log_timing(logger, f"{prefix} validate"):
            if df.empty:
                msg = f"season={cfg.season} no returning production returned; skipping without DB changes"
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
            _validate_df(df, cfg.season)

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
