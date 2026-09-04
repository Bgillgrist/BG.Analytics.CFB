#!/usr/bin/env python3
"""
ETL Job: import_teamrankings_team_map

Refresh the TeamRankings-to-CFBD team-name map in Neon from a local CSV.
"""

from __future__ import annotations

import argparse
import io
import os
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
import psycopg

from etl.common_config import load_config
from etl.common_logging import format_step_prefix, log_timing, setup_logging
from etl.common_types import StepResult
from etl.jobs.nightly_data_updates.update_teamrankings_predictive import (
    CREATE_MAP_TABLE_SQL,
    MAP_TABLE,
    _normalize_map_key,
)


STEP_NAME = "import_teamrankings_team_map"
COLS = ["teamrankings_team", "cfbd_team", "active", "notes"]
SOURCE_COL_ALIASES = [
    "teamrankings_team",
    "teamrankings_name",
    "team_rankings_team",
    "teamrankings",
    "team rankings",
    "source_team",
    "source",
]
CFBD_COL_ALIASES = [
    "cfbd_team",
    "cfb_name",
    "cfbd",
    "team",
    "school",
    "canonical_team",
    "canonical",
]
ACTIVE_COL_ALIASES = ["active", "is_active", "enabled"]
NOTES_COL_ALIASES = ["notes", "note", "comment", "comments"]

COPY_SQL = f"""
COPY temp_teamrankings_team_map (
  {", ".join(COLS)}
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL (notes)
)
"""

CREATE_TEMP_TABLE_SQL = """
CREATE TEMP TABLE temp_teamrankings_team_map (
  teamrankings_team TEXT NOT NULL,
  cfbd_team TEXT NOT NULL,
  active BOOLEAN NOT NULL,
  notes TEXT
) ON COMMIT DROP;
"""

INSERT_FROM_TEMP_SQL = f"""
INSERT INTO {MAP_TABLE} (
  teamrankings_team,
  cfbd_team,
  active,
  notes,
  created_at,
  updated_at
)
SELECT
  teamrankings_team,
  cfbd_team,
  active,
  notes,
  NOW(),
  NOW()
FROM temp_teamrankings_team_map;
"""


def _normalize_col(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {_normalize_col(col): col for col in columns}
    for alias in aliases:
        found = normalized.get(_normalize_col(alias))
        if found is not None:
            return found
    return None


def _parse_active(value) -> bool:
    if pd.isna(value) or value == "":
        return True
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "t", "yes", "y", "active", "enabled"}:
        return True
    if text in {"0", "false", "f", "no", "n", "inactive", "disabled"}:
        return False
    raise ValueError(f"Could not parse active value {value!r}.")


def _read_map_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    source_col = _find_column(list(df.columns), SOURCE_COL_ALIASES)
    cfbd_col = _find_column(list(df.columns), CFBD_COL_ALIASES)
    active_col = _find_column(list(df.columns), ACTIVE_COL_ALIASES)
    notes_col = _find_column(list(df.columns), NOTES_COL_ALIASES)

    if source_col is None:
        raise ValueError(f"Map CSV is missing a TeamRankings team column. Tried: {SOURCE_COL_ALIASES}")
    if cfbd_col is None:
        raise ValueError(f"Map CSV is missing a CFBD team column. Tried: {CFBD_COL_ALIASES}")

    out = pd.DataFrame()
    out["teamrankings_team"] = df[source_col].astype("string").str.strip()
    out["cfbd_team"] = df[cfbd_col].astype("string").str.strip()
    out["active"] = df[active_col].map(_parse_active) if active_col else True
    out["notes"] = df[notes_col].astype("string").str.strip() if notes_col else None
    out["notes"] = out["notes"].replace({"": None, pd.NA: None})

    _validate_map_df(out)
    return out[COLS]


def _validate_map_df(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("TeamRankings team map CSV is empty; refusing to refresh map table.")

    missing_source = df["teamrankings_team"].isna() | df["teamrankings_team"].astype(str).str.strip().eq("")
    if missing_source.any():
        raise ValueError("Found team map rows with missing TeamRankings team names.")

    missing_cfbd = df["cfbd_team"].isna() | df["cfbd_team"].astype(str).str.strip().eq("")
    if missing_cfbd.any():
        raise ValueError("Found team map rows with missing CFBD team names.")

    normalized_sources = df["teamrankings_team"].astype(str).map(_normalize_map_key)
    dupes = normalized_sources.duplicated().sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate TeamRankings team map rows; aborting.")


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _refresh_map_table(pg_dsn: str, df: pd.DataFrame, logger) -> tuple[int, int]:
    csv_bytes = _df_to_csv_bytes(df)

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute(CREATE_MAP_TABLE_SQL)
            cur.execute(CREATE_TEMP_TABLE_SQL)

            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            cur.execute(f"DELETE FROM {MAP_TABLE};")
            deleted = int(cur.rowcount or 0)
            logger.info(f"Deleted {deleted} existing rows from {MAP_TABLE}")

            cur.execute(INSERT_FROM_TEMP_SQL)
            inserted = int(cur.rowcount or 0)
            cur.execute("COMMIT;")
            return deleted, inserted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import TeamRankings-to-CFBD team-name map.")
    parser.add_argument(
        "--csv-path",
        default=os.getenv("TEAMRANKINGS_TEAM_MAP_CSV", "").strip() or None,
        help="Path to a CSV with TeamRankings and CFBD team-name columns.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> StepResult:
    args = _build_parser().parse_args(argv)
    cfg = load_config()
    logger = setup_logging()
    prefix = format_step_prefix(cfg.run_id, STEP_NAME)

    if not args.csv_path:
        raise RuntimeError("Pass --csv-path or set TEAMRANKINGS_TEAM_MAP_CSV.")

    logger.info(f"{prefix} start (csv_path={args.csv_path})")

    try:
        with log_timing(logger, f"{prefix} read"):
            df = _read_map_csv(args.csv_path)

        with log_timing(logger, f"{prefix} load"):
            deleted, inserted = _refresh_map_table(cfg.pg_dsn, df, logger)

        msg = f"deleted={deleted} inserted={inserted}"
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
