#!/usr/bin/env python3
"""
ETL Job: import_venue_map

Fetch CFBD venue metadata and fully refresh public.venue_map.
"""

from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import psycopg

from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import format_step_prefix, log_timing, setup_logging
from etl.common_types import StepResult


STEP_NAME = "import_venue_map"
API_PATH = "/venues"
TABLE = "public.venue_map"
DUMMY_SEASON = 0

COLS = [
    "Id",
    "Name",
    "Capacity",
    "Grass",
    "Dome",
    "City",
    "State",
    "Zip",
    "CountryCode",
    "Timezone",
    "Latitude",
    "Longitude",
    "Elevation",
    "ConstructionYear",
]

INTEGER_COLS = ["Id", "Capacity", "ConstructionYear"]
FLOAT_COLS = ["Latitude", "Longitude", "Elevation"]
NUMERIC_COLS = INTEGER_COLS + FLOAT_COLS
BOOLEAN_COLS = ["Grass", "Dome"]
FORCE_NULL_SQL = ", ".join(f'"{c}"' for c in NUMERIC_COLS + BOOLEAN_COLS)

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    "Id" BIGINT,
    "Name" TEXT,
    "Capacity" INT,
    "Grass" BOOLEAN,
    "Dome" BOOLEAN,
    "City" TEXT,
    "State" TEXT,
    "Zip" TEXT,
    "CountryCode" TEXT,
    "Timezone" TEXT,
    "Latitude" DOUBLE PRECISION,
    "Longitude" DOUBLE PRECISION,
    "Elevation" DOUBLE PRECISION,
    "ConstructionYear" INT
);
"""

COPY_SQL = f"""
COPY {TABLE} (
  {", ".join(f'"{c}"' for c in COLS)}
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL ({FORCE_NULL_SQL})
)
"""


@dataclass(frozen=True)
class VenueConfig:
    pg_dsn: str
    cfbd_api_key: Optional[str]
    run_id: str


def _load_config() -> VenueConfig:
    pg_dsn = os.getenv("PG_DSN", "").strip()
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var is required.")
    return VenueConfig(
        pg_dsn=pg_dsn,
        cfbd_api_key=os.getenv("CFBD_API_KEY", "").strip() or None,
        run_id=os.getenv("RUN_ID", "").strip() or uuid.uuid4().hex[:10],
    )


def _pick(d: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _fetch_venue_map(api_key: Optional[str], logger) -> list[dict[str, Any]]:
    session = build_retry_session(api_key=api_key, timeout_seconds=60, total_retries=6)

    logger.info("Fetching CFBD venues")
    resp = cfbd_get(session, API_PATH)
    if resp.status_code >= 400:
        raise RuntimeError(f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CFBD response type for venues: {type(data)}")

    logger.info(f"Retrieved {len(data)} rows from CFBD.")
    return data


def _transform_rows(payload: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Id": _pick(venue, "id", "Id"),
            "Name": _pick(venue, "name", "Name"),
            "Capacity": _pick(venue, "capacity", "Capacity"),
            "Grass": _pick(venue, "grass", "Grass"),
            "Dome": _pick(venue, "dome", "Dome"),
            "City": _pick(venue, "city", "City"),
            "State": _pick(venue, "state", "State"),
            "Zip": _pick(venue, "zip", "Zip"),
            "CountryCode": _pick(venue, "country_code", "countryCode", "CountryCode"),
            "Timezone": _pick(venue, "timezone", "Timezone"),
            "Latitude": _pick(venue, "latitude", "Latitude"),
            "Longitude": _pick(venue, "longitude", "Longitude"),
            "Elevation": _pick(venue, "elevation", "Elevation"),
            "ConstructionYear": _pick(venue, "construction_year", "constructionYear", "ConstructionYear"),
        }
        for venue in payload
    ]

    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return df

    for col in INTEGER_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _validate_df(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("CFBD returned 0 venue rows; refusing to refresh venue_map.")

    if df["Id"].isna().all():
        raise ValueError("All venue Id values are NULL; CFBD schema likely changed. Aborting.")

    dupes = df.duplicated(subset=["Id"]).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate rows on key (Id); aborting.")


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _ensure_table(pg_dsn: str, logger) -> None:
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
    logger.info(f"Ensured {TABLE} exists")


def _refresh_table(pg_dsn: str, df: pd.DataFrame, logger) -> tuple[int, int]:
    csv_bytes = _df_to_csv_bytes(df)

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute(CREATE_TABLE_SQL)

            cur.execute(f"DELETE FROM {TABLE};")
            deleted = int(cur.rowcount or 0)
            logger.info(f"Deleted {deleted} existing rows from {TABLE}")

            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
            inserted = int(cur.fetchone()[0])

            cur.execute("COMMIT;")
            return deleted, inserted


def run() -> StepResult:
    cfg = _load_config()
    logger = setup_logging()
    prefix = format_step_prefix(cfg.run_id, STEP_NAME)

    logger.info(f"{prefix} start")

    try:
        with log_timing(logger, f"{prefix} fetch"):
            payload = _fetch_venue_map(cfg.cfbd_api_key, logger)

        with log_timing(logger, f"{prefix} transform"):
            df = _transform_rows(payload)

        with log_timing(logger, f"{prefix} validate"):
            if df.empty:
                _ensure_table(cfg.pg_dsn, logger)
                msg = "no venue rows returned; skipping refresh"
                logger.info(f"{prefix} skipped | {msg}")
                return StepResult(step_name=STEP_NAME, season=DUMMY_SEASON, status="skipped", message=msg)
            _validate_df(df)

        with log_timing(logger, f"{prefix} load"):
            deleted, inserted = _refresh_table(cfg.pg_dsn, df, logger)

        msg = f"deleted={deleted} inserted={inserted}"
        logger.info(f"{prefix} success | {msg}")
        return StepResult(
            step_name=STEP_NAME,
            season=DUMMY_SEASON,
            status="success",
            rows_fetched=len(payload),
            rows_deleted=deleted,
            rows_inserted=inserted,
            message=msg,
        )

    except Exception as e:
        logger.exception(f"{prefix} FAILED: {e}")
        return StepResult(
            step_name=STEP_NAME,
            season=DUMMY_SEASON,
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
