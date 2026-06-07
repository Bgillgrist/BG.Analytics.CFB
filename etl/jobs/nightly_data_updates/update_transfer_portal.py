#!/usr/bin/env python3
"""
ETL Job: update_transfer_portal

Fetch transfer portal data from CFBD and replace one season in Neon.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Optional

import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import format_step_prefix, log_timing, setup_logging
from etl.common_types import StepResult


STEP_NAME = "update_transfer_portal"
API_PATH = "/player/portal"
TABLE = "public.transfer_portal"

COLS = [
    "Season",
    "FirstName",
    "LastName",
    "Position",
    "Origin",
    "Destination",
    "TransferDate",
    "Rating",
    "Stars",
    "Eligibility",
]
NUMERIC_OR_DATE_COLS = ["Season", "TransferDate", "Rating", "Stars"]
FORCE_NULL_SQL = ", ".join(f'"{c}"' for c in NUMERIC_OR_DATE_COLS)

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

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  "Season" INTEGER NOT NULL,
  "FirstName" TEXT,
  "LastName" TEXT,
  "Position" TEXT,
  "Origin" TEXT,
  "Destination" TEXT,
  "TransferDate" TIMESTAMPTZ,
  "Rating" DOUBLE PRECISION,
  "Stars" INTEGER,
  "Eligibility" TEXT
);

ALTER TABLE {TABLE}
  ADD COLUMN IF NOT EXISTS "Season" INTEGER,
  ADD COLUMN IF NOT EXISTS "FirstName" TEXT,
  ADD COLUMN IF NOT EXISTS "LastName" TEXT,
  ADD COLUMN IF NOT EXISTS "Position" TEXT,
  ADD COLUMN IF NOT EXISTS "Origin" TEXT,
  ADD COLUMN IF NOT EXISTS "Destination" TEXT,
  ADD COLUMN IF NOT EXISTS "TransferDate" TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS "Rating" DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS "Stars" INTEGER,
  ADD COLUMN IF NOT EXISTS "Eligibility" TEXT;

CREATE INDEX IF NOT EXISTS idx_transfer_portal_season
ON {TABLE} ("Season");
"""


def _pick(d: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _stringify_if_structured(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _cfbd_to_row(player: Dict[str, Any], season: int) -> Dict[str, Any]:
    row = {
        "Season": _pick(player, "season", "Season") or season,
        "FirstName": _pick(player, "firstName", "first_name", "FirstName"),
        "LastName": _pick(player, "lastName", "last_name", "LastName"),
        "Position": _pick(player, "position", "Position"),
        "Origin": _pick(player, "origin", "Origin"),
        "Destination": _pick(player, "destination", "Destination"),
        "TransferDate": _pick(player, "transferDate", "transfer_date", "TransferDate"),
        "Rating": _pick(player, "rating", "Rating"),
        "Stars": _pick(player, "stars", "Stars"),
        "Eligibility": _pick(player, "eligibility", "Eligibility"),
    }
    row["Eligibility"] = _stringify_if_structured(row["Eligibility"])
    return row


def _fetch_transfer_portal(season: int, api_key: Optional[str], logger) -> List[Dict[str, Any]]:
    session = build_retry_session(api_key=api_key, timeout_seconds=60, total_retries=6)

    logger.info(f"Fetching CFBD transfer portal data for season={season}")
    resp = cfbd_get(session, API_PATH, params={"year": season})
    if resp.status_code >= 400:
        raise RuntimeError(f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CFBD response type for transfer portal: {type(data)}")

    rows = [_cfbd_to_row(player, season) for player in data]
    logger.info(f"Fetched {len(rows)} transfer portal rows from CFBD.")
    return rows


def _validate_rows(rows: List[Dict[str, Any]], season: int) -> None:
    for row in rows:
        missing = [c for c in COLS if c not in row]
        if missing:
            raise ValueError(f"Mapped transfer portal row missing keys for: {missing}")

    wrong_seasons = sorted(
        {
            int(row["Season"])
            for row in rows
            if row.get("Season") not in (None, "") and int(row["Season"]) != season
        }
    )
    if wrong_seasons:
        raise ValueError(f"CFBD payload contained seasons other than {season}: {wrong_seasons}")


def _to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLS})
    return buf.getvalue().encode("utf-8")


def _replace_season_partition(
    pg_dsn: str,
    season: int,
    rows: List[Dict[str, Any]],
    logger,
) -> tuple[int, int]:
    csv_bytes = _to_csv_bytes(rows)

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")

            cur.execute(CREATE_TABLE_SQL)

            cur.execute(f'DELETE FROM {TABLE} WHERE "Season" = %s;', (season,))
            deleted = int(cur.rowcount or 0)
            logger.info(f"Deleted {deleted} existing rows from {TABLE} for Season={season}")

            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            cur.execute(f'SELECT COUNT(*) FROM {TABLE} WHERE "Season" = %s;', (season,))
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
            rows = _fetch_transfer_portal(cfg.season, cfg.cfbd_api_key, logger)

        with log_timing(logger, f"{prefix} validate"):
            _validate_rows(rows, cfg.season)

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
