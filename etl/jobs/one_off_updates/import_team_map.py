#!/usr/bin/env python3
"""
ETL Job: import_team_map

Fetch CFBD team metadata for a season and fully refresh public.team_map.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import pandas as pd
import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session, cfbd_get
from etl.common_logging import format_step_prefix, log_timing, setup_logging
from etl.common_types import StepResult


STEP_NAME = "import_team_map"
API_PATH = "/teams"
TABLE = "public.team_map"

COLS = [
    "Season",
    "Id",
    "School",
    "Mascot",
    "Abbreviation",
    "AlternateNames",
    "Conference",
    "Division",
    "Classification",
    "Color",
    "AlternateColor",
    "Logo",
    "Logo_Dark",
    "Twitter",
    "Location_Id",
    "Location_Name",
    "Location_City",
    "Location_State",
    "Location_Zip",
    "Location_CountryCode",
    "Location_Timezone",
    "Location_Latitude",
    "Location_Longitude",
    "Location_Elevation",
    "Location_Capacity",
    "Location_ConstructionYear",
    "Location_Grass",
    "Location_Dome",
]

INTEGER_COLS = [
    "Season",
    "Id",
    "Location_Id",
    "Location_Capacity",
    "Location_ConstructionYear",
]
FLOAT_COLS = ["Location_Latitude", "Location_Longitude", "Location_Elevation"]
NUMERIC_COLS = INTEGER_COLS + FLOAT_COLS
BOOLEAN_COLS = ["Location_Grass", "Location_Dome"]
FORCE_NULL_SQL = ", ".join(f'"{c}"' for c in NUMERIC_COLS + BOOLEAN_COLS)

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    "Season" INT NOT NULL,
    "Id" BIGINT,
    "School" TEXT,
    "Mascot" TEXT,
    "Abbreviation" TEXT,
    "AlternateNames" TEXT,
    "Conference" TEXT,
    "Division" TEXT,
    "Classification" TEXT,
    "Color" TEXT,
    "AlternateColor" TEXT,
    "Logo" TEXT,
    "Logo_Dark" TEXT,
    "Twitter" TEXT,
    "Location_Id" BIGINT,
    "Location_Name" TEXT,
    "Location_City" TEXT,
    "Location_State" TEXT,
    "Location_Zip" TEXT,
    "Location_CountryCode" TEXT,
    "Location_Timezone" TEXT,
    "Location_Latitude" DOUBLE PRECISION,
    "Location_Longitude" DOUBLE PRECISION,
    "Location_Elevation" DOUBLE PRECISION,
    "Location_Capacity" INT,
    "Location_ConstructionYear" INT,
    "Location_Grass" BOOLEAN,
    "Location_Dome" BOOLEAN
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


def _pick(d: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def _join_names(team: dict[str, Any]) -> str | None:
    names = _pick(team, "alternateNames", "alternate_names", "altNames")
    if isinstance(names, list):
        return ",".join(str(v) for v in names if v)
    if names:
        return str(names)

    alt_names = [
        _pick(team, "alt_name1", "altName1"),
        _pick(team, "alt_name2", "altName2"),
        _pick(team, "alt_name3", "altName3"),
    ]
    out = [str(v) for v in alt_names if v]
    return ",".join(out) if out else None


def _split_logos(team: dict[str, Any]) -> tuple[str | None, str | None]:
    logos = team.get("logos") or team.get("Logos") or []
    if isinstance(logos, str):
        logos = [v.strip() for v in logos.split(",") if v.strip()]
    if not isinstance(logos, list):
        return None, None

    logo = next((str(v) for v in logos if v and "500-dark" not in str(v).lower()), None)
    logo_dark = next((str(v) for v in logos if v and "500-dark" in str(v).lower()), None)

    if logo is None and logos:
        logo = str(logos[0])
    if logo_dark is None and len(logos) > 1:
        logo_dark = str(logos[1])

    return logo, logo_dark


def _fetch_team_map(season: int, api_key: Optional[str], logger) -> list[dict[str, Any]]:
    session = build_retry_session(api_key=api_key, timeout_seconds=60, total_retries=6)

    logger.info(f"Fetching CFBD teams for season={season}")
    resp = cfbd_get(session, API_PATH, params={"year": season})
    if resp.status_code >= 400:
        raise RuntimeError(f"CFBD GET {API_PATH} failed: status={resp.status_code} body={resp.text[:500]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected CFBD response type for teams: {type(data)}")

    logger.info(f"Retrieved {len(data)} rows from CFBD.")
    return data


def _transform_rows(payload: list[dict[str, Any]], season: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for team in payload:
        location = team.get("location") or {}
        logo, logo_dark = _split_logos(team)
        rows.append(
            {
                "Season": season,
                "Id": _pick(team, "id", "Id"),
                "School": _pick(team, "school", "School"),
                "Mascot": _pick(team, "mascot", "Mascot"),
                "Abbreviation": _pick(team, "abbreviation", "Abbreviation"),
                "AlternateNames": _join_names(team),
                "Conference": _pick(team, "conference", "Conference"),
                "Division": _pick(team, "division", "Division"),
                "Classification": _pick(team, "classification", "Classification"),
                "Color": _pick(team, "color", "Color"),
                "AlternateColor": _pick(team, "alternateColor", "alternate_color", "AlternateColor"),
                "Logo": logo,
                "Logo_Dark": logo_dark,
                "Twitter": _pick(team, "twitter", "Twitter"),
                "Location_Id": _pick(location, "venue_id", "id", "venueId", "Id"),
                "Location_Name": _pick(location, "name", "Name"),
                "Location_City": _pick(location, "city", "City"),
                "Location_State": _pick(location, "state", "State"),
                "Location_Zip": _pick(location, "zip", "Zip"),
                "Location_CountryCode": _pick(location, "country_code", "countryCode", "CountryCode"),
                "Location_Timezone": _pick(location, "timezone", "Timezone"),
                "Location_Latitude": _pick(location, "latitude", "Latitude"),
                "Location_Longitude": _pick(location, "longitude", "Longitude"),
                "Location_Elevation": _pick(location, "elevation", "Elevation"),
                "Location_Capacity": _pick(location, "capacity", "Capacity"),
                "Location_ConstructionYear": _pick(
                    location,
                    "construction_year",
                    "constructionYear",
                    "ConstructionYear",
                ),
                "Location_Grass": _pick(location, "grass", "Grass"),
                "Location_Dome": _pick(location, "dome", "Dome"),
            }
        )

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
        raise ValueError("CFBD returned 0 team rows; refusing to refresh team_map.")

    if df["Id"].isna().all():
        raise ValueError("All team Id values are NULL; CFBD schema likely changed. Aborting.")

    dupes = df.duplicated(subset=["Season", "Id"]).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate rows on key (Season, Id); aborting.")


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
    cfg = load_config()
    logger = setup_logging()
    prefix = format_step_prefix(cfg.run_id, STEP_NAME)

    logger.info(f"{prefix} start (season={cfg.season})")

    try:
        with log_timing(logger, f"{prefix} fetch"):
            payload = _fetch_team_map(cfg.season, cfg.cfbd_api_key, logger)

        with log_timing(logger, f"{prefix} transform"):
            df = _transform_rows(payload, cfg.season)

        with log_timing(logger, f"{prefix} validate"):
            if df.empty:
                _ensure_table(cfg.pg_dsn, logger)
                msg = f"season={cfg.season} no team rows returned; skipping refresh"
                logger.info(f"{prefix} skipped | {msg}")
                return StepResult(step_name=STEP_NAME, season=cfg.season, status="skipped", message=msg)
            _validate_df(df)

        with log_timing(logger, f"{prefix} load"):
            deleted, inserted = _refresh_table(cfg.pg_dsn, df, logger)

        msg = f"season={cfg.season} deleted={deleted} inserted={inserted}"
        logger.info(f"{prefix} success | {msg}")
        return StepResult(
            step_name=STEP_NAME,
            season=cfg.season,
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
