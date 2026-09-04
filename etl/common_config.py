from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ETLConfig:
    pg_dsn: str
    cfbd_api_key: Optional[str]
    season: int
    run_id: str


def _infer_current_cfb_season(today_utc: datetime) -> int:
    """
    Simple rule-of-thumb:
    - CFB season is the calendar year of the fall (Aug–Jan).
    - If it's Jan/Feb/Mar, you're still likely in the prior season's postseason.
    Adjust if you prefer a different cutoff.
    """
    year = today_utc.year
    if today_utc.month <= 3:
        return year - 1
    return year


def _normalize_pg_dsn(pg_dsn: str) -> str:
    """
    Accept SQLAlchemy-style Postgres URLs from the dashboard in ETL jobs that
    connect directly with psycopg.
    """
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if pg_dsn.startswith(prefix):
            return "postgresql://" + pg_dsn.removeprefix(prefix)
    return pg_dsn


def _load_local_secrets_env() -> None:
    secrets_path = Path(__file__).resolve().with_name("secrets.env")
    if not secrets_path.exists():
        return

    for raw in secrets_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def load_config() -> ETLConfig:
    _load_local_secrets_env()

    pg_dsn = os.getenv("PG_DSN", "").strip() or os.getenv("NEON_DATABASE_URL", "").strip()
    pg_dsn = _normalize_pg_dsn(pg_dsn)
    if not pg_dsn:
        raise RuntimeError("PG_DSN or NEON_DATABASE_URL env var is required.")

    cfbd_api_key = os.getenv("CFBD_API_KEY", "").strip() or None

    season_raw = os.getenv("SEASON", "").strip()
    if season_raw:
        season = int(season_raw)
    else:
        season = _infer_current_cfb_season(datetime.utcnow())

    run_id = os.getenv("RUN_ID", "").strip() or uuid.uuid4().hex[:10]

    return ETLConfig(
        pg_dsn=pg_dsn,
        cfbd_api_key=cfbd_api_key,
        season=season,
        run_id=run_id,
    )
