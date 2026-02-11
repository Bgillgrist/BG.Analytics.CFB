from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
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


def load_config() -> ETLConfig:
    pg_dsn = os.getenv("PG_DSN", "").strip()
    if not pg_dsn:
        raise RuntimeError("PG_DSN env var is required.")

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