#!/usr/bin/env python3
import os
import sys
import io
import csv
import time
import psycopg
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List, Iterable, Optional

# ── Config ─────────────────────────────────────────────────────────────────────
PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://neondb_owner:npg_Up1blnYS0oxs@ep-delicate-brook-a4yhfswb-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require",
)
CFBD_API_KEY = os.getenv("CFBD_API_KEY")  # REQUIRED
BETTING_TABLE = "public.betting_odds"

# Choose season: env var or current year
SEASON = int(os.getenv("SEASON", str(datetime.now(timezone.utc).year)))

# CFBD endpoints
CFBD_BASE = "https://api.collegefootballdata.com"
CFBD_LINES = f"{CFBD_BASE}/lines"

# Backoff/retry for CFBD
MAX_TRIES = 3
SLEEP_SECONDS = 1.5

# ── COPY template that matches your table (quoted to preserve CSV header case) ──
COPY_SQL = f"""
COPY {BETTING_TABLE} (
  "Id",
  "HomeTeam",
  "HomeScore",
  "AwayTeam",
  "AwayScore",
  "FormattedSpread",
  "LineProvider",
  "OverUnder",
  "Spread",
  "OpeningSpread",
  "OpeningOverUnder",
  "HomeMoneyline",
  "AwayMoneyline"
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL (
    "HomeScore",
    "AwayScore",
    "OverUnder",
    "Spread",
    "OpeningSpread",
    "OpeningOverUnder",
    "HomeMoneyline",
    "AwayMoneyline"
  )
)
"""

# ── Helpers ────────────────────────────────────────────────────────────────────
def _cfbd_get(url: str, params: Dict[str, Any]) -> Any:
    if not CFBD_API_KEY:
        raise RuntimeError("CFBD_API_KEY env var is required for CFBD requests.")
    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            # 429 or 5xx: back off
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(SLEEP_SECONDS * attempt)
                continue
            r.raise_for_status()
        except Exception as e:
            last_exc = e
            time.sleep(SLEEP_SECONDS * attempt)
    # If we’re here, give up
    raise RuntimeError(f"CFBD request failed after retries: {url} params={params} err={last_exc}")

def fetch_lines_for_season(season: int) -> List[Dict[str, Any]]:
    """
    Pulls all available line snapshots for the given season.
    CFBD returns an array; each element contains game info and a 'lines' array (per provider).
    We’ll fetch both 'regular' and 'postseason', weeks 1..20 (stop early on empty).
    """
    print(f"• Fetching CFBD lines for season {season} …")
    all_rows: List[Dict[str, Any]] = []
    for season_type in ("regular", "postseason"):
        for week in range(1, 21):
            params = {"year": season, "seasonType": season_type, "week": week}
            data = _cfbd_get(CFBD_LINES, params)
            if not data:
                # Stop early if we’ve gone past available weeks for this season type
                if season_type == "regular":
                    # continue to postseason later
                    pass
                else:
                    # postseason: likely done
                    pass
                continue
            for game in data:
                # Some fields may be absent in early weeks or incomplete records
                game_id = game.get("id")
                home = game.get("home_team")
                away = game.get("away_team")
                home_pts = game.get("home_points")
                away_pts = game.get("away_points")

                lines = game.get("lines") or []
                if not lines:
                    # No provider lines — we can still emit a row with NULLs so Id gets tracked
                    all_rows.append({
                        "Id": game_id,
                        "HomeTeam": home,
                        "HomeScore": home_pts,
                        "AwayTeam": away,
                        "AwayScore": away_pts,
                        "FormattedSpread": None,
                        "LineProvider": None,
                        "OverUnder": None,
                        "Spread": None,
                        "OpeningSpread": None,
                        "OpeningOverUnder": None,
                        "HomeMoneyline": None,
                        "AwayMoneyline": None,
                    })
                    continue

                for ln in lines:
                    # Provider-specific fields (presence varies by provider/snapshot)
                    provider = ln.get("provider")
                    spread = ln.get("spread")                 # numeric, home-based usually
                    over_under = ln.get("overUnder")
                    formatted_spread = ln.get("formattedSpread")  # sometimes present
                    # These often missing in CFBD; keep if provided by some providers
                    opening_spread = ln.get("spreadOpen")
                    opening_ou = ln.get("overUnderOpen")
                    money_home = ln.get("homeMoneyline")
                    money_away = ln.get("awayMoneyline")

                    all_rows.append({
                        "Id": game_id,
                        "HomeTeam": home,
                        "HomeScore": home_pts,
                        "AwayTeam": away,
                        "AwayScore": away_pts,
                        "FormattedSpread": formatted_spread,
                        "LineProvider": provider,
                        "OverUnder": over_under,
                        "Spread": spread,
                        "OpeningSpread": opening_spread,
                        "OpeningOverUnder": opening_ou,
                        "HomeMoneyline": money_home,
                        "AwayMoneyline": money_away,
                    })
        # loop season_type
    print(f"  → fetched {len(all_rows):,} line rows (provider×game)")
    return all_rows

def rows_to_csv_buffer(rows: List[Dict[str, Any]]) -> io.StringIO:
    """
    Convert rows to an in-memory CSV matching the DB table schema exactly.
    Postgres COPY will coerce '' (empty string) to NULL thanks to FORCE_NULL above.
    """
    headers = [
        "Id",
        "HomeTeam",
        "HomeScore",
        "AwayTeam",
        "AwayScore",
        "FormattedSpread",
        "LineProvider",
        "OverUnder",
        "Spread",
        "OpeningSpread",
        "OpeningOverUnder",
        "HomeMoneyline",
        "AwayMoneyline",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=headers, lineterminator="\n")
    w.writeheader()
    for r in rows:
        # normalize None -> '' for numeric/text fields; COPY FORCE_NULL will handle numerics
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in headers})
    buf.seek(0)
    return buf

def iter_id_batches(rows: List[Dict[str, Any]], batch_size: int = 10_000) -> Iterable[List[int]]:
    batch: List[int] = []
    for r in rows:
        v = r.get("Id")
        if v is None:
            continue
        try:
            v = int(v)
        except Exception:
            continue
        batch.append(v)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# ── Main ETL ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"🔄 Nightly betting ETL (CFBD) starting…  season={SEASON}")
    rows = fetch_lines_for_season(SEASON)

    if not rows:
        print("  ! No rows returned from CFBD; aborting without DB changes.")
        return

    # Prepare CSV buffer for COPY
    csv_buf = rows_to_csv_buffer(rows)

    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            # 1) Temp table of Ids to delete/replace
            print("• Creating temp table temp_betting_ids …")
            cur.execute('CREATE TEMP TABLE temp_betting_ids ("Id" BIGINT) ON COMMIT DROP;')

            print("• Inserting Ids into temp table …")
            total_ids = 0
            for batch in iter_id_batches(rows):
                cur.executemany('INSERT INTO temp_betting_ids ("Id") VALUES (%s)', [(i,) for i in batch])
                total_ids += len(batch)
            print(f"  → Inserted {total_ids:,} Ids into temp table")

            # 2) Delete existing rows for those Ids
            print("• Deleting existing rows in betting_odds for those Ids …")
            cur.execute(f'''
                DELETE FROM {BETTING_TABLE} bo
                USING temp_betting_ids t
                WHERE bo."Id" = t."Id";
            ''')
            print(f"  → Deleted {cur.rowcount:,} rows")

            # 3) COPY new rows
            print("• COPY fresh CFBD rows into betting_odds …")
            with cur.copy(COPY_SQL) as cp:
                # Stream from in-memory CSV buffer in chunks
                CHUNK = 1_000_000
                while True:
                    chunk = csv_buf.read(CHUNK)
                    if not chunk:
                        break
                    cp.write(chunk)

            # 4) Commit and report
            conn.commit()
            cur.execute(f'SELECT COUNT(*) FROM {BETTING_TABLE}')
            final_count = cur.fetchone()[0]
            print(f"✅ Load complete. betting_odds row count is now: {final_count:,}")

    print("🏁 Nightly betting ETL done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ ETL failed: {e}")
        sys.exit(1)
