import os
import requests
import psycopg
import pandas as pd
import datetime as dt
from io import StringIO

# Connect to your Neon DB
PG_DSN = os.getenv("PG_DSN")

# Constants
def current_cfb_season(today_utc: dt.date) -> int:
    """Return current season (CFB seasons start in August)."""
    return today_utc.year if today_utc.month >= 8 else today_utc.year - 1

SEASON = int(os.getenv("SEASON") or current_cfb_season(dt.datetime.utcnow().date()))
API_URL = f"https://api.collegefootballdata.com/stats/season/advanced?year={SEASON}"

# Headers if you use the CFBD API (insert your key if needed)
HEADERS = {"Authorization": f"Bearer {os.getenv('CFBD_API_KEY', '')}"}

print(f"Fetching team advanced stats for {SEASON}...")

resp = requests.get(API_URL, headers=HEADERS)
resp.raise_for_status()
data = resp.json()

# Convert to DataFrame
df = pd.json_normalize(data)
print(f"Retrieved {len(df)} rows from API.")

# Match DB column names exactly (case + spelling)
df.rename(columns=lambda c: c.strip()
          .lower()
          .replace(" ", "")
          .replace("totalopportunities", "totalopportunies"), inplace=True)

# Clean up missing values
df = df.fillna("")

# Write CSV in memory
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

COPY_SQL = """
COPY public.team_advanced_season_stats FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL (*)
)
"""

with psycopg.connect(PG_DSN) as conn:
    with conn.cursor() as cur:
        # Delete existing season first
        cur.execute(
            "DELETE FROM public.team_advanced_season_stats WHERE season = %s;",
            (SEASON,),
        )
        print(f"Deleted existing rows for season {SEASON}")

        # Load new rows
        with cur.copy(COPY_SQL) as cp:
            cp.write(csv_buffer.getvalue().encode("utf-8"))
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM public.team_advanced_season_stats WHERE season = %s;", (SEASON,))
        print(f"✅ Inserted {cur.fetchone()[0]} rows for {SEASON}")
