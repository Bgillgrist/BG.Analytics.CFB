import os
import requests
import psycopg
import pandas as pd
from io import StringIO
import datetime as dt

PG_DSN = os.getenv("PG_DSN")
CFBD_API_KEY = os.getenv("CFBD_API_KEY")

def current_cfb_season(today_utc: dt.date) -> int:
    return today_utc.year if today_utc.month >= 8 else today_utc.year - 1

SEASON = int(os.getenv("SEASON") or current_cfb_season(dt.datetime.utcnow().date()))
API_URL = f"https://api.collegefootballdata.com/stats/season/advanced?year={SEASON}"
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}

print(f"Fetching team advanced stats for {SEASON}...")
resp = requests.get(API_URL, headers=HEADERS, timeout=90)
resp.raise_for_status()
data = resp.json()

df = pd.json_normalize(data)
print(f"Retrieved {len(df)} rows from API.")

# Normalize headers to match your SQL: lowercase + spaces -> underscores
def normalize(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    # keep your table's misspelling
    s = s.replace("totalopportunities", "totalopportunies")
    return s

df.rename(columns=lambda c: normalize(c), inplace=True)

# Exact column order for your table
COLS = [
    "season","team","conference",
    "offense_plays","offense_drives","offense_ppa","offense_totalppa",
    "offense_successrate","offense_explosiveness","offense_powersuccess",
    "offense_stuffrate","offense_lineyards","offense_lineyardstotal",
    "offense_secondlevelyards","offense_secondlevelyardstotal",
    "offense_openfieldyards","offense_openfieldyardstotal",
    "offense_totalopportunies","offense_pointsperopportunity",
    "offense_fieldposition_averagestart","offense_fieldposition_averagepredictedpoints",
    "offense_havoc_total","offense_havoc_frontseven","offense_havoc_db",
    "offense_standarddowns_rate","offense_standarddowns_ppa",
    "offense_standarddowns_successrate","offense_standarddowns_explosiveness",
    "offense_passingdowns_rate","offense_passingdowns_ppa",
    "offense_passingdowns_successrate","offense_passingdowns_explosiveness",
    "offense_rushingplays_rate","offense_rushingplays_ppa",
    "offense_rushingplays_totalppa","offense_rushingplays_successrate",
    "offense_rushingplays_explosiveness",
    "offense_passingplays_rate","offense_passingplays_ppa",
    "offense_passingplays_totalppa","offense_passingplays_successrate",
    "offense_passingplays_explosiveness",
    "defense_plays","defense_drives","defense_ppa","defense_totalppa",
    "defense_successrate","defense_explosiveness","defense_powersuccess",
    "defense_stuffrate","defense_lineyards","defense_lineyardstotal",
    "defense_secondlevelyards","defense_secondlevelyardstotal",
    "defense_openfieldyards","defense_openfieldyardstotal",
    "defense_totalopportunies","defense_pointsperopportunity",
    "defense_fieldposition_averagestart","defense_fieldposition_averagepredictedpoints",
    "defense_havoc_total","defense_havoc_frontseven","defense_havoc_db",
    "defense_standarddowns_rate","defense_standarddowns_ppa",
    "defense_standarddowns_successrate","defense_standarddowns_explosiveness",
    "defense_passingdowns_rate","defense_passingdowns_ppa",
    "defense_passingdowns_totalppa","defense_passingdowns_successrate",
    "defense_passingdowns_explosiveness",
    "defense_rushingplays_rate","defense_rushingplays_ppa",
    "defense_rushingplays_totalppa","defense_rushingplays_successrate",
    "defense_rushingplays_explosiveness",
    "defense_passingplays_rate","defense_passingplays_ppa",
    "defense_passingplays_totalppa","defense_passingplays_successrate",
    "defense_passingplays_explosiveness",
]

# Ensure all expected columns exist; if missing, add empty
for c in COLS:
    if c not in df.columns:
        df[c] = ""

# Reorder and keep only expected columns
df = df[COLS]

# Build FORCE_NULL list: all numeric cols (everything except team, conference)
NUMERIC_COLS = [c for c in COLS if c not in ("team", "conference")]
force_null_sql = ", ".join(NUMERIC_COLS)

# Create CSV in memory
csv_buffer = StringIO()
# Keep empty fields truly empty (not 'NaN')
df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

COPY_SQL = f"""
COPY public.team_advanced_season_stats (
  {", ".join(COLS)}
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL ({force_null_sql})
)
"""

with psycopg.connect(PG_DSN) as conn:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM public.team_advanced_season_stats WHERE season = %s;", (SEASON,))
        print(f"Deleted existing rows for season {SEASON}")

        with cur.copy(COPY_SQL) as cp:
            cp.write(csv_buffer.getvalue().encode("utf-8"))
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM public.team_advanced_season_stats WHERE season = %s;", (SEASON,))
        print(f"✅ Inserted {cur.fetchone()[0]} rows for {SEASON}")
