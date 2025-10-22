#!/usr/bin/env python3
import os
import re
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

API_URL = "https://api.collegefootballdata.com/stats/game/advanced"
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}
PARAMS = {"year": SEASON}

print(f"Fetching team advanced *game* stats for {SEASON} ...")
r = requests.get(API_URL, headers=HEADERS, params=PARAMS, timeout=120)
r.raise_for_status()
data = r.json()
print(f"Retrieved {len(data)} rows.")

df = pd.json_normalize(data)

# --- robust camelCase -> snake_case (and spaces -> underscores) ---
def to_snake(s: str) -> str:
    # insert underscore before capitals (camelCase/PascalCase)
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s)
    s = s.strip().replace(" ", "_").lower()
    return s

df.rename(columns=lambda c: to_snake(c), inplace=True)

# Explicitly map variations to our schema names
rename_map = {}
if "gameid" in df.columns:
    rename_map["gameid"] = "game_id"
if "id" in df.columns:
    rename_map["id"] = "game_id"  # some endpoints use 'id' for the game id
df.rename(columns=rename_map, inplace=True)

# Exact table column order
COLS = [
    "game_id","season","season_type","week","team","opponent",

    "offense_plays","offense_drives","offense_ppa","offense_totalppa",
    "offense_successrate","offense_explosiveness","offense_powersuccess",
    "offense_stuffrate","offense_lineyards","offense_lineyardstotal",
    "offense_secondlevelyards","offense_secondlevelyardstotal",
    "offense_openfieldyards","offense_openfieldyardstotal",

    "offense_standarddowns_ppa","offense_standarddowns_successrate","offense_standarddowns_explosiveness",
    "offense_passingdowns_ppa","offense_passingdowns_successrate","offense_passingdowns_explosiveness",
    "offense_rushingplays_ppa","offense_rushingplays_totalppa","offense_rushingplays_successrate","offense_rushingplays_explosiveness",
    "offense_passingplays_ppa","offense_passingplays_totalppa","offense_passingplays_successrate","offense_passingplays_explosiveness",

    "defense_plays","defense_drives","defense_ppa","defense_totalppa",
    "defense_successrate","defense_explosiveness","defense_powersuccess",
    "defense_stuffrate","defense_lineyards","defense_lineyardstotal",
    "defense_secondlevelyards","defense_secondlevelyardstotal",
    "defense_openfieldyards","defense_openfieldyardstotal",

    "defense_standarddowns_ppa","defense_standarddowns_successrate","defense_standarddowns_explosiveness",
    "defense_passingdowns_ppa","defense_passingdowns_successrate","defense_passingdowns_explosiveness",
    "defense_rushingplays_ppa","defense_rushingplays_totalppa","defense_rushingplays_successrate","defense_rushingplays_explosiveness",
    "defense_passingplays_ppa","defense_passingplays_totalppa","defense_passingplays_successrate","defense_passingplays_explosiveness",
]

# Add any missing expected columns as empty strings so COPY aligns
for c in COLS:
    if c not in df.columns:
        df[c] = ""

# (Critical) Drop rows without a game_id to satisfy PK (game_id, team)
missing_gid = df["game_id"].isna().sum()
if missing_gid:
    print(f"Warning: dropping {missing_gid} rows with null game_id (API sometimes omits for odd cases).")
df = df[df["game_id"].notna()]

# Reorder and keep only expected columns
df = df[COLS].fillna("")

# Build FORCE_NULL: treat everything except these text columns as numeric
TEXT_COLS = {"season_type","team","opponent"}
NUMERIC_COLS = [c for c in COLS if c not in TEXT_COLS]

csv_buf = StringIO()
df.to_csv(csv_buf, index=False)
csv_buf.seek(0)

COPY_SQL = f"""
COPY public.team_advanced_game_stats (
  {", ".join(COLS)}
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL ({", ".join(NUMERIC_COLS)})
)
"""

with psycopg.connect(PG_DSN) as conn:
    with conn.cursor() as cur:
        cur.execute("BEGIN;")
        cur.execute("DELETE FROM public.team_advanced_game_stats WHERE season = %s;", (SEASON,))
        print(f"Deleted existing rows for season {SEASON}")

        with cur.copy(COPY_SQL) as cp:
            cp.write(csv_buf.getvalue().encode("utf-8"))

        cur.execute("COMMIT;")
        cur.execute("SELECT COUNT(*) FROM public.team_advanced_game_stats WHERE season = %s;", (SEASON,))
        print(f"✅ Inserted {cur.fetchone()[0]} rows for {SEASON}")
