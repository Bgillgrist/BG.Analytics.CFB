#!/usr/bin/env python3
import os
import re
import requests
import psycopg
import pandas as pd
import numpy as np
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

# --- Flatten nested offense/defense dicts with underscores ---
df = pd.json_normalize(data, sep="_")

# --- Map top-level keys to our schema names ---
rename_map = {
    "gameId": "game_id",
    "seasonType": "season_type",
}
df.rename(columns=rename_map, inplace=True)

# --- Exact table column order (must match your SQL table) ---
COLS = [
    "game_id","season","season_type","week","team","opponent",

    "offense_plays","offense_drives","offense_ppa","offense_totalPPA",
    "offense_successRate","offense_explosiveness","offense_powerSuccess",
    "offense_stuffRate","offense_lineYards","offense_lineYardsTotal",
    "offense_secondLevelYards","offense_secondLevelYardsTotal",
    "offense_openFieldYards","offense_openFieldYardsTotal",

    "offense_standardDowns_ppa","offense_standardDowns_successRate","offense_standardDowns_explosiveness",
    "offense_passingDowns_ppa","offense_passingDowns_successRate","offense_passingDowns_explosiveness",
    "offense_rushingPlays_ppa","offense_rushingPlays_totalPPA","offense_rushingPlays_successRate","offense_rushingPlays_explosiveness",
    "offense_passingPlays_ppa","offense_passingPlays_totalPPA","offense_passingPlays_successRate","offense_passingPlays_explosiveness",

    "defense_plays","defense_drives","defense_ppa","defense_totalPPA",
    "defense_successRate","defense_explosiveness","defense_powerSuccess",
    "defense_stuffRate","defense_lineYards","defense_lineYardsTotal",
    "defense_secondLevelYards","defense_secondLevelYardsTotal",
    "defense_openFieldYards","defense_openFieldYardsTotal",

    "defense_standardDowns_ppa","defense_standardDowns_successRate","defense_standardDowns_explosiveness",
    "defense_passingDowns_ppa","defense_passingDowns_successRate","defense_passingDowns_explosiveness",
    "defense_rushingPlays_ppa","defense_rushingPlays_totalPPA","defense_rushingPlays_successRate","defense_rushingPlays_explosiveness",
    "defense_passingPlays_ppa","defense_passingPlays_totalPPA","defense_passingPlays_successRate","defense_passingPlays_explosiveness",
]

# The CFBD JSON uses camelCase within these flattened names (e.g., totalPPA).
# Convert those CFBD keys to your all-lowercase snake_case table columns:
to_table = {
    "offense_totalPPA":"offense_totalppa",
    "offense_successRate":"offense_successrate",
    "offense_powerSuccess":"offense_powersuccess",
    "offense_stuffRate":"offense_stuffrate",
    "offense_lineYards":"offense_lineyards",
    "offense_lineYardsTotal":"offense_lineyardstotal",
    "offense_secondLevelYards":"offense_secondlevelyards",
    "offense_secondLevelYardsTotal":"offense_secondlevelyardstotal",
    "offense_openFieldYards":"offense_openfieldyards",
    "offense_openFieldYardsTotal":"offense_openfieldyardstotal",
    "offense_standardDowns_ppa":"offense_standarddowns_ppa",
    "offense_standardDowns_successRate":"offense_standarddowns_successrate",
    "offense_standardDowns_explosiveness":"offense_standarddowns_explosiveness",
    "offense_passingDowns_ppa":"offense_passingdowns_ppa",
    "offense_passingDowns_successRate":"offense_passingdowns_successrate",
    "offense_passingDowns_explosiveness":"offense_passingdowns_explosiveness",
    "offense_rushingPlays_ppa":"offense_rushingplays_ppa",
    "offense_rushingPlays_totalPPA":"offense_rushingplays_totalppa",
    "offense_rushingPlays_successRate":"offense_rushingplays_successrate",
    "offense_rushingPlays_explosiveness":"offense_rushingplays_explosiveness",
    "offense_passingPlays_ppa":"offense_passingplays_ppa",
    "offense_passingPlays_totalPPA":"offense_passingplays_totalppa",
    "offense_passingPlays_successRate":"offense_passingplays_successrate",
    "offense_passingPlays_explosiveness":"offense_passingplays_explosiveness",

    "defense_totalPPA":"defense_totalppa",
    "defense_successRate":"defense_successrate",
    "defense_powerSuccess":"defense_powersuccess",
    "defense_stuffRate":"defense_stuffrate",
    "defense_lineYards":"defense_lineyards",
    "defense_lineYardsTotal":"defense_lineyardstotal",
    "defense_secondLevelYards":"defense_secondlevelyards",
    "defense_secondLevelYardsTotal":"defense_secondlevelyardstotal",
    "defense_openFieldYards":"defense_openfieldyards",
    "defense_openFieldYardsTotal":"defense_openfieldyardstotal",
    "defense_standardDowns_ppa":"defense_standarddowns_ppa",
    "defense_standardDowns_successRate":"defense_standarddowns_successrate",
    "defense_standardDowns_explosiveness":"defense_standarddowns_explosiveness",
    "defense_passingDowns_ppa":"defense_passingdowns_ppa",
    "defense_passingDowns_successRate":"defense_passingdowns_successrate",
    "defense_passingDowns_explosiveness":"defense_passingdowns_explosiveness",
    "defense_rushingPlays_ppa":"defense_rushingplays_ppa",
    "defense_rushingPlays_totalPPA":"defense_rushingplays_totalppa",
    "defense_rushingPlays_successRate":"defense_rushingplays_successrate",
    "defense_rushingPlays_explosiveness":"defense_rushingplays_explosiveness",
    "defense_passingPlays_ppa":"defense_passingplays_ppa",
    "defense_passingPlays_totalPPA":"defense_passingplays_totalppa",
    "defense_passingPlays_successRate":"defense_passingplays_successrate",
    "defense_passingPlays_explosiveness":"defense_passingplays_explosiveness",
}

# The base id columns already align after rename_map; we only need to map the camelCase segments to lower snake:
df.rename(columns=to_table, inplace=True)

# Build the final column list exactly as in your SQL table (all lower snake_case)
FINAL_COLS = [
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

# Ensure every expected column exists; add missing as NaN
for c in FINAL_COLS:
    if c not in df.columns:
        df[c] = np.nan

# Drop rows without a game_id
if df["game_id"].isna().any():
    dropped = int(df["game_id"].isna().sum())
    print(f"⚠️  Dropping {dropped} rows with null game_id.")
    df = df[df["game_id"].notna()]

# Keep only FINAL_COLS and validate that metrics aren’t all null
df = df[FINAL_COLS]

num_cols = [c for c in FINAL_COLS if c not in ("game_id","season","season_type","week","team","opponent")]
non_null_metric_cells = df[num_cols].notna().sum().sum()
if non_null_metric_cells == 0:
    # Print a small hint to debug changed keys if CFBD schema changed
    print("❌ All advanced metric fields are NULL. Aborting load.")
    print("Sample incoming columns:", sorted(df.columns)[:20])
    raise SystemExit(1)

# Write CSV to buffer
csv_buf = StringIO()
df.to_csv(csv_buf, index=False)
csv_buf.seek(0)

# Treat all metric columns as numeric (NULLable), text-only for these:
TEXT_COLS = {"season_type","team","opponent"}
NUMERIC_COLS = [c for c in FINAL_COLS if c not in TEXT_COLS]

COPY_SQL = f"""
COPY public.team_advanced_game_stats (
  {", ".join(FINAL_COLS)}
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
