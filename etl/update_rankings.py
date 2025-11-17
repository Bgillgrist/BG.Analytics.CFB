#!/usr/bin/env python3
"""
update_rankings.py

Purpose:
  Fetch CFBD poll rankings for the current season and reload them into
  public.rankings every night (delete that season, then re-insert).

Table schema expected:

  CREATE TABLE IF NOT EXISTS public.rankings (
      season            int    NOT NULL,
      season_type       text   NOT NULL,
      week              int    NOT NULL,
      poll              text   NOT NULL,
      rank              int    NOT NULL,
      school            text   NOT NULL,
      conference        text,
      first_place_votes int,
      points            int,
      PRIMARY KEY (season, week, poll, rank, school)
  );
"""

import os
import requests
import psycopg
import pandas as pd
from io import StringIO
import datetime as dt

PG_DSN = os.getenv("PG_DSN")
CFBD_API_KEY = os.getenv("CFBD_API_KEY")


def current_cfb_season(today_utc: dt.date) -> int:
    """Return the current CFB season (Aug–Dec belong to that calendar year)."""
    return today_utc.year if today_utc.month >= 8 else today_utc.year - 1


SEASON = int(os.getenv("SEASON") or current_cfb_season(dt.datetime.utcnow().date()))

API_URL = f"https://api.collegefootballdata.com/rankings?year={SEASON}"
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}

print(f"Fetching rankings for {SEASON} from {API_URL}...")

resp = requests.get(API_URL, headers=HEADERS, timeout=90)
resp.raise_for_status()
data = resp.json()

# data is a list of objects like:
# {
#   "season": 2019,
#   "seasonType": "regular",
#   "week": 10,
#   "polls": [
#       { "poll": "AP Top 25", "ranks": [ { "rank": 1, "school": "...", ... }, ... ] },
#       ...
#   ]
# }

rows = []
for entry in data:
    season = entry.get("season")
    season_type = entry.get("seasonType")
    week = entry.get("week")
    for poll in entry.get("polls", []):
        poll_name = poll.get("poll")
        for r in poll.get("ranks", []):
            rows.append(
                {
                    "season": season,
                    "season_type": season_type,
                    "week": week,
                    "poll": poll_name,
                    "rank": r.get("rank"),
                    "school": r.get("school"),
                    "conference": r.get("conference"),
                    "first_place_votes": r.get("firstPlaceVotes"),
                    "points": r.get("points"),
                }
            )

df = pd.DataFrame(rows)
print(f"Flattened to {len(df)} ranking rows.")

if df.empty:
    print("No rankings returned from API; aborting.")
    raise SystemExit(0)

# Ensure correct column order / types
COLS = [
    "season",
    "season_type",
    "week",
    "poll",
    "rank",
    "school",
    "conference",
    "first_place_votes",
    "points",
]

# Make sure all expected columns exist (in case some fields missing from API)
for c in COLS:
    if c not in df.columns:
        df[c] = None

df = df[COLS]

# Coerce numeric fields
df["season"] = df["season"].astype(int)
df["week"] = df["week"].astype(int)
df["rank"] = df["rank"].astype(int)

# Let first_place_votes / points be nullable ints
df["first_place_votes"] = pd.to_numeric(df["first_place_votes"], errors="coerce")
df["points"] = pd.to_numeric(df["points"], errors="coerce")

# Create CSV in memory (empty string = NULL in COPY)
csv_buffer = StringIO()
df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

COPY_SQL = """
COPY public.rankings (
  season,
  season_type,
  week,
  poll,
  rank,
  school,
  conference,
  first_place_votes,
  points
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL (first_place_votes, points)
)
"""

with psycopg.connect(PG_DSN) as conn:
    with conn.cursor() as cur:
        # Drop current-season rankings so we always have a clean snapshot
        cur.execute("DELETE FROM public.rankings WHERE season = %s;", (SEASON,))
        print(f"Deleted existing rankings rows for season {SEASON}")

        # Bulk load via COPY
        with cur.copy(COPY_SQL) as cp:
            cp.write(csv_buffer.getvalue().encode("utf-8"))

        conn.commit()

        cur.execute(
            "SELECT COUNT(*) FROM public.rankings WHERE season = %s;", (SEASON,)
        )
        count = cur.fetchone()[0]
        print(f"✅ Inserted {count} rankings rows for {SEASON}")
