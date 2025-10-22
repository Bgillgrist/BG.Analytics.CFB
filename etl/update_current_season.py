#!/usr/bin/env python3
import os, sys, io, csv, time, datetime as dt
import psycopg
import requests
from dotenv import load_dotenv

load_dotenv()
PG_DSN = os.getenv("PG_DSN")  # set in GitHub Secrets and/or etl/.env
CFBD_API_KEY = os.getenv("CFBD_API_KEY")  # if using CFBD; else remove header

# --- CONFIG ---
DIVISION = "fbs"  # adjust if you load more
SEASON_OVERRIDE = os.getenv("SEASON")  # optional manual override (e.g., 2025)

def current_cfb_season(today_utc: dt.date) -> int:
    # CFB season "year" usually maps to fall term; games in Jan belong to prior season
    return today_utc.year if today_utc.month >= 8 else today_utc.year - 1

SEASON = int(SEASON_OVERRIDE) if SEASON_OVERRIDE else current_cfb_season(dt.datetime.utcnow().date())

API_URL = "https://api.collegefootballdata.com/games"
API_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}
API_PARAMS = {"year": SEASON, "division": DIVISION}
# You can also iterate seasonType = regular/postseason if you prefer

# Columns must match public.game_data EXACTLY (names & order) except last_modified (filled by default later if you want)
COLS = [
    "id","season","week","seasontype","startdate","starttimetbd","completed","neutralsite",
    "conferencegame","attendance","venueid","venue",
    "homeid","hometeam","homeclassification","homeconference","homepoints","homelinescores",
    "homepostgamewinprobability","homepregameelo","homepostgameelo",
    "awayid","awayteam","awayclassification","awayconference","awaypoints","awaylinescores",
    "awaypostgamewinprobability","awaypregameelo","awaypostgameelo",
    "excitementindex","highlights","notes"
]

COPY_SQL = f"""
COPY public.game_data (
  {", ".join(COLS)}
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL (
    attendance, venueid,
    homeid, homepoints,
    awayid, awaypoints,
    starttimetbd, completed, neutralsite, conferencegame,
    homepregameelo, homepostgameelo,
    awaypregameelo, awaypostgameelo,
    homepostgamewinprobability, awaypostgamewinprobability,
    excitementindex
  )
)
"""

def fetch_games() -> list[dict]:
    # Basic retry (handles Neon cold starts on later steps too)
    for attempt in range(5):
        try:
            r = requests.get(API_URL, params=API_PARAMS, headers=API_HEADERS, timeout=60)
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(1 + attempt)
    # Map API JSON -> our schema keys
    rows = []
    for g in data:
        # CFBD fields (adjust if your API differs)
        # lineScores can be a list; join as comma-separated string
        home_ls = ",".join(str(x) for x in (g.get("home_line_scores") or [])) or None
        away_ls = ",".join(str(x) for x in (g.get("away_line_scores") or [])) or None

        row = {
            "id": g.get("id"),
            "season": g.get("season"),
            "week": g.get("week"),
            "seasontype": g.get("season_type"),
            "startdate": g.get("start_date") or g.get("start_time_tbd") or g.get("start_time"),  # ISO string
            "starttimetbd": g.get("start_time_tbd"),
            "completed": g.get("completed"),
            "neutralsite": g.get("neutral_site"),
            "conferencegame": g.get("conference_game"),
            "attendance": g.get("attendance"),
            "venueid": g.get("venue_id"),
            "venue": g.get("venue"),
            "homeid": g.get("home_id"),
            "hometeam": g.get("home_team"),
            "homeclassification": g.get("home_conference") and ("fbs" if DIVISION=="fbs" else "fcs"),  # or your own map
            "homeconference": g.get("home_conference"),
            "homepoints": g.get("home_points"),
            "homelinescores": home_ls,
            "homepostgamewinprobability": g.get("home_postgame_win_prob"),
            "homepregameelo": g.get("home_pregame_elo"),
            "homepostgameelo": g.get("home_postgame_elo"),
            "awayid": g.get("away_id"),
            "awayteam": g.get("away_team"),
            "awayclassification": g.get("away_conference") and ("fbs" if DIVISION=="fbs" else "fcs"),
            "awayconference": g.get("away_conference"),
            "awaypoints": g.get("away_points"),
            "awaylinescores": away_ls,
            "awaypostgamewinprobability": g.get("away_postgame_win_prob"),
            "awaypregameelo": g.get("away_pregame_elo"),
            "awaypostgameelo": g.get("away_postgame_elo"),
            "excitementindex": g.get("excitement_index"),
            "highlights": g.get("highlights"),
            "notes": g.get("notes"),
        }
        rows.append(row)
    return rows

def rows_to_csv_bytes(rows: list[dict]) -> bytes:
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        # normalize: None stays None -> becomes empty in CSV; COPY NULL '' + FORCE_NULL will cast correctly
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS})
    return sio.getvalue().encode("utf-8")

def main():
    print(f"Fetching {SEASON} {DIVISION} games…")
    rows = fetch_games()
    print(f"Fetched {len(rows)} rows.")

    csv_bytes = rows_to_csv_bytes(rows)

    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            # 1) Create a same-shape staging table (ephemeral)
            cur.execute("""
                CREATE TEMP TABLE game_data_staging (LIKE public.game_data INCLUDING ALL);
            """)
            # 2) COPY into staging
            print("Copying into staging…")
            with cur.copy(COPY_SQL.replace("public.game_data", "game_data_staging")) as cp:
                # stream bytes into staging
                cp.write(csv_bytes)
            # 3) Atomic swap for the season
            print(f"Replacing season {SEASON} atomically…")
            cur.execute("BEGIN;")
            cur.execute("DELETE FROM public.game_data WHERE season = %s;", (SEASON,))
            cur.execute("""
                INSERT INTO public.game_data (
                  id, season, week, seasontype, startdate, starttimetbd, completed, neutralsite,
                  conferencegame, attendance, venueid, venue,
                  homeid, hometeam, homeclassification, homeconference, homepoints, homelinescores,
                  homepostgamewinprobability, homepregameelo, homepostgameelo,
                  awayid, awayteam, awayclassification, awayconference, awaypoints, awaylinescores,
                  awaypostgamewinprobability, awaypregameelo, awaypostgameelo,
                  excitementindex, highlights, notes
                )
                SELECT
                  id, season, week, seasontype, startdate, starttimetbd, completed, neutralsite,
                  conferencegame, attendance, venueid, venue,
                  homeid, hometeam, homeclassification, homeconference, homepoints, homelinescores,
                  homepostgamewinprobability, homepregameelo, homepostgameelo,
                  awayid, awayteam, awayclassification, awayconference, awaypoints, awaylinescores,
                  awaypostgamewinprobability, awaypregameelo, awaypostgameelo,
                  excitementindex, highlights, notes
                FROM game_data_staging;
            """)
            cur.execute("COMMIT;")

            # Verify count
            cur.execute("SELECT COUNT(*) FROM public.game_data WHERE season = %s;", (SEASON,))
            print("Season row count now:", cur.fetchone()[0])

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
