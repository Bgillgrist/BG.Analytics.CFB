#!/usr/bin/env python3
import os, sys, io, csv, datetime as dt, psycopg, requests
from dotenv import load_dotenv

load_dotenv()
PG_DSN = os.getenv("PG_DSN")
CFBD_API_KEY = os.getenv("CFBD_API_KEY")

API_URL = "https://api.collegefootballdata.com/games"
API_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}

# ---------- Helper ----------
def current_cfb_season(today_utc: dt.date) -> int:
    """Return current season (CFB seasons start in August)."""
    return today_utc.year if today_utc.month >= 8 else today_utc.year - 1

SEASON = int(os.getenv("SEASON") or current_cfb_season(dt.datetime.utcnow().date()))

# ---------- Columns ----------
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

# ---------- Step 1: Pull all current-season games ----------
def fetch_games() -> list[dict]:
    print(f"Fetching all games for season {SEASON} ...")
    r = requests.get(API_URL, params={"year": SEASON}, headers=API_HEADERS, timeout=90)
    r.raise_for_status()
    data = r.json()
    print(f"Fetched {len(data)} total games.")
    rows = []
    for g in data:
        home_ls = ",".join(str(x) for x in (g.get("home_line_scores") or [])) or None
        away_ls = ",".join(str(x) for x in (g.get("away_line_scores") or [])) or None
        rows.append({
            "id": g.get("id"),
            "season": g.get("season"),
            "week": g.get("week"),
            "seasontype": g.get("season_type"),
            "startdate": g.get("start_date") or g.get("start_time"),
            "starttimetbd": g.get("start_time_tbd"),
            "completed": g.get("completed"),
            "neutralsite": g.get("neutral_site"),
            "conferencegame": g.get("conference_game"),
            "attendance": g.get("attendance"),
            "venueid": g.get("venue_id"),
            "venue": g.get("venue"),
            "homeid": g.get("home_id"),
            "hometeam": g.get("home_team"),
            "homeclassification": g.get("home_division"),  # e.g., fbs, fcs, ii, iii
            "homeconference": g.get("home_conference"),
            "homepoints": g.get("home_points"),
            "homelinescores": home_ls,
            "homepostgamewinprobability": g.get("home_postgame_win_prob"),
            "homepregameelo": g.get("home_pregame_elo"),
            "homepostgameelo": g.get("home_postgame_elo"),
            "awayid": g.get("away_id"),
            "awayteam": g.get("away_team"),
            "awayclassification": g.get("away_division"),
            "awayconference": g.get("away_conference"),
            "awaypoints": g.get("away_points"),
            "awaylinescores": away_ls,
            "awaypostgamewinprobability": g.get("away_postgame_win_prob"),
            "awaypregameelo": g.get("away_pregame_elo"),
            "awaypostgameelo": g.get("away_postgame_elo"),
            "excitementindex": g.get("excitement_index"),
            "highlights": g.get("highlights"),
            "notes": g.get("notes"),
        })
    return rows

# ---------- Step 2: Convert to CSV bytes ----------
def to_csv_bytes(rows: list[dict]) -> bytes:
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS})
    return sio.getvalue().encode("utf-8")

# ---------- Step 3: Delete + Re-Insert ----------
def main():
    rows = fetch_games()
    csv_bytes = to_csv_bytes(rows)

    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute("DELETE FROM public.game_data WHERE season = %s;", (SEASON,))
            print(f"Deleted old {SEASON} rows.")
            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)
            cur.execute("COMMIT;")
            cur.execute("SELECT COUNT(*) FROM public.game_data WHERE season=%s;", (SEASON,))
            print(f"Reimport complete: {cur.fetchone()[0]} rows now in season {SEASON}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
