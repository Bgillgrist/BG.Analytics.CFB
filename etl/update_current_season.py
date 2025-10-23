#!/usr/bin/env python3
import os, sys, io, csv, datetime as dt
import psycopg, requests
from dotenv import load_dotenv

load_dotenv()
PG_DSN = os.getenv("PG_DSN")
CFBD_API_KEY = os.getenv("CFBD_API_KEY")

API_URL = "https://api.collegefootballdata.com/games"
API_HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}

def current_cfb_season(today_utc: dt.date) -> int:
    return today_utc.year if today_utc.month >= 8 else today_utc.year - 1

SEASON = int(os.getenv("SEASON") or current_cfb_season(dt.datetime.utcnow().date()))

# Target columns in Postgres (match your table)
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

# ------- helpers -------
def first_of(d: dict, *keys):
    """Return first present, non-None value from d among keys."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def join_linescores(xs):
    if not xs:
        return None
    if isinstance(xs, (list, tuple)):
        return ",".join(str(x) for x in xs)
    return str(xs)

# ------- fetch -------
def fetch_games() -> list[dict]:
    print(f"Fetching all games for season {SEASON} ...")
    r = requests.get(API_URL, params={"year": SEASON}, headers=API_HEADERS, timeout=90)
    r.raise_for_status()
    data = r.json()
    print(f"Fetched {len(data)} total games.")

    rows = []
    for g in data:
        # map with robust fallbacks for fields CFBD sometimes renames
        startdate = first_of(g, "start_date", "start_time", "startDate", "startTime")
        # CFBD booleans occasionally vary in naming
        starttimetbd = first_of(g, "start_time_tbd", "startTimeTbd")
        neutralsite  = first_of(g, "neutral_site", "neutralSite")
        conferencegame = first_of(g, "conference_game", "conferenceGame")
        home_ls = join_linescores(first_of(g, "home_line_scores", "homeLineScores"))
        away_ls = join_linescores(first_of(g, "away_line_scores", "awayLineScores"))
        home_pg_wp = first_of(g,
                              "home_postgame_win_prob", "home_post_win_prob",
                              "homePostgameWinProb", "homePostWinProb")
        away_pg_wp = first_of(g,
                              "away_postgame_win_prob", "away_post_win_prob",
                              "awayPostgameWinProb", "awayPostWinProb")

        row = {
            "id":                 first_of(g, "id", "game_id", "gameId"),
            "season":             g.get("season"),
            "week":               g.get("week"),
            "seasontype":         first_of(g, "season_type", "seasonType"),
            "startdate":          startdate,
            "starttimetbd":       starttimetbd,
            "completed":          g.get("completed"),
            "neutralsite":        neutralsite,
            "conferencegame":     conferencegame,
            "attendance":         g.get("attendance"),
            "venueid":            first_of(g, "venue_id", "venueId"),
            "venue":              g.get("venue"),

            "homeid":             first_of(g, "home_id", "homeId"),
            "hometeam":           first_of(g, "home_team", "homeTeam"),
            "homeclassification": first_of(g, "home_division", "homeDivision"),
            "homeconference":     first_of(g, "home_conference", "homeConference"),
            "homepoints":         first_of(g, "home_points", "homePoints"),
            "homelinescores":     home_ls,
            "homepostgamewinprobability": home_pg_wp,
            "homepregameelo":     first_of(g, "home_pregame_elo", "homePregameElo"),
            "homepostgameelo":    first_of(g, "home_postgame_elo", "homePostgameElo"),

            "awayid":             first_of(g, "away_id", "awayId"),
            "awayteam":           first_of(g, "away_team", "awayTeam"),
            "awayclassification": first_of(g, "away_division", "awayDivision"),
            "awayconference":     first_of(g, "away_conference", "awayConference"),
            "awaypoints":         first_of(g, "away_points", "awayPoints"),
            "awaylinescores":     away_ls,
            "awaypostgamewinprobability": away_pg_wp,
            "awaypregameelo":     first_of(g, "away_pregame_elo", "awayPregameElo"),
            "awaypostgameelo":    first_of(g, "away_postgame_elo", "awayPostgameElo"),

            "excitementindex":    first_of(g, "excitement_index", "excitementIndex"),
            "highlights":         g.get("highlights"),
            "notes":              g.get("notes"),
        }
        rows.append(row)
    return rows

def to_csv_bytes(rows: list[dict]) -> bytes:
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        # empty strings for NULLs lets COPY + FORCE_NULL coerce correctly
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS})
    return sio.getvalue().encode("utf-8")

def sanity_print(rows: list[dict]):
    # quick visibility to catch schema drifts
    def nn(col): return sum(1 for r in rows if r.get(col) not in (None, ""))
    print("Non-null counts:",
          {c: nn(c) for c in [
              "id","startdate","hometeam","awayteam",
              "homepoints","awaypoints",
              "homepregameelo","homepostgameelo","awaypregameelo","awaypostgameelo"
          ]})

def main():
    rows = fetch_games()
    sanity_print(rows)
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
