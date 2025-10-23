#!/usr/bin/env python3
import os, sys, io, csv, re, datetime as dt
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

# ---------- helpers ----------
def to_snake(s: str) -> str:
    # camelCase/PascalCase -> snake_case; also collapse spaces
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).replace(" ", "_")
    return s.lower()

def normalize_keys(d: dict) -> dict:
    return {to_snake(k): v for k, v in d.items()}

def join_linescores(xs):
    if not xs:
        return None
    if isinstance(xs, (list, tuple)):
        return ",".join(str(x) for x in xs)
    return str(xs)

# ---------- fetch ----------
def fetch_games() -> list[dict]:
    print(f"Fetching all games for season {SEASON} ...")
    r = requests.get(API_URL, params={"year": SEASON}, headers=API_HEADERS, timeout=120)
    r.raise_for_status()
    data = r.json()
    print(f"Fetched {len(data)} total games.")

    rows = []
    for g in data:
        g2 = normalize_keys(g)

        # Robust fallbacks for a few fields that have drifted names
        startdate = g2.get("start_date") or g2.get("start_time")
        starttimetbd = g2.get("start_time_tbd") or g2.get("StartTimeTBD") or g2.get("starttimetbd")
        homeclassification = g2.get("home_classification") or g2.get("home_division")
        awayclassification = g2.get("away_classification") or g2.get("away_division")
        home_pg_wp = (
            g2.get("home_postgame_win_prob")
            or g2.get("home_post_win_prob")
            or g2.get("home_postgame_win_probability")
        )
        away_pg_wp = (
            g2.get("away_postgame_win_prob")
            or g2.get("away_post_win_prob")
            or g2.get("away_postgame_win_probability")
        )

        row = {
            # core
            "id":                 g2.get("id") or g2.get("game_id"),
            "season":             g2.get("season"),
            "week":               g2.get("week"),
            "seasontype":         g2.get("season_type"),
            "startdate":          startdate,
            "starttimetbd":       starttimetbd,
            "completed":          g2.get("completed"),
            "neutralsite":        g2.get("neutral_site"),
            "conferencegame":     g2.get("conference_game"),
            "attendance":         g2.get("attendance"),
            "venueid":            g2.get("venue_id"),
            "venue":              g2.get("venue"),

            # home
            "homeid":             g2.get("home_id"),
            "hometeam":           g2.get("home_team"),
            "homeclassification": homeclassification,
            "homeconference":     g2.get("home_conference"),
            "homepoints":         g2.get("home_points"),
            "homelinescores":     join_linescores(g2.get("home_line_scores")),
            "homepostgamewinprobability": home_pg_wp,
            "homepregameelo":     g2.get("home_pregame_elo"),
            "homepostgameelo":    g2.get("home_postgame_elo"),

            # away
            "awayid":             g2.get("away_id"),
            "awayteam":           g2.get("away_team"),
            "awayclassification": awayclassification,
            "awayconference":     g2.get("away_conference"),
            "awaypoints":         g2.get("away_points"),
            "awaylinescores":     join_linescores(g2.get("away_line_scores")),
            "awaypostgamewinprobability": away_pg_wp,
            "awaypregameelo":     g2.get("away_pregame_elo"),
            "awaypostgameelo":    g2.get("away_postgame_elo"),

            # misc
            "excitementindex":    g2.get("excitement_index"),
            "highlights":         g2.get("highlights"),
            "notes":              g2.get("notes"),
        }
        rows.append(row)
    return rows

# ---------- csv ----------
def to_csv_bytes(rows: list[dict]) -> bytes:
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        # empty string → NULL via COPY + FORCE_NULL
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLS})
    return sio.getvalue().encode("utf-8")

def sanity_print(rows: list[dict]):
    def nn(col): return sum(1 for r in rows if r.get(col) not in (None, ""))
    print("Non-null counts:",
          {c: nn(c) for c in [
              "id","startdate","starttimetbd",
              "hometeam","homeclassification",
              "awayteam","awayclassification",
              "homepoints","awaypoints",
              "homepregameelo","homepostgameelo",
              "awaypregameelo","awaypostgameelo",
          ]})

# ---------- main ----------
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
