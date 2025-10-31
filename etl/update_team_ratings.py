#!/usr/bin/env python3
"""
backfill_history.py

Purpose:
  Backfill the `public.team_ratings` table for EVERY DAY from the day BEFORE the
  first game of each season through the day AFTER the last game of each season.

What it writes:
  Two rows per team per day (FBS & FCS teams included)
    • rating_model = "market_v1"      (market-based rating)
    • rating_model = "performance_v1" (results-based rating)

Inputs:
  - Environment:
      PG_DSN                : Postgres connection string
      START_SEASON (opt)    : lower bound season (int)
      END_SEASON   (opt)    : upper bound season (int)
  - Code hooks below where you will paste the logic from your
    `market_v1` and `performance_v1` modules.

Assumptions:
  - Your table exists as: public.team_ratings(season, asof_date, week, team, rating_model, rating_value, created_at)
  - Primary key is (season, asof_date, team, rating_model) so ON CONFLICT works.

How to use:
  1) Paste your market & performance functions into the two hook sections below.
  2) Run, optionally bounding seasons with env vars:
       START_SEASON=2015 END_SEASON=2025 python backfill_history.py
"""

from __future__ import annotations
import os
import sys
import psycopg
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Tuple, Optional

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────
PG_DSN = "postgresql://neondb_owner:npg_Up1blnYS0oxs@ep-delicate-brook-a4yhfswb-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require"
HFA = 2.5
FUTURE_WEEKS_AHEAD = 4
RATINGS_TABLE = "public.team_ratings"
GAME_TABLE    = "public.game_data"

# Toggle prints per date to keep logs readable
VERBOSE_DATES = True

# Optional date caps for partial backfills
ASOF_MIN_STR = os.getenv("ASOF_MIN")  # e.g., "2025-08-23"
ASOF_MAX_STR = os.getenv("ASOF_MAX")  # e.g., "2025-10-30" (defaults to today if not set)

def _parse_date_or_none(s: str | None) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()

ASOF_MIN = _parse_date_or_none(ASOF_MIN_STR)
ASOF_MAX = _parse_date_or_none(ASOF_MAX_STR) or date.today()

# ────────────────────────────────────────────
# Utility SQL helpers
# ────────────────────────────────────────────

def fetch_season_bounds() -> pd.DataFrame:
    """Return one row per season with first/last game dates."""
    sql = f"""
        SELECT
            season,
            MIN(startdate)::date AS first_game,
            MAX(startdate)::date AS last_game
        FROM {GAME_TABLE}
        WHERE startdate IS NOT NULL
          AND homeclassification IN ('fbs','fcs')
          AND awayclassification IN ('fbs','fcs')
        GROUP BY season
        ORDER BY season;
    """
    with psycopg.connect(PG_DSN) as conn:
        return pd.read_sql(sql, conn)


def season_week_for_date(season: int, asof: date) -> Optional[int]:
    """Return the latest week number for games with startdate <= asof (or None)."""
    sql = f"""
        SELECT MAX(week) AS week
        FROM {GAME_TABLE}
        WHERE season = %s
          AND startdate IS NOT NULL
          AND startdate::date <= %s
          AND homeclassification IN ('fbs','fcs')
          AND awayclassification IN ('fbs','fcs');
    """
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, (season, asof))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None


def daterange_inclusive(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

# ────────────────────────────────────────────
# ⛳️ HOOK 2: PERFORMANCE RATINGS (paste your logic here)
# ────────────────────────────────────────────

def compute_performance_ratings(season: int, asof_date: str) -> pd.DataFrame:
    """
    Calculate team performance ratings based on actual game results.
    Each team's rating represents average MOV vs. an average team on a neutral field.
    """
    print(f"Connecting to database...")
    with psycopg.connect(PG_DSN) as conn:
        print(f"Querying completed games for season {season} up to {asof_date}...")
        sql = """
            SELECT
                g.id,
                g.hometeam AS home_team,
                g.awayteam AS away_team,
                g.homepoints AS home_score,
                g.awaypoints AS away_score,
                g.neutralsite AS neutral_site
            FROM public.game_data g
            WHERE g.season = %s
              AND g.startdate <= %s
              AND g.homepoints IS NOT NULL
              AND g.awaypoints IS NOT NULL
              AND g.homeclassification IN ('fbs', 'fcs')
              AND g.awayclassification IN ('fbs', 'fcs')
        """
        df = pd.read_sql(sql, conn, params=(season, asof_date))

    if df.empty:
        print("❌ No completed games found — cannot calculate performance ratings.")
        return pd.DataFrame()

    print(f"  → Loaded {len(df)} completed games.")
    df["neutral_site"] = df["neutral_site"].fillna(False)
    df["mov"] = df["home_score"] - df["away_score"]
    df["mov_adj"] = np.where(df["neutral_site"], df["mov"], df["mov"] - HFA)

    # ── Build team list and matrix ───────────────────────────
    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    team_index = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    A = np.zeros((len(df), n))
    b = np.zeros(len(df))

    for i, row in df.iterrows():
        hi, ai = team_index[row["home_team"]], team_index[row["away_team"]]
        A[i, hi] = 1
        A[i, ai] = -1
        b[i] = row["mov_adj"]

    # Add mean-zero constraint
    A = np.vstack([A, np.ones(n)])
    b = np.append(b, 0.0)

    # Solve via least squares
    ratings, *_ = np.linalg.lstsq(A, b, rcond=None)
    ratings_df = pd.DataFrame({"team": teams, "rating": ratings})

    print("✅ Performance ratings calculated.")
    return ratings_df.sort_values("rating", ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────
# Helper for market ratings
# ────────────────────────────────────────────

def _determine_future_weeks(season: int, asof_date: str) -> list[int]:
    """
    Return up to FUTURE_WEEKS_AHEAD distinct weeks strictly after asof_date,
    ordered by the earliest startdate of each (seasonType, week) bucket.
    (Postseason weeks restart at 1; ordering by date fixes the [1,2,3,4] issue.)
    """
    with psycopg.connect(PG_DSN) as conn:
        sql = """
            WITH wk AS (
                SELECT
                    COALESCE(seasontype, 'regular') AS st,
                    week,
                    MIN(startdate)::date AS first_day
                FROM public.game_data
                WHERE season = %s
                  AND startdate IS NOT NULL
                  AND startdate > %s
                  AND homeclassification IN ('fbs','fcs')
                  AND awayclassification IN ('fbs','fcs')
                GROUP BY st, week
            )
            SELECT week
            FROM wk
            WHERE week IS NOT NULL
            ORDER BY first_day
            LIMIT %s;
        """
        wdf = pd.read_sql(sql, conn, params=(season, asof_date, FUTURE_WEEKS_AHEAD))
    return [int(x) for x in wdf["week"].tolist()] if not wdf.empty else []

# ────────────────────────────────────────────
# ⛳️ HOOK 1: MARKET RATINGS (paste your logic here)
# ────────────────────────────────────────────

def compute_market_ratings(season: int, asof_date: str) -> pd.DataFrame:
    """
    Calculate market-based ratings using betting spreads.
    Includes future opening lines for upcoming weeks.
    """
    print(f"Connecting to database...")
    future_weeks = _determine_future_weeks(season, asof_date)
    print(f"Including future opening lines for weeks: {future_weeks} (ahead={FUTURE_WEEKS_AHEAD})")

    with psycopg.connect(PG_DSN) as conn:
        query = f"""
            WITH lines AS (
                SELECT
                    g.id                           AS game_id,
                    g.hometeam                     AS home_team,
                    g.awayteam                     AS away_team,
                    COALESCE(g.neutralsite, FALSE) AS neutral_site,
                    g.startdate                    AS startdate,
                    g.week                         AS week,
                    CASE
                      WHEN g.startdate <= %s THEN b."Spread"                             -- past: closing proxy
                      ELSE COALESCE(b."OpeningSpread", b."Spread")                       -- future: opening (fallback to spread)
                    END AS raw_line
                FROM public.game_data g
                JOIN public.betting_odds b
                  ON b."Id" = g.id
                WHERE g.season = %s
                  AND g.startdate IS NOT NULL
                  AND (
                        g.startdate <= %s
                     OR (
                            %s = TRUE
                        AND %s = TRUE
                        AND g.startdate > %s
                        AND g.week = ANY(%s)
                     )
                  )
                  AND g.homeclassification IN ('fbs','fcs')
                  AND g.awayclassification IN ('fbs','fcs')
                  AND (
                        (g.startdate <= %s AND b."Spread" IS NOT NULL)
                     OR (g.startdate > %s AND COALESCE(b."OpeningSpread", b."Spread") IS NOT NULL)
                  )
            )
            SELECT
                game_id,
                home_team,
                away_team,
                neutral_site,
                AVG(raw_line) AS home_spread
            FROM lines
            GROUP BY game_id, home_team, away_team, neutral_site
            ORDER BY game_id;
        """
        params = (
            asof_date,     # for CASE past/future
            season,
            asof_date,     # bound for past
            True,          # include_future flag (always true in this module)
            bool(future_weeks),
            asof_date,
            future_weeks,
            asof_date,     # null guard for past branch
            asof_date,     # null guard for future branch
        )
        print(f"Querying betting lines for season {season} up to {asof_date} and future weeks {future_weeks}...")
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        print("No data found for given season and date.")
        return pd.DataFrame(columns=["team", "season", "asof_date", "games", "market_rating"])

    print("Using per-game AVG spread; converting to home-favored-positive before solving...")
    avg_spread = df.copy()
    # CFBD stores home spread as negative when home is favored.
    # Convert to a home-favored-positive convention and ensure numeric.
    avg_spread["spread_home_pos"] = -pd.to_numeric(avg_spread["home_spread"], errors="coerce")
    print(f"Number of games to process: {len(avg_spread)}")

    teams = pd.unique(avg_spread[["home_team", "away_team"]].values.ravel())
    team_index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    print(f"Number of teams: {n}")

    A = np.zeros((len(avg_spread)+1, n))
    b = np.zeros(len(avg_spread)+1)

    print("Building linear system...")
    for i, row in avg_spread.iterrows():
        h = team_index[row.home_team]
        a = team_index[row.away_team]
        A[i, h] = 1.0
        A[i, a] = -1.0
        hfa = 0.0 if bool(row.neutral_site) else HFA
        b[i] = float(row.spread_home_pos) - hfa

    # Add constraint to fix sum of ratings to zero (to avoid singular matrix)
    A[-1, :] = 1
    b[-1] = 0

    print("Solving linear system for market ratings...")
    ratings = np.linalg.lstsq(A, b, rcond=None)[0]

    print("Counting games played per team...")
    games_home = avg_spread["home_team"].value_counts()
    games_away = avg_spread["away_team"].value_counts()
    games = games_home.add(games_away, fill_value=0).reindex(teams).fillna(0).astype(int)

    print("Preparing results DataFrame...")
    results = pd.DataFrame({
        "team": teams,
        "season": season,
        "asof_date": asof_date,
        "games": games.values,
        "market_rating": ratings
    })

    print("Market ratings calculation complete.")
    return results.rename(columns={"market_rating": "rating"})


# ────────────────────────────────────────────
# Insert helpers
# ────────────────────────────────────────────

INSERT_SQL = f"""
INSERT INTO {RATINGS_TABLE} (season, asof_date, week, team, rating_model, rating_value)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (season, asof_date, team, rating_model)
DO UPDATE SET
  rating_value = EXCLUDED.rating_value,
  week         = EXCLUDED.week,
  created_at   = now();
"""


def upsert_ratings(rows: list[tuple]) -> None:
    if not rows:
        return
    with psycopg.connect(PG_DSN) as conn, conn.cursor() as cur:
        cur.executemany(INSERT_SQL, rows)
        conn.commit()


# ────────────────────────────────────────────
# Main backfill
# ────────────────────────────────────────────

def main() -> None:
    # Choose season & as-of date
    season = int(os.getenv("SEASON", str(date.today().year)))
    asof_str = os.getenv("ASOF_DATE", date.today().isoformat())
    asof_dt = datetime.strptime(asof_str, "%Y-%m-%d").date()

    all_bounds = fetch_season_bounds()
    bounds = all_bounds[all_bounds["season"] == season]

    if bounds.empty:
        print(f"No season {season} found in game_data; nothing to update for {asof_str}.")
        return

    first_game = bounds.iloc[0]["first_game"]
    last_game  = bounds.iloc[0]["last_game"]
    # Allow running any day; you can clamp if you want:
    # asof_dt = max(min(asof_dt, last_game + timedelta(days=1)), first_game - timedelta(days=1))

    print(f"Update daily ratings — season={season} as-of={asof_str}  window=({first_game}..{last_game})")

    week = season_week_for_date(season, asof_dt)
    print(f"  [asof={asof_str}] week={week}")

    # --- Market ratings ---
    mk = compute_market_ratings(season, asof_str)
    if mk is None:
        mk = pd.DataFrame(columns=["team", "rating"])
    mk = mk.rename(columns={"rating": "rating_value"})
    mk["rating_model"] = "market_v1"

    # --- Performance ratings ---
    pf = compute_performance_ratings(season, asof_str)
    if pf is None:
        pf = pd.DataFrame(columns=["team", "rating"])
    pf = pf.rename(columns={"rating": "rating_value"})
    pf["rating_model"] = "performance_v1"

    # --- Combine + upsert ---
    combined = pd.concat([mk, pf], ignore_index=True, sort=False)
    if combined.empty:
        print("No ratings to upsert today.")
        return

    combined = combined[["team", "rating_value", "rating_model"]].dropna(subset=["team", "rating_value"])
    rows = [
        (season, asof_dt, week, str(r.team), str(r.rating_model), float(r.rating_value))
        for r in combined.itertuples(index=False)
    ]

    upsert_ratings(rows)
    print(f"✅ Daily ratings updated for {asof_str} ({len(rows)} rows).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Backfill failed: {e}")
        sys.exit(1)
