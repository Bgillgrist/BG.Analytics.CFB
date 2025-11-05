#!/usr/bin/env python3
"""
daily_ratings_update_bgv1_mv2.py

Purpose:
  Compute and write daily team ratings into `public.team_ratings` for a
  single season and as-of date.

What it writes:
  One row per team for the chosen season/as-of date for each model:
    • rating_model = "market_v2"      (market-based rating)
    • rating_model = "bg_v1" (SRS and Market Combo)

Inputs (via environment variables):
  - PG_DSN     : Postgres connection string (falls back to default in code)
  - SEASON     : season year (int). Defaults to current year if not set.
  - ASOF_DATE  : as-of date in YYYY-MM-DD. Defaults to today if not set.

Assumptions:
  - Ratings table: public.team_ratings(season, asof_date, week, team,
    rating_model, rating_value, created_at)
  - Primary key: (season, asof_date, team, rating_model) so ON CONFLICT works.
"""

from __future__ import annotations
import os
import sys
import psycopg
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Optional

# ────────────────────────────────────────────
# Config
# ────────────────────────────────────────────
PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://neondb_owner:npg_Up1blnYS0oxs@ep-delicate-brook-a4yhfswb-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require",
)
HFA = 2.5
FUTURE_WEEKS_AHEAD = 4
RATINGS_TABLE = "public.team_ratings"
GAME_TABLE = "public.game_data"


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


# ────────────────────────────────────────────
# BG Ratings (SRS and Market V2 Combo)
# ────────────────────────────────────────────


def compute_bg_ratings(season: int, asof_date: str) -> pd.DataFrame:
    """
    Calculate BG hybrid ratings:
      - MOV for past games (homepoints - awaypoints)
      - Spread for future games (home-positive convention)

    Returns one row per team with columns: ['team', 'rating'].
    """
    print("Connecting to database for BG ratings...")

    with psycopg.connect(PG_DSN) as conn:
        query = """
            WITH lines AS (
                SELECT
                    g.id                           AS game_id,
                    g.hometeam                     AS home_team,
                    g.awayteam                     AS away_team,
                    COALESCE(g.neutralsite, FALSE) AS neutral_site,
                    g.startdate                    AS startdate,
                    g.week                         AS week,
                    CASE
                        WHEN g.startdate <= %s THEN (g.homepoints - g.awaypoints)  -- past: actual MOV, home positive
                        ELSE -b."Spread"                                           -- future: expected MOV from spread, home positive
                    END AS home_value
                FROM public.game_data g
                JOIN public.betting_odds b
                  ON b."Id" = g.id
                WHERE g.season = %s
                  AND g.startdate IS NOT NULL
                  AND g.homeclassification IN ('fbs','fcs')
                  AND g.awayclassification IN ('fbs','fcs')
                  AND (
                        (g.startdate <= %s AND g.homepoints IS NOT NULL AND g.awaypoints IS NOT NULL)
                     OR (g.startdate > %s AND b."Spread" IS NOT NULL)
                  )
            )
            SELECT
                game_id,
                home_team,
                away_team,
                neutral_site,
                AVG(home_value) AS home_value
            FROM lines
            GROUP BY game_id, home_team, away_team, neutral_site
            ORDER BY game_id;
        """
        params = (
            asof_date,  # CASE: past vs future
            season,
            asof_date,  # WHERE past branch
            asof_date,  # WHERE future branch
        )
        print(f"Querying MOV (past) + Spread (future) for season {season} as of {asof_date}...")
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        print("No data found for given season and date (BG ratings).")
        return pd.DataFrame(columns=["team", "rating"])

    print("Using MOV for past games and spread (home-positive) for future games...")
    game_values = df.copy()
    game_values["home_value"] = pd.to_numeric(game_values["home_value"], errors="coerce")
    print(f"Number of games to process: {len(game_values)}")

    teams = pd.unique(game_values[["home_team", "away_team"]].values.ravel())
    team_index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    print(f"Number of teams: {n}")

    A = np.zeros((len(game_values) + 1, n))
    b = np.zeros(len(game_values) + 1)

    print("Building linear system for BG ratings...")
    for i, row in game_values.iterrows():
        h = team_index[row.home_team]
        a = team_index[row.away_team]
        A[i, h] = 1.0
        A[i, a] = -1.0
        hfa = 0.0 if bool(row.neutral_site) else HFA
        b[i] = float(row.home_value) - hfa

    # Mean-zero constraint
    A[-1, :] = 1.0
    b[-1] = 0.0

    print("Solving linear system for BG ratings...")
    ratings = np.linalg.lstsq(A, b, rcond=None)[0]

    print("Preparing BG results DataFrame...")
    results = pd.DataFrame({
        "team": teams,
        "rating": ratings,
    })

    print("BG ratings calculation complete.")
    return results.sort_values("rating", ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────
# MARKET RATINGS V2 (spread-based)
# ────────────────────────────────────────────


def _determine_future_weeks(season: int, asof_date: str) -> list[int]:
    """Return up to FUTURE_WEEKS_AHEAD distinct weeks strictly after asof_date."""
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


def compute_market_ratings(season: int, asof_date: str) -> pd.DataFrame:
    """
    Calculate market-based ratings using closing spreads:

      - Past games: uses b."Spread" (closing line proxy)
      - Future games (up to FUTURE_WEEKS_AHEAD weeks): also uses b."Spread"

    All spreads are converted to a home-favored-positive convention
    before solving the rating system.
    """
    print("Connecting to database for market_v2 ratings…")
    future_weeks = _determine_future_weeks(season, asof_date)
    print(f"Including future weeks: {future_weeks} (ahead={FUTURE_WEEKS_AHEAD})")

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
                    b."Spread"                     AS raw_line
                FROM public.game_data g
                JOIN public.betting_odds b
                  ON b."Id" = g.id
                WHERE g.season = %s
                  AND g.startdate IS NOT NULL
                  AND (
                        g.startdate <= %s
                     OR (
                            %s = TRUE           -- include_future flag
                        AND %s = TRUE           -- only if we actually have future_weeks
                        AND g.startdate > %s
                        AND g.week = ANY(%s)
                     )
                  )
                  AND g.homeclassification IN ('fbs','fcs')
                  AND g.awayclassification IN ('fbs','fcs')
                  AND (
                        (g.startdate <= %s AND b."Spread" IS NOT NULL)
                     OR (g.startdate > %s AND b."Spread" IS NOT NULL)
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
        # Placeholder mapping:
        #  1: season
        #  2: asof_date         (g.startdate <= %s)
        #  3: include_future    (True)
        #  4: bool(future_weeks)
        #  5: asof_date         (g.startdate > %s)
        #  6: future_weeks      (ARRAY[int])
        #  7: asof_date         (g.startdate <= %s in last clause)
        #  8: asof_date         (g.startdate > %s in last clause)
        params = (
            season,
            asof_date,
            True,
            bool(future_weeks),
            asof_date,
            future_weeks,
            asof_date,
            asof_date,
        )
        print(f"Querying betting lines for season {season} up to {asof_date} and future weeks {future_weeks}…")
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        print("No data found for given season and date.")
        return pd.DataFrame(columns=["team", "rating"])

    print("Using per-game AVG spread; converting to home-favored-positive before solving…")
    avg_spread = df.copy()
    # CFBD stores home spread as negative when home is favored.
    avg_spread["spread_home_pos"] = -pd.to_numeric(avg_spread["home_spread"], errors="coerce")
    print(f"Number of games to process: {len(avg_spread)}")

    teams = pd.unique(avg_spread[["home_team", "away_team"]].values.ravel())
    team_index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    print(f"Number of teams: {n}")

    A = np.zeros((len(avg_spread) + 1, n))
    b = np.zeros(len(avg_spread) + 1)

    print("Building linear system…")
    for i, row in avg_spread.iterrows():
        h = team_index[row.home_team]
        a = team_index[row.away_team]
        A[i, h] = 1.0
        A[i, a] = -1.0
        hfa = 0.0 if bool(row.neutral_site) else HFA
        b[i] = float(row.spread_home_pos) - hfa

    # Add constraint to fix sum of ratings to zero (to avoid singular matrix)
    A[-1, :] = 1.0
    b[-1] = 0.0

    print("Solving linear system for market_v2 ratings…")
    ratings = np.linalg.lstsq(A, b, rcond=None)[0]

    print("Preparing results DataFrame…")
    results = pd.DataFrame({
        "team": teams,
        "rating": ratings,
    })

    print("Market_v2 ratings calculation complete.")
    return results.sort_values("rating", ascending=False).reset_index(drop=True)


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
# Main: one-day update for a single season/as-of date
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
    last_game = bounds.iloc[0]["last_game"]

    print(f"Updating daily ratings — season={season} as-of={asof_str}  window=({first_game}..{last_game})")

    week = season_week_for_date(season, asof_dt)
    print(f"  [asof={asof_str}] week={week}")

    # --- Market ratings ---
    mk = compute_market_ratings(season, asof_str)
    if mk is None or mk.empty:
        mk = pd.DataFrame(columns=["team", "rating"])
    mk = mk.rename(columns={"rating": "rating_value"})
    mk["rating_model"] = "market_v2"

    # --- BG hybrid ratings (MOV + Spread) ---
    bg = compute_bg_ratings(season, asof_str)
    if bg is None or bg.empty:
        bg = pd.DataFrame(columns=["team", "rating"])
    bg = bg.rename(columns={"rating": "rating_value"})
    bg["rating_model"] = "bg_v1"

    # --- Combine + upsert ---
    combined = pd.concat([mk, bg], ignore_index=True, sort=False)
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
        print(f"❌ Ratings update failed: {e}")
        sys.exit(1)
