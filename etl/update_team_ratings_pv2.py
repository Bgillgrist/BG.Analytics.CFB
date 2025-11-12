#!/usr/bin/env python3
"""
daily_ratings_update.py

Purpose:
  Compute and write daily team ratings into `public.team_ratings` for a
  single season and as-of date.

What it writes:
  One row per team for the chosen season/as-of date for each model:
    • rating_model = "market_v1"      (market-based rating)
    • rating_model = "performance_v1" (results-based rating)

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
MOV_CAP = 28
RIDGE = 1e-6
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
# PERFORMANCE RATINGS V2
# ────────────────────────────────────────────

def compute_performance_v2_ratings(season: int, asof_date: str) -> pd.DataFrame:
    """
    Calculate performance_v2 ratings using all completed games up to asof_date.
    - MOV is capped at ±MOV_CAP
    - HFA is removed (neutralized) so ratings are on neutral field
    - Mean-zero constraint + tiny ridge for stability
    Returns: DataFrame(team, rating)
    """
    with psycopg.connect(PG_DSN) as conn:
        sql = """
            SELECT
                g.hometeam      AS home_team,
                g.awayteam      AS away_team,
                g.homepoints    AS home_score,
                g.awaypoints    AS away_score,
                COALESCE(g.neutralsite, FALSE) AS neutral_site
            FROM public.game_data g
            WHERE g.season = %s
              AND g.startdate <= %s
              AND g.homepoints IS NOT NULL
              AND g.awaypoints IS NOT NULL
              AND g.homeclassification IN ('fbs', 'fcs')
              AND g.awayclassification IN ('fbs', 'fcs');
        """
        df = pd.read_sql(sql, conn, params=(season, asof_date))

    if df.empty:
        return pd.DataFrame(columns=["team", "rating"])

    # MOV (home - away), cap blowouts
    mov = (df["home_score"] - df["away_score"]).to_numpy(dtype=float)
    mov = np.clip(mov, -MOV_CAP, MOV_CAP)

    # Neutralize HFA: subtract HFA from home MOV when not neutral
    hfa_term = np.where(df["neutral_site"].to_numpy(dtype=bool), 0.0, HFA)
    target = mov - hfa_term

    # Build design: +1 for home team, -1 for away team
    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    m = len(df)

    # We solve (A^T A + R) r = A^T y with mean-zero constraint
    # Build A^T A and A^T y directly (saves memory over explicit A)
    AtA = np.zeros((n, n), dtype=float)
    Aty = np.zeros(n, dtype=float)

    ht = df["home_team"].to_numpy()
    at = df["away_team"].to_numpy()

    for k in range(m):
        i = idx[ht[k]]
        j = idx[at[k]]

        # Row vector is e_i - e_j
        AtA[i, i] += 1.0
        AtA[j, j] += 1.0
        AtA[i, j] -= 1.0
        AtA[j, i] -= 1.0

        Aty[i] += target[k]
        Aty[j] -= target[k]

    # Mean-zero constraint: add 1*11^T
    AtA += np.ones((n, n), dtype=float)
    # Tiny ridge
    AtA[np.diag_indices_from(AtA)] += RIDGE

    ratings = np.linalg.solve(AtA, Aty)
    ratings -= ratings.mean()  # ensure exact mean zero

    return pd.DataFrame({"team": teams, "rating": ratings}).sort_values("rating", ascending=False)

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

    # --- Performance v2 ratings ---
    pf = compute_performance_v2_ratings(season, asof_str)
    if pf is None or pf.empty:
        pf = pd.DataFrame(columns=["team", "rating"])
    pf = pf.rename(columns={"rating": "rating_value"})
    pf["rating_model"] = "performance_v2"

    # --- Upsert only performance_v2 today ---
    to_write = pf[["team", "rating_value", "rating_model"]].dropna(subset=["team", "rating_value"])
    if to_write.empty:
        print("No ratings to upsert today.")
        return

    rows = [
        (season, asof_dt, week, str(r.team), str(r.rating_model), float(r.rating_value))
        for r in to_write.itertuples(index=False)
    ]
    upsert_ratings(rows)
    print(f"✅ Daily performance_v2 updated for {asof_str} ({len(rows)} rows).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Ratings update failed: {e}")
        sys.exit(1)
