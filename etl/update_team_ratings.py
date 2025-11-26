#!/usr/bin/env python3
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
MOV_CAP = 28
RIDGE = 1e-6
RATINGS_TABLE = "public.team_ratings"
GAME_TABLE    = "public.game_data"
FUTURE_WEEKS_AHEAD = 1

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

def current_cfb_season(today: Optional[date] = None) -> int:
    """
    Return the current CFB season based on calendar date.
    Rule of thumb:
      - Jan–Jul: belong to previous season
      - Aug–Dec: belong to current calendar year
    """
    if today is None:
        today = date.today()
    return today.year if today.month >= 8 else today.year - 1

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


def season_week_for_date(season: int, asof: date, conn) -> Optional[int]:
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
    with conn.cursor() as cur:
        cur.execute(sql, (season, asof))
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None


def daterange_inclusive(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ────────────────────────────────────────────
# 🧮 PERFORMANCE_V2 RATINGS (results-based SRS-style)
# ────────────────────────────────────────────

def fetch_game_data_for_season(conn, season: int) -> pd.DataFrame:
    """
    Fetch all completed FBS/FCS games for a season with scores.
    """
    sql = f"""
        SELECT
            id,
            season,
            startdate::date AS startdate,
            hometeam,
            awayteam,
            homepoints,
            awaypoints,
            neutralsite
        FROM {GAME_TABLE}
        WHERE season = %s
          AND startdate IS NOT NULL
          AND homeclassification IN ('fbs','fcs')
          AND awayclassification IN ('fbs','fcs')
          AND homepoints IS NOT NULL
          AND awaypoints IS NOT NULL;
    """
    return pd.read_sql(sql, conn, params=(season,))


def build_team_rows(base: pd.DataFrame) -> pd.DataFrame:
    """
    Expand game rows to team/opponent perspective with home/away/neutral flags.
    """
    home = base.rename(columns={
        "hometeam": "team",
        "awayteam": "opponent",
        "homepoints": "team_points",
        "awaypoints": "opponent_points",
    }).assign(home_away="H")

    away = base.rename(columns={
        "awayteam": "team",
        "hometeam": "opponent",
        "awaypoints": "team_points",
        "homepoints": "opponent_points",
    }).assign(home_away="A")

    df = pd.concat([home, away], ignore_index=True)

    if "neutralsite" in df.columns:
        df["home_away"] = np.where(df["neutralsite"].fillna(False), "N", df["home_away"])

    df["startdate"] = pd.to_datetime(df["startdate"])
    df = df[["season", "startdate", "team", "opponent", "team_points", "opponent_points", "home_away"]]
    df = df.dropna(subset=["team_points", "opponent_points"])
    return df

def compute_market_v1_ratings(
    conn,
    season: int,
    asof_date: str,
) -> pd.DataFrame:
    """
    Compute market_v1 ratings using betting lines and future weeks.

    This implementation:
      - Finds up to FUTURE_WEEKS_AHEAD distinct future weeks after asof_date.
      - Uses past spreads for games on or before asof_date.
      - Uses opening or actual spreads for future games in those weeks.
      - Converts spreads to home-favored-positive convention.
      - Builds and solves a linear system to estimate team ratings.
      - Returns a DataFrame with columns 'team' and 'rating'.
    """
    # Determine future weeks after asof_date, limited by FUTURE_WEEKS_AHEAD
    future_weeks_sql = f"""
        SELECT DISTINCT week
        FROM {GAME_TABLE}
        WHERE season = %s
          AND startdate::date > %s
          AND seasontype = 'regular'
          AND homeclassification IN ('fbs','fcs')
          AND awayclassification IN ('fbs','fcs')
        ORDER BY week
        LIMIT {FUTURE_WEEKS_AHEAD}
    """
    future_weeks_df = pd.read_sql(future_weeks_sql, conn, params=(season, asof_date))
    future_weeks = future_weeks_df["week"].tolist()

    # Query betting lines for past and selected future regular-season games only
    sql = f"""
        SELECT
            g.id,
            g.hometeam,
            g.awayteam,
            g.neutralsite,
            AVG(
                CASE
                    WHEN g.startdate::date <= %s THEN b."Spread"
                    WHEN g.startdate::date > %s
                         AND g.week = ANY(%s)
                         AND g.seasontype = 'regular'
                    THEN COALESCE(b."OpeningSpread", b."Spread")
                    ELSE NULL
                END
            ) AS home_spread
        FROM {GAME_TABLE} g
        JOIN public.betting_odds b ON g.id = b."Id"
        WHERE g.season = %s
          AND g.homeclassification IN ('fbs','fcs')
          AND g.awayclassification IN ('fbs','fcs')
          AND (
            (g.startdate::date <= %s AND b."Spread" IS NOT NULL)
            OR
            (g.startdate::date > %s
             AND g.week = ANY(%s)
             AND g.seasontype = 'regular'
             AND (b."OpeningSpread" IS NOT NULL OR b."Spread" IS NOT NULL))
          )
        GROUP BY g.id, g.hometeam, g.awayteam, g.neutralsite
    """
    df = pd.read_sql(
        sql,
        conn,
        params=(asof_date, asof_date, future_weeks, season, asof_date, asof_date, future_weeks),
    )

    if df.empty:
        return pd.DataFrame(columns=["team", "rating"])

    # Convert CFBD spread to home-favored-positive: spread_home_pos = -home_spread
    df["spread_home_pos"] = -df["home_spread"]

    # Build linear system: one row per game, columns per team
    teams = pd.Index(sorted(set(df["hometeam"]).union(df["awayteam"])))
    team_idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    m = len(df)
    A = np.zeros((m + 1, n))
    b_vec = np.zeros(m + 1)

    for i, row in df.iterrows():
        h = row["hometeam"]
        a = row["awayteam"]
        neutral = row["neutralsite"] if pd.notna(row["neutralsite"]) else False
        spread = row["spread_home_pos"]
        hfa_adj = 0 if neutral else HFA
        rhs = spread - hfa_adj
        A[i, team_idx[h]] = 1
        A[i, team_idx[a]] = -1
        b_vec[i] = rhs

    # Constraint: sum of all ratings = 0
    A[m, :] = 1
    b_vec[m] = 0

    # Solve least squares
    try:
        ratings = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    except Exception:
        return pd.DataFrame(columns=["team", "rating"])

    out = pd.DataFrame({"team": teams, "rating": ratings})
    return out


def compute_market_v2_ratings(
    conn,
    season: int,
    asof_date: str,
) -> pd.DataFrame:
    """
    Compute market_v2 ratings using ONLY closing spreads up to asof_date.

    Differences from market_v1:
      - No future weeks are included (no look-ahead).
      - Uses b."Spread" for all games (no OpeningSpread).
    """

    sql = f"""
        SELECT
            g.id,
            g.hometeam,
            g.awayteam,
            g.neutralsite,
            AVG(b."Spread") AS home_spread
        FROM {GAME_TABLE} g
        JOIN public.betting_odds b
          ON g.id = b."Id"
        WHERE g.season = %s
          AND g.startdate IS NOT NULL
          AND g.startdate::date <= %s
          AND g.homeclassification IN ('fbs','fcs')
          AND g.awayclassification IN ('fbs','fcs')
          AND b."Spread" IS NOT NULL
        GROUP BY g.id, g.hometeam, g.awayteam, g.neutralsite
    """

    df = pd.read_sql(sql, conn, params=(season, asof_date))

    if df.empty:
        return pd.DataFrame(columns=["team", "rating"])

    # Convert CFBD spread to home-favored-positive: spread_home_pos = -home_spread
    df["spread_home_pos"] = -df["home_spread"]

    # Build linear system: one row per game, columns per team
    teams = pd.Index(sorted(set(df["hometeam"]).union(df["awayteam"])))
    team_idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    m = len(df)

    A = np.zeros((m + 1, n))
    b_vec = np.zeros(m + 1)

    for i, row in df.iterrows():
        h = row["hometeam"]
        a = row["awayteam"]
        neutral = bool(row["neutralsite"]) if pd.notna(row["neutralsite"]) else False
        spread = float(row["spread_home_pos"])
        hfa_adj = 0.0 if neutral else HFA
        rhs = spread - hfa_adj

        A[i, team_idx[h]] = 1.0
        A[i, team_idx[a]] = -1.0
        b_vec[i] = rhs

    # Constraint: sum of all ratings = 0
    A[m, :] = 1.0
    b_vec[m] = 0.0

    try:
        ratings = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    except Exception:
        return pd.DataFrame(columns=["team", "rating"])

    out = pd.DataFrame({"team": teams, "rating": ratings})
    return out


def compute_bg_v1_ratings(
    conn,
    season: int,
    asof_date: str,
) -> pd.DataFrame:
    """
    Compute bg_v1 ratings as a hybrid of:
      - Past games: use capped margin of victory (MOV) up to asof_date.
      - Future games (up to FUTURE_WEEKS_AHEAD weeks ahead): use market spreads.

    For each included game, we create a linear equation:
        rating_home - rating_away = target

    where:
      - For past games, target is capped MOV with HFA neutralized.
      - For future games, target is market-implied edge (spread, converted to
        home-favored-positive and neutralized for HFA).

    Returns a DataFrame with columns:
      - 'team'
      - 'rating'
    """

    # 1) Determine future weeks after asof_date (limited by FUTURE_WEEKS_AHEAD)
    future_weeks_sql = f"""
        SELECT DISTINCT week
        FROM {GAME_TABLE}
        WHERE season = %s
          AND startdate::date > %s
          AND homeclassification IN ('fbs','fcs')
          AND awayclassification IN ('fbs','fcs')
        ORDER BY week
        LIMIT {FUTURE_WEEKS_AHEAD}
    """
    future_weeks_df = pd.read_sql(future_weeks_sql, conn, params=(season, asof_date))
    future_weeks = future_weeks_df["week"].tolist()

    # 2) Pull combined game + line data for:
    #    - All games on/before asof_date (past)
    #    - Games in the selected future weeks (for spread-only equations)
    sql = f"""
        SELECT
            g.id,
            g.hometeam,
            g.awayteam,
            g.neutralsite,
            g.startdate::date AS gamedate,
            g.homepoints,
            g.awaypoints,
            g.week,
            AVG(b."Spread") AS home_spread
        FROM {GAME_TABLE} g
        JOIN public.betting_odds b
          ON g.id = b."Id"
        WHERE g.season = %s
          AND g.startdate IS NOT NULL
          AND g.homeclassification IN ('fbs','fcs')
          AND g.awayclassification IN ('fbs','fcs')
          AND (
                g.startdate::date <= %s
          AND seasontype = 'regular'
          )
          AND (
                (g.startdate::date <= %s AND b."Spread" IS NOT NULL)
             OR (g.week = ANY(%s)
                 AND g.startdate::date > %s
                 AND g.seasontype = 'regular'
                 AND b."Spread" IS NOT NULL)
          )
        GROUP BY
            g.id,
            g.hometeam,
            g.awayteam,
            g.neutralsite,
            g.startdate::date,
            g.homepoints,
            g.awaypoints,
            g.week
    """
    params = (
        season,
        asof_date,
        future_weeks,
        asof_date,
        asof_date,
        future_weeks,
        asof_date,
    )
    df = pd.read_sql(sql, conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=["team", "rating"])

    df["gamedate"] = pd.to_datetime(df["gamedate"])

    # 3) Build hybrid target: MOV for past, spread for future
    df["target"] = np.nan

    # Past games: use capped MOV with HFA neutralized to neutral field
    past_mask = (
        (df["gamedate"] <= pd.to_datetime(asof_date))
        & df["homepoints"].notna()
        & df["awaypoints"].notna()
    )
    if past_mask.any():
        mov = df.loc[past_mask, "homepoints"] - df.loc[past_mask, "awaypoints"]
        mov = mov.clip(lower=-MOV_CAP, upper=MOV_CAP)

        # Apply HFA for non-neutral games so ratings are on neutral field
        neutrals = df.loc[past_mask, "neutralsite"].fillna(False).astype(bool)
        hfa_adj = np.where(neutrals, 0.0, HFA)

        df.loc[past_mask, "target"] = mov - hfa_adj

    # Future games (in the selected weeks): use spreads, converted to neutral field
    future_mask = (
        (df["gamedate"] > pd.to_datetime(asof_date))
        & df["home_spread"].notna()
        & df["week"].isin(future_weeks)
    )
    if future_mask.any():
        # CFBD convention: home spread is negative when home is favored.
        spread_home_pos = -df.loc[future_mask, "home_spread"].astype(float)
        neutrals = df.loc[future_mask, "neutralsite"].fillna(False).astype(bool)
        hfa_adj = np.where(neutrals, 0.0, HFA)
        df.loc[future_mask, "target"] = spread_home_pos - hfa_adj

    # Keep only rows that actually got a target value
    df = df[df["target"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["team", "rating"])

    # 4) Build and solve linear system over all included games
    teams = pd.Index(sorted(set(df["hometeam"]).union(df["awayteam"])))
    team_idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    m = len(df)

    A = np.zeros((m + 1, n))
    b_vec = np.zeros(m + 1)

    for i, row in df.iterrows():
        h = row["hometeam"]
        a = row["awayteam"]
        target = float(row["target"])

        A[i, team_idx[h]] = 1.0
        A[i, team_idx[a]] = -1.0
        b_vec[i] = target

    # Constraint: sum of all ratings = 0 (identifiability)
    A[m, :] = 1.0
    b_vec[m] = 0.0

    try:
        ratings = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    except Exception:
        return pd.DataFrame(columns=["team", "rating"])

    out = pd.DataFrame({"team": teams, "rating": ratings})
    return out


def compute_performance_v1_ratings(
    team_rows: pd.DataFrame,
    season: int,
    asof_date: str,
    hfa: float = HFA,
    mov_cap: int = MOV_CAP,
    ridge: float = RIDGE,
) -> pd.DataFrame:
    """
    TODO: Implement performance_v1 ratings.

    Suggested pattern:
      - Use the same inputs as performance_v2 (team_rows filtered by season
        and through_date), but with whatever MOV / transformation logic you
        want for performance_v1.
      - Return a DataFrame with columns:
          - 'team'
          - 'rating'
    """
    # Filter rows to games played on or before asof_date
    df = team_rows[team_rows["startdate"] <= pd.to_datetime(asof_date)].copy()
    if df.empty:
        return pd.DataFrame(columns=["team", "rating"])

    # Cap MOV
    df["mov"] = df["team_points"] - df["opponent_points"]
    df["mov"] = df["mov"].clip(lower=-mov_cap, upper=mov_cap)

    # Neutralize HFA
    hfa_mask = df["home_away"] == "H"
    afa_mask = df["home_away"] == "A"
    df.loc[hfa_mask, "mov"] = df.loc[hfa_mask, "mov"] - hfa / 2
    df.loc[afa_mask, "mov"] = df.loc[afa_mask, "mov"] + hfa / 2
    # Neutral games: no adjustment

    # Build SRS system
    teams = pd.Index(sorted(set(df["team"]).union(df["opponent"])))
    team_idx = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    A = np.zeros((len(df), n))
    y = df["mov"].to_numpy()
    for i, row in enumerate(df.itertuples(index=False)):
        A[i, team_idx[row.team]] = 1
        A[i, team_idx[row.opponent]] = -1

    # Ridge regularization
    AtA = A.T @ A + ridge * np.eye(n)
    Aty = A.T @ y

    # Enforce mean-zero constraint
    # Add a row of ones to AtA and zero to Aty to constrain mean to zero
    AtA_aug = np.vstack([AtA, np.ones(n)])
    last_row = np.append(np.ones(n), 0)
    AtA_aug = np.column_stack([AtA_aug, last_row])
    AtA_aug[-1, -1] = 0
    Aty_aug = np.append(Aty, 0)

    # Solve
    try:
        sol = np.linalg.lstsq(AtA_aug, Aty_aug, rcond=None)[0]
        ratings = sol[:n]
    except Exception:
        # If solving fails, return empty
        return pd.DataFrame(columns=["team", "rating"])

    out = pd.DataFrame({"team": teams, "rating": ratings})
    return out

def _prepare_model_df(df: Optional[pd.DataFrame], model_name: str) -> pd.DataFrame:
    """
    Standardize a per-model ratings DataFrame so it can be concatenated:
      - If df is None or empty, returns an empty DataFrame with
        ['team', 'rating_value', 'rating_model'].
      - Otherwise renames 'rating' -> 'rating_value' and tags 'rating_model'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["team", "rating_value", "rating_model"])
    df = df.rename(columns={"rating": "rating_value"}).copy()
    df["rating_model"] = model_name
    return df[["team", "rating_value", "rating_model"]]

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
    all_bounds = fetch_season_bounds()
    if all_bounds.empty:
        print("No seasons found in game_data; nothing to backfill.")
        return

    # Bound by env vars if provided
    # Bound by env vars if provided; otherwise, default to *current* CFB season
    start_env = os.getenv("START_SEASON")
    end_env   = os.getenv("END_SEASON")

    if start_env or end_env:
        # Manual override (useful for big backfills)
        start_season = int(start_env or all_bounds["season"].min())
        end_season   = int(end_env   or start_season)
    else:
        # Automatic nightly behavior: just do the current CFB season
        auto_season = current_cfb_season()
        # Clamp to known seasons in game_data
        min_season = int(all_bounds["season"].min())
        max_season = int(all_bounds["season"].max())
        auto_season = max(min_season, min(max_season, auto_season))
        start_season = end_season = auto_season

    bounds = all_bounds[
        (all_bounds["season"] >= start_season) &
        (all_bounds["season"] <= end_season)
    ]

    print(f"Backfilling seasons {start_season}..{end_season} ({len(bounds)} seasons)")

    for _, row in bounds.iterrows():
        season = int(row.season)
        first_game: date = row.first_game
        last_game:  date = row.last_game
        if pd.isna(first_game) or pd.isna(last_game):
            print(f"  • Season {season}: missing first/last game dates; skipping.")
            continue

        start_day = first_game - timedelta(days=1)
        end_day   = last_game  + timedelta(days=1)
        # Apply optional global caps for partial backfills
        if ASOF_MIN and start_day < ASOF_MIN:
            start_day = ASOF_MIN
        if ASOF_MAX and end_day > ASOF_MAX:
            end_day = ASOF_MAX

        if start_day > end_day:
            print(f"  • Season {season}: capped range empty (start {start_day} > end {end_day}); skipping.")
            continue

        print(f"\nSeason {season}: {start_day} → {end_day} (daily)")

        with psycopg.connect(PG_DSN) as conn:
            # Preload all completed games for this season once (for performance_v2)
            base_games = fetch_game_data_for_season(conn, season)
            team_rows = build_team_rows(base_games)

            for asof in daterange_inclusive(start_day, end_day):
                asof_str = asof.strftime("%Y-%m-%d")
                week = season_week_for_date(season, asof, conn)
                if VERBOSE_DATES:
                    print(f"  [{asof_str}] week={week}")

                # ── Compute all rating models for this as-of date ──
                # 1) market_v1
                mk1 = compute_market_v1_ratings(conn, season, asof_str)
                mk1 = _prepare_model_df(mk1, "market_v1")

                # 2) market_v2
                mk2 = compute_market_v2_ratings(conn, season, asof_str)
                mk2 = _prepare_model_df(mk2, "market_v2")

                # 3) bg_v1
                bg1 = compute_bg_v1_ratings(conn, season, asof_str)
                bg1 = _prepare_model_df(bg1, "bg_v1")

                # 4) performance_v1
                perf1 = compute_performance_v1_ratings(
                    team_rows=team_rows,
                    season=season,
                    asof_date=asof_str,
                    hfa=HFA,
                    mov_cap=MOV_CAP,
                    ridge=RIDGE,
                )
                perf1 = _prepare_model_df(perf1, "performance_v1")

                # Combine all models; some may be empty until implemented
                combined = pd.concat(
                    [mk1, mk2, bg1, perf1],
                    ignore_index=True,
                )

                if combined.empty:
                    # Nothing to write for this date/season (e.g., before any games played)
                    continue

                rows = [
                    (
                        season,
                        asof,      # as date
                        week,
                        str(r.team),
                        str(r.rating_model),
                        float(r.rating_value),
                    )
                    for r in combined.itertuples(index=False)
                ]

                upsert_ratings(rows)

            print(f"  ✓ Season {season} complete.")

    print("\nAll done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Backfill failed: {e}")
        sys.exit(1)
