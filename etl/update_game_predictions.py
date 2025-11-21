#!/usr/bin/env python3
"""
nightly_predictions.py

Purpose:
  - Train a win probability model and a spread model using the SAME covariates:
      * 4 rating models available (market_v1, performance_v1, bg_v1, market_v2)
      * Base features use ONLY 3 models for each side:
          - team/opp: performance_v1, bg_v1, market_v2  (6 numeric features)
      * market_v1 (team & opp) is used ONLY through week interactions:
          - team_market_rating_week, opp_market_rating_week
      * So: 6 base rating features + 8 interaction features total
      * plus missing flags for the 6 base ratings
      * plus week, location, teamclassification, opponentclassification
  - Train on seasons 2015..(current_season - 1)
  - Predict for ALL games in the current season
  - Store one row per game (home perspective) in public.game_predictions
  - Set totalpred = average total points of completed games in the current season.

Config via environment:
  - PG_DSN         : Postgres connection string
  - RATING_LAG_DAYS: how many days prior to gameday ratings must be as-of (default 1)
  - MODEL_VERSION  : a tag for this model version (default 'wp_spread_v1')
"""

import os
import sys
import psycopg
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression

# ─────────────────────────────
# Config
# ─────────────────────────────
PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://neondb_owner:npg_Up1blnYS0oxs@ep-delicate-brook-a4yhfswb-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require",
)
RATING_LAG_DAYS = int(os.getenv("RATING_LAG_DAYS", "1"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "wp_spread_v1")


# ─────────────────────────────
# Helpers: DB + current season
# ─────────────────────────────
def get_current_season(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(season) FROM public.game_data;")
        row = cur.fetchone()
        if not row or row[0] is None:
            raise RuntimeError("Could not determine current season from game_data.")
        return int(row[0])


# ─────────────────────────────
# Build modeling table
# ─────────────────────────────
def build_modeling_table(conn, max_season: int) -> pd.DataFrame:
    """
    Build a team-centric modeling table:
      - seasons 2015..max_season
      - one row per team per game
      - joins 4 rating models for team & opponent:
          market_v1, performance_v1, bg_v1, market_v2
      - but market_v1 is only used later via week interactions
      - builds an 'asof_target' = gamedate - RATING_LAG_DAYS for rating lookups
    """

    sql = f"""
    WITH g AS (
      SELECT
        id,
        season,
        week,
        CAST(startdate AS date) AS gamedate,
        hometeam,
        awayteam,
        homepoints,
        awaypoints,
        conferencegame,
        COALESCE(neutralsite, false) AS neutralsite,
        homeclassification,
        awayclassification
      FROM public.game_data
      WHERE season BETWEEN 2015 AND %s
        AND homeclassification IN ('fbs', 'fcs')
        AND awayclassification IN ('fbs', 'fcs')
        AND startdate IS NOT NULL
    ),
    team_rows AS (
      SELECT
        id,
        season,
        week,
        gamedate,
        conferencegame,
        hometeam AS team,
        awayteam AS opponent,
        hometeam AS home_team,
        awayteam AS away_team,
        homepoints AS teampoints,
        awaypoints AS opponentpoints,
        CASE WHEN neutralsite THEN 'N' ELSE 'H' END AS location,
        homeclassification  AS teamclassification,
        awayclassification  AS opponentclassification,
        CASE WHEN homepoints > awaypoints THEN 1
             WHEN homepoints < awaypoints THEN 0
             ELSE NULL
        END AS team_win,
        (gamedate - INTERVAL '{RATING_LAG_DAYS} days') AS asof_target
      FROM g
      UNION ALL
      SELECT
        id,
        season,
        week,
        gamedate,
        conferencegame,
        awayteam AS team,
        hometeam AS opponent,
        hometeam AS home_team,
        awayteam AS away_team,
        awaypoints AS teampoints,
        homepoints AS opponentpoints,
        CASE WHEN neutralsite THEN 'N' ELSE 'A' END AS location,
        awayclassification  AS teamclassification,
        homeclassification  AS opponentclassification,
        CASE WHEN awaypoints > homepoints THEN 1
             WHEN awaypoints < homepoints THEN 0
             ELSE NULL
        END AS team_win,
        (gamedate - INTERVAL '{RATING_LAG_DAYS} days') AS asof_target
      FROM g
    ),
    rating_spans AS (
      SELECT
        team,
        rating_model,
        asof_date,
        LEAD(asof_date) OVER (
          PARTITION BY team, rating_model
          ORDER BY asof_date
        ) AS next_asof_date,
        rating_value
      FROM public.team_ratings
      WHERE rating_model IN (
        'market_v1', 'performance_v1', 'bg_v1', 'market_v2'
      )
    ),
    joined AS (
      SELECT
        tr.*,

        -- TEAM ratings
        tm_m1.rating_value  AS team_market_rating,
        tm_p1.rating_value  AS team_perf_rating,
        tm_bg.rating_value  AS team_bg_rating,
        tm_m2.rating_value  AS team_market_v2_rating,

        -- OPPONENT ratings
        op_m1.rating_value  AS opp_market_rating,
        op_p1.rating_value  AS opp_perf_rating,
        op_bg.rating_value  AS opp_bg_rating,
        op_m2.rating_value  AS opp_market_v2_rating

      FROM team_rows tr

      -- TEAM side
      LEFT JOIN rating_spans tm_m1
        ON tm_m1.team = tr.team
       AND tm_m1.rating_model = 'market_v1'
       AND tr.asof_target >= tm_m1.asof_date
       AND (tr.asof_target < tm_m1.next_asof_date OR tm_m1.next_asof_date IS NULL)

      LEFT JOIN rating_spans tm_p1
        ON tm_p1.team = tr.team
       AND tm_p1.rating_model = 'performance_v1'
       AND tr.asof_target >= tm_p1.asof_date
       AND (tr.asof_target < tm_p1.next_asof_date OR tm_p1.next_asof_date IS NULL)

      LEFT JOIN rating_spans tm_bg
        ON tm_bg.team = tr.team
       AND tm_bg.rating_model = 'bg_v1'
       AND tr.asof_target >= tm_bg.asof_date
       AND (tr.asof_target < tm_bg.next_asof_date OR tm_bg.next_asof_date IS NULL)

      LEFT JOIN rating_spans tm_m2
        ON tm_m2.team = tr.team
       AND tm_m2.rating_model = 'market_v2'
       AND tr.asof_target >= tm_m2.asof_date
       AND (tr.asof_target < tm_m2.next_asof_date OR tm_m2.next_asof_date IS NULL)

      -- OPPONENT side
      LEFT JOIN rating_spans op_m1
        ON op_m1.team = tr.opponent
       AND op_m1.rating_model = 'market_v1'
       AND tr.asof_target >= op_m1.asof_date
       AND (tr.asof_target < op_m1.next_asof_date OR op_m1.next_asof_date IS NULL)

      LEFT JOIN rating_spans op_p1
        ON op_p1.team = tr.opponent
       AND op_p1.rating_model = 'performance_v1'
       AND tr.asof_target >= op_p1.asof_date
       AND (tr.asof_target < op_p1.next_asof_date OR op_p1.next_asof_date IS NULL)

      LEFT JOIN rating_spans op_bg
        ON op_bg.team = tr.opponent
       AND op_bg.rating_model = 'bg_v1'
       AND tr.asof_target >= op_bg.asof_date
       AND (tr.asof_target < op_bg.next_asof_date OR op_bg.next_asof_date IS NULL)

      LEFT JOIN rating_spans op_m2
        ON op_m2.team = tr.opponent
       AND op_m2.rating_model = 'market_v2'
       AND tr.asof_target >= op_m2.asof_date
       AND (tr.asof_target < op_m2.next_asof_date OR op_m2.next_asof_date IS NULL)
    )
    SELECT *
    FROM joined
    ORDER BY season, week, gamedate, team;
    """

    df = pd.read_sql(sql, conn, params=(max_season,))
    if df.empty:
        raise RuntimeError("Modeling query returned no rows.")
    return df


# ─────────────────────────────
# Train win + spread models
# ─────────────────────────────
def train_models(df: pd.DataFrame, current_season: int):
    """
    Covariates:
      - Base rating features (6):
          team_perf_rating, team_bg_rating, team_market_v2_rating,
          opp_perf_rating,  opp_bg_rating,  opp_market_v2_rating
      - Missing flags for those 6 ratings.
      - Week (numeric).
      - Week interactions for 8 ratings:
          * the 6 above
          * team_market_rating, opp_market_rating (market_v1, interaction-only)
      - Categorical: location, teamclassification, opponentclassification

    Trains:
      - win_model   (LogisticRegression, target = team_win)
      - spread_model(LinearRegression, target = teampoints - opponentpoints)
    """

    df = df.copy()

    # Base rating columns: 3 models × (team, opp) = 6 features
    rating_cols = [
        "team_perf_rating",
        "team_bg_rating",
        "team_market_v2_rating",
        "opp_perf_rating",
        "opp_bg_rating",
        "opp_market_v2_rating",
    ]

    # Missing flags for these 6 ratings + fillna(0) (ratings are mean-zero-ish)
    rating_missing_cols = []
    for col in rating_cols:
        miss_col = f"{col}_missing"
        rating_missing_cols.append(miss_col)
        df[miss_col] = df[col].isna().astype(int)
        df[col] = df[col].fillna(0.0)

    # market_v1 ratings used only via interactions:
    # - team_market_rating, opp_market_rating
    # make sure they exist and are filled
    for col in ["team_market_rating", "opp_market_rating"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Week numeric
    df["week"] = df["week"].astype(float)

    # Interaction base columns:
    #   - 6 base rating features
    #   - plus team/opp market_v1 (interaction-only)
    interaction_base_cols = rating_cols + ["team_market_rating", "opp_market_rating"]

    # Create week interactions for these 8
    interaction_cols = []
    for col in interaction_base_cols:
        inter_col = f"{col}_week"
        interaction_cols.append(inter_col)
        df[inter_col] = df[col] * df["week"]

    # Categorical features
    categorical_features = ["teamclassification", "opponentclassification", "location"]
    for c in categorical_features:
        df[c] = df[c].astype("category")

    # Spread target
    df["spread"] = df["teampoints"] - df["opponentpoints"]

    # Features used for BOTH models
    numeric_features = rating_cols + rating_missing_cols + ["week"] + interaction_cols

    # Training sets: seasons before current_season
    train_mask = (df["season"] < current_season) & df["team_win"].notna()
    train_win = df[train_mask].copy()

    train_spread_mask = train_mask & df["spread"].notna()
    train_spread = df[train_spread_mask].copy()

    if train_win.empty:
        raise RuntimeError("No training data for win model (team_win missing).")
    if train_spread.empty:
        raise RuntimeError("No training data for spread model (spread missing).")

    # Common preprocessing
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # WIN MODEL (logistic)
    X_win = train_win[numeric_features + categorical_features]
    y_win = train_win["team_win"].astype(int)

    win_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("logreg", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )

    print(f"Training WIN model on {len(train_win)} rows...")
    win_model.fit(X_win, y_win)
    print("✅ Win model trained.")

    # SPREAD MODEL (linear regression) – same covariates
    X_spread = train_spread[numeric_features + categorical_features]
    y_spread = train_spread["spread"].astype(float)

    spread_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("linreg", LinearRegression()),
        ]
    )

    print(f"Training SPREAD model on {len(train_spread)} rows...")
    spread_model.fit(X_spread, y_spread)
    print("✅ Spread model trained.")

    # store feature lists in attrs for scoring later
    df.attrs["numeric_features"] = numeric_features
    df.attrs["categorical_features"] = categorical_features

    return win_model, spread_model, df


# ─────────────────────────────
# Predictions table helpers
# ─────────────────────────────
CREATE_PRED_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.game_predictions (
    gameid        TEXT PRIMARY KEY,
    season        INT NOT NULL,
    week          INT NOT NULL,
    home_team     TEXT NOT NULL,
    away_team     TEXT NOT NULL,

    homepoints    DOUBLE PRECISION,
    awaypoints    DOUBLE PRECISION,
    homespread    DOUBLE PRECISION,
    awayspread    DOUBLE PRECISION,
    totalpred     DOUBLE PRECISION,

    homewinprob   DOUBLE PRECISION,
    awaywinprob   DOUBLE PRECISION,

    model_version TEXT NOT NULL
);
"""

INSERT_PRED_SQL = """
INSERT INTO public.game_predictions (
    gameid, season, week,
    home_team, away_team,
    homepoints, awaypoints,
    homespread, awayspread, totalpred,
    homewinprob, awaywinprob,
    model_version
)
VALUES (
    %(gameid)s, %(season)s, %(week)s,
    %(home_team)s, %(away_team)s,
    %(homepoints)s, %(awaypoints)s,
    %(homespread)s, %(awayspread)s, %(totalpred)s,
    %(homewinprob)s, %(awaywinprob)s,
    %(model_version)s
)
ON CONFLICT (gameid)
DO UPDATE SET
    season        = EXCLUDED.season,
    week          = EXCLUDED.week,
    home_team     = EXCLUDED.home_team,
    away_team     = EXCLUDED.away_team,
    homepoints    = EXCLUDED.homepoints,
    awaypoints    = EXCLUDED.awaypoints,
    homespread    = EXCLUDED.homespread,
    awayspread    = EXCLUDED.awayspread,
    totalpred     = EXCLUDED.totalpred,
    homewinprob   = EXCLUDED.homewinprob,
    awaywinprob   = EXCLUDED.awaywinprob,
    model_version = EXCLUDED.model_version;
"""


def ensure_predictions_table(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_PRED_TABLE_SQL)
    conn.commit()


# ─────────────────────────────
# Main
# ─────────────────────────────
def main():
    print("Connecting to database...")
    with psycopg.connect(PG_DSN) as conn:
        current_season = get_current_season(conn)
        print(f"Current season detected: {current_season}")

        print("Building modeling table...")
        df = build_modeling_table(conn, max_season=current_season)
        print(f"Modeling table rows: {len(df)}")

        # Train models
        win_model, spread_model, df = train_models(df, current_season=current_season)
        numeric_features = df.attrs["numeric_features"]
        categorical_features = df.attrs["categorical_features"]

        # Compute average total for current season (completed games only)
        completed_mask = (
            (df["season"] == current_season)
            & df["teampoints"].notna()
            & df["opponentpoints"].notna()
        )
        if completed_mask.any():
            # Each game appears twice in team-rows, but total is symmetrical.
            # We'll compute on home perspective only (team == home_team).
            tmp = df[completed_mask & (df["team"] == df["home_team"])].copy()
            tmp["total_points"] = tmp["teampoints"] + tmp["opponentpoints"]
            avg_total = float(tmp["total_points"].mean())
        else:
            avg_total = None
        print(f"Average total for season {current_season}: {avg_total}")

        # Prepare current-season rows for prediction
        current_df = df[df["season"] == current_season].copy()
        if current_df.empty:
            print(f"No rows for current season {current_season}; nothing to predict.")
            return

        for c in categorical_features:
            current_df[c] = current_df[c].astype("category")

        X_curr = current_df[numeric_features + categorical_features]

        print(f"Scoring {len(current_df)} team-rows for season {current_season}...")
        current_df["team_win_prob"] = win_model.predict_proba(X_curr)[:, 1]
        current_df["team_spread_pred"] = spread_model.predict(X_curr)

        # Reduce to one row per game: home perspective
        home_side = current_df[current_df["team"] == current_df["home_team"]].copy()
        home_side["gameid"] = home_side["id"].astype(str)

        # Predictions from home perspective
        home_side["homewinprob"] = home_side["team_win_prob"]
        home_side["awaywinprob"] = 1.0 - home_side["homewinprob"]

        home_side["homespread"] = home_side["team_spread_pred"]
        home_side["awayspread"] = -home_side["homespread"]

        # No point predictions yet; we only set totalpred = avg_total
        home_side["homepoints"] = None
        home_side["awaypoints"] = None
        home_side["totalpred"] = avg_total

        home_side["model_version"] = MODEL_VERSION

        preds = home_side[
            [
                "gameid",
                "season",
                "week",
                "home_team",
                "away_team",
                "homepoints",
                "awaypoints",
                "homespread",
                "awayspread",
                "totalpred",
                "homewinprob",
                "awaywinprob",
                "model_version",
            ]
        ].copy()

        print(f"Prepared {len(preds)} game-level predictions for season {current_season}.")

        ensure_predictions_table(conn)

        with conn.cursor() as cur:
            print(f"Deleting existing predictions for season {current_season}...")
            cur.execute("DELETE FROM public.game_predictions WHERE season = %s;", (current_season,))

            print("Inserting new predictions...")
            records = preds.to_dict(orient="records")
            cur.executemany(INSERT_PRED_SQL, records)

        conn.commit()
        print(f"✅ Finished updating game_predictions for season {current_season}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ nightly_predictions.py failed: {e}")
        sys.exit(1)
