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
  - MODEL_VERSION  : a tag for this model version (default 'wp_spread_v1')
"""

import sys
import psycopg
import pandas as pd
import numpy as np

from etl.common_config import load_config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression

MODEL_VERSION = "preds_2026"

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
    """

    sql = f"""
    WITH g AS (
      SELECT
        id,
        season,
        week,
        CAST(startdate AS date) AS gamedate,
        seasontype,
        neutralsite,
        conferencegame,
        hometeam,
        awayteam,
        homepoints,
        awaypoints
      FROM public.game_data
      WHERE season BETWEEN 2015 AND %s
        AND homeclassification = 'fbs'
        AND awayclassification = 'fbs'
        AND startdate IS NOT NULL
    ),
    odds AS (
      SELECT
        CAST("Id" AS BIGINT) AS id,
        AVG("Spread") AS avg_spread,
        AVG("OpeningSpread") AS avg_opening_spread,
        AVG("OverUnder") AS avg_over_under,
        AVG("OpeningOverUnder") AS avg_opening_over_under
      FROM public.betting_odds
      GROUP BY 1
    ),
    advanced_stats_prior AS (
      SELECT
        t.game_id,
        t.team,
        AVG(t.offense_totalppa) OVER (
          PARTITION BY t.season, t.team
          ORDER BY
            t.phase_order,
            t.gamedate,
            t.game_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS offense_totalppa_prior_avg,
        AVG(t.offense_successrate) OVER (
          PARTITION BY t.season, t.team
          ORDER BY
            t.phase_order,
            t.gamedate,
            t.game_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS offense_successrate_prior_avg,
        AVG(t.offense_explosiveness) OVER (
          PARTITION BY t.season, t.team
          ORDER BY
            t.phase_order,
            t.gamedate,
            t.game_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS offense_explosiveness_prior_avg,
        AVG(t.defense_totalppa) OVER (
          PARTITION BY t.season, t.team
          ORDER BY
            t.phase_order,
            t.gamedate,
            t.game_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS defense_totalppa_prior_avg,
        AVG(t.defense_successrate) OVER (
          PARTITION BY t.season, t.team
          ORDER BY
            t.phase_order,
            t.gamedate,
            t.game_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS defense_successrate_prior_avg,
        AVG(t.defense_explosiveness) OVER (
          PARTITION BY t.season, t.team
          ORDER BY
            t.phase_order,
            t.gamedate,
            t.game_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS defense_explosiveness_prior_avg
      FROM (
        SELECT
          tags.game_id,
          tags.season,
          tags.team,
          CAST(gd_adv.startdate AS date) AS gamedate,
          CASE
            WHEN LOWER(COALESCE(tags.season_type, gd_adv.seasontype)) = 'regular' THEN 1
            WHEN LOWER(COALESCE(tags.season_type, gd_adv.seasontype)) = 'postseason' THEN 2
            ELSE 3
          END AS phase_order,
          tags.offense_totalppa,
          tags.offense_successrate,
          tags.offense_explosiveness,
          tags.defense_totalppa,
          tags.defense_successrate,
          tags.defense_explosiveness
        FROM public.team_advanced_game_stats tags
        INNER JOIN public.game_data gd_adv
          ON gd_adv.id = tags.game_id
        WHERE tags.season BETWEEN 2015 AND %s
          AND gd_adv.startdate IS NOT NULL
      ) t
    )
    SELECT
      g.*,
      o.avg_spread,
      o.avg_opening_spread,
      o.avg_over_under,
      o.avg_opening_over_under,
      arr.points AS away_recruiting_points,
      hrr.points AS home_recruiting_points,
      atc.talent AS away_talent,
      htc.talent AS home_talent,
      arp.total_ppa AS away_returning_production_total_ppa,
      hrp.total_ppa AS home_returning_production_total_ppa,
      aasp.offense_totalppa_prior_avg AS away_offense_totalppa_prior_avg,
      hasp.offense_totalppa_prior_avg AS home_offense_totalppa_prior_avg,
      aasp.offense_successrate_prior_avg AS away_offense_successrate_prior_avg,
      hasp.offense_successrate_prior_avg AS home_offense_successrate_prior_avg,
      aasp.offense_explosiveness_prior_avg AS away_offense_explosiveness_prior_avg,
      hasp.offense_explosiveness_prior_avg AS home_offense_explosiveness_prior_avg,
      aasp.defense_totalppa_prior_avg AS away_defense_totalppa_prior_avg,
      hasp.defense_totalppa_prior_avg AS home_defense_totalppa_prior_avg,
      aasp.defense_successrate_prior_avg AS away_defense_successrate_prior_avg,
      hasp.defense_successrate_prior_avg AS home_defense_successrate_prior_avg,
      aasp.defense_explosiveness_prior_avg AS away_defense_explosiveness_prior_avg,
      hasp.defense_explosiveness_prior_avg AS home_defense_explosiveness_prior_avg
    FROM g
    LEFT JOIN odds o
      ON o.id = g.id
    LEFT JOIN public.team_recruiting_rankings arr
      ON arr.team = g.awayteam
     AND arr.year = g.season
    LEFT JOIN public.team_recruiting_rankings hrr
      ON hrr.team = g.hometeam
     AND hrr.year = g.season
    LEFT JOIN public.team_talent_composite atc
      ON atc.team = g.awayteam
     AND atc.year = g.season
    LEFT JOIN public.team_talent_composite htc
      ON htc.team = g.hometeam
     AND htc.year = g.season
    LEFT JOIN public.team_returning_production arp
      ON arp.team = g.awayteam
     AND arp.season = g.season
    LEFT JOIN public.team_returning_production hrp
      ON hrp.team = g.hometeam
     AND hrp.season = g.season
    LEFT JOIN advanced_stats_prior aasp
      ON aasp.game_id = g.id
     AND aasp.team = g.awayteam
    LEFT JOIN advanced_stats_prior hasp
      ON hasp.game_id = g.id
     AND hasp.team = g.hometeam
    ORDER BY g.season, g.week, g.gamedate, g.id
    """
    df = pd.read_sql(sql, conn, params=(max_season, max_season))
    if df.empty:
        raise RuntimeError("Modeling query returned no rows.")
    return df


# ─────────────────────────────
# Train win + spread models
# ─────────────────────────────
def train_models(df: pd.DataFrame, current_season: int):
    """
    Train line-aware and fallback game-level models.

    Common features for every version:
      - home/away returning production
      - week
      - advanced-stat-by-week interaction terms

    Line-aware versions add:
      - avg_spread for win + spread models
      - avg_over_under for total-points model

    Fallback versions:
      - use only the common features
    """

    df = df.copy()

    base_numeric_cols = [
        "home_returning_production_total_ppa",
        "away_returning_production_total_ppa",
        "week",
    ]

    advanced_stat_cols = [
        "home_offense_totalppa_prior_avg",
        "away_offense_totalppa_prior_avg",
        "home_offense_successrate_prior_avg",
        "away_offense_successrate_prior_avg",
        "home_offense_explosiveness_prior_avg",
        "away_offense_explosiveness_prior_avg",
        "home_defense_totalppa_prior_avg",
        "away_defense_totalppa_prior_avg",
        "home_defense_successrate_prior_avg",
        "away_defense_successrate_prior_avg",
        "home_defense_explosiveness_prior_avg",
        "away_defense_explosiveness_prior_avg",
    ]

    # Week numeric
    df["week"] = df["week"].astype(float)

    # Track whether market inputs are actually available before filling nulls.
    df["has_spread_line"] = df["avg_spread"].notna() if "avg_spread" in df.columns else False
    df["has_total_line"] = df["avg_over_under"].notna() if "avg_over_under" in df.columns else False

    # Ensure all numeric predictors exist, coerce to numeric, then fill missing.
    numeric_source_cols = base_numeric_cols + advanced_stat_cols + ["avg_spread", "avg_over_under"]
    for col in numeric_source_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create week interactions for the advanced stats.
    interaction_cols = []
    for col in advanced_stat_cols:
        inter_col = f"{col}_week"
        interaction_cols.append(inter_col)
        df[inter_col] = df[col] * df["week"]

    # Fill predictors after interaction creation so week 1 / missing priors resolve to 0.
    model_numeric_cols = base_numeric_cols + advanced_stat_cols + interaction_cols + ["avg_spread", "avg_over_under"]
    for col in model_numeric_cols:
        df[col] = df[col].fillna(0.0)

    # Targets from the home-team perspective.
    df["spread_target"] = pd.to_numeric(df["homepoints"], errors="coerce") - pd.to_numeric(df["awaypoints"], errors="coerce")
    df["win_target"] = (pd.to_numeric(df["homepoints"], errors="coerce") > pd.to_numeric(df["awaypoints"], errors="coerce")).astype(float)
    df["total_points_target"] = pd.to_numeric(df["homepoints"], errors="coerce") + pd.to_numeric(df["awaypoints"], errors="coerce")

    common_train_mask = (
        (df["season"] < current_season)
        & df["homepoints"].notna()
        & df["awaypoints"].notna()
    )

    train_df = df[common_train_mask].copy()
    if train_df.empty:
        raise RuntimeError("No completed historical games available for model training.")

    common_features = base_numeric_cols + interaction_cols
    spread_win_line_features = common_features + ["avg_spread"]
    spread_win_fallback_features = common_features
    total_line_features = common_features + ["avg_over_under"]
    total_fallback_features = common_features

    train_df_spread_line = train_df[train_df["has_spread_line"]].copy()
    train_df_total_line = train_df[train_df["has_total_line"]].copy()

    def build_linear_pipeline(feature_cols):
        preprocess = ColumnTransformer(
            transformers=[("num", StandardScaler(), feature_cols)]
        )
        return Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("linreg", LinearRegression()),
            ]
        )

    def build_logistic_pipeline(feature_cols):
        preprocess = ColumnTransformer(
            transformers=[("num", StandardScaler(), feature_cols)]
        )
        return Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("logreg", LogisticRegression(max_iter=1000, solver="lbfgs")),
            ]
        )

    if train_df_spread_line.empty:
        raise RuntimeError("No historical games with spread data available for line-aware win/spread models.")
    if train_df_total_line.empty:
        raise RuntimeError("No historical games with total-line data available for line-aware total model.")

    # Line-aware WIN/SPREAD models
    win_model = build_logistic_pipeline(spread_win_line_features)
    print(f"Training WIN model (line-aware) on {len(train_df_spread_line)} rows...")
    win_model.fit(
        train_df_spread_line[spread_win_line_features],
        train_df_spread_line["win_target"].astype(int),
    )
    print("✅ Win model (line-aware) trained.")

    spread_model = build_linear_pipeline(spread_win_line_features)
    print(f"Training SPREAD model (line-aware) on {len(train_df_spread_line)} rows...")
    spread_model.fit(
        train_df_spread_line[spread_win_line_features],
        train_df_spread_line["spread_target"].astype(float),
    )
    print("✅ Spread model (line-aware) trained.")

    # Fallback WIN/SPREAD models
    win_fallback_model = build_logistic_pipeline(spread_win_fallback_features)
    print(f"Training WIN model (fallback) on {len(train_df)} rows...")
    win_fallback_model.fit(
        train_df[spread_win_fallback_features],
        train_df["win_target"].astype(int),
    )
    print("✅ Win model (fallback) trained.")

    spread_fallback_model = build_linear_pipeline(spread_win_fallback_features)
    print(f"Training SPREAD model (fallback) on {len(train_df)} rows...")
    spread_fallback_model.fit(
        train_df[spread_win_fallback_features],
        train_df["spread_target"].astype(float),
    )
    print("✅ Spread model (fallback) trained.")

    # Line-aware TOTAL model
    total_model = build_linear_pipeline(total_line_features)
    print(f"Training TOTAL model (line-aware) on {len(train_df_total_line)} rows...")
    total_model.fit(
        train_df_total_line[total_line_features],
        train_df_total_line["total_points_target"].astype(float),
    )
    print("✅ Total model (line-aware) trained.")

    # Fallback TOTAL model
    total_fallback_model = build_linear_pipeline(total_fallback_features)
    print(f"Training TOTAL model (fallback) on {len(train_df)} rows...")
    total_fallback_model.fit(
        train_df[total_fallback_features],
        train_df["total_points_target"].astype(float),
    )
    print("✅ Total model (fallback) trained.")

    # Store feature lists in attrs for scoring later.
    df.attrs["spread_win_line_features"] = spread_win_line_features
    df.attrs["spread_win_fallback_features"] = spread_win_fallback_features
    df.attrs["total_line_features"] = total_line_features
    df.attrs["total_fallback_features"] = total_fallback_features
    df.attrs["categorical_features"] = []

    model_bundle = {
        "win_line": win_model,
        "spread_line": spread_model,
        "total_line": total_model,
        "win_fallback": win_fallback_model,
        "spread_fallback": spread_fallback_model,
        "total_fallback": total_fallback_model,
    }

    return model_bundle, df


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
    cfg = load_config()
    print("Connecting to database...")
    with psycopg.connect(cfg.pg_dsn) as conn:
        current_season = cfg.season
        print(f"Target season: {current_season}")

        print("Building modeling table...")
        df = build_modeling_table(conn, max_season=current_season)
        print(f"Modeling table rows: {len(df)}")

        # Train models
        model_bundle, modeled_df = train_models(df, current_season)

        current_df = modeled_df[modeled_df["season"] == current_season].copy()
        if current_df.empty:
            raise RuntimeError(f"No rows found for current season {current_season}.")

        spread_win_line_features = modeled_df.attrs["spread_win_line_features"]
        spread_win_fallback_features = modeled_df.attrs["spread_win_fallback_features"]
        total_line_features = modeled_df.attrs["total_line_features"]
        total_fallback_features = modeled_df.attrs["total_fallback_features"]

        print(f"Scoring {len(current_df)} games for season {current_season}...")
        spread_line_mask = current_df["has_spread_line"].astype(bool)
        total_line_mask = current_df["has_total_line"].astype(bool)

        current_df["homewinprob"] = np.nan
        current_df["homespread"] = np.nan
        current_df["totalpred"] = np.nan

        if spread_line_mask.any():
            current_df.loc[spread_line_mask, "homewinprob"] = model_bundle["win_line"].predict_proba(
                current_df.loc[spread_line_mask, spread_win_line_features]
            )[:, 1]
            current_df.loc[spread_line_mask, "homespread"] = model_bundle["spread_line"].predict(
                current_df.loc[spread_line_mask, spread_win_line_features]
            )

        if (~spread_line_mask).any():
            current_df.loc[~spread_line_mask, "homewinprob"] = model_bundle["win_fallback"].predict_proba(
                current_df.loc[~spread_line_mask, spread_win_fallback_features]
            )[:, 1]
            current_df.loc[~spread_line_mask, "homespread"] = model_bundle["spread_fallback"].predict(
                current_df.loc[~spread_line_mask, spread_win_fallback_features]
            )

        if total_line_mask.any():
            current_df.loc[total_line_mask, "totalpred"] = model_bundle["total_line"].predict(
                current_df.loc[total_line_mask, total_line_features]
            )

        if (~total_line_mask).any():
            current_df.loc[~total_line_mask, "totalpred"] = model_bundle["total_fallback"].predict(
                current_df.loc[~total_line_mask, total_fallback_features]
            )

        current_df["awaywinprob"] = 1.0 - current_df["homewinprob"]
        current_df["awayspread"] = -current_df["homespread"]

        current_df["gameid"] = current_df["id"].astype(str)
        current_df["home_team"] = current_df["hometeam"]
        current_df["away_team"] = current_df["awayteam"]
        current_df["model_version"] = MODEL_VERSION

        preds = current_df[
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
