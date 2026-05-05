#!/usr/bin/env python3
"""
Train game prediction models from Neon data and update public.game_predictions.
"""

from __future__ import annotations

import os
import sys

# Keep model runs single-threaded to avoid OpenMP SHM issues in small runners.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import psycopg
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from etl.common_config import load_config


MODEL_VERSION = "preds_2026"
ADVANCED_GAME_KEY_COLS = ["game_id", "season", "season_type", "week", "team", "opponent"]

BASE_DIFF_FEATURES = [
    "recruiting_diff",
    "talent_diff",
    "returning_diff",
    "offense_ppa_prior_avg_diff",
    "offense_totalppa_prior_avg_diff",
    "offense_successrate_prior_avg_diff",
    "offense_explosiveness_prior_avg_diff",
    "defense_ppa_prior_avg_diff",
    "defense_totalppa_prior_avg_diff",
    "defense_successrate_prior_avg_diff",
    "defense_explosiveness_prior_avg_diff",
]

MODEL_2_EXTRA_DIFF_FEATURES = [
    "offense_stuffrate_prior_avg_diff",
    "offense_openfieldyardstotal_prior_avg_diff",
    "offense_standarddowns_ppa_prior_avg_diff",
    "offense_standarddowns_successrate_prior_avg_diff",
    "offense_passingdowns_ppa_prior_avg_diff",
    "offense_passingdowns_successrate_prior_avg_diff",
    "offense_rushingplays_totalppa_prior_avg_diff",
    "offense_rushingplays_successrate_prior_avg_diff",
    "offense_rushingplays_explosiveness_prior_avg_diff",
    "offense_passingplays_totalppa_prior_avg_diff",
    "offense_passingplays_explosiveness_prior_avg_diff",
    "defense_plays_prior_avg_diff",
    "defense_drives_prior_avg_diff",
    "defense_powersuccess_prior_avg_diff",
    "defense_secondlevelyardstotal_prior_avg_diff",
    "defense_standarddowns_ppa_prior_avg_diff",
    "defense_passingdowns_successrate_prior_avg_diff",
    "defense_passingdowns_explosiveness_prior_avg_diff",
    "defense_rushingplays_ppa_prior_avg_diff",
    "defense_passingplays_totalppa_prior_avg_diff",
    "defense_passingplays_explosiveness_prior_avg_diff",
]

ADVANCED_DIFF_FEATURES = [
    col
    for col in BASE_DIFF_FEATURES + MODEL_2_EXTRA_DIFF_FEATURES
    if col.endswith("_prior_avg_diff")
]
ADVANCED_METRICS = [
    col.replace("_prior_avg_diff", "")
    for col in ADVANCED_DIFF_FEATURES
]

PRESEASON_VALUE_COLS = {
    "recruiting": ("away_recruiting_points", "home_recruiting_points"),
    "talent": ("away_talent", "home_talent"),
    "returning": (
        "away_returning_production_total_ppa",
        "home_returning_production_total_ppa",
    ),
}

PRESEASON_IMPUTER_FEATURES = [
    "away_recruiting_points",
    "away_talent",
    "away_returning_production_total_ppa",
    "home_recruiting_points",
    "home_talent",
    "home_returning_production_total_ppa",
]

LINE_AWARE_IMPUTER_FEATURES = PRESEASON_IMPUTER_FEATURES + ["avg_spread", "avg_over_under"]
MIN_AUXILIARY_TRAINING_ROWS = 10


def build_modeling_table(conn, max_season: int) -> pd.DataFrame:
    prior_select_sql = ",\n        ".join(
        f"""
        AVG(t.{metric}) OVER (
          PARTITION BY t.season, t.team
          ORDER BY t.phase_order, t.gamedate, t.game_id
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS {metric}_prior_avg""".strip()
        for metric in ADVANCED_METRICS
    )
    advanced_base_cols_sql = ",\n          ".join(f"tags.{metric}" for metric in ADVANCED_METRICS)
    away_prior_select_sql = ",\n      ".join(
        f"aasp.{metric}_prior_avg AS away_{metric}_prior_avg,\n"
        f"      hasp.{metric}_prior_avg AS home_{metric}_prior_avg"
        for metric in ADVANCED_METRICS
    )
    away_actual_select_sql = ",\n      ".join(
        f"aactual.{metric} AS away_{metric}_actual,\n"
        f"      hactual.{metric} AS home_{metric}_actual"
        for metric in ADVANCED_METRICS
    )

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
    fbs_team_seasons AS (
      SELECT hometeam AS team, season
      FROM public.game_data
      WHERE homeclassification = 'fbs'
        AND hometeam IS NOT NULL
      UNION
      SELECT awayteam AS team, season
      FROM public.game_data
      WHERE awayclassification = 'fbs'
        AND awayteam IS NOT NULL
    ),
    first_fbs_seasons AS (
      SELECT team, MIN(season) AS first_fbs_season
      FROM fbs_team_seasons
      GROUP BY team
    ),
    odds AS (
      SELECT
        CAST("Id" AS BIGINT) AS id,
        AVG("Spread") AS avg_spread,
        AVG("OverUnder") AS avg_over_under
      FROM public.betting_odds
      GROUP BY 1
    ),
    advanced_base AS (
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
        {advanced_base_cols_sql}
      FROM public.team_advanced_game_stats tags
      INNER JOIN public.game_data gd_adv
        ON gd_adv.id = tags.game_id
      WHERE tags.season BETWEEN 2015 AND %s
        AND gd_adv.startdate IS NOT NULL
    ),
    advanced_stats_prior AS (
      SELECT
        t.game_id,
        t.team,
        {prior_select_sql}
      FROM advanced_base t
    )
    SELECT
      g.*,
      o.avg_spread,
      o.avg_over_under,
      afs.first_fbs_season AS away_first_fbs_season,
      hfs.first_fbs_season AS home_first_fbs_season,
      arr.points AS away_recruiting_points,
      hrr.points AS home_recruiting_points,
      atc.talent AS away_talent,
      htc.talent AS home_talent,
      arp.total_ppa AS away_returning_production_total_ppa,
      hrp.total_ppa AS home_returning_production_total_ppa,
      {away_prior_select_sql},
      {away_actual_select_sql}
    FROM g
    LEFT JOIN odds o
      ON o.id = g.id
    LEFT JOIN first_fbs_seasons afs
      ON afs.team = g.awayteam
    LEFT JOIN first_fbs_seasons hfs
      ON hfs.team = g.hometeam
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
    LEFT JOIN advanced_base aactual
      ON aactual.game_id = g.id
     AND aactual.team = g.awayteam
    LEFT JOIN advanced_base hactual
      ON hactual.game_id = g.id
     AND hactual.team = g.hometeam
    ORDER BY g.season, g.week, g.gamedate, g.id
    """

    df = pd.read_sql(sql, conn, params=(max_season, max_season))
    if df.empty:
        raise RuntimeError("Modeling query returned no rows.")

    diff_pairs = {
        "recruiting_diff": ("away_recruiting_points", "home_recruiting_points"),
        "talent_diff": ("away_talent", "home_talent"),
        "returning_diff": (
            "away_returning_production_total_ppa",
            "home_returning_production_total_ppa",
        ),
    }
    for metric in ADVANCED_METRICS:
        diff_pairs[f"{metric}_prior_avg_diff"] = (
            f"away_{metric}_prior_avg",
            f"home_{metric}_prior_avg",
        )
        diff_pairs[f"{metric}_game_diff"] = (
            f"away_{metric}_actual",
            f"home_{metric}_actual",
        )

    for diff_col, (away_col, home_col) in diff_pairs.items():
        df[diff_col] = (
            pd.to_numeric(df.get(away_col), errors="coerce")
            - pd.to_numeric(df.get(home_col), errors="coerce")
        )

    return df


def recompute_preseason_diffs(df: pd.DataFrame) -> None:
    for metric, (away_col, home_col) in PRESEASON_VALUE_COLS.items():
        df[f"{metric}_diff"] = (
            pd.to_numeric(df[away_col], errors="coerce")
            - pd.to_numeric(df[home_col], errors="coerce")
        )


def fill_preseason_inputs_with_second_fbs_averages(
    df: pd.DataFrame,
    current_season: int,
) -> None:
    team_frames = []
    side_specs = [
        (
            "awayteam",
            "away_first_fbs_season",
            {
                "recruiting": "away_recruiting_points",
                "talent": "away_talent",
                "returning": "away_returning_production_total_ppa",
            },
        ),
        (
            "hometeam",
            "home_first_fbs_season",
            {
                "recruiting": "home_recruiting_points",
                "talent": "home_talent",
                "returning": "home_returning_production_total_ppa",
            },
        ),
    ]

    for team_col, first_fbs_col, value_cols in side_specs:
        frame = df[["season", team_col, first_fbs_col] + list(value_cols.values())].rename(
            columns={
                team_col: "team",
                first_fbs_col: "first_fbs_season",
                **{col: metric for metric, col in value_cols.items()},
            }
        )
        team_frames.append(frame)

    team_seasons = (
        pd.concat(team_frames, ignore_index=True)
        .dropna(subset=["team", "season"])
        .drop_duplicates(subset=["team", "season"])
    )
    for metric in PRESEASON_VALUE_COLS:
        team_seasons[metric] = pd.to_numeric(team_seasons[metric], errors="coerce")

    historical = team_seasons[pd.to_numeric(team_seasons["season"], errors="coerce") < current_season]
    second_fbs_historical = historical[
        pd.to_numeric(historical["season"], errors="coerce")
        == pd.to_numeric(historical["first_fbs_season"], errors="coerce") + 1
    ]
    fill_values = (
        second_fbs_historical[list(PRESEASON_VALUE_COLS.keys())]
        .mean()
        .fillna(historical[list(PRESEASON_VALUE_COLS.keys())].mean())
        .fillna(0.0)
    )

    for metric, (away_col, home_col) in PRESEASON_VALUE_COLS.items():
        for col in [away_col, home_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col].isna(), col] = float(fill_values[metric])

    recompute_preseason_diffs(df)


def advanced_feature_target_col(feature_col: str) -> str:
    return feature_col.replace("_prior_avg_diff", "_game_diff")


def build_linear_pipeline(feature_cols: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(transformers=[("num", StandardScaler(), feature_cols)])
    return Pipeline(steps=[("preprocess", preprocess), ("linreg", LinearRegression())])


def build_logistic_pipeline(feature_cols: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(transformers=[("num", StandardScaler(), feature_cols)])
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("logreg", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ]
    )


def predict_auxiliary_values(
    df: pd.DataFrame,
    fit_mask: pd.Series,
    predict_mask: pd.Series,
    feature_cols: list[str],
    target_col: str,
):
    if not predict_mask.any():
        return None

    fit_cols = feature_cols + [target_col]
    fit_df = df.loc[fit_mask, fit_cols].copy()
    for col in fit_cols:
        fit_df[col] = pd.to_numeric(fit_df[col], errors="coerce")
    fit_df = fit_df.dropna()
    if len(fit_df) < MIN_AUXILIARY_TRAINING_ROWS:
        return None

    predict_df = df.loc[predict_mask, feature_cols].copy()
    for col in feature_cols:
        predict_df[col] = pd.to_numeric(predict_df[col], errors="coerce")
    predict_df = predict_df.fillna(0.0)

    model = build_linear_pipeline(feature_cols)
    model.fit(fit_df[feature_cols], fit_df[target_col].astype(float))
    return pd.Series(model.predict(predict_df[feature_cols]), index=predict_df.index)


def fill_week1_advanced_diffs_with_auxiliary_models(
    df: pd.DataFrame,
    current_season: int,
) -> None:
    for col in PRESEASON_IMPUTER_FEATURES + ["avg_spread", "avg_over_under"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["_week_numeric"] = pd.to_numeric(df["week"], errors="coerce")
    df["_season_numeric"] = pd.to_numeric(df["season"], errors="coerce")
    df["_has_both_lines"] = df["avg_spread"].notna() & df["avg_over_under"].notna()

    seasons = sorted(
        int(s)
        for s in df["_season_numeric"].dropna().unique().tolist()
        if int(s) <= current_season
    )
    for feature_col in ADVANCED_DIFF_FEATURES:
        target_col = advanced_feature_target_col(feature_col)
        if target_col not in df.columns:
            continue

        df[feature_col] = pd.to_numeric(df[feature_col], errors="coerce")
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        for season in seasons:
            prediction_mask = (
                df["_season_numeric"].eq(season)
                & df["_week_numeric"].eq(1)
                & df[feature_col].isna()
            )
            if not prediction_mask.any():
                continue

            historical_week1_mask = (
                df["_season_numeric"].lt(season)
                & df["_week_numeric"].eq(1)
                & df[target_col].notna()
            )
            if not historical_week1_mask.any():
                continue

            line_prediction_mask = prediction_mask & df["_has_both_lines"]
            fallback_prediction_mask = prediction_mask & ~df["_has_both_lines"]

            line_predictions = predict_auxiliary_values(
                df,
                historical_week1_mask & df["_has_both_lines"],
                line_prediction_mask,
                LINE_AWARE_IMPUTER_FEATURES,
                target_col,
            )
            if line_predictions is None:
                line_predictions = predict_auxiliary_values(
                    df,
                    historical_week1_mask,
                    line_prediction_mask,
                    PRESEASON_IMPUTER_FEATURES,
                    target_col,
                )

            fallback_predictions = predict_auxiliary_values(
                df,
                historical_week1_mask,
                fallback_prediction_mask,
                PRESEASON_IMPUTER_FEATURES,
                target_col,
            )

            for predictions in [line_predictions, fallback_predictions]:
                if predictions is not None:
                    df.loc[predictions.index, feature_col] = predictions

    df.drop(columns=["_week_numeric", "_season_numeric", "_has_both_lines"], inplace=True)


def train_models(df: pd.DataFrame, current_season: int):
    df = df.copy()
    df["has_spread_line"] = df["avg_spread"].notna() if "avg_spread" in df.columns else False
    df["has_total_line"] = df["avg_over_under"].notna() if "avg_over_under" in df.columns else False

    fill_preseason_inputs_with_second_fbs_averages(df, current_season)
    fill_week1_advanced_diffs_with_auxiliary_models(df, current_season)

    model_1_spread_features = ["avg_spread"] + BASE_DIFF_FEATURES
    model_1_total_features = ["avg_over_under"] + BASE_DIFF_FEATURES
    model_2_features = BASE_DIFF_FEATURES + MODEL_2_EXTRA_DIFF_FEATURES
    model_numeric_cols = sorted(
        set(model_1_spread_features + model_1_total_features + model_2_features)
    )

    for col in model_numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["spread_target"] = (
        pd.to_numeric(df["homepoints"], errors="coerce")
        - pd.to_numeric(df["awaypoints"], errors="coerce")
    )
    df["win_target"] = (
        pd.to_numeric(df["homepoints"], errors="coerce")
        > pd.to_numeric(df["awaypoints"], errors="coerce")
    ).astype(float)
    df["total_points_target"] = (
        pd.to_numeric(df["homepoints"], errors="coerce")
        + pd.to_numeric(df["awaypoints"], errors="coerce")
    )

    train_df = df[
        (pd.to_numeric(df["season"], errors="coerce") < current_season)
        & df["homepoints"].notna()
        & df["awaypoints"].notna()
    ].copy()
    if train_df.empty:
        raise RuntimeError("No completed historical games available for model training.")

    train_df_spread_line = train_df[train_df["has_spread_line"]].copy()
    train_df_total_line = train_df[train_df["has_total_line"]].copy()
    if train_df_spread_line.empty:
        raise RuntimeError("No historical games with spread data available for line-aware win/spread models.")
    if train_df_total_line.empty:
        raise RuntimeError("No historical games with total-line data available for line-aware total model.")

    model_bundle = {
        "win_line": build_logistic_pipeline(model_1_spread_features),
        "spread_line": build_linear_pipeline(model_1_spread_features),
        "total_line": build_linear_pipeline(model_1_total_features),
        "win_model_2": build_logistic_pipeline(model_2_features),
        "spread_model_2": build_linear_pipeline(model_2_features),
        "total_model_2": build_linear_pipeline(model_2_features),
    }

    print(f"Training WIN model (line-aware) on {len(train_df_spread_line)} rows...")
    model_bundle["win_line"].fit(
        train_df_spread_line[model_1_spread_features],
        train_df_spread_line["win_target"].astype(int),
    )
    print(f"Training SPREAD model (line-aware) on {len(train_df_spread_line)} rows...")
    model_bundle["spread_line"].fit(
        train_df_spread_line[model_1_spread_features],
        train_df_spread_line["spread_target"].astype(float),
    )
    print(f"Training TOTAL model (line-aware) on {len(train_df_total_line)} rows...")
    model_bundle["total_line"].fit(
        train_df_total_line[model_1_total_features],
        train_df_total_line["total_points_target"].astype(float),
    )

    print(f"Training WIN model (Model 2, no line) on {len(train_df)} rows...")
    model_bundle["win_model_2"].fit(
        train_df[model_2_features],
        train_df["win_target"].astype(int),
    )
    print(f"Training SPREAD model (Model 2, no line) on {len(train_df)} rows...")
    model_bundle["spread_model_2"].fit(
        train_df[model_2_features],
        train_df["spread_target"].astype(float),
    )
    print(f"Training TOTAL model (Model 2, no line) on {len(train_df)} rows...")
    model_bundle["total_model_2"].fit(
        train_df[model_2_features],
        train_df["total_points_target"].astype(float),
    )

    df.attrs["win_line_features"] = model_1_spread_features
    df.attrs["spread_line_features"] = model_1_spread_features
    df.attrs["total_line_features"] = model_1_total_features
    df.attrs["win_model_2_features"] = model_2_features
    df.attrs["spread_model_2_features"] = model_2_features
    df.attrs["total_model_2_features"] = model_2_features
    return model_bundle, df


def score_current_season(model_bundle: dict, modeled_df: pd.DataFrame, current_season: int) -> pd.DataFrame:
    current_df = modeled_df[pd.to_numeric(modeled_df["season"], errors="coerce").eq(current_season)].copy()
    if current_df.empty:
        raise RuntimeError(f"No rows found for current season {current_season}.")

    current_df["homewinprob_with_spread"] = model_bundle["win_line"].predict_proba(
        current_df[modeled_df.attrs["win_line_features"]]
    )[:, 1]
    current_df["homewinprob_without_spread"] = model_bundle["win_model_2"].predict_proba(
        current_df[modeled_df.attrs["win_model_2_features"]]
    )[:, 1]
    current_df["homespread_with_spread"] = -model_bundle["spread_line"].predict(
        current_df[modeled_df.attrs["spread_line_features"]]
    )
    current_df["homespread_without_spread"] = -model_bundle["spread_model_2"].predict(
        current_df[modeled_df.attrs["spread_model_2_features"]]
    )
    current_df["totalpred_with_total"] = model_bundle["total_line"].predict(
        current_df[modeled_df.attrs["total_line_features"]]
    )
    current_df["totalpred_without_total"] = model_bundle["total_model_2"].predict(
        current_df[modeled_df.attrs["total_model_2_features"]]
    )

    current_df["homewinprob"] = np.where(
        current_df["has_spread_line"],
        current_df["homewinprob_with_spread"],
        current_df["homewinprob_without_spread"],
    )
    current_df["homespread"] = np.where(
        current_df["has_spread_line"],
        current_df["homespread_with_spread"],
        current_df["homespread_without_spread"],
    )
    current_df["totalpred"] = np.where(
        current_df["has_total_line"],
        current_df["totalpred_with_total"],
        current_df["totalpred_without_total"],
    )
    current_df["awaywinprob"] = 1.0 - current_df["homewinprob"]
    current_df["awayspread"] = -current_df["homespread"]
    return current_df.sort_values(["week", "id"]).reset_index(drop=True)


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


def ensure_predictions_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_PRED_TABLE_SQL)
    conn.commit()


def prediction_records(preds: pd.DataFrame) -> list[dict]:
    output = preds.copy()
    output["gameid"] = output["id"].astype(str)
    output["home_team"] = output["hometeam"]
    output["away_team"] = output["awayteam"]
    output["model_version"] = MODEL_VERSION
    output = output[
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
    ]
    output = output.astype(object).where(pd.notna(output), None)
    return output.to_dict(orient="records")


def main() -> None:
    cfg = load_config()
    print("Connecting to database...")
    with psycopg.connect(cfg.pg_dsn) as conn:
        current_season = cfg.season
        print(f"Target season: {current_season}")

        print("Building modeling table...")
        df = build_modeling_table(conn, max_season=current_season)
        print(f"Modeling table rows: {len(df)}")

        model_bundle, modeled_df = train_models(df, current_season)
        preds = score_current_season(model_bundle, modeled_df, current_season)
        records = prediction_records(preds)
        print(f"Prepared {len(records)} game-level predictions for season {current_season}.")

        ensure_predictions_table(conn)
        with conn.cursor() as cur:
            print(f"Deleting existing predictions for season {current_season}...")
            cur.execute("DELETE FROM public.game_predictions WHERE season = %s;", (current_season,))

            print("Inserting new predictions...")
            cur.executemany(INSERT_PRED_SQL, records)

        conn.commit()
        print(f"Finished updating game_predictions for season {current_season}.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"update_game_predictions.py failed: {e}")
        sys.exit(1)
