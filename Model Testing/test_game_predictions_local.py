#!/usr/bin/env python3
"""
Standalone local test runner for game predictions.

Reads CSV snapshots from the same folder as this file:
  - game_data.csv
  - betting_odds.csv
  - team_recruiting_rankings.csv
  - team_talent_composite.csv
  - team_returning_production.csv
  - team_advanced_game_stats.csv

Recreates the same modeling table and trains the same win/spread/total
models as etl/jobs/update_game_predictions.py, but does not write to Neon.
It prints the current-season predictions to the terminal instead.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

# Keep local model-testing runs single-threaded to avoid OpenMP SHM issues.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.simplefilter("ignore", PerformanceWarning)

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_TEST_SEASON = 2025
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
FCS_PREDICTION_TYPE = "FCS"
FBS_PREDICTION_TYPE = "FBS"


def load_csv(table_name: str) -> pd.DataFrame:
    path = MODEL_DIR / f"{table_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path, low_memory=False)


def build_advanced_stats_prior(
    team_advanced_game_stats: pd.DataFrame,
    game_data: pd.DataFrame,
    max_season: int,
) -> pd.DataFrame:
    tags = team_advanced_game_stats.copy()
    gd_adv = game_data.copy()

    tags["season"] = pd.to_numeric(tags["season"], errors="coerce")
    tags["game_id"] = pd.to_numeric(tags["game_id"], errors="coerce")
    gd_adv["id"] = pd.to_numeric(gd_adv["id"], errors="coerce")
    gd_adv["startdate"] = pd.to_datetime(gd_adv["startdate"], errors="coerce")

    merged = tags.merge(
        gd_adv[["id", "startdate", "seasontype"]],
        left_on="game_id",
        right_on="id",
        how="inner",
    )

    merged = merged[
        merged["season"].between(2015, max_season, inclusive="both")
        & merged["startdate"].notna()
    ].copy()

    season_type = (
        merged["season_type"]
        .fillna(merged["seasontype"])
        .astype(str)
        .str.lower()
    )
    merged["phase_order"] = np.select(
        [season_type.eq("regular"), season_type.eq("postseason")],
        [1, 2],
        default=3,
    )
    merged["gamedate"] = merged["startdate"].dt.date

    metric_cols = [c for c in tags.columns if c not in ADVANCED_GAME_KEY_COLS]
    for col in metric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged = merged.sort_values(["season", "team", "phase_order", "gamedate", "game_id"]).copy()

    out = merged[["game_id", "team"]].copy()
    grouped = merged.groupby(["season", "team"], sort=False)
    for col in metric_cols:
        out[f"{col}_prior_avg"] = grouped[col].transform(lambda s: s.shift(1).expanding().mean())
    return out


def build_modeling_table(max_season: int) -> pd.DataFrame:
    game_data = load_csv("game_data")
    betting_odds = load_csv("betting_odds")
    recruiting = load_csv("team_recruiting_rankings")
    talent = load_csv("team_talent_composite")
    returning = load_csv("team_returning_production")
    advanced_game_stats = load_csv("team_advanced_game_stats")

    g = game_data.copy()
    g["season"] = pd.to_numeric(g["season"], errors="coerce")
    g["id"] = pd.to_numeric(g["id"], errors="coerce")
    g["startdate"] = pd.to_datetime(g["startdate"], errors="coerce")
    g["gamedate"] = g["startdate"].dt.date
    g["homepoints"] = pd.to_numeric(g["homepoints"], errors="coerce")
    g["awaypoints"] = pd.to_numeric(g["awaypoints"], errors="coerce")
    g["homeclassification"] = g["homeclassification"].fillna("").astype(str).str.lower()
    g["awayclassification"] = g["awayclassification"].fillna("").astype(str).str.lower()

    home_fbs_seasons = g[g["homeclassification"].eq("fbs")][["hometeam", "season"]].rename(
        columns={"hometeam": "team"}
    )
    away_fbs_seasons = g[g["awayclassification"].eq("fbs")][["awayteam", "season"]].rename(
        columns={"awayteam": "team"}
    )
    first_fbs_seasons = (
        pd.concat([home_fbs_seasons, away_fbs_seasons], ignore_index=True)
        .dropna(subset=["team", "season"])
        .groupby("team", as_index=False)["season"]
        .min()
        .rename(columns={"season": "first_fbs_season"})
    )

    g = g[
        g["season"].between(2015, max_season, inclusive="both")
        & (
            (g["homeclassification"].eq("fbs") & g["awayclassification"].isin(["fbs", "fcs"]))
            | (g["awayclassification"].eq("fbs") & g["homeclassification"].isin(["fbs", "fcs"]))
        )
        & g["startdate"].notna()
    ][
        [
            "id",
            "season",
            "week",
            "gamedate",
            "seasontype",
            "neutralsite",
            "conferencegame",
            "hometeam",
            "awayteam",
            "homeclassification",
            "awayclassification",
            "homepoints",
            "awaypoints",
        ]
    ].copy()

    odds = betting_odds.copy()
    odds["Id"] = pd.to_numeric(odds["Id"], errors="coerce")
    for col in ["Spread", "OverUnder"]:
        odds[col] = pd.to_numeric(odds[col], errors="coerce")
    odds = (
        odds.groupby("Id", dropna=True, as_index=False)
        .agg(
            avg_spread=("Spread", "mean"),
            avg_over_under=("OverUnder", "mean"),
        )
        .rename(columns={"Id": "id"})
    )

    recruiting = recruiting.copy()
    recruiting["year"] = pd.to_numeric(recruiting["year"], errors="coerce")
    recruiting["points"] = pd.to_numeric(recruiting["points"], errors="coerce")

    talent = talent.copy()
    talent["year"] = pd.to_numeric(talent["year"], errors="coerce")
    talent["talent"] = pd.to_numeric(talent["talent"], errors="coerce")

    returning = returning.copy()
    returning["season"] = pd.to_numeric(returning["season"], errors="coerce")
    returning["total_ppa"] = pd.to_numeric(returning["total_ppa"], errors="coerce")

    advanced_prior = build_advanced_stats_prior(advanced_game_stats, game_data, max_season)
    advanced_actual = advanced_game_stats.copy()
    advanced_actual["game_id"] = pd.to_numeric(advanced_actual["game_id"], errors="coerce")
    advanced_actual_metric_cols = [
        c for c in advanced_actual.columns if c not in ADVANCED_GAME_KEY_COLS
    ]
    for col in advanced_actual_metric_cols:
        advanced_actual[col] = pd.to_numeric(advanced_actual[col], errors="coerce")

    df = g.merge(odds, on="id", how="left")
    df = df.merge(
        first_fbs_seasons.rename(
            columns={"team": "awayteam", "first_fbs_season": "away_first_fbs_season"}
        ),
        on="awayteam",
        how="left",
    )
    df = df.merge(
        first_fbs_seasons.rename(
            columns={"team": "hometeam", "first_fbs_season": "home_first_fbs_season"}
        ),
        on="hometeam",
        how="left",
    )
    df = df.merge(
        recruiting[["team", "year", "points"]].rename(
            columns={"team": "awayteam", "year": "season", "points": "away_recruiting_points"}
        ),
        on=["awayteam", "season"],
        how="left",
    )
    df = df.merge(
        recruiting[["team", "year", "points"]].rename(
            columns={"team": "hometeam", "year": "season", "points": "home_recruiting_points"}
        ),
        on=["hometeam", "season"],
        how="left",
    )
    df = df.merge(
        talent[["team", "year", "talent"]].rename(
            columns={"team": "awayteam", "year": "season", "talent": "away_talent"}
        ),
        on=["awayteam", "season"],
        how="left",
    )
    df = df.merge(
        talent[["team", "year", "talent"]].rename(
            columns={"team": "hometeam", "year": "season", "talent": "home_talent"}
        ),
        on=["hometeam", "season"],
        how="left",
    )
    df = df.merge(
        returning[["team", "season", "total_ppa"]].rename(
            columns={"team": "awayteam", "total_ppa": "away_returning_production_total_ppa"}
        ),
        on=["awayteam", "season"],
        how="left",
    )
    df = df.merge(
        returning[["team", "season", "total_ppa"]].rename(
            columns={"team": "hometeam", "total_ppa": "home_returning_production_total_ppa"}
        ),
        on=["hometeam", "season"],
        how="left",
    )
    advanced_prior_cols = [c for c in advanced_prior.columns if c not in ["game_id", "team"]]
    away_rename = {"game_id": "id", "team": "awayteam"}
    home_rename = {"game_id": "id", "team": "hometeam"}
    for col in advanced_prior_cols:
        away_rename[col] = f"away_{col}"
        home_rename[col] = f"home_{col}"

    df = df.merge(
        advanced_prior.rename(columns=away_rename),
        on=["id", "awayteam"],
        how="left",
    )
    df = df.merge(
        advanced_prior.rename(columns=home_rename),
        on=["id", "hometeam"],
        how="left",
    )

    away_actual_rename = {"game_id": "id", "team": "awayteam"}
    home_actual_rename = {"game_id": "id", "team": "hometeam"}
    for col in advanced_actual_metric_cols:
        away_actual_rename[col] = f"away_{col}_actual"
        home_actual_rename[col] = f"home_{col}_actual"

    df = df.merge(
        advanced_actual[["game_id", "team"] + advanced_actual_metric_cols].rename(
            columns=away_actual_rename
        ),
        on=["id", "awayteam"],
        how="left",
    )
    df = df.merge(
        advanced_actual[["game_id", "team"] + advanced_actual_metric_cols].rename(
            columns=home_actual_rename
        ),
        on=["id", "hometeam"],
        how="left",
    )

    diff_pairs = {
        "recruiting_diff": ("away_recruiting_points", "home_recruiting_points"),
        "talent_diff": ("away_talent", "home_talent"),
        "returning_diff": (
            "away_returning_production_total_ppa",
            "home_returning_production_total_ppa",
        ),
    }
    for col in advanced_prior_cols:
        diff_pairs[f"{col}_diff"] = (f"away_{col}", f"home_{col}")
    for col in advanced_actual_metric_cols:
        diff_pairs[f"{col}_game_diff"] = (f"away_{col}_actual", f"home_{col}_actual")

    for diff_col, (away_col, home_col) in diff_pairs.items():
        if away_col not in df.columns:
            df[away_col] = np.nan
        if home_col not in df.columns:
            df[home_col] = np.nan
        df[diff_col] = (
            pd.to_numeric(df[away_col], errors="coerce")
            - pd.to_numeric(df[home_col], errors="coerce")
        )

    return df.sort_values(["season", "week", "gamedate", "id"]).reset_index(drop=True)


def recompute_preseason_diffs(df: pd.DataFrame) -> None:
    for metric, (away_col, home_col) in PRESEASON_VALUE_COLS.items():
        df[f"{metric}_diff"] = (
            pd.to_numeric(df[away_col], errors="coerce")
            - pd.to_numeric(df[home_col], errors="coerce")
        )


def normalized_classification(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index)
    return df[col].fillna("").astype(str).str.lower()


def assign_prediction_type(df: pd.DataFrame) -> None:
    is_fcs_game = (
        normalized_classification(df, "homeclassification").eq("fcs")
        | normalized_classification(df, "awayclassification").eq("fcs")
    )
    df["prediction_type"] = np.where(is_fcs_game, FCS_PREDICTION_TYPE, FBS_PREDICTION_TYPE)


def fcs_advanced_baseline_quantile(metric: str) -> float:
    if metric.startswith("defense_") or metric == "offense_stuffrate":
        return 0.95
    return 0.05


def recompute_advanced_diffs(df: pd.DataFrame) -> None:
    for feature_col in ADVANCED_DIFF_FEATURES:
        metric = feature_col.replace("_prior_avg_diff", "")
        away_col = f"away_{metric}_prior_avg"
        home_col = f"home_{metric}_prior_avg"
        if away_col not in df.columns:
            df[away_col] = np.nan
        if home_col not in df.columns:
            df[home_col] = np.nan
        df[feature_col] = (
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
            "away",
            "awayteam",
            "awayclassification",
            "away_first_fbs_season",
            {
                "recruiting": "away_recruiting_points",
                "talent": "away_talent",
                "returning": "away_returning_production_total_ppa",
            },
        ),
        (
            "home",
            "hometeam",
            "homeclassification",
            "home_first_fbs_season",
            {
                "recruiting": "home_recruiting_points",
                "talent": "home_talent",
                "returning": "home_returning_production_total_ppa",
            },
        ),
    ]

    for _, team_col, class_col, first_fbs_col, value_cols in side_specs:
        frame = df[
            ["season", team_col, class_col, first_fbs_col] + list(value_cols.values())
        ].rename(
            columns={
                team_col: "team",
                class_col: "classification",
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

    historical = team_seasons[
        (pd.to_numeric(team_seasons["season"], errors="coerce") < current_season)
        & team_seasons["classification"].fillna("").astype(str).str.lower().eq("fbs")
    ]
    second_fbs_historical = historical[
        pd.to_numeric(historical["season"], errors="coerce")
        == pd.to_numeric(historical["first_fbs_season"], errors="coerce") + 1
    ]

    second_fbs_means = second_fbs_historical[list(PRESEASON_VALUE_COLS.keys())].mean()
    overall_means = historical[list(PRESEASON_VALUE_COLS.keys())].mean()
    fill_values = second_fbs_means.fillna(overall_means).fillna(0.0)
    fcs_fill_values = {
        "recruiting": 0.0,
        "talent": 0.0,
        "returning": float(historical["returning"].quantile(0.10))
        if historical["returning"].notna().any()
        else 0.0,
    }
    historical_by_team = historical.sort_values(["team", "season"])
    latest_historical_values = {
        metric: historical_by_team.dropna(subset=[metric]).groupby("team")[metric].last()
        for metric in PRESEASON_VALUE_COLS
    }

    season_numeric = pd.to_numeric(df["season"], errors="coerce")
    for _, team_col, class_col, _, value_cols in side_specs:
        class_lower = normalized_classification(df, class_col)
        for metric, col in value_cols.items():
            df[col] = pd.to_numeric(df[col], errors="coerce")
            current_fbs_missing = (
                df[col].isna()
                & class_lower.eq("fbs")
                & season_numeric.eq(current_season)
            )
            if current_fbs_missing.any():
                df.loc[current_fbs_missing, col] = df.loc[current_fbs_missing, team_col].map(
                    latest_historical_values[metric]
                )
            fbs_missing = df[col].isna() & class_lower.eq("fbs")
            df.loc[fbs_missing, col] = float(fill_values[metric])
            df.loc[class_lower.eq("fcs"), col] = float(fcs_fill_values[metric])

    recompute_preseason_diffs(df)


def fill_fcs_advanced_inputs_with_baselines(
    df: pd.DataFrame,
    current_season: int,
) -> None:
    season_numeric = pd.to_numeric(df["season"], errors="coerce")
    historical_mask = season_numeric.lt(current_season)

    for feature_col in ADVANCED_DIFF_FEATURES:
        metric = feature_col.replace("_prior_avg_diff", "")
        side_values = []
        for side in ["away", "home"]:
            class_col = f"{side}classification"
            value_col = f"{side}_{metric}_prior_avg"
            if value_col not in df.columns:
                df[value_col] = np.nan
            fbs_mask = historical_mask & normalized_classification(df, class_col).eq("fbs")
            side_values.append(pd.to_numeric(df.loc[fbs_mask, value_col], errors="coerce"))

        historical_values = pd.concat(side_values).dropna()
        quantile = fcs_advanced_baseline_quantile(metric)
        baseline = float(historical_values.quantile(quantile)) if not historical_values.empty else 0.0

        for side in ["away", "home"]:
            class_col = f"{side}classification"
            value_col = f"{side}_{metric}_prior_avg"
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            df.loc[normalized_classification(df, class_col).eq("fcs"), value_col] = baseline

    recompute_advanced_diffs(df)


def advanced_feature_target_col(feature_col: str) -> str:
    return feature_col.replace("_prior_avg_diff", "_game_diff")


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

    model = Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers=[("num", StandardScaler(), feature_cols)])),
            ("linreg", LinearRegression()),
        ]
    )
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
                if predictions is None:
                    continue
                df.loc[predictions.index, feature_col] = predictions

    df.drop(columns=["_week_numeric", "_season_numeric", "_has_both_lines"], inplace=True)


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


def train_models(df: pd.DataFrame, current_season: int):
    df = df.copy()
    assign_prediction_type(df)

    df["has_spread_line"] = df["avg_spread"].notna() if "avg_spread" in df.columns else False
    df["has_total_line"] = df["avg_over_under"].notna() if "avg_over_under" in df.columns else False

    fill_preseason_inputs_with_second_fbs_averages(df, current_season)
    fill_fcs_advanced_inputs_with_baselines(df, current_season)
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
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0.0)

    df["spread_target"] = pd.to_numeric(df["homepoints"], errors="coerce") - pd.to_numeric(df["awaypoints"], errors="coerce")
    df["win_target"] = (pd.to_numeric(df["homepoints"], errors="coerce") > pd.to_numeric(df["awaypoints"], errors="coerce")).astype(float)
    df["total_points_target"] = pd.to_numeric(df["homepoints"], errors="coerce") + pd.to_numeric(df["awaypoints"], errors="coerce")

    train_df = df[
        (df["season"] < current_season)
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

    print("Training models")
    print(f"Model 1 WIN/SPREAD features: {len(model_1_spread_features)}")
    print(f"Model 1 TOTAL features: {len(model_1_total_features)}")
    print(f"Model 2 features: {len(model_2_features)}")

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
    current_df = modeled_df[modeled_df["season"] == current_season].copy()
    if current_df.empty:
        raise RuntimeError(f"No rows found for current season {current_season}.")

    win_line_features = modeled_df.attrs["win_line_features"]
    spread_line_features = modeled_df.attrs["spread_line_features"]
    total_line_features = modeled_df.attrs["total_line_features"]
    win_model_2_features = modeled_df.attrs["win_model_2_features"]
    spread_model_2_features = modeled_df.attrs["spread_model_2_features"]
    total_model_2_features = modeled_df.attrs["total_model_2_features"]

    current_df["homewinprob_with_spread"] = model_bundle["win_line"].predict_proba(
        current_df[win_line_features]
    )[:, 1]
    current_df["homewinprob_without_spread"] = model_bundle["win_model_2"].predict_proba(
        current_df[win_model_2_features]
    )[:, 1]
    current_df["homespread_with_spread"] = -model_bundle["spread_line"].predict(
        current_df[spread_line_features]
    )
    current_df["homespread_without_spread"] = -model_bundle["spread_model_2"].predict(
        current_df[spread_model_2_features]
    )
    current_df["totalpred_with_total"] = model_bundle["total_line"].predict(
        current_df[total_line_features]
    )
    current_df["totalpred_without_total"] = model_bundle["total_model_2"].predict(
        current_df[total_model_2_features]
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
    current_df = current_df.sort_values(["week", "id"]).reset_index(drop=True)
    return current_df


def compute_win_metrics(actual: pd.Series, predicted_prob: pd.Series) -> dict[str, float]:
    y_true = pd.to_numeric(pd.Series(actual), errors="coerce")
    y_prob = pd.to_numeric(pd.Series(predicted_prob), errors="coerce")
    mask = y_true.notna() & y_prob.notna()
    y_true = y_true[mask].astype(float)
    y_prob = y_prob[mask].clip(1e-6, 1 - 1e-6).astype(float)
    if y_true.empty:
        return {"count": 0.0}

    y_pred = (y_prob >= 0.5).astype(float)
    accuracy = float((y_pred == y_true).mean())
    brier = float(((y_prob - y_true) ** 2).mean())
    log_loss = float((-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))).mean())
    return {
        "count": float(len(y_true)),
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
    }


def compute_regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    y_true = pd.to_numeric(pd.Series(actual), errors="coerce")
    y_pred = pd.to_numeric(pd.Series(predicted), errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask].astype(float)
    y_pred = y_pred[mask].astype(float)
    if y_true.empty:
        return {"count": 0.0}

    err = y_pred - y_true
    mae = float(err.abs().mean())
    rmse = float(np.sqrt((err**2).mean()))
    bias = float(err.mean())
    return {
        "count": float(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


def format_metric_dict(metrics: dict[str, float]) -> str:
    if metrics.get("count", 0.0) == 0.0:
        return "count=0"
    parts = []
    for key, value in metrics.items():
        if key == "count":
            parts.append(f"count={int(value)}")
        else:
            parts.append(f"{key}={value:.4f}")
    return " ".join(parts)


def print_performance(preds: pd.DataFrame, current_season: int) -> None:
    completed = preds[preds["homepoints"].notna() & preds["awaypoints"].notna()].copy()
    completed["actual_home_win"] = (completed["homepoints"] > completed["awaypoints"]).astype(float)
    completed["actual_homespread"] = completed["awaypoints"] - completed["homepoints"]
    completed["actual_total_points"] = completed["homepoints"] + completed["awaypoints"]

    completed_spread_subset = completed[completed["has_spread_line"].astype(bool)].copy()
    completed_total_subset = completed[completed["has_total_line"].astype(bool)].copy()

    print()
    print(f"Summary for season {current_season}")
    print(f"Games scored: {len(preds)}")
    print(f"Completed games: {len(completed)}")
    print(f"Games with spread line: {int(preds['has_spread_line'].sum())}")
    print(f"Games with total line: {int(preds['has_total_line'].sum())}")
    if "prediction_type" in preds.columns:
        print(f"FCS flagged games: {int(preds['prediction_type'].eq(FCS_PREDICTION_TYPE).sum())}")

    print()
    print("Prediction performance on completed games")
    if completed.empty:
        print("No completed games with actual scores available in the target season.")
    else:
        print(
            "chosen home win: "
            + format_metric_dict(compute_win_metrics(completed["actual_home_win"], completed["homewinprob"]))
        )
        print(
            "chosen homespread: "
            + format_metric_dict(compute_regression_metrics(completed["actual_homespread"], completed["homespread"]))
        )
        print(
            "chosen totalpred: "
            + format_metric_dict(compute_regression_metrics(completed["actual_total_points"], completed["totalpred"]))
        )

    if "prediction_type" in preds.columns:
        fcs_games = preds[preds["prediction_type"].eq(FCS_PREDICTION_TYPE)].copy()
    else:
        fcs_games = pd.DataFrame()
    if not fcs_games.empty:
        print()
        print("FCS flagged prediction sample")
        sample_cols = [
            "week",
            "awayteam",
            "hometeam",
            "has_spread_line",
            "has_total_line",
            "homespread",
            "totalpred",
            "homewinprob",
        ]
        sample = fcs_games[sample_cols].head(25).copy()
        for col in ["homespread", "totalpred", "homewinprob"]:
            sample[col] = pd.to_numeric(sample[col], errors="coerce").round(3)
        print(sample.to_string(index=False))

    print()
    print("Model 1 vs Model 2 performance on completed games with spread lines")
    if completed_spread_subset.empty:
        print("No completed games with spread lines available in the target season.")
    else:
        print(
            "win Model 1: "
            + format_metric_dict(
                compute_win_metrics(
                    completed_spread_subset["actual_home_win"],
                    completed_spread_subset["homewinprob_with_spread"],
                )
            )
        )
        print(
            "win Model 2: "
            + format_metric_dict(
                compute_win_metrics(
                    completed_spread_subset["actual_home_win"],
                    completed_spread_subset["homewinprob_without_spread"],
                )
            )
        )
        print(
            "spread Model 1: "
            + format_metric_dict(
                compute_regression_metrics(
                    completed_spread_subset["actual_homespread"],
                    completed_spread_subset["homespread_with_spread"],
                )
            )
        )
        print(
            "spread Model 2: "
            + format_metric_dict(
                compute_regression_metrics(
                    completed_spread_subset["actual_homespread"],
                    completed_spread_subset["homespread_without_spread"],
                )
            )
        )

    print()
    print("Model 1 vs Model 2 performance on completed games with total lines")
    if completed_total_subset.empty:
        print("No completed games with total lines available in the target season.")
    else:
        print(
            "total Model 1: "
            + format_metric_dict(
                compute_regression_metrics(
                    completed_total_subset["actual_total_points"],
                    completed_total_subset["totalpred_with_total"],
                )
            )
        )
        print(
            "total Model 2: "
            + format_metric_dict(
                compute_regression_metrics(
                    completed_total_subset["actual_total_points"],
                    completed_total_subset["totalpred_without_total"],
                )
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Local CSV-based game prediction tester.")
    parser.add_argument(
        "--season",
        type=int,
        default=DEFAULT_TEST_SEASON,
        help=f"Season to score. Defaults to {DEFAULT_TEST_SEASON}.",
    )
    args = parser.parse_args()

    current_season = args.season

    print(f"Building modeling table from CSVs in {MODEL_DIR} ...")
    df = build_modeling_table(max_season=current_season)
    print(f"Modeling table rows: {len(df)}")

    model_bundle, modeled_df = train_models(df, current_season)
    preds = score_current_season(model_bundle, modeled_df, current_season)
    print_performance(preds, current_season)


if __name__ == "__main__":
    main()
