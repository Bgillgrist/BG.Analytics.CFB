#!/usr/bin/env python3
"""
Train XGBoost game prediction models from Neon data and update public.game_predictions.
"""

from __future__ import annotations

import os
import sys
import warnings

# Keep model runs single-threaded to avoid OpenMP SHM issues in small runners.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import psycopg

from etl.common_config import load_config


warnings.simplefilter("ignore", PerformanceWarning)

XGB_FBS_AWARE_MODEL_VERSION = "xgb_fbs_aware_2026"
XGB_FBS_INCOMPLETE_MODEL_VERSION = "xgb_fbs_incomplete_2026"
XGB_FCS_AWARE_MODEL_VERSION = "xgb_fcs_aware_2026"
XGB_FCS_INCOMPLETE_MODEL_VERSION = "xgb_fcs_incomplete_2026"
MODEL_VERSION_LABELS = (
    XGB_FBS_AWARE_MODEL_VERSION,
    XGB_FBS_INCOMPLETE_MODEL_VERSION,
    XGB_FCS_AWARE_MODEL_VERSION,
    XGB_FCS_INCOMPLETE_MODEL_VERSION,
)

# Kept for downstream imports/tests that reference the older names.
LINE_AWARE_MODEL_VERSION = XGB_FBS_AWARE_MODEL_VERSION
INCOMPLETE_MODEL_VERSION = XGB_FBS_INCOMPLETE_MODEL_VERSION

FCS_PREDICTION_TYPE = "FCS"
FBS_PREDICTION_TYPE = "FBS"
RANDOM_STATE = 42
XGB_N_ESTIMATORS = 500

FBS_BASE_FEATURES = [
    "home_teamrankings_rating",
    "away_teamrankings_rating",
    "home_talent",
    "away_talent",
    "home_recruiting_points",
    "away_recruiting_points",
    "home_returning_total_ppa",
    "away_returning_total_ppa",
    "home_returning_percent_ppa",
    "away_returning_percent_ppa",
    "neutral_site",
]
FBS_SPREAD_FEATURES = ["avg_spread"] + FBS_BASE_FEATURES
FBS_TOTAL_FEATURES = ["avg_over_under"] + FBS_BASE_FEATURES

FCS_BASE_FEATURES = [
    "fbs_teamrankings_rating",
    "fbs_talent",
    "fbs_recruiting_points",
    "fbs_returning_total_ppa",
    "fbs_returning_percent_ppa",
    "fbs_is_home",
    "neutral_site",
    "is_fcs_game",
]
FCS_SPREAD_FEATURES = ["fbs_spread"] + FCS_BASE_FEATURES
FCS_TOTAL_FEATURES = ["avg_over_under"] + FCS_BASE_FEATURES


def build_modeling_table(conn, max_season: int) -> pd.DataFrame:
    sql = """
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
        LOWER(homeclassification) AS homeclassification,
        LOWER(awayclassification) AS awayclassification,
        homepoints,
        awaypoints
      FROM public.game_data
      WHERE season BETWEEN 2015 AND %s
        AND (
          (LOWER(homeclassification) = 'fbs' AND LOWER(awayclassification) IN ('fbs', 'fcs'))
          OR (LOWER(awayclassification) = 'fbs' AND LOWER(homeclassification) IN ('fbs', 'fcs'))
        )
        AND startdate IS NOT NULL
    ),
    odds AS (
      SELECT
        CAST("Id" AS BIGINT) AS id,
        AVG("Spread") AS avg_spread,
        AVG("OverUnder") AS avg_over_under
      FROM public.betting_odds
      GROUP BY 1
    )
    SELECT
      g.*,
      o.avg_spread,
      o.avg_over_under,
      hrr.points AS home_recruiting_points,
      arr.points AS away_recruiting_points,
      htc.talent AS home_talent,
      atc.talent AS away_talent,
      hrp.total_ppa AS home_returning_total_ppa,
      arp.total_ppa AS away_returning_total_ppa,
      hrp.percent_ppa AS home_returning_percent_ppa,
      arp.percent_ppa AS away_returning_percent_ppa,
      hrp.usage AS home_returning_usage,
      arp.usage AS away_returning_usage,
      htr.pull_date AS home_teamrankings_pull_date,
      atr.pull_date AS away_teamrankings_pull_date,
      htr.rank AS home_teamrankings_rank,
      atr.rank AS away_teamrankings_rank,
      htr.rating AS home_teamrankings_rating,
      atr.rating AS away_teamrankings_rating
    FROM g
    LEFT JOIN odds o
      ON o.id = g.id
    LEFT JOIN public.team_recruiting_rankings hrr
      ON hrr.team = g.hometeam
     AND hrr.year = g.season
    LEFT JOIN public.team_recruiting_rankings arr
      ON arr.team = g.awayteam
     AND arr.year = g.season
    LEFT JOIN public.team_talent_composite htc
      ON htc.team = g.hometeam
     AND htc.year = g.season
    LEFT JOIN public.team_talent_composite atc
      ON atc.team = g.awayteam
     AND atc.year = g.season
    LEFT JOIN public.team_returning_production hrp
      ON hrp.team = g.hometeam
     AND hrp.season = g.season
    LEFT JOIN public.team_returning_production arp
      ON arp.team = g.awayteam
     AND arp.season = g.season
    LEFT JOIN LATERAL (
      SELECT pull_date, rank, rating
      FROM public.teamrankings_predictive_ratings tr
      WHERE tr.season = g.season
        AND tr.team = g.hometeam
        AND tr.pull_date < g.gamedate
      ORDER BY tr.pull_date DESC
      LIMIT 1
    ) htr ON TRUE
    LEFT JOIN LATERAL (
      SELECT pull_date, rank, rating
      FROM public.teamrankings_predictive_ratings tr
      WHERE tr.season = g.season
        AND tr.team = g.awayteam
        AND tr.pull_date < g.gamedate
      ORDER BY tr.pull_date DESC
      LIMIT 1
    ) atr ON TRUE
    ORDER BY g.season, g.week, g.gamedate, g.id
    """

    df = pd.read_sql(sql, conn, params=(max_season,))
    if df.empty:
        raise RuntimeError("Modeling query returned no rows.")
    return df


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


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _indicator_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    raw = df[col]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False).astype(int)
    text = raw.fillna("").astype(str).str.strip().str.lower()
    truthy = text.isin({"true", "t", "1", "yes", "y"})
    numeric = pd.to_numeric(raw, errors="coerce").fillna(0).ne(0)
    return (truthy | numeric).astype(int)


def prepare_modeling_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assign_prediction_type(df)

    home_class = normalized_classification(df, "homeclassification")
    away_class = normalized_classification(df, "awayclassification")
    fbs_home = home_class.eq("fbs") & away_class.eq("fcs")
    fbs_away = away_class.eq("fbs") & home_class.eq("fcs")

    df["has_spread_line"] = _numeric_series(df, "avg_spread").notna()
    df["has_total_line"] = _numeric_series(df, "avg_over_under").notna()
    df["neutral_site"] = _indicator_series(df, "neutralsite")
    df["is_fcs_game"] = df["prediction_type"].eq(FCS_PREDICTION_TYPE).astype(int)
    df["fbs_is_home"] = fbs_home.astype(int)

    numeric_cols = sorted(
        set(
            FBS_SPREAD_FEATURES
            + FBS_TOTAL_FEATURES
            + FCS_SPREAD_FEATURES
            + FCS_TOTAL_FEATURES
            + [
                "homepoints",
                "awaypoints",
                "home_teamrankings_rank",
                "away_teamrankings_rank",
                "home_returning_usage",
                "away_returning_usage",
            ]
        )
    )
    for col in numeric_cols:
        if col not in {"neutral_site", "is_fcs_game", "fbs_is_home"}:
            df[col] = _numeric_series(df, col)

    df["fbs_teamrankings_rating"] = np.where(
        fbs_home,
        df["home_teamrankings_rating"],
        np.where(fbs_away, df["away_teamrankings_rating"], np.nan),
    )
    df["fbs_talent"] = np.where(
        fbs_home,
        df["home_talent"],
        np.where(fbs_away, df["away_talent"], np.nan),
    )
    df["fbs_recruiting_points"] = np.where(
        fbs_home,
        df["home_recruiting_points"],
        np.where(fbs_away, df["away_recruiting_points"], np.nan),
    )
    df["fbs_returning_total_ppa"] = np.where(
        fbs_home,
        df["home_returning_total_ppa"],
        np.where(fbs_away, df["away_returning_total_ppa"], np.nan),
    )
    df["fbs_returning_percent_ppa"] = np.where(
        fbs_home,
        df["home_returning_percent_ppa"],
        np.where(fbs_away, df["away_returning_percent_ppa"], np.nan),
    )
    df["fbs_spread"] = np.where(
        fbs_home,
        df["avg_spread"],
        np.where(fbs_away, -df["avg_spread"], np.nan),
    )

    home_points = _numeric_series(df, "homepoints")
    away_points = _numeric_series(df, "awaypoints")
    df["spread_target"] = home_points - away_points
    df["win_target"] = (home_points > away_points).where(home_points.notna() & away_points.notna())
    df["total_points_target"] = home_points + away_points
    df["fbs_points"] = np.where(fbs_home, home_points, np.where(fbs_away, away_points, np.nan))
    df["fcs_points"] = np.where(fbs_home, away_points, np.where(fbs_away, home_points, np.nan))
    df["fbs_margin_target"] = df["fbs_points"] - df["fcs_points"]
    df["fbs_win_target"] = (
        pd.Series(df["fbs_points"], index=df.index)
        > pd.Series(df["fcs_points"], index=df.index)
    ).where(pd.notna(df["fbs_points"]) & pd.notna(df["fcs_points"]))

    return df


def _xgboost_classes():
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except Exception as exc:
        raise RuntimeError(
            "Unable to import xgboost. Install the Python package from etl/requirements.txt; "
            "on macOS you may also need `brew install libomp` so libxgboost can load."
        ) from exc
    return XGBClassifier, XGBRegressor


def build_xgb_classifier():
    XGBClassifier, _ = _xgboost_classes()
    return XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def build_xgb_regressor():
    _, XGBRegressor = _xgboost_classes()
    return XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def _training_frame(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    label: str,
) -> pd.DataFrame:
    model_df = train_df[feature_cols + [target_col]].copy()
    model_df = model_df[model_df[target_col].notna()]
    if model_df.empty:
        raise RuntimeError(f"No completed training rows available for {label}.")
    return model_df


def _fit_classifier(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    label: str,
):
    model_df = _training_frame(train_df, feature_cols, target_col, label)
    y = model_df[target_col].astype(int)
    if y.nunique() < 2:
        raise RuntimeError(f"{label} has only one target class; cannot train classifier.")
    print(f"Training {label} on {len(model_df)} rows...")
    model = build_xgb_classifier()
    model.fit(model_df[feature_cols], y)
    return model


def _fit_regressor(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    label: str,
):
    model_df = _training_frame(train_df, feature_cols, target_col, label)
    print(f"Training {label} on {len(model_df)} rows...")
    model = build_xgb_regressor()
    model.fit(model_df[feature_cols], model_df[target_col].astype(float))
    return model


def train_models_for_mask(
    df: pd.DataFrame,
    current_season: int,
    train_mask: pd.Series,
    training_message: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    modeled_df = prepare_modeling_dataframe(df)
    completed = modeled_df["homepoints"].notna() & modeled_df["awaypoints"].notna()
    train_df = modeled_df[train_mask & completed].copy()
    if train_df.empty:
        raise RuntimeError("No completed games available for model training.")

    if training_message:
        print(training_message)

    fbs_train = train_df[train_df["prediction_type"].eq(FBS_PREDICTION_TYPE)].copy()
    fcs_train = train_df[train_df["prediction_type"].eq(FCS_PREDICTION_TYPE)].copy()
    if fbs_train.empty:
        raise RuntimeError("No completed FBS-vs-FBS games available for model training.")
    if fcs_train.empty:
        raise RuntimeError("No completed FBS-vs-FCS games available for model training.")

    fbs_spread_train = fbs_train[fbs_train["has_spread_line"]].copy()
    fbs_total_train = fbs_train[fbs_train["has_total_line"]].copy()
    fcs_spread_train = fcs_train[fcs_train["has_spread_line"]].copy()
    fcs_total_train = fcs_train[fcs_train["has_total_line"]].copy()
    if fbs_spread_train.empty:
        raise RuntimeError("No FBS-vs-FBS games with spread data available.")
    if fbs_total_train.empty:
        raise RuntimeError("No FBS-vs-FBS games with total-line data available.")
    if fcs_spread_train.empty:
        raise RuntimeError("No FBS-vs-FCS games with spread data available.")
    if fcs_total_train.empty:
        raise RuntimeError("No FBS-vs-FCS games with total-line data available.")

    model_bundle = {
        "fbs_win_with_spread": _fit_classifier(
            fbs_spread_train, FBS_SPREAD_FEATURES, "win_target", "FBS win XGBoost with spread"
        ),
        "fbs_win_no_spread": _fit_classifier(
            fbs_train, FBS_BASE_FEATURES, "win_target", "FBS win XGBoost without spread"
        ),
        "fbs_spread_with_spread": _fit_regressor(
            fbs_spread_train, FBS_SPREAD_FEATURES, "spread_target", "FBS spread XGBoost with spread"
        ),
        "fbs_spread_no_spread": _fit_regressor(
            fbs_train, FBS_BASE_FEATURES, "spread_target", "FBS spread XGBoost without spread"
        ),
        "fbs_total_with_total": _fit_regressor(
            fbs_total_train, FBS_TOTAL_FEATURES, "total_points_target", "FBS total XGBoost with total"
        ),
        "fbs_total_no_total": _fit_regressor(
            fbs_train, FBS_BASE_FEATURES, "total_points_target", "FBS total XGBoost without total"
        ),
        "fcs_win_with_spread": _fit_classifier(
            fcs_spread_train, FCS_SPREAD_FEATURES, "fbs_win_target", "FCS win XGBoost with spread"
        ),
        "fcs_win_no_spread": _fit_classifier(
            fcs_train, FCS_BASE_FEATURES, "fbs_win_target", "FCS win XGBoost without spread"
        ),
        "fcs_margin_with_spread": _fit_regressor(
            fcs_spread_train, FCS_SPREAD_FEATURES, "fbs_margin_target", "FCS margin XGBoost with spread"
        ),
        "fcs_margin_no_spread": _fit_regressor(
            fcs_train, FCS_BASE_FEATURES, "fbs_margin_target", "FCS margin XGBoost without spread"
        ),
        "fcs_total_with_total": _fit_regressor(
            fcs_total_train, FCS_TOTAL_FEATURES, "total_points_target", "FCS total XGBoost with total"
        ),
        "fcs_total_no_total": _fit_regressor(
            fcs_train, FCS_BASE_FEATURES, "total_points_target", "FCS total XGBoost without total"
        ),
    }

    modeled_df.attrs["fbs_spread_features"] = FBS_SPREAD_FEATURES
    modeled_df.attrs["fbs_base_features"] = FBS_BASE_FEATURES
    modeled_df.attrs["fbs_total_features"] = FBS_TOTAL_FEATURES
    modeled_df.attrs["fcs_spread_features"] = FCS_SPREAD_FEATURES
    modeled_df.attrs["fcs_base_features"] = FCS_BASE_FEATURES
    modeled_df.attrs["fcs_total_features"] = FCS_TOTAL_FEATURES
    return model_bundle, modeled_df


def train_models(df: pd.DataFrame, current_season: int):
    season_numeric = pd.to_numeric(df["season"], errors="coerce")
    train_mask = season_numeric.lt(current_season)
    return train_models_for_mask(
        df,
        current_season,
        train_mask,
        training_message=f"Training on completed seasons before {current_season}.",
    )


def _predict_probability(model, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return model.predict_proba(frame[feature_cols])[:, 1]


def _score_fbs_rows(current_df: pd.DataFrame, modeled_df: pd.DataFrame, model_bundle: dict) -> None:
    fbs_mask = current_df["prediction_type"].eq(FBS_PREDICTION_TYPE)
    spread_mask = fbs_mask & current_df["has_spread_line"].astype(bool)
    no_spread_mask = fbs_mask & ~current_df["has_spread_line"].astype(bool)
    total_mask = fbs_mask & current_df["has_total_line"].astype(bool)
    no_total_mask = fbs_mask & ~current_df["has_total_line"].astype(bool)

    for mask, win_model, spread_model, feature_key in [
        (spread_mask, "fbs_win_with_spread", "fbs_spread_with_spread", "fbs_spread_features"),
        (no_spread_mask, "fbs_win_no_spread", "fbs_spread_no_spread", "fbs_base_features"),
    ]:
        if not mask.any():
            continue
        features = modeled_df.attrs[feature_key]
        frame = current_df.loc[mask, features]
        home_prob = _predict_probability(model_bundle[win_model], current_df.loc[mask], features)
        home_margin = model_bundle[spread_model].predict(frame)
        current_df.loc[mask, "homewinprob"] = home_prob
        current_df.loc[mask, "awaywinprob"] = 1.0 - home_prob
        current_df.loc[mask, "homespread"] = -home_margin
        current_df.loc[mask, "awayspread"] = home_margin

    for mask, total_model, feature_key in [
        (total_mask, "fbs_total_with_total", "fbs_total_features"),
        (no_total_mask, "fbs_total_no_total", "fbs_base_features"),
    ]:
        if not mask.any():
            continue
        features = modeled_df.attrs[feature_key]
        current_df.loc[mask, "totalpred"] = model_bundle[total_model].predict(
            current_df.loc[mask, features]
        )


def _score_fcs_rows(current_df: pd.DataFrame, modeled_df: pd.DataFrame, model_bundle: dict) -> None:
    fcs_mask = current_df["prediction_type"].eq(FCS_PREDICTION_TYPE)
    spread_mask = fcs_mask & current_df["has_spread_line"].astype(bool)
    no_spread_mask = fcs_mask & ~current_df["has_spread_line"].astype(bool)
    total_mask = fcs_mask & current_df["has_total_line"].astype(bool)
    no_total_mask = fcs_mask & ~current_df["has_total_line"].astype(bool)

    for mask, win_model, margin_model, feature_key in [
        (spread_mask, "fcs_win_with_spread", "fcs_margin_with_spread", "fcs_spread_features"),
        (no_spread_mask, "fcs_win_no_spread", "fcs_margin_no_spread", "fcs_base_features"),
    ]:
        if not mask.any():
            continue
        features = modeled_df.attrs[feature_key]
        fbs_prob = _predict_probability(model_bundle[win_model], current_df.loc[mask], features)
        fbs_margin = model_bundle[margin_model].predict(current_df.loc[mask, features])
        fbs_home = current_df.loc[mask, "fbs_is_home"].astype(bool)

        current_df.loc[mask, "homewinprob"] = np.where(fbs_home, fbs_prob, 1.0 - fbs_prob)
        current_df.loc[mask, "awaywinprob"] = 1.0 - current_df.loc[mask, "homewinprob"]
        current_df.loc[mask, "homespread"] = np.where(fbs_home, -fbs_margin, fbs_margin)
        current_df.loc[mask, "awayspread"] = -current_df.loc[mask, "homespread"]

    for mask, total_model, feature_key in [
        (total_mask, "fcs_total_with_total", "fcs_total_features"),
        (no_total_mask, "fcs_total_no_total", "fcs_base_features"),
    ]:
        if not mask.any():
            continue
        features = modeled_df.attrs[feature_key]
        current_df.loc[mask, "totalpred"] = model_bundle[total_model].predict(
            current_df.loc[mask, features]
        )


def score_current_season(model_bundle: dict, modeled_df: pd.DataFrame, current_season: int) -> pd.DataFrame:
    current_df = modeled_df[pd.to_numeric(modeled_df["season"], errors="coerce").eq(current_season)].copy()
    if current_df.empty:
        raise RuntimeError(f"No rows found for current season {current_season}.")

    for col in ["homewinprob", "awaywinprob", "homespread", "awayspread", "totalpred"]:
        current_df[col] = np.nan

    _score_fbs_rows(current_df, modeled_df, model_bundle)
    _score_fcs_rows(current_df, modeled_df, model_bundle)
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

    model_version TEXT NOT NULL,
    prediction_type TEXT NOT NULL DEFAULT 'FBS'
);
"""

ALTER_PRED_TABLE_SQL = """
ALTER TABLE public.game_predictions
ADD COLUMN IF NOT EXISTS prediction_type TEXT NOT NULL DEFAULT 'FBS';
"""

INSERT_PRED_SQL = """
INSERT INTO public.game_predictions (
    gameid, season, week,
    home_team, away_team,
    homepoints, awaypoints,
    homespread, awayspread, totalpred,
    homewinprob, awaywinprob,
    model_version, prediction_type
)
VALUES (
    %(gameid)s, %(season)s, %(week)s,
    %(home_team)s, %(away_team)s,
    %(homepoints)s, %(awaypoints)s,
    %(homespread)s, %(awayspread)s, %(totalpred)s,
    %(homewinprob)s, %(awaywinprob)s,
    %(model_version)s, %(prediction_type)s
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
    model_version = EXCLUDED.model_version,
    prediction_type = EXCLUDED.prediction_type;
"""


def ensure_predictions_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_PRED_TABLE_SQL)
        cur.execute(ALTER_PRED_TABLE_SQL)
    conn.commit()


def _row_model_version(output: pd.DataFrame) -> np.ndarray:
    is_fcs = output["prediction_type"].eq(FCS_PREDICTION_TYPE)
    fully_line_aware = output["has_spread_line"].astype(bool) & output["has_total_line"].astype(bool)
    return np.select(
        [
            is_fcs & fully_line_aware,
            is_fcs & ~fully_line_aware,
            ~is_fcs & fully_line_aware,
            ~is_fcs & ~fully_line_aware,
        ],
        [
            XGB_FCS_AWARE_MODEL_VERSION,
            XGB_FCS_INCOMPLETE_MODEL_VERSION,
            XGB_FBS_AWARE_MODEL_VERSION,
            XGB_FBS_INCOMPLETE_MODEL_VERSION,
        ],
        default=XGB_FBS_INCOMPLETE_MODEL_VERSION,
    )


def prediction_records(preds: pd.DataFrame) -> list[dict]:
    output = preds.copy()
    output["gameid"] = output["id"].astype(str)
    output["home_team"] = output["hometeam"]
    output["away_team"] = output["awayteam"]
    if "prediction_type" not in output.columns:
        output["prediction_type"] = FBS_PREDICTION_TYPE
    output["model_version"] = _row_model_version(output)
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
            "prediction_type",
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
        fcs_count = int(preds["prediction_type"].eq(FCS_PREDICTION_TYPE).sum())
        print(
            f"Prepared {len(records)} game-level predictions for season {current_season} "
            f"({fcs_count} FCS flagged)."
        )

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
