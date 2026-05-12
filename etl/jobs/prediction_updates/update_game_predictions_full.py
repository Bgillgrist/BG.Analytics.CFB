#!/usr/bin/env python3
"""
Train game prediction models from Neon data and append changed full snapshots.

This job intentionally leaves public.game_predictions untouched. It reuses the
modeling, training, scoring, and record-shaping functions from
etl.jobs.prediction_updates.update_game_predictions so the model output stays identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import psycopg

from etl.common_config import load_config
from etl.jobs.prediction_updates.update_game_predictions import (
    BASE_DIFF_FEATURES,
    FCS_PREDICTION_TYPE,
    INCOMPLETE_MODEL_VERSION,
    LINE_AWARE_MODEL_VERSION,
    MODEL_2_EXTRA_DIFF_FEATURES,
    assign_prediction_type,
    build_linear_pipeline,
    build_logistic_pipeline,
    build_modeling_table,
    fill_fcs_advanced_inputs_with_baselines,
    fill_preseason_inputs_with_second_fbs_averages,
    fill_week1_advanced_diffs_with_auxiliary_models,
    prediction_records,
    score_current_season,
)


RUN_TYPE_ENV = "GAME_PREDICTION_RUN_TYPE"
RUN_DATE_ENV = "GAME_PREDICTION_RUN_DATE"
NOTES_ENV = "GAME_PREDICTION_RUN_NOTES"
HASH_EXCLUDED_COLUMNS = {"homepoints", "awaypoints"}
HASH_FLOAT_DECIMAL_PLACES = 4
NORMAL_SNAPSHOT_RUN_TYPES = ("manual", "nightly")
DETAIL_HASH_COLUMNS = (
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
)

CREATE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.game_prediction_runs (
    game_prediction_run_id UUID PRIMARY KEY,
    season                 INT NOT NULL,
    run_date               DATE NOT NULL,
    run_type               TEXT NOT NULL DEFAULT 'nightly',
    etl_run_id             TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at           TIMESTAMPTZ,
    status                 TEXT NOT NULL DEFAULT 'running',
    model_version          TEXT NOT NULL,
    prediction_hash        TEXT,
    duplicate_of_run_id    UUID REFERENCES public.game_prediction_runs(game_prediction_run_id),
    row_count              INT NOT NULL DEFAULT 0,
    inserted_row_count     INT NOT NULL DEFAULT 0,
    fcs_count              INT NOT NULL DEFAULT 0,
    notes                  TEXT,
    error_message          TEXT,
    CONSTRAINT game_prediction_runs_status_check
      CHECK (status IN ('running', 'success', 'duplicate', 'failed'))
);
"""

CREATE_DETAILS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.game_predictions_full (
    game_prediction_run_id UUID NOT NULL
      REFERENCES public.game_prediction_runs(game_prediction_run_id)
      ON DELETE CASCADE,
    gameid                TEXT NOT NULL,
    season                INT NOT NULL,
    week                  INT NOT NULL,
    home_team             TEXT NOT NULL,
    away_team             TEXT NOT NULL,
    homepoints            DOUBLE PRECISION,
    awaypoints            DOUBLE PRECISION,
    homespread            DOUBLE PRECISION,
    awayspread            DOUBLE PRECISION,
    totalpred             DOUBLE PRECISION,
    homewinprob           DOUBLE PRECISION,
    awaywinprob           DOUBLE PRECISION,
    model_version         TEXT NOT NULL,
    prediction_type       TEXT NOT NULL DEFAULT 'FBS',
    prediction_row_hash   TEXT NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_prediction_run_id, gameid)
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_game_prediction_runs_lookup
  ON public.game_prediction_runs (season, run_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_game_prediction_runs_hash
  ON public.game_prediction_runs (season, run_type, prediction_hash)
  WHERE status = 'success';

CREATE INDEX IF NOT EXISTS idx_game_predictions_full_game_lookup
  ON public.game_predictions_full (season, gameid);

CREATE INDEX IF NOT EXISTS idx_game_predictions_full_team_week
  ON public.game_predictions_full (season, week, home_team, away_team);
"""

INSERT_RUN_SQL = """
INSERT INTO public.game_prediction_runs (
    game_prediction_run_id,
    season,
    run_date,
    run_type,
    etl_run_id,
    status,
    model_version,
    row_count,
    inserted_row_count,
    fcs_count,
    notes
)
VALUES (
    %(game_prediction_run_id)s,
    %(season)s,
    %(run_date)s,
    %(run_type)s,
    %(etl_run_id)s,
    'running',
    %(model_version)s,
    %(row_count)s,
    0,
    %(fcs_count)s,
    %(notes)s
);
"""

MARK_RUN_SUCCESS_SQL = """
UPDATE public.game_prediction_runs
SET
    completed_at = NOW(),
    status = 'success',
    prediction_hash = %(prediction_hash)s,
    inserted_row_count = %(inserted_row_count)s
WHERE game_prediction_run_id = %(game_prediction_run_id)s;
"""

MARK_RUN_DUPLICATE_SQL = """
UPDATE public.game_prediction_runs
SET
    completed_at = NOW(),
    status = 'duplicate',
    prediction_hash = %(prediction_hash)s,
    duplicate_of_run_id = %(duplicate_of_run_id)s,
    inserted_row_count = 0
WHERE game_prediction_run_id = %(game_prediction_run_id)s;
"""

MARK_RUN_FAILED_SQL = """
UPDATE public.game_prediction_runs
SET
    completed_at = NOW(),
    status = 'failed',
    error_message = %(error_message)s
WHERE game_prediction_run_id = %(game_prediction_run_id)s;
"""

LATEST_SUCCESSFUL_RUN_SQL = """
SELECT game_prediction_run_id, prediction_hash
FROM public.game_prediction_runs
WHERE season = %s
  AND run_type IN ({run_type_placeholders})
  AND status = 'success'
  AND prediction_hash IS NOT NULL
ORDER BY created_at DESC
LIMIT 1;
"""

LATEST_SUCCESSFUL_RUN_BY_DATE_SQL = """
SELECT game_prediction_run_id, prediction_hash
FROM public.game_prediction_runs
WHERE season = %s
  AND run_type = %s
  AND run_date = %s
  AND status = 'success'
  AND prediction_hash IS NOT NULL
ORDER BY created_at DESC
LIMIT 1;
"""

INSERT_DETAIL_SQL = """
INSERT INTO public.game_predictions_full (
    game_prediction_run_id,
    gameid,
    season,
    week,
    home_team,
    away_team,
    homepoints,
    awaypoints,
    homespread,
    awayspread,
    totalpred,
    homewinprob,
    awaywinprob,
    model_version,
    prediction_type,
    prediction_row_hash
)
VALUES (
    %(game_prediction_run_id)s,
    %(gameid)s,
    %(season)s,
    %(week)s,
    %(home_team)s,
    %(away_team)s,
    %(homepoints)s,
    %(awaypoints)s,
    %(homespread)s,
    %(awayspread)s,
    %(totalpred)s,
    %(homewinprob)s,
    %(awaywinprob)s,
    %(model_version)s,
    %(prediction_type)s,
    %(prediction_row_hash)s
);
"""

FINALIZE_SCORE_SQL = """
UPDATE public.game_predictions_full
SET
    homepoints = %(homepoints)s,
    awaypoints = %(awaypoints)s
WHERE season = %(season)s
  AND gameid = %(gameid)s
  AND (homepoints IS NULL OR awaypoints IS NULL);
"""

DETAIL_RECORDS_FOR_RUN_SQL = f"""
SELECT {", ".join(DETAIL_HASH_COLUMNS)}
FROM public.game_predictions_full
WHERE game_prediction_run_id = %s
ORDER BY gameid;
"""


def _run_date_from_env() -> date:
    raw = os.getenv(RUN_DATE_ENV, "").strip()
    if raw:
        return date.fromisoformat(raw)
    return datetime.now(timezone.utc).date()


def _run_date_was_supplied() -> bool:
    return bool(os.getenv(RUN_DATE_ENV, "").strip())


def _run_type_from_env() -> str:
    return os.getenv(RUN_TYPE_ENV, "nightly").strip().lower() or "nightly"


def _model_version_label() -> str:
    return f"{LINE_AWARE_MODEL_VERSION}+{INCOMPLETE_MODEL_VERSION}"


def _as_of_training_mask(df, current_season: int, as_of_date: date):
    season_numeric = pd.to_numeric(df["season"], errors="coerce")
    gamedate = pd.to_datetime(df["gamedate"], errors="coerce").dt.date
    has_final_score = df["homepoints"].notna() & df["awaypoints"].notna()
    return season_numeric.lt(current_season) | (
        season_numeric.eq(current_season)
        & has_final_score
        & gamedate.lt(as_of_date)
    )


def train_models_as_of(df, current_season: int, as_of_date: date):
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

    train_mask = _as_of_training_mask(df, current_season, as_of_date)
    train_df = df[train_mask & df["homepoints"].notna() & df["awaypoints"].notna()].copy()
    if train_df.empty:
        raise RuntimeError("No completed games available for model training.")

    current_train_count = int(
        pd.to_numeric(train_df["season"], errors="coerce").eq(current_season).sum()
    )
    print(
        f"Training through {as_of_date.isoformat()} "
        f"({current_train_count} completed {current_season} games included)."
    )

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


def filter_predictions_for_run_type(
    preds,
    run_type: str,
    run_date: date,
    target_game_date: date | None = None,
):
    gamedate = pd.to_datetime(preds["gamedate"], errors="coerce").dt.date
    if run_type == "backfill":
        target_date = target_game_date or run_date + timedelta(days=1)
        return preds[gamedate.eq(target_date)].copy()
    return preds[gamedate.gt(run_date) & preds["homepoints"].isna() & preds["awaypoints"].isna()].copy()


def backfill_game_dates(df, current_season: int) -> list[date]:
    season_numeric = pd.to_numeric(df["season"], errors="coerce")
    gamedates = pd.to_datetime(
        df.loc[season_numeric.eq(current_season), "gamedate"],
        errors="coerce",
    ).dropna()
    return sorted({value.date() for value in gamedates})


def _canonical_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return round(value, HASH_FLOAT_DECIMAL_PLACES)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return round(float(value), HASH_FLOAT_DECIMAL_PLACES)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _canonical_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _canonical_value(record.get(key))
        for key in sorted(record)
        if key not in HASH_EXCLUDED_COLUMNS
    }


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add_prediction_hashes(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    enriched: list[dict[str, Any]] = []
    canonical_records = []
    for record in sorted(records, key=lambda item: str(item["gameid"])):
        canonical = _canonical_record(record)
        row_hash = _hash_payload(canonical)
        enriched_record = dict(record)
        enriched_record["prediction_row_hash"] = row_hash
        enriched.append(enriched_record)
        canonical_records.append(canonical)
    return enriched, _hash_payload(canonical_records)


def _prediction_hash_from_records(records: list[dict[str, Any]]) -> str:
    canonical_records = [
        _canonical_record(record)
        for record in sorted(records, key=lambda item: str(item["gameid"]))
    ]
    return _hash_payload(canonical_records)


def _comparable_run_types(run_type: str) -> tuple[str, ...]:
    if run_type in NORMAL_SNAPSHOT_RUN_TYPES:
        return NORMAL_SNAPSHOT_RUN_TYPES
    return (run_type,)


def ensure_full_prediction_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_RUNS_TABLE_SQL)
        cur.execute(CREATE_DETAILS_TABLE_SQL)
        cur.execute(CREATE_INDEXES_SQL)
    conn.commit()


def completed_score_records(df, current_season: int) -> list[dict[str, Any]]:
    season_numeric = pd.to_numeric(df["season"], errors="coerce")
    scored = df[
        season_numeric.eq(current_season)
        & df["homepoints"].notna()
        & df["awaypoints"].notna()
    ].copy()
    if scored.empty:
        return []

    scored["gameid"] = scored["id"].astype(str)
    scored = scored[["gameid", "season", "homepoints", "awaypoints"]]
    scored = scored.drop_duplicates(subset=["gameid"], keep="last")
    scored = scored.astype(object).where(pd.notna(scored), None)
    return scored.to_dict(orient="records")


def finalize_completed_scores(conn, df, current_season: int) -> int:
    records = completed_score_records(df, current_season)
    if not records:
        return 0

    with conn.cursor() as cur:
        cur.executemany(FINALIZE_SCORE_SQL, records)
        updated_count = cur.rowcount
    conn.commit()
    return max(updated_count, 0)


def get_latest_successful_run(conn, season: int, run_type: str) -> tuple[str, str] | None:
    run_types = _comparable_run_types(run_type)
    placeholders = ", ".join(["%s"] * len(run_types))
    sql = LATEST_SUCCESSFUL_RUN_SQL.format(run_type_placeholders=placeholders)
    with conn.cursor() as cur:
        cur.execute(sql, (season, *run_types))
        row = cur.fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1])


def get_latest_successful_run_for_date(
    conn,
    season: int,
    run_type: str,
    run_date: date,
) -> tuple[str, str] | None:
    with conn.cursor() as cur:
        cur.execute(LATEST_SUCCESSFUL_RUN_BY_DATE_SQL, (season, run_type, run_date))
        row = cur.fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1])


def get_prediction_records_for_run(conn, run_id: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(DETAIL_RECORDS_FOR_RUN_SQL, (run_id,))
        rows = cur.fetchall()
    return [dict(zip(DETAIL_HASH_COLUMNS, row)) for row in rows]


def prediction_hash_matches_run(
    conn,
    *,
    run_id: str,
    stored_prediction_hash: str,
    prediction_hash: str,
) -> bool:
    if stored_prediction_hash == prediction_hash:
        return True

    prior_records = get_prediction_records_for_run(conn, run_id)
    return _prediction_hash_from_records(prior_records) == prediction_hash


def create_run(
    conn,
    *,
    run_id: uuid.UUID,
    season: int,
    run_date: date,
    run_type: str,
    etl_run_id: str,
    row_count: int,
    fcs_count: int,
    notes: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            INSERT_RUN_SQL,
            {
                "game_prediction_run_id": run_id,
                "season": season,
                "run_date": run_date,
                "run_type": run_type,
                "etl_run_id": etl_run_id,
                "model_version": _model_version_label(),
                "row_count": row_count,
                "fcs_count": fcs_count,
                "notes": notes,
            },
        )
    conn.commit()


def mark_run_failed(conn, run_id: uuid.UUID, error_message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            MARK_RUN_FAILED_SQL,
            {
                "game_prediction_run_id": run_id,
                "error_message": error_message[:2000],
            },
        )
    conn.commit()


def write_changed_snapshot(
    conn,
    *,
    run_id: uuid.UUID,
    records: list[dict[str, Any]],
    prediction_hash: str,
) -> None:
    with conn.cursor() as cur:
        cur.executemany(INSERT_DETAIL_SQL, records)
        cur.execute(
            MARK_RUN_SUCCESS_SQL,
            {
                "game_prediction_run_id": run_id,
                "prediction_hash": prediction_hash,
                "inserted_row_count": len(records),
            },
        )
    conn.commit()


def write_duplicate_run(
    conn,
    *,
    run_id: uuid.UUID,
    prediction_hash: str,
    duplicate_of_run_id: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            MARK_RUN_DUPLICATE_SQL,
            {
                "game_prediction_run_id": run_id,
                "prediction_hash": prediction_hash,
                "duplicate_of_run_id": duplicate_of_run_id,
            },
        )
    conn.commit()


def process_prediction_snapshot(
    conn,
    *,
    df,
    current_season: int,
    run_date: date,
    run_type: str,
    etl_run_id: str,
    notes: str | None,
    target_game_date: date | None = None,
) -> tuple[uuid.UUID, int, str]:
    run_id = uuid.uuid4()
    print(f"Prediction run id: {run_id}")
    print(f"Run date/type: {run_date} / {run_type}")
    if target_game_date is not None:
        print(f"Backfill target game date: {target_game_date}")

    model_bundle, modeled_df = train_models_as_of(df, current_season, run_date)
    preds = score_current_season(model_bundle, modeled_df, current_season)
    preds = filter_predictions_for_run_type(
        preds,
        run_type,
        run_date,
        target_game_date=target_game_date,
    )
    records = prediction_records(preds)
    records, prediction_hash = add_prediction_hashes(records)
    fcs_count = int(preds["prediction_type"].eq(FCS_PREDICTION_TYPE).sum())
    print(
        f"Prepared {len(records)} game-level predictions for season {current_season} "
        f"({fcs_count} FCS flagged)."
    )
    print(f"Prediction hash: {prediction_hash}")

    create_run(
        conn,
        run_id=run_id,
        season=current_season,
        run_date=run_date,
        run_type=run_type,
        etl_run_id=etl_run_id,
        row_count=len(records),
        fcs_count=fcs_count,
        notes=notes,
    )

    try:
        latest = (
            get_latest_successful_run_for_date(conn, current_season, run_type, run_date)
            if run_type == "backfill"
            else get_latest_successful_run(conn, current_season, run_type)
        )
        if latest is not None:
            latest_run_id, latest_hash = latest
            if prediction_hash_matches_run(
                conn,
                run_id=latest_run_id,
                stored_prediction_hash=latest_hash,
                prediction_hash=prediction_hash,
            ):
                print(
                    "Predictions match latest comparable successful snapshot; "
                    f"marking this run as duplicate of {latest_run_id}."
                )
                write_duplicate_run(
                    conn,
                    run_id=run_id,
                    prediction_hash=prediction_hash,
                    duplicate_of_run_id=latest_run_id,
                )
                print("Finished without inserting duplicate prediction detail rows.")
                return run_id, 0, "duplicate"

        print("Prediction set changed; inserting full snapshot rows...")
        for record in records:
            record["game_prediction_run_id"] = run_id
        write_changed_snapshot(
            conn,
            run_id=run_id,
            records=records,
            prediction_hash=prediction_hash,
        )
        print(f"Finished inserting {len(records)} rows into game_predictions_full.")
        return run_id, len(records), "success"
    except Exception as exc:
        conn.rollback()
        mark_run_failed(conn, run_id, str(exc))
        raise


def main() -> None:
    cfg = load_config()
    current_season = cfg.season
    run_date = _run_date_from_env()
    run_date_was_supplied = _run_date_was_supplied()
    run_type = _run_type_from_env()
    notes = os.getenv(NOTES_ENV, "").strip() or None

    print("Connecting to database...")
    with psycopg.connect(cfg.pg_dsn) as conn:
        print(f"Target season: {current_season}")

        print("Building modeling table...")
        df = build_modeling_table(conn, max_season=current_season)
        print(f"Modeling table rows: {len(df)}")

        ensure_full_prediction_tables(conn)
        finalized_count = finalize_completed_scores(conn, df, current_season)
        print(f"Finalized scores on {finalized_count} existing prediction rows.")

        if run_type == "backfill" and not run_date_was_supplied:
            game_dates = backfill_game_dates(df, current_season)
            print(
                f"Backfilling {len(game_dates)} game dates for season {current_season}; "
                "each run_date will be the day before its game date."
            )
            total_inserted = 0
            for game_date in game_dates:
                snapshot_run_date = game_date - timedelta(days=1)
                _, inserted_count, _ = process_prediction_snapshot(
                    conn,
                    df=df,
                    current_season=current_season,
                    run_date=snapshot_run_date,
                    run_type=run_type,
                    etl_run_id=cfg.run_id,
                    notes=notes,
                    target_game_date=game_date,
                )
                total_inserted += inserted_count
            print(f"Finished season backfill; inserted {total_inserted} prediction rows.")
            return

        process_prediction_snapshot(
            conn,
            df=df,
            current_season=current_season,
            run_date=run_date,
            run_type=run_type,
            etl_run_id=cfg.run_id,
            notes=notes,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"update_game_predictions_full.py failed: {e}")
        sys.exit(1)
