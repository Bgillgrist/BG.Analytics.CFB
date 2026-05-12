#!/usr/bin/env python3
"""
Build team ranking projection snapshots from the latest game prediction snapshot.

The job follows the same run/detail snapshot pattern as
etl.jobs.prediction_updates.update_game_predictions_full. A ranking projection run is tied to one
successful game_prediction_run_id, so every remaining-game win probability in
the ranking table comes from the same model snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import psycopg

from etl.common_config import load_config


RUN_TYPE_ENV = "RANKING_PROJECTION_RUN_TYPE"
RUN_DATE_ENV = "RANKING_PROJECTION_RUN_DATE"
NOTES_ENV = "RANKING_PROJECTION_RUN_NOTES"
GAME_PREDICTION_RUN_ID_ENV = "RANKING_PROJECTION_GAME_PREDICTION_RUN_ID"
POLL_THROUGH_WEEK_ENV = "RANKING_PROJECTION_POLL_THROUGH_WEEK"

HASH_FLOAT_DECIMAL_PLACES = 4
NORMAL_SNAPSHOT_RUN_TYPES = ("manual", "nightly")
AP_POLL_NAMES = {"AP Top 25", "AP Poll"}
COACHES_POLL_NAMES = {"Coaches Poll", "USA Today Coaches Poll"}
CFP_POLL_NAMES = {"Playoff Committee Rankings", "College Football Playoff", "CFP Rankings"}
REGULAR_SEASON_TYPES = {"regular", "regular season", "2"}

DETAIL_HASH_EXCLUDED_COLUMNS = {
    "ranking_projection_run_id",
    "run_date",
    "run_type",
    "created_at",
    "notes",
    "prediction_hash",
    "game_prediction_run_id",
}

DETAIL_HASH_COLUMNS = (
    "season",
    "model_version",
    "team",
    "conference",
    "classification",
    "projected_ap_ranking",
    "projected_end_ap_ranking",
    "projected_cfp_ranking",
    "projected_end_cfp_ranking",
    "projected_ap_score",
    "projected_end_ap_score",
    "projected_cfp_score",
    "projected_end_cfp_score",
    "resume_score",
    "projected_resume_score",
    "power_score",
    "poll_inertia_score",
    "current_wins",
    "current_losses",
    "current_conference_wins",
    "current_conference_losses",
    "projected_wins",
    "projected_losses",
    "projected_conference_wins",
    "projected_conference_losses",
    "current_ap_rank",
    "previous_ap_rank",
    "current_coaches_rank",
    "current_cfp_rank",
    "previous_cfp_rank",
    "strength_of_schedule",
    "remaining_strength_of_schedule",
    "team_strength",
    "talent_score",
    "recruiting_score",
    "returning_production_score",
    "advanced_stats_season",
    "prediction_type",
)

CREATE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.ranking_projection_runs (
    ranking_projection_run_id UUID PRIMARY KEY,
    season                    INT NOT NULL,
    run_date                  DATE NOT NULL,
    run_type                  TEXT NOT NULL DEFAULT 'nightly',
    etl_run_id                TEXT,
    game_prediction_run_id    UUID REFERENCES public.game_prediction_runs(game_prediction_run_id),
    created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at              TIMESTAMPTZ,
    status                    TEXT NOT NULL DEFAULT 'running',
    model_version             TEXT NOT NULL,
    prediction_hash           TEXT,
    duplicate_of_run_id       UUID REFERENCES public.ranking_projection_runs(ranking_projection_run_id),
    row_count                 INT NOT NULL DEFAULT 0,
    inserted_row_count        INT NOT NULL DEFAULT 0,
    notes                     TEXT,
    error_message             TEXT,
    CONSTRAINT ranking_projection_runs_status_check
      CHECK (status IN ('running', 'success', 'duplicate', 'failed'))
);
"""

CREATE_DETAILS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.ranking_projections_full (
    ranking_projection_run_id UUID NOT NULL
      REFERENCES public.ranking_projection_runs(ranking_projection_run_id)
      ON DELETE CASCADE,
    season                              INT NOT NULL,
    run_date                            DATE NOT NULL,
    run_type                            TEXT NOT NULL DEFAULT 'nightly',
    model_version                       TEXT NOT NULL,
    team                                TEXT NOT NULL,
    conference                          TEXT,
    classification                      TEXT,
    created_at                          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    projected_ap_ranking                INT,
    projected_end_ap_ranking            INT,
    projected_cfp_ranking               INT,
    projected_end_cfp_ranking           INT,
    projected_ap_score                  DOUBLE PRECISION,
    projected_end_ap_score              DOUBLE PRECISION,
    projected_cfp_score                 DOUBLE PRECISION,
    projected_end_cfp_score             DOUBLE PRECISION,
    resume_score                        DOUBLE PRECISION,
    projected_resume_score              DOUBLE PRECISION,
    power_score                         DOUBLE PRECISION,
    poll_inertia_score                  DOUBLE PRECISION,
    current_wins                        DOUBLE PRECISION,
    current_losses                      DOUBLE PRECISION,
    current_conference_wins             DOUBLE PRECISION,
    current_conference_losses           DOUBLE PRECISION,
    projected_wins                      DOUBLE PRECISION,
    projected_losses                    DOUBLE PRECISION,
    projected_conference_wins           DOUBLE PRECISION,
    projected_conference_losses         DOUBLE PRECISION,
    current_ap_rank                     INT,
    previous_ap_rank                    INT,
    current_coaches_rank                INT,
    current_cfp_rank                    INT,
    previous_cfp_rank                   INT,
    strength_of_schedule                DOUBLE PRECISION,
    remaining_strength_of_schedule      DOUBLE PRECISION,
    team_strength                       DOUBLE PRECISION,
    talent_score                        DOUBLE PRECISION,
    recruiting_score                    DOUBLE PRECISION,
    returning_production_score          DOUBLE PRECISION,
    advanced_stats_season               INT,
    game_prediction_run_id              UUID
      REFERENCES public.game_prediction_runs(game_prediction_run_id),
    prediction_hash                     TEXT NOT NULL,
    prediction_type                     TEXT NOT NULL DEFAULT 'FBS',
    notes                               TEXT,
    PRIMARY KEY (ranking_projection_run_id, team)
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_ranking_projection_runs_lookup
  ON public.ranking_projection_runs (season, run_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ranking_projection_runs_hash
  ON public.ranking_projection_runs (season, run_type, prediction_hash)
  WHERE status = 'success';

CREATE INDEX IF NOT EXISTS idx_ranking_projections_full_team_lookup
  ON public.ranking_projections_full (season, team, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_ranking_projections_full_rank_lookup
  ON public.ranking_projections_full (season, projected_cfp_ranking, projected_ap_ranking);
"""

INSERT_RUN_SQL = """
INSERT INTO public.ranking_projection_runs (
    ranking_projection_run_id,
    season,
    run_date,
    run_type,
    etl_run_id,
    game_prediction_run_id,
    status,
    model_version,
    row_count,
    inserted_row_count,
    notes
)
VALUES (
    %(ranking_projection_run_id)s,
    %(season)s,
    %(run_date)s,
    %(run_type)s,
    %(etl_run_id)s,
    %(game_prediction_run_id)s,
    'running',
    %(model_version)s,
    %(row_count)s,
    0,
    %(notes)s
);
"""

MARK_RUN_SUCCESS_SQL = """
UPDATE public.ranking_projection_runs
SET
    completed_at = NOW(),
    status = 'success',
    prediction_hash = %(prediction_hash)s,
    inserted_row_count = %(inserted_row_count)s
WHERE ranking_projection_run_id = %(ranking_projection_run_id)s;
"""

MARK_RUN_DUPLICATE_SQL = """
UPDATE public.ranking_projection_runs
SET
    completed_at = NOW(),
    status = 'duplicate',
    prediction_hash = %(prediction_hash)s,
    duplicate_of_run_id = %(duplicate_of_run_id)s,
    inserted_row_count = 0
WHERE ranking_projection_run_id = %(ranking_projection_run_id)s;
"""

MARK_RUN_FAILED_SQL = """
UPDATE public.ranking_projection_runs
SET
    completed_at = NOW(),
    status = 'failed',
    error_message = %(error_message)s
WHERE ranking_projection_run_id = %(ranking_projection_run_id)s;
"""

LATEST_SUCCESSFUL_RUN_SQL = """
SELECT ranking_projection_run_id, prediction_hash
FROM public.ranking_projection_runs
WHERE season = %s
  AND run_type IN ({run_type_placeholders})
  AND status = 'success'
  AND prediction_hash IS NOT NULL
ORDER BY created_at DESC
LIMIT 1;
"""

LATEST_SUCCESSFUL_RUN_BY_DATE_SQL = """
SELECT ranking_projection_run_id, prediction_hash
FROM public.ranking_projection_runs
WHERE season = %s
  AND run_type = %s
  AND run_date = %s
  AND status = 'success'
  AND prediction_hash IS NOT NULL
ORDER BY created_at DESC
LIMIT 1;
"""

LATEST_GAME_PREDICTION_RUN_SQL = """
SELECT game_prediction_run_id, model_version
FROM public.game_prediction_runs
WHERE season = %s
  AND run_type IN ({run_type_placeholders})
  AND status = 'success'
ORDER BY created_at DESC
LIMIT 1;
"""

LATEST_GAME_PREDICTION_RUN_BY_DATE_SQL = """
SELECT game_prediction_run_id, model_version
FROM public.game_prediction_runs
WHERE season = %s
  AND run_type = %s
  AND run_date = %s
  AND status = 'success'
ORDER BY created_at DESC
LIMIT 1;
"""

EXPLICIT_GAME_PREDICTION_RUN_SQL = """
SELECT game_prediction_run_id, model_version
FROM public.game_prediction_runs
WHERE game_prediction_run_id = %s
  AND season = %s
  AND status = 'success'
LIMIT 1;
"""

BACKFILL_GAME_RUN_DATES_SQL = """
SELECT DISTINCT run_date
FROM public.game_prediction_runs
WHERE season = %s
  AND run_type = 'backfill'
  AND status = 'success'
ORDER BY run_date;
"""

DETAIL_RECORDS_FOR_RUN_SQL = f"""
SELECT {", ".join(DETAIL_HASH_COLUMNS)}
FROM public.ranking_projections_full
WHERE ranking_projection_run_id = %s
ORDER BY team;
"""

INSERT_DETAIL_SQL = """
INSERT INTO public.ranking_projections_full (
    ranking_projection_run_id,
    season,
    run_date,
    run_type,
    model_version,
    team,
    conference,
    classification,
    projected_ap_ranking,
    projected_end_ap_ranking,
    projected_cfp_ranking,
    projected_end_cfp_ranking,
    projected_ap_score,
    projected_end_ap_score,
    projected_cfp_score,
    projected_end_cfp_score,
    resume_score,
    projected_resume_score,
    power_score,
    poll_inertia_score,
    current_wins,
    current_losses,
    current_conference_wins,
    current_conference_losses,
    projected_wins,
    projected_losses,
    projected_conference_wins,
    projected_conference_losses,
    current_ap_rank,
    previous_ap_rank,
    current_coaches_rank,
    current_cfp_rank,
    previous_cfp_rank,
    strength_of_schedule,
    remaining_strength_of_schedule,
    team_strength,
    talent_score,
    recruiting_score,
    returning_production_score,
    advanced_stats_season,
    game_prediction_run_id,
    prediction_hash,
    prediction_type,
    notes
)
VALUES (
    %(ranking_projection_run_id)s,
    %(season)s,
    %(run_date)s,
    %(run_type)s,
    %(model_version)s,
    %(team)s,
    %(conference)s,
    %(classification)s,
    %(projected_ap_ranking)s,
    %(projected_end_ap_ranking)s,
    %(projected_cfp_ranking)s,
    %(projected_end_cfp_ranking)s,
    %(projected_ap_score)s,
    %(projected_end_ap_score)s,
    %(projected_cfp_score)s,
    %(projected_end_cfp_score)s,
    %(resume_score)s,
    %(projected_resume_score)s,
    %(power_score)s,
    %(poll_inertia_score)s,
    %(current_wins)s,
    %(current_losses)s,
    %(current_conference_wins)s,
    %(current_conference_losses)s,
    %(projected_wins)s,
    %(projected_losses)s,
    %(projected_conference_wins)s,
    %(projected_conference_losses)s,
    %(current_ap_rank)s,
    %(previous_ap_rank)s,
    %(current_coaches_rank)s,
    %(current_cfp_rank)s,
    %(previous_cfp_rank)s,
    %(strength_of_schedule)s,
    %(remaining_strength_of_schedule)s,
    %(team_strength)s,
    %(talent_score)s,
    %(recruiting_score)s,
    %(returning_production_score)s,
    %(advanced_stats_season)s,
    %(game_prediction_run_id)s,
    %(prediction_hash)s,
    %(prediction_type)s,
    %(notes)s
);
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


def _poll_through_week_from_env() -> int | None:
    raw = os.getenv(POLL_THROUGH_WEEK_ENV, "").strip()
    return int(raw) if raw else None


def _model_version_label(game_model_version: str | None) -> str:
    suffix = game_model_version or "unknown_game_model"
    return f"ranking_projection_2026+{suffix}"


def _comparable_run_types(run_type: str) -> tuple[str, ...]:
    if run_type in NORMAL_SNAPSHOT_RUN_TYPES:
        return NORMAL_SNAPSHOT_RUN_TYPES
    return (run_type,)


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
        if key not in DETAIL_HASH_EXCLUDED_COLUMNS
    }


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add_prediction_hashes(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    enriched: list[dict[str, Any]] = []
    canonical_records = []
    for record in sorted(records, key=lambda item: str(item["team"])):
        canonical = _canonical_record(record)
        row_hash = _hash_payload(canonical)
        enriched_record = dict(record)
        enriched_record["prediction_hash"] = row_hash
        enriched.append(enriched_record)
        canonical_records.append(canonical)
    return enriched, _hash_payload(canonical_records)


def _prediction_hash_from_records(records: list[dict[str, Any]]) -> str:
    canonical_records = [
        _canonical_record(record)
        for record in sorted(records, key=lambda item: str(item["team"]))
    ]
    return _hash_payload(canonical_records)


def ensure_ranking_projection_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_RUNS_TABLE_SQL)
        cur.execute(CREATE_DETAILS_TABLE_SQL)
        cur.execute(CREATE_INDEXES_SQL)
    conn.commit()


def _read_sql(conn, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql(sql, conn, params=params)


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


def select_game_prediction_run(
    conn,
    *,
    season: int,
    run_type: str,
    run_date: date,
    explicit_run_id: str | None,
) -> tuple[str, str | None]:
    if explicit_run_id:
        with conn.cursor() as cur:
            cur.execute(EXPLICIT_GAME_PREDICTION_RUN_SQL, (explicit_run_id, season))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                f"No successful game_prediction_runs row found for {explicit_run_id} and season={season}."
            )
        return str(row[0]), row[1]

    if run_type == "backfill":
        with conn.cursor() as cur:
            cur.execute(LATEST_GAME_PREDICTION_RUN_BY_DATE_SQL, (season, run_type, run_date))
            row = cur.fetchone()
    else:
        run_types = _comparable_run_types(run_type)
        placeholders = ", ".join(["%s"] * len(run_types))
        sql = LATEST_GAME_PREDICTION_RUN_SQL.format(run_type_placeholders=placeholders)
        with conn.cursor() as cur:
            cur.execute(sql, (season, *run_types))
            row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"No successful game prediction snapshot found for season={season}, "
            f"run_type={run_type}, run_date={run_date}."
        )
    return str(row[0]), row[1]


def backfill_run_dates(conn, season: int) -> list[date]:
    with conn.cursor() as cur:
        cur.execute(BACKFILL_GAME_RUN_DATES_SQL, (season,))
        rows = cur.fetchall()
    return [row[0] for row in rows]


def create_run(
    conn,
    *,
    run_id: uuid.UUID,
    season: int,
    run_date: date,
    run_type: str,
    etl_run_id: str,
    game_prediction_run_id: str,
    model_version: str,
    row_count: int,
    notes: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            INSERT_RUN_SQL,
            {
                "ranking_projection_run_id": run_id,
                "season": season,
                "run_date": run_date,
                "run_type": run_type,
                "etl_run_id": etl_run_id,
                "game_prediction_run_id": game_prediction_run_id,
                "model_version": model_version,
                "row_count": row_count,
                "notes": notes,
            },
        )
    conn.commit()


def mark_run_failed(conn, run_id: uuid.UUID, error_message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            MARK_RUN_FAILED_SQL,
            {
                "ranking_projection_run_id": run_id,
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
                "ranking_projection_run_id": run_id,
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
                "ranking_projection_run_id": run_id,
                "prediction_hash": prediction_hash,
                "duplicate_of_run_id": duplicate_of_run_id,
            },
        )
    conn.commit()


def _rank_desc(values_by_team: dict[str, float], teams: list[str]) -> dict[str, int]:
    ordered = sorted(teams, key=lambda team: (-values_by_team.get(team, 0.0), team))
    return {team: rank + 1 for rank, team in enumerate(ordered)}


def _percentile_scores(
    values_by_team: dict[str, float | None],
    teams: list[str],
    *,
    lower_is_better: bool = False,
) -> dict[str, float]:
    usable = {team: values_by_team[team] for team in teams if values_by_team.get(team) is not None}
    if not usable:
        return {team: 50.0 for team in teams}
    ordered = sorted(usable, key=usable.get, reverse=not lower_is_better)
    if len(ordered) == 1:
        return {team: 50.0 for team in teams}
    scores = {
        team: 100.0 * (len(ordered) - 1 - index) / (len(ordered) - 1)
        for index, team in enumerate(ordered)
    }
    for team in teams:
        scores.setdefault(team, 50.0)
    return scores


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def _number(value: Any, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def _is_truthy(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "t", "true", "yes", "y"}
    return bool(value)


def _poll_rank_to_score(rank: Any, *, unranked_score: float = 20.0) -> float:
    if rank is None or pd.isna(rank):
        return unranked_score
    return max(0.0, min(100.0, 105.0 - int(rank) * 3.2))


def fetch_game_data(conn, season: int) -> pd.DataFrame:
    sql = """
    SELECT
        id::text AS gameid,
        season,
        week,
        CAST(startdate AS date) AS gamedate,
        seasontype,
        conferencegame,
        hometeam,
        awayteam,
        LOWER(homeclassification) AS homeclassification,
        LOWER(awayclassification) AS awayclassification,
        homeconference,
        awayconference,
        homepoints,
        awaypoints
    FROM public.game_data
    WHERE season = %s
      AND LOWER(seasontype) = 'regular'
      AND startdate IS NOT NULL
      AND (
          (LOWER(homeclassification) = 'fbs' AND LOWER(awayclassification) IN ('fbs', 'fcs'))
          OR (LOWER(awayclassification) = 'fbs' AND LOWER(homeclassification) IN ('fbs', 'fcs'))
      )
    ORDER BY week, startdate, id;
    """
    return _read_sql(conn, sql, (season,))


def fetch_game_predictions(conn, game_prediction_run_id: str) -> pd.DataFrame:
    sql = """
    SELECT
        game_prediction_run_id::text AS game_prediction_run_id,
        gameid::text AS gameid,
        season,
        week,
        home_team,
        away_team,
        homewinprob,
        awaywinprob,
        model_version,
        prediction_type
    FROM public.game_predictions_full
    WHERE game_prediction_run_id = %s;
    """
    return _read_sql(conn, sql, (game_prediction_run_id,))


def build_game_snapshot(
    *,
    game_data: pd.DataFrame,
    predictions: pd.DataFrame,
    run_date: date,
) -> pd.DataFrame:
    if game_data.empty:
        raise RuntimeError("No regular-season game_data rows available for ranking projection.")

    preds = predictions.copy()
    prediction_gameids = set(preds["gameid"].astype(str)) if not preds.empty else set()
    games = game_data.copy()
    games["gameid"] = games["gameid"].astype(str)
    games["gamedate"] = pd.to_datetime(games["gamedate"], errors="coerce").dt.date
    games["completed_before_run"] = (
        games["homepoints"].notna()
        & games["awaypoints"].notna()
        & games["gamedate"].lt(run_date)
    )
    games["needs_prediction"] = ~games["completed_before_run"]
    missing = games[games["needs_prediction"] & ~games["gameid"].isin(prediction_gameids)]
    if not missing.empty:
        preview = "; ".join(
            f"{row.gameid}: {row.awayteam} at {row.hometeam} week {row.week}"
            for row in missing.head(8).itertuples()
        )
        extra = "" if len(missing) <= 8 else f"; +{len(missing) - 8} more"
        raise RuntimeError(
            "Selected game_prediction_run_id does not cover every unplayed regular-season game. "
            "Ranking projections require one complete remaining-season probability snapshot. "
            f"Missing: {preview}{extra}"
        )

    pred_probs = preds[["gameid", "homewinprob"]].copy() if not preds.empty else pd.DataFrame(columns=["gameid", "homewinprob"])
    merged = games.merge(pred_probs, on="gameid", how="left")
    completed_home_win = merged["homepoints"] > merged["awaypoints"]
    merged["effective_homewinprob"] = np.where(
        merged["completed_before_run"],
        np.where(completed_home_win, 1.0, 0.0),
        pd.to_numeric(merged["homewinprob"], errors="coerce").fillna(0.5),
    )
    merged.loc[~merged["completed_before_run"], ["homepoints", "awaypoints"]] = np.nan
    return merged


def fetch_rankings(conn, season: int) -> pd.DataFrame:
    return _read_sql(conn, "SELECT * FROM public.rankings WHERE season = %s;", (season,))


def latest_poll_rankings(
    rankings: pd.DataFrame,
    *,
    poll_names: set[str],
    through_week: int | None,
) -> tuple[dict[str, int], dict[str, int]]:
    if rankings.empty:
        return {}, {}
    rows = rankings[
        rankings["poll"].isin(poll_names)
        & rankings["season_type"].fillna("").astype(str).str.lower().isin(REGULAR_SEASON_TYPES)
    ].copy()
    if through_week is not None:
        rows = rows[pd.to_numeric(rows["week"], errors="coerce").le(through_week)]
    if rows.empty:
        return {}, {}
    weeks = sorted(pd.to_numeric(rows["week"], errors="coerce").dropna().astype(int).unique())
    if not weeks:
        return {}, {}
    current_week = weeks[-1]
    previous_week = weeks[-2] if len(weeks) >= 2 else None
    current = {
        row.school: int(row.rank)
        for row in rows[pd.to_numeric(rows["week"], errors="coerce").eq(current_week)].itertuples()
        if pd.notna(row.rank)
    }
    previous = {}
    if previous_week is not None:
        previous = {
            row.school: int(row.rank)
            for row in rows[pd.to_numeric(rows["week"], errors="coerce").eq(previous_week)].itertuples()
            if pd.notna(row.rank)
        }
    return current, previous


def fetch_supporting_tables(conn, season: int, *, run_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    advanced_max_season = season - 1 if run_type == "backfill" else season
    advanced = _read_sql(
        conn,
        "SELECT * FROM public.team_advanced_season_stats WHERE season <= %s;",
        (advanced_max_season,),
    )
    recruiting = _read_sql(conn, "SELECT * FROM public.team_recruiting_rankings WHERE year <= %s;", (season,))
    talent = _read_sql(conn, "SELECT * FROM public.team_talent_composite WHERE year <= %s;", (season,))
    returning = _read_sql(conn, "SELECT * FROM public.team_returning_production WHERE season <= %s;", (season,))
    return advanced, recruiting, talent, returning


def _best_row_by_season(rows: pd.DataFrame, *, team: str, season: int, season_col: str) -> pd.Series | None:
    if rows.empty or "team" not in rows:
        return None
    candidates = rows[
        rows["team"].eq(team)
        & pd.to_numeric(rows[season_col], errors="coerce").le(season)
    ].copy()
    if candidates.empty:
        return None
    candidates["_sort_season"] = pd.to_numeric(candidates[season_col], errors="coerce")
    return candidates.sort_values("_sort_season").iloc[-1]


def build_supporting_scores(
    *,
    teams: list[str],
    season: int,
    advanced: pd.DataFrame,
    recruiting: pd.DataFrame,
    talent: pd.DataFrame,
    returning: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    raw_efficiency: dict[str, float] = {}
    raw_talent: dict[str, float] = {}
    raw_recruiting: dict[str, float] = {}
    raw_returning: dict[str, float] = {}
    advanced_season: dict[str, int | None] = {}

    for team in teams:
        advanced_row = _best_row_by_season(advanced, team=team, season=season, season_col="season")
        advanced_season[team] = int(advanced_row["season"]) if advanced_row is not None and pd.notna(advanced_row["season"]) else None
        if advanced_row is not None:
            raw_efficiency[team] = (
                _number(advanced_row.get("offense_ppa")) * 45.0
                - _number(advanced_row.get("defense_ppa")) * 45.0
                + _number(advanced_row.get("offense_successrate")) * 22.0
                - _number(advanced_row.get("defense_successrate")) * 22.0
                + _number(advanced_row.get("offense_pointsperopportunity")) * 2.5
                - _number(advanced_row.get("defense_pointsperopportunity")) * 2.5
            )

        talent_row = _best_row_by_season(talent, team=team, season=season, season_col="year")
        if talent_row is not None and pd.notna(talent_row.get("talent")):
            raw_talent[team] = _number(talent_row["talent"])

        recruiting_row = _best_row_by_season(recruiting, team=team, season=season, season_col="year")
        if recruiting_row is not None:
            points = recruiting_row.get("points")
            rank = recruiting_row.get("rank")
            raw_recruiting[team] = _number(points) if pd.notna(points) else -_number(rank, 999.0)

        returning_row = _best_row_by_season(returning, team=team, season=season, season_col="season")
        if returning_row is not None:
            value = returning_row.get("percent_ppa")
            if pd.isna(value):
                value = returning_row.get("total_ppa")
            raw_returning[team] = _number(value)

    efficiency_scores = _percentile_scores(raw_efficiency, teams)
    talent_scores = _percentile_scores(raw_talent, teams)
    recruiting_scores = _percentile_scores(raw_recruiting, teams)
    returning_scores = _percentile_scores(raw_returning, teams)
    return {
        team: {
            "power_score": (
                efficiency_scores[team] * 0.55
                + talent_scores[team] * 0.20
                + recruiting_scores[team] * 0.15
                + returning_scores[team] * 0.10
            ),
            "talent_score": talent_scores[team],
            "recruiting_score": recruiting_scores[team],
            "returning_production_score": returning_scores[team],
            "advanced_stats_season": advanced_season[team],
        }
        for team in teams
    }


def team_profiles(games: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side in ("home", "away"):
        frames.append(
            games[
                [
                    f"{side}team",
                    f"{side}conference",
                    f"{side}classification",
                ]
            ].rename(
                columns={
                    f"{side}team": "team",
                    f"{side}conference": "conference",
                    f"{side}classification": "classification",
                }
            )
        )
    teams = pd.concat(frames, ignore_index=True).dropna(subset=["team"])
    teams["classification"] = teams["classification"].fillna("").astype(str).str.lower()
    return (
        teams.groupby("team", as_index=False)
        .agg(
            conference=("conference", lambda values: values.dropna().iloc[-1] if not values.dropna().empty else None),
            classification=("classification", lambda values: values.dropna().iloc[-1] if not values.dropna().empty else None),
        )
        .sort_values("team")
        .reset_index(drop=True)
    )


def build_team_metrics(games: pd.DataFrame, teams: list[str]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {
        team: {
            "current_wins": 0.0,
            "current_losses": 0.0,
            "current_conference_wins": 0.0,
            "current_conference_losses": 0.0,
            "projected_wins": 0.0,
            "projected_losses": 0.0,
            "projected_conference_wins": 0.0,
            "projected_conference_losses": 0.0,
            "points_for": 0.0,
            "points_against": 0.0,
            "games_played": 0.0,
            "opponents": [],
            "remaining_opponents": [],
        }
        for team in teams
    }

    for game in games.itertuples():
        home = game.hometeam
        away = game.awayteam
        if home not in metrics or away not in metrics:
            continue
        same_conference_game = (
            _is_truthy(game.conferencegame)
            and pd.notna(game.homeconference)
            and pd.notna(game.awayconference)
            and game.homeconference == game.awayconference
        )
        homewinprob = float(np.clip(game.effective_homewinprob, 0.0, 1.0))
        home_completed_win = bool(pd.notna(game.homepoints) and pd.notna(game.awaypoints) and game.homepoints > game.awaypoints)
        completed = bool(game.completed_before_run)

        metrics[home]["opponents"].append(away)
        metrics[away]["opponents"].append(home)

        if completed:
            metrics[home]["current_wins"] += 1.0 if home_completed_win else 0.0
            metrics[home]["current_losses"] += 0.0 if home_completed_win else 1.0
            metrics[away]["current_wins"] += 0.0 if home_completed_win else 1.0
            metrics[away]["current_losses"] += 1.0 if home_completed_win else 0.0
            metrics[home]["points_for"] += float(game.homepoints)
            metrics[home]["points_against"] += float(game.awaypoints)
            metrics[away]["points_for"] += float(game.awaypoints)
            metrics[away]["points_against"] += float(game.homepoints)
            metrics[home]["games_played"] += 1.0
            metrics[away]["games_played"] += 1.0
            if same_conference_game:
                metrics[home]["current_conference_wins"] += 1.0 if home_completed_win else 0.0
                metrics[home]["current_conference_losses"] += 0.0 if home_completed_win else 1.0
                metrics[away]["current_conference_wins"] += 0.0 if home_completed_win else 1.0
                metrics[away]["current_conference_losses"] += 1.0 if home_completed_win else 0.0

        metrics[home]["projected_wins"] += 1.0 if completed and home_completed_win else homewinprob
        metrics[home]["projected_losses"] += 0.0 if completed and home_completed_win else 1.0 - homewinprob
        metrics[away]["projected_wins"] += 0.0 if completed and home_completed_win else 1.0 - homewinprob
        metrics[away]["projected_losses"] += 1.0 if completed and home_completed_win else homewinprob

        if same_conference_game:
            metrics[home]["projected_conference_wins"] += 1.0 if completed and home_completed_win else homewinprob
            metrics[home]["projected_conference_losses"] += 0.0 if completed and home_completed_win else 1.0 - homewinprob
            metrics[away]["projected_conference_wins"] += 0.0 if completed and home_completed_win else 1.0 - homewinprob
            metrics[away]["projected_conference_losses"] += 1.0 if completed and home_completed_win else homewinprob

        if not completed:
            metrics[home]["remaining_opponents"].append(away)
            metrics[away]["remaining_opponents"].append(home)

    for team, row in metrics.items():
        current_games = row["current_wins"] + row["current_losses"]
        projected_games = row["projected_wins"] + row["projected_losses"]
        margin = _safe_ratio(row["points_for"] - row["points_against"], current_games, 0.0)
        current_win_pct = _safe_ratio(row["current_wins"], current_games, 0.5)
        projected_win_pct = _safe_ratio(row["projected_wins"], projected_games, current_win_pct)
        row["current_win_pct"] = current_win_pct
        row["projected_win_pct"] = projected_win_pct
        row["average_point_margin"] = margin

    team_strength = {
        team: (
            metrics[team]["projected_win_pct"] * 0.70
            + _safe_ratio(metrics[team]["average_point_margin"] + 28.0, 56.0, 0.5) * 0.30
        )
        for team in teams
    }
    for team, row in metrics.items():
        row["team_strength"] = team_strength[team]
        row["strength_of_schedule"] = _safe_mean([team_strength[opponent] for opponent in row["opponents"]])
        row["remaining_strength_of_schedule"] = _safe_mean(
            [team_strength[opponent] for opponent in row["remaining_opponents"]]
        )
    return metrics


def build_ranking_projection_records(
    *,
    season: int,
    run_date: date,
    run_type: str,
    model_version: str,
    game_prediction_run_id: str,
    games: pd.DataFrame,
    rankings: pd.DataFrame,
    advanced: pd.DataFrame,
    recruiting: pd.DataFrame,
    talent: pd.DataFrame,
    returning: pd.DataFrame,
    notes: str | None,
    poll_through_week: int | None,
) -> list[dict[str, Any]]:
    profiles = team_profiles(games)
    fbs_profiles = profiles[profiles["classification"].eq("fbs")].copy()
    teams = fbs_profiles["team"].tolist()
    if not teams:
        raise RuntimeError("No FBS teams found for ranking projection.")

    metrics = build_team_metrics(games, teams)
    support = build_supporting_scores(
        teams=teams,
        season=season,
        advanced=advanced,
        recruiting=recruiting,
        talent=talent,
        returning=returning,
    )
    current_ap, previous_ap = latest_poll_rankings(rankings, poll_names=AP_POLL_NAMES, through_week=poll_through_week)
    current_coaches, _ = latest_poll_rankings(rankings, poll_names=COACHES_POLL_NAMES, through_week=poll_through_week)
    current_cfp, previous_cfp = latest_poll_rankings(rankings, poll_names=CFP_POLL_NAMES, through_week=poll_through_week)

    current_win_pct_scores = _percentile_scores({team: metrics[team]["current_win_pct"] for team in teams}, teams)
    projected_win_pct_scores = _percentile_scores({team: metrics[team]["projected_win_pct"] for team in teams}, teams)
    margin_scores = _percentile_scores({team: metrics[team]["average_point_margin"] for team in teams}, teams)
    sos_scores = _percentile_scores({team: metrics[team]["strength_of_schedule"] or 0.5 for team in teams}, teams)
    remaining_sos_scores = _percentile_scores({team: metrics[team]["remaining_strength_of_schedule"] or 0.5 for team in teams}, teams)
    projected_win_total_scores = _percentile_scores({team: metrics[team]["projected_wins"] for team in teams}, teams)
    projected_conference_scores = _percentile_scores({team: metrics[team]["projected_conference_wins"] for team in teams}, teams)

    score_rows: dict[str, dict[str, float]] = {}
    for team in teams:
        power_score = float(support[team]["power_score"] or 50.0)
        poll_inertia_score = (
            _poll_rank_to_score(current_ap.get(team))
            + _poll_rank_to_score(previous_ap.get(team), unranked_score=18.0)
            + _poll_rank_to_score(current_coaches.get(team), unranked_score=18.0)
            + _poll_rank_to_score(current_cfp.get(team), unranked_score=18.0)
        ) / 4.0
        in_season_resume_score = (
            current_win_pct_scores[team] * 0.38
            + margin_scores[team] * 0.22
            + sos_scores[team] * 0.30
            + metrics[team]["current_wins"] * 1.3
            - metrics[team]["current_losses"] * 3.5
        )
        projected_resume_score = (
            projected_win_pct_scores[team] * 0.30
            + projected_win_total_scores[team] * 0.26
            + projected_conference_scores[team] * 0.14
            + sos_scores[team] * 0.18
            + remaining_sos_scores[team] * 0.12
        )
        preseason_resume_score = (
            projected_win_total_scores[team] * 0.36
            + projected_win_pct_scores[team] * 0.22
            + projected_conference_scores[team] * 0.12
            + power_score * 0.20
            + sos_scores[team] * 0.10
        )
        current_games_played = metrics[team]["current_wins"] + metrics[team]["current_losses"]
        in_season_weight = min(current_games_played / 5.0, 1.0)
        resume_score = (
            in_season_resume_score * in_season_weight
            + preseason_resume_score * (1.0 - in_season_weight)
        )
        projected_ap_score = (
            resume_score * 0.42
            + power_score * 0.23
            + poll_inertia_score * 0.30
            + support[team]["talent_score"] * 0.05
        )
        projected_cfp_score = (
            resume_score * 0.52
            + power_score * 0.25
            + sos_scores[team] * 0.13
            + poll_inertia_score * 0.10
        )
        projected_end_ap_score = (
            projected_resume_score * 0.43
            + projected_ap_score * 0.30
            + power_score * 0.17
            + poll_inertia_score * 0.10
        )
        projected_end_cfp_score = (
            projected_resume_score * 0.50
            + projected_cfp_score * 0.25
            + power_score * 0.13
            + sos_scores[team] * 0.07
            + remaining_sos_scores[team] * 0.05
        )
        score_rows[team] = {
            "poll_inertia_score": poll_inertia_score,
            "resume_score": resume_score,
            "projected_resume_score": projected_resume_score,
            "power_score": power_score,
            "projected_ap_score": projected_ap_score,
            "projected_cfp_score": projected_cfp_score,
            "projected_end_ap_score": projected_end_ap_score,
            "projected_end_cfp_score": projected_end_cfp_score,
        }

    projected_ap_ranks = _rank_desc({team: score_rows[team]["projected_ap_score"] for team in teams}, teams)
    projected_cfp_ranks = _rank_desc({team: score_rows[team]["projected_cfp_score"] for team in teams}, teams)
    projected_end_ap_ranks = _rank_desc({team: score_rows[team]["projected_end_ap_score"] for team in teams}, teams)
    projected_end_cfp_ranks = _rank_desc({team: score_rows[team]["projected_end_cfp_score"] for team in teams}, teams)
    profiles_by_team = fbs_profiles.set_index("team").to_dict(orient="index")

    records: list[dict[str, Any]] = []
    for team in teams:
        metric = metrics[team]
        scores = score_rows[team]
        profile = profiles_by_team[team]
        records.append(
            {
                "season": season,
                "run_date": run_date,
                "run_type": run_type,
                "model_version": model_version,
                "team": team,
                "conference": profile.get("conference"),
                "classification": profile.get("classification"),
                "projected_ap_ranking": projected_ap_ranks[team],
                "projected_end_ap_ranking": projected_end_ap_ranks[team],
                "projected_cfp_ranking": projected_cfp_ranks[team],
                "projected_end_cfp_ranking": projected_end_cfp_ranks[team],
                "projected_ap_score": scores["projected_ap_score"],
                "projected_end_ap_score": scores["projected_end_ap_score"],
                "projected_cfp_score": scores["projected_cfp_score"],
                "projected_end_cfp_score": scores["projected_end_cfp_score"],
                "resume_score": scores["resume_score"],
                "projected_resume_score": scores["projected_resume_score"],
                "power_score": scores["power_score"],
                "poll_inertia_score": scores["poll_inertia_score"],
                "current_wins": metric["current_wins"],
                "current_losses": metric["current_losses"],
                "current_conference_wins": metric["current_conference_wins"],
                "current_conference_losses": metric["current_conference_losses"],
                "projected_wins": metric["projected_wins"],
                "projected_losses": metric["projected_losses"],
                "projected_conference_wins": metric["projected_conference_wins"],
                "projected_conference_losses": metric["projected_conference_losses"],
                "current_ap_rank": current_ap.get(team),
                "previous_ap_rank": previous_ap.get(team),
                "current_coaches_rank": current_coaches.get(team),
                "current_cfp_rank": current_cfp.get(team),
                "previous_cfp_rank": previous_cfp.get(team),
                "strength_of_schedule": metric["strength_of_schedule"],
                "remaining_strength_of_schedule": metric["remaining_strength_of_schedule"],
                "team_strength": metric["team_strength"],
                "talent_score": support[team]["talent_score"],
                "recruiting_score": support[team]["recruiting_score"],
                "returning_production_score": support[team]["returning_production_score"],
                "advanced_stats_season": support[team]["advanced_stats_season"],
                "game_prediction_run_id": game_prediction_run_id,
                "prediction_type": "FBS",
                "notes": notes,
            }
        )

    records_df = pd.DataFrame(records)
    records_df = records_df.astype(object).where(pd.notna(records_df), None)
    return records_df.to_dict(orient="records")


def process_ranking_projection_snapshot(
    conn,
    *,
    season: int,
    run_date: date,
    run_type: str,
    etl_run_id: str,
    notes: str | None,
    explicit_game_prediction_run_id: str | None,
    poll_through_week: int | None,
) -> tuple[uuid.UUID, int, str]:
    run_id = uuid.uuid4()
    game_prediction_run_id, game_model_version = select_game_prediction_run(
        conn,
        season=season,
        run_type=run_type,
        run_date=run_date,
        explicit_run_id=explicit_game_prediction_run_id,
    )
    model_version = _model_version_label(game_model_version)
    print(f"Ranking projection run id: {run_id}")
    print(f"Run date/type: {run_date} / {run_type}")
    print(f"Using game_prediction_run_id: {game_prediction_run_id}")
    if poll_through_week is not None:
        print(f"Poll inputs limited through week: {poll_through_week}")

    game_data = fetch_game_data(conn, season)
    game_predictions = fetch_game_predictions(conn, game_prediction_run_id)
    games = build_game_snapshot(
        game_data=game_data,
        predictions=game_predictions,
        run_date=run_date,
    )
    rankings = fetch_rankings(conn, season)
    advanced, recruiting, talent, returning = fetch_supporting_tables(conn, season, run_type=run_type)
    records = build_ranking_projection_records(
        season=season,
        run_date=run_date,
        run_type=run_type,
        model_version=model_version,
        game_prediction_run_id=game_prediction_run_id,
        games=games,
        rankings=rankings,
        advanced=advanced,
        recruiting=recruiting,
        talent=talent,
        returning=returning,
        notes=notes,
        poll_through_week=poll_through_week,
    )
    records, prediction_hash = add_prediction_hashes(records)
    print(f"Prepared {len(records)} ranking projections for season {season}.")
    print(f"Prediction hash: {prediction_hash}")

    create_run(
        conn,
        run_id=run_id,
        season=season,
        run_date=run_date,
        run_type=run_type,
        etl_run_id=etl_run_id,
        game_prediction_run_id=game_prediction_run_id,
        model_version=model_version,
        row_count=len(records),
        notes=notes,
    )

    try:
        latest = (
            get_latest_successful_run_for_date(conn, season, run_type, run_date)
            if run_type == "backfill"
            else get_latest_successful_run(conn, season, run_type)
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
                    "Ranking projections match latest comparable successful snapshot; "
                    f"marking this run as duplicate of {latest_run_id}."
                )
                write_duplicate_run(
                    conn,
                    run_id=run_id,
                    prediction_hash=prediction_hash,
                    duplicate_of_run_id=latest_run_id,
                )
                print("Finished without inserting duplicate ranking projection detail rows.")
                return run_id, 0, "duplicate"

        print("Ranking projection set changed; inserting full snapshot rows...")
        for record in records:
            record["ranking_projection_run_id"] = run_id
        write_changed_snapshot(
            conn,
            run_id=run_id,
            records=records,
            prediction_hash=prediction_hash,
        )
        print(f"Finished inserting {len(records)} rows into ranking_projections_full.")
        return run_id, len(records), "success"
    except Exception as exc:
        conn.rollback()
        mark_run_failed(conn, run_id, str(exc))
        raise


def main() -> None:
    cfg = load_config()
    season = cfg.season
    run_date = _run_date_from_env()
    run_date_was_supplied = _run_date_was_supplied()
    run_type = _run_type_from_env()
    notes = os.getenv(NOTES_ENV, "").strip() or None
    explicit_game_prediction_run_id = os.getenv(GAME_PREDICTION_RUN_ID_ENV, "").strip() or None
    poll_through_week = _poll_through_week_from_env()

    print("Connecting to database...")
    with psycopg.connect(cfg.pg_dsn) as conn:
        print(f"Target season: {season}")
        ensure_ranking_projection_tables(conn)

        if run_type == "backfill" and not run_date_was_supplied and not explicit_game_prediction_run_id:
            dates = backfill_run_dates(conn, season)
            print(f"Backfilling {len(dates)} ranking projection run dates for season {season}.")
            total_inserted = 0
            for backfill_date in dates:
                _, inserted_count, _ = process_ranking_projection_snapshot(
                    conn,
                    season=season,
                    run_date=backfill_date,
                    run_type=run_type,
                    etl_run_id=cfg.run_id,
                    notes=notes,
                    explicit_game_prediction_run_id=None,
                    poll_through_week=poll_through_week,
                )
                total_inserted += inserted_count
            print(f"Finished ranking projection backfill; inserted {total_inserted} detail rows.")
            return

        process_ranking_projection_snapshot(
            conn,
            season=season,
            run_date=run_date,
            run_type=run_type,
            etl_run_id=cfg.run_id,
            notes=notes,
            explicit_game_prediction_run_id=explicit_game_prediction_run_id,
            poll_through_week=poll_through_week,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"update_ranking_projections.py failed: {e}")
        sys.exit(1)
