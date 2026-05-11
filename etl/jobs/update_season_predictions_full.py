#!/usr/bin/env python3
"""
Build season-level team prediction snapshots from game and ranking snapshots.

The job follows the same run/detail snapshot pattern as
etl.jobs.update_game_predictions_full. It uses the game_prediction_run_id tied
to a successful ranking_projection_run_id so remaining-season win probabilities,
projected rankings, CFP selection, and playoff simulation all share one coherent
snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import uuid
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import psycopg

from etl.common_config import load_config
from etl.jobs.conference_championship_rules import (
    SEC_RELATIVE_OFFENSE_CAP,
    select_conference_championship_teams,
    simulate_conference_championship_game,
    sun_belt_division_for_team,
)
from etl.jobs.update_game_predictions import FBS_PREDICTION_TYPE, FCS_PREDICTION_TYPE


RUN_TYPE_ENV = "SEASON_PREDICTION_RUN_TYPE"
RUN_DATE_ENV = "SEASON_PREDICTION_RUN_DATE"
NOTES_ENV = "SEASON_PREDICTION_RUN_NOTES"
SIMULATIONS_ENV = "SEASON_PREDICTION_SIMULATIONS"
RANDOM_SEED_ENV = "SEASON_PREDICTION_RANDOM_SEED"
RANKING_PROJECTION_RUN_ID_ENV = "SEASON_PREDICTION_RANKING_PROJECTION_RUN_ID"
DEFAULT_SIMULATIONS = 10_000
MAX_WIN_BUCKET = 13
HASH_FLOAT_DECIMAL_PLACES = 4
NORMAL_SNAPSHOT_RUN_TYPES = ("manual", "nightly")
INDEPENDENT_CONFERENCES = {"FBS Independents", "FCS Independents", "Independent"}
CFP_POWER_AUTO_CONFERENCES = {"ACC", "Big Ten", "Big 12", "SEC"}
CFP_GROUP_AUTO_CONFERENCES = {
    "American Athletic",
    "Conference USA",
    "Mid-American",
    "Mountain West",
    "Pac-12",
    "Sun Belt",
}
NOTRE_DAME = "Notre Dame"

WIN_PROBABILITY_COLUMNS = tuple(f"probability_{wins}_wins" for wins in range(MAX_WIN_BUCKET + 1))
DETAIL_HASH_EXCLUDED_COLUMNS = {
    "season_prediction_run_id",
    "run_date",
    "run_type",
    "created_at",
    "notes",
    "prediction_hash",
}
DETAIL_HASH_COLUMNS = (
    "season",
    "model_version",
    "team",
    "conference",
    "division",
    "classification",
    "projected_wins",
    "projected_losses",
    "projected_conference_wins",
    "projected_conference_losses",
    *WIN_PROBABILITY_COLUMNS,
    "conference_championship_game_prob",
    "conference_champion_prob",
    "playoff_prob",
    "cfp_bye_prob",
    "cfp_at_large_prob",
    "cfp_auto_bid_prob",
    "national_championship_game_prob",
    "national_champion_prob",
    "bowl_eligible_prob",
    "projected_ap_ranking",
    "projected_cfp_ranking",
    "resume_ranking",
    "strength_of_schedule",
    "remaining_strength_of_schedule",
    "expected_number_of_wins",
    "simulations",
    "prediction_type",
)


CREATE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.season_prediction_runs (
    season_prediction_run_id UUID PRIMARY KEY,
    season                   INT NOT NULL,
    run_date                 DATE NOT NULL,
    run_type                 TEXT NOT NULL DEFAULT 'nightly',
    etl_run_id               TEXT,
    ranking_projection_run_id UUID REFERENCES public.ranking_projection_runs(ranking_projection_run_id),
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at             TIMESTAMPTZ,
    status                   TEXT NOT NULL DEFAULT 'running',
    model_version            TEXT NOT NULL,
    prediction_hash          TEXT,
    duplicate_of_run_id      UUID REFERENCES public.season_prediction_runs(season_prediction_run_id),
    row_count                INT NOT NULL DEFAULT 0,
    inserted_row_count       INT NOT NULL DEFAULT 0,
    simulations              INT NOT NULL DEFAULT 0,
    notes                    TEXT,
    error_message            TEXT,
    CONSTRAINT season_prediction_runs_status_check
      CHECK (status IN ('running', 'success', 'duplicate', 'failed'))
);
"""

CREATE_DETAILS_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS public.season_predictions_full (
    season_prediction_run_id UUID NOT NULL
      REFERENCES public.season_prediction_runs(season_prediction_run_id)
      ON DELETE CASCADE,
    season                              INT NOT NULL,
    run_date                            DATE NOT NULL,
    run_type                            TEXT NOT NULL DEFAULT 'nightly',
    model_version                       TEXT NOT NULL,
    team                                TEXT NOT NULL,
    conference                          TEXT,
    division                            TEXT,
    classification                      TEXT,
    created_at                          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    projected_wins                      DOUBLE PRECISION,
    projected_losses                    DOUBLE PRECISION,
    projected_conference_wins           DOUBLE PRECISION,
    projected_conference_losses         DOUBLE PRECISION,
    {", ".join(f"{column} DOUBLE PRECISION" for column in WIN_PROBABILITY_COLUMNS)},
    conference_championship_game_prob   DOUBLE PRECISION,
    conference_champion_prob            DOUBLE PRECISION,
    playoff_prob                        DOUBLE PRECISION,
    cfp_bye_prob                        DOUBLE PRECISION,
    cfp_at_large_prob                   DOUBLE PRECISION,
    cfp_auto_bid_prob                   DOUBLE PRECISION,
    national_championship_game_prob     DOUBLE PRECISION,
    national_champion_prob              DOUBLE PRECISION,
    bowl_eligible_prob                  DOUBLE PRECISION,
    projected_ap_ranking                INT,
    projected_cfp_ranking               INT,
    resume_ranking                      INT,
    strength_of_schedule                DOUBLE PRECISION,
    remaining_strength_of_schedule      DOUBLE PRECISION,
    expected_number_of_wins             DOUBLE PRECISION,
    simulations                         INT NOT NULL,
    prediction_hash                     TEXT NOT NULL,
    prediction_type                     TEXT NOT NULL DEFAULT 'FBS',
    notes                               TEXT,
    PRIMARY KEY (season_prediction_run_id, team)
);
"""

CREATE_INDEXES_SQL = """
ALTER TABLE public.season_prediction_runs
ADD COLUMN IF NOT EXISTS ranking_projection_run_id UUID;

CREATE INDEX IF NOT EXISTS idx_season_prediction_runs_lookup
  ON public.season_prediction_runs (season, run_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_season_prediction_runs_hash
  ON public.season_prediction_runs (season, run_type, prediction_hash)
  WHERE status = 'success';

CREATE INDEX IF NOT EXISTS idx_season_prediction_runs_ranking_projection
  ON public.season_prediction_runs (ranking_projection_run_id);

CREATE INDEX IF NOT EXISTS idx_season_predictions_full_team_lookup
  ON public.season_predictions_full (season, team, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_season_predictions_full_conference_lookup
  ON public.season_predictions_full (season, conference, team);
"""

INSERT_RUN_SQL = """
INSERT INTO public.season_prediction_runs (
    season_prediction_run_id,
    season,
    run_date,
    run_type,
    etl_run_id,
    ranking_projection_run_id,
    status,
    model_version,
    row_count,
    inserted_row_count,
    simulations,
    notes
)
VALUES (
    %(season_prediction_run_id)s,
    %(season)s,
    %(run_date)s,
    %(run_type)s,
    %(etl_run_id)s,
    %(ranking_projection_run_id)s,
    'running',
    %(model_version)s,
    %(row_count)s,
    0,
    %(simulations)s,
    %(notes)s
);
"""

MARK_RUN_SUCCESS_SQL = """
UPDATE public.season_prediction_runs
SET
    completed_at = NOW(),
    status = 'success',
    prediction_hash = %(prediction_hash)s,
    inserted_row_count = %(inserted_row_count)s
WHERE season_prediction_run_id = %(season_prediction_run_id)s;
"""

MARK_RUN_DUPLICATE_SQL = """
UPDATE public.season_prediction_runs
SET
    completed_at = NOW(),
    status = 'duplicate',
    prediction_hash = %(prediction_hash)s,
    duplicate_of_run_id = %(duplicate_of_run_id)s,
    inserted_row_count = 0
WHERE season_prediction_run_id = %(season_prediction_run_id)s;
"""

MARK_RUN_FAILED_SQL = """
UPDATE public.season_prediction_runs
SET
    completed_at = NOW(),
    status = 'failed',
    error_message = %(error_message)s
WHERE season_prediction_run_id = %(season_prediction_run_id)s;
"""

LATEST_SUCCESSFUL_RUN_SQL = """
SELECT season_prediction_run_id, prediction_hash
FROM public.season_prediction_runs
WHERE season = %s
  AND run_type IN ({run_type_placeholders})
  AND status = 'success'
  AND prediction_hash IS NOT NULL
ORDER BY created_at DESC
LIMIT 1;
"""

LATEST_SUCCESSFUL_RUN_BY_DATE_SQL = """
SELECT season_prediction_run_id, prediction_hash
FROM public.season_prediction_runs
WHERE season = %s
  AND run_type = %s
  AND run_date = %s
  AND status = 'success'
  AND prediction_hash IS NOT NULL
ORDER BY created_at DESC
LIMIT 1;
"""

INSERT_DETAIL_SQL = f"""
INSERT INTO public.season_predictions_full (
    season_prediction_run_id,
    season,
    run_date,
    run_type,
    model_version,
    team,
    conference,
    division,
    classification,
    projected_wins,
    projected_losses,
    projected_conference_wins,
    projected_conference_losses,
    {", ".join(WIN_PROBABILITY_COLUMNS)},
    conference_championship_game_prob,
    conference_champion_prob,
    playoff_prob,
    cfp_bye_prob,
    cfp_at_large_prob,
    cfp_auto_bid_prob,
    national_championship_game_prob,
    national_champion_prob,
    bowl_eligible_prob,
    projected_ap_ranking,
    projected_cfp_ranking,
    resume_ranking,
    strength_of_schedule,
    remaining_strength_of_schedule,
    expected_number_of_wins,
    simulations,
    prediction_hash,
    prediction_type,
    notes
)
VALUES (
    %(season_prediction_run_id)s,
    %(season)s,
    %(run_date)s,
    %(run_type)s,
    %(model_version)s,
    %(team)s,
    %(conference)s,
    %(division)s,
    %(classification)s,
    %(projected_wins)s,
    %(projected_losses)s,
    %(projected_conference_wins)s,
    %(projected_conference_losses)s,
    {", ".join(f"%({column})s" for column in WIN_PROBABILITY_COLUMNS)},
    %(conference_championship_game_prob)s,
    %(conference_champion_prob)s,
    %(playoff_prob)s,
    %(cfp_bye_prob)s,
    %(cfp_at_large_prob)s,
    %(cfp_auto_bid_prob)s,
    %(national_championship_game_prob)s,
    %(national_champion_prob)s,
    %(bowl_eligible_prob)s,
    %(projected_ap_ranking)s,
    %(projected_cfp_ranking)s,
    %(resume_ranking)s,
    %(strength_of_schedule)s,
    %(remaining_strength_of_schedule)s,
    %(expected_number_of_wins)s,
    %(simulations)s,
    %(prediction_hash)s,
    %(prediction_type)s,
    %(notes)s
);
"""

DETAIL_RECORDS_FOR_RUN_SQL = f"""
SELECT {", ".join(DETAIL_HASH_COLUMNS)}
FROM public.season_predictions_full
WHERE season_prediction_run_id = %s
ORDER BY team;
"""

LATEST_RANKING_PROJECTION_RUN_SQL = """
SELECT ranking_projection_run_id, model_version, game_prediction_run_id
FROM public.ranking_projection_runs
WHERE season = %s
  AND run_type IN ({run_type_placeholders})
  AND status = 'success'
ORDER BY created_at DESC
LIMIT 1;
"""

LATEST_RANKING_PROJECTION_RUN_BY_DATE_SQL = """
SELECT ranking_projection_run_id, model_version, game_prediction_run_id
FROM public.ranking_projection_runs
WHERE season = %s
  AND run_type = %s
  AND run_date = %s
  AND status = 'success'
ORDER BY created_at DESC
LIMIT 1;
"""

EXPLICIT_RANKING_PROJECTION_RUN_SQL = """
SELECT ranking_projection_run_id, model_version, game_prediction_run_id
FROM public.ranking_projection_runs
WHERE ranking_projection_run_id = %s
  AND season = %s
  AND status = 'success'
LIMIT 1;
"""

BACKFILL_RANKING_RUN_DATES_SQL = """
SELECT DISTINCT run_date
FROM public.ranking_projection_runs
WHERE season = %s
  AND run_type = 'backfill'
  AND status = 'success'
ORDER BY run_date;
"""

RANKING_PROJECTION_ROWS_SQL = """
SELECT
    team,
    projected_ap_ranking,
    projected_end_ap_ranking,
    projected_cfp_ranking,
    projected_end_cfp_ranking,
    projected_end_cfp_score,
    projected_wins,
    projected_conference_wins,
    team_strength,
    strength_of_schedule
FROM public.ranking_projections_full
WHERE ranking_projection_run_id = %s;
"""

GAME_SNAPSHOT_SQL = """
SELECT
    gd.id::text AS id,
    gd.season,
    gd.week,
    CAST(gd.startdate AS date) AS gamedate,
    gd.seasontype,
    gd.conferencegame,
    gd.hometeam,
    gd.awayteam,
    LOWER(gd.homeclassification) AS homeclassification,
    LOWER(gd.awayclassification) AS awayclassification,
    gd.homeconference,
    gd.awayconference,
    gd.homepoints,
    gd.awaypoints,
    gp.homewinprob,
    gp.homespread,
    gp.totalpred,
    gp.model_version AS game_prediction_model_version,
    gp.prediction_type
FROM public.game_data gd
LEFT JOIN public.game_predictions_full gp
  ON gp.gameid = gd.id::text
 AND gp.game_prediction_run_id = %s
WHERE gd.season = %s
  AND LOWER(gd.seasontype) = 'regular'
  AND gd.startdate IS NOT NULL
  AND (
      (LOWER(gd.homeclassification) = 'fbs' AND LOWER(gd.awayclassification) IN ('fbs', 'fcs'))
      OR (LOWER(gd.awayclassification) = 'fbs' AND LOWER(gd.homeclassification) IN ('fbs', 'fcs'))
  )
ORDER BY gd.week, gd.startdate, gd.id;
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


def _simulations_from_env() -> int:
    raw = os.getenv(SIMULATIONS_ENV, "").strip()
    if not raw:
        return DEFAULT_SIMULATIONS
    simulations = int(raw)
    if simulations <= 0:
        raise ValueError(f"{SIMULATIONS_ENV} must be positive.")
    return simulations


def _rng_from_env(season: int) -> np.random.Generator:
    raw = os.getenv(RANDOM_SEED_ENV, "").strip()
    seed = int(raw) if raw else int(season)
    return np.random.default_rng(seed)


def _model_version_label(ranking_model_version: str | None = None) -> str:
    suffix = ranking_model_version or "unknown_ranking_projection"
    return f"season_sim_2026+{suffix}"


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


def ensure_full_prediction_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(CREATE_RUNS_TABLE_SQL)
        cur.execute(CREATE_DETAILS_TABLE_SQL)
        cur.execute(CREATE_INDEXES_SQL)
    conn.commit()


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
    ranking_projection_run_id: str,
    model_version: str,
    row_count: int,
    simulations: int,
    notes: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            INSERT_RUN_SQL,
            {
                "season_prediction_run_id": run_id,
                "season": season,
                "run_date": run_date,
                "run_type": run_type,
                "etl_run_id": etl_run_id,
                "ranking_projection_run_id": ranking_projection_run_id,
                "model_version": model_version,
                "row_count": row_count,
                "simulations": simulations,
                "notes": notes,
            },
        )
    conn.commit()


def mark_run_failed(conn, run_id: uuid.UUID, error_message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            MARK_RUN_FAILED_SQL,
            {
                "season_prediction_run_id": run_id,
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
                "season_prediction_run_id": run_id,
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
                "season_prediction_run_id": run_id,
                "prediction_hash": prediction_hash,
                "duplicate_of_run_id": duplicate_of_run_id,
            },
        )
    conn.commit()


def _first_non_null(values: pd.Series) -> Any:
    cleaned = values.dropna()
    if cleaned.empty:
        return None
    modes = cleaned.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return cleaned.iloc[-1]


def _team_profiles(games: pd.DataFrame) -> pd.DataFrame:
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
            conference=("conference", _first_non_null),
            classification=("classification", _first_non_null),
        )
        .sort_values("team")
        .reset_index(drop=True)
    )


def _is_truthy(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "t", "true", "yes", "y"}
    return bool(value)


def _is_completed_as_of(games: pd.DataFrame, run_date: date) -> pd.Series:
    gamedate = pd.to_datetime(games["gamedate"], errors="coerce").dt.date
    return (
        games["homepoints"].notna()
        & games["awaypoints"].notna()
        & gamedate.lt(run_date)
    )


def _regular_season_games(preds: pd.DataFrame) -> pd.DataFrame:
    season_type = preds["seasontype"].fillna("").astype(str).str.lower()
    return preds[season_type.eq("regular")].copy()


def _has_explicit_fbs_championship_games(games: pd.DataFrame) -> bool:
    notes = games.get("notes", pd.Series("", index=games.index)).fillna("").astype(str).str.lower()
    same_conference = games["homeconference"].fillna("") == games["awayconference"].fillna("")
    both_fbs = (
        games["homeclassification"].fillna("").astype(str).str.lower().eq("fbs")
        & games["awayclassification"].fillna("").astype(str).str.lower().eq("fbs")
    )
    conference_game = games["conferencegame"].map(_is_truthy)
    return bool((notes.str.contains("championship") & same_conference & both_fbs & conference_game).any())


def _rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _simulated_scores(game: pd.Series, home_win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homepoints = pd.to_numeric(game.get("homepoints"), errors="coerce")
    awaypoints = pd.to_numeric(game.get("awaypoints"), errors="coerce")
    if pd.notna(homepoints) and pd.notna(awaypoints):
        return (
            np.full(len(home_win), float(homepoints), dtype=float),
            np.full(len(home_win), float(awaypoints), dtype=float),
        )

    homespread = pd.to_numeric(game.get("homespread"), errors="coerce")
    expected_home_margin = -float(homespread) if pd.notna(homespread) else 1.0
    magnitude = abs(expected_home_margin) or 1.0
    signed_margin = np.where(home_win, magnitude, -magnitude)
    totalpred = pd.to_numeric(game.get("totalpred"), errors="coerce")
    total = float(totalpred) if pd.notna(totalpred) else 50.0
    total = max(total, magnitude)
    home_score = np.maximum((total + signed_margin) / 2.0, 0.0)
    away_score = np.maximum((total - signed_margin) / 2.0, 0.0)
    return home_score.astype(float), away_score.astype(float)


def _relative_scoring_margin(
    *,
    points_for: float,
    points_allowed: float,
    opponent_avg_scored: float,
    opponent_avg_allowed: float,
) -> float:
    if opponent_avg_allowed <= 0.0:
        offense_pct = SEC_RELATIVE_OFFENSE_CAP if points_for > 0.0 else 100.0
    else:
        offense_pct = min((points_for / opponent_avg_allowed) * 100.0, SEC_RELATIVE_OFFENSE_CAP)

    if opponent_avg_scored <= 0.0:
        defense_pct = 0.0 if points_allowed == 0.0 else SEC_RELATIVE_OFFENSE_CAP
    else:
        defense_pct = max((points_allowed / opponent_avg_scored) * 100.0, 0.0)

    return float(offense_pct - defense_pct)


def _build_sec_relative_scoring_margins(
    conference_game_scores_by_sim: list[list[tuple[int, int, float, float]]],
    n_teams: int,
) -> np.ndarray:
    margins = np.zeros((len(conference_game_scores_by_sim), n_teams), dtype=float)
    for sim, scores in enumerate(conference_game_scores_by_sim):
        points_for = np.zeros(n_teams, dtype=float)
        points_allowed = np.zeros(n_teams, dtype=float)
        games_played = np.zeros(n_teams, dtype=float)

        for home_idx, away_idx, home_score, away_score in scores:
            points_for[home_idx] += home_score
            points_allowed[home_idx] += away_score
            games_played[home_idx] += 1.0
            points_for[away_idx] += away_score
            points_allowed[away_idx] += home_score
            games_played[away_idx] += 1.0

        avg_scored = np.divide(
            points_for,
            games_played,
            out=np.zeros(n_teams, dtype=float),
            where=games_played > 0,
        )
        avg_allowed = np.divide(
            points_allowed,
            games_played,
            out=np.zeros(n_teams, dtype=float),
            where=games_played > 0,
        )
        relative_sums = np.zeros(n_teams, dtype=float)
        relative_counts = np.zeros(n_teams, dtype=float)

        for home_idx, away_idx, home_score, away_score in scores:
            relative_sums[home_idx] += _relative_scoring_margin(
                points_for=home_score,
                points_allowed=away_score,
                opponent_avg_scored=avg_scored[away_idx],
                opponent_avg_allowed=avg_allowed[away_idx],
            )
            relative_counts[home_idx] += 1.0
            relative_sums[away_idx] += _relative_scoring_margin(
                points_for=away_score,
                points_allowed=home_score,
                opponent_avg_scored=avg_scored[home_idx],
                opponent_avg_allowed=avg_allowed[home_idx],
            )
            relative_counts[away_idx] += 1.0

        margins[sim] = np.divide(
            relative_sums,
            relative_counts,
            out=np.zeros(n_teams, dtype=float),
            where=relative_counts > 0,
        )
    return margins


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def _ranking_projection_value(
    ranking_projection: dict[str, dict[str, Any]],
    team: str,
    key: str,
    default: Any = None,
) -> Any:
    return ranking_projection.get(team, {}).get(key, default)


def _ranking_projection_score(
    ranking_projection: dict[str, dict[str, Any]],
    team: str,
    *,
    fallback_rank: int,
    fallback_score: float,
) -> float:
    score = _ranking_projection_value(ranking_projection, team, "projected_end_cfp_score")
    if score is not None and pd.notna(score):
        return float(score)
    rank = (
        _ranking_projection_value(ranking_projection, team, "projected_end_cfp_ranking")
        or _ranking_projection_value(ranking_projection, team, "projected_cfp_ranking")
        or fallback_rank
    )
    if rank is not None and pd.notna(rank) and int(rank) > 0:
        return max(0.0, 130.0 - int(rank) * 2.0)
    return fallback_score


def _select_cfp_field(
    *,
    ranked_fbs: list[int],
    conference_champion_row: np.ndarray,
    conferences: list[Any],
    team_names: list[str],
    ranking_scores: np.ndarray,
) -> tuple[list[int], list[int], list[int]]:
    rank_order = {idx: rank + 1 for rank, idx in enumerate(ranked_fbs)}
    champion_indices = [idx for idx in ranked_fbs if bool(conference_champion_row[idx])]
    champions_by_conference = {
        conferences[idx]: idx
        for idx in champion_indices
        if conferences[idx] is not None
    }

    auto_bids: list[int] = []
    for conference in sorted(CFP_POWER_AUTO_CONFERENCES):
        champion = champions_by_conference.get(conference)
        if champion is not None:
            auto_bids.append(champion)

    group_candidates = [
        idx
        for idx in ranked_fbs
        if conferences[idx] in CFP_GROUP_AUTO_CONFERENCES
    ]
    if group_candidates:
        auto_bids.append(group_candidates[0])

    notre_dame_idx = next((idx for idx, team in enumerate(team_names) if team == NOTRE_DAME), None)
    if (
        notre_dame_idx is not None
        and notre_dame_idx in rank_order
        and rank_order[notre_dame_idx] <= 12
    ):
        auto_bids.append(notre_dame_idx)

    auto_bids = list(dict.fromkeys(auto_bids))
    auto_set = set(auto_bids)
    at_large_slots = max(0, 12 - len(auto_bids))
    at_large = [idx for idx in ranked_fbs if idx not in auto_set][:at_large_slots]
    playoff = auto_bids + at_large

    auto_set = set(auto_bids)

    def seed_key(idx: int) -> tuple[int, int, float, str]:
        unranked_auto_bid = idx in auto_set and rank_order.get(idx, 999) > 25
        return (
            1 if unranked_auto_bid else 0,
            rank_order.get(idx, 999),
            -float(ranking_scores[idx]),
            team_names[idx],
        )

    seeded = sorted(playoff, key=seed_key)[:12]
    return seeded, auto_bids, [idx for idx in seeded if idx not in auto_set]


def _cfp_game_win_probability(
    team_a: int,
    team_b: int,
    *,
    ranking_scores: np.ndarray,
    team_strength: np.ndarray,
    total_wins: np.ndarray,
    sim: int,
) -> float:
    ranking_delta = (ranking_scores[team_a] - ranking_scores[team_b]) / 18.0
    strength_delta = (team_strength[team_a] - team_strength[team_b]) * 4.0
    win_delta = (total_wins[sim, team_a] - total_wins[sim, team_b]) * 0.12
    probability = _sigmoid(float(ranking_delta + strength_delta + win_delta))
    return float(np.clip(probability, 0.05, 0.95))


def _simulate_cfp_game(
    team_a: int,
    team_b: int,
    *,
    ranking_scores: np.ndarray,
    team_strength: np.ndarray,
    total_wins: np.ndarray,
    sim: int,
    rng: np.random.Generator,
) -> int:
    probability = _cfp_game_win_probability(
        team_a,
        team_b,
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
    )
    return team_a if rng.random() < probability else team_b


def _simulate_cfp_bracket(
    seeded: list[int],
    *,
    ranking_scores: np.ndarray,
    team_strength: np.ndarray,
    total_wins: np.ndarray,
    sim: int,
    rng: np.random.Generator,
) -> tuple[list[int], int | None]:
    if len(seeded) < 12:
        return [], seeded[0] if seeded else None

    seed = {idx + 1: team for idx, team in enumerate(seeded)}
    winner_5_12 = _simulate_cfp_game(
        seed[5], seed[12], ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    winner_6_11 = _simulate_cfp_game(
        seed[6], seed[11], ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    winner_7_10 = _simulate_cfp_game(
        seed[7], seed[10], ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    winner_8_9 = _simulate_cfp_game(
        seed[8], seed[9], ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    quarter_1 = _simulate_cfp_game(
        seed[1], winner_8_9, ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    quarter_4 = _simulate_cfp_game(
        seed[4], winner_5_12, ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    quarter_2 = _simulate_cfp_game(
        seed[2], winner_7_10, ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    quarter_3 = _simulate_cfp_game(
        seed[3], winner_6_11, ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    semifinal_1 = _simulate_cfp_game(
        quarter_1, quarter_4, ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    semifinal_2 = _simulate_cfp_game(
        quarter_2, quarter_3, ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    champion = _simulate_cfp_game(
        semifinal_1, semifinal_2, ranking_scores=ranking_scores, team_strength=team_strength, total_wins=total_wins, sim=sim, rng=rng
    )
    return [semifinal_1, semifinal_2], champion


def _ranking_projection_lookup(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if rows.empty:
        raise RuntimeError("Selected ranking projection run has no detail rows.")
    return rows.set_index("team").to_dict(orient="index")


def build_season_prediction_records(
    preds: pd.DataFrame,
    *,
    ranking_projection: dict[str, dict[str, Any]],
    ranking_model_version: str | None,
    season: int,
    run_date: date,
    run_type: str,
    simulations: int,
    notes: str | None,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    games = _regular_season_games(preds)
    if games.empty:
        raise RuntimeError(f"No regular-season games available for season {season}.")

    profiles = _team_profiles(games)
    team_names = profiles["team"].tolist()
    team_to_idx = {team: idx for idx, team in enumerate(team_names)}
    n_teams = len(team_names)

    classification = profiles["classification"].fillna("").astype(str).str.lower().to_numpy()
    conferences = profiles["conference"].astype(object).where(pd.notna(profiles["conference"]), None).tolist()
    fbs_mask = classification == "fbs"
    output_indices = np.where(fbs_mask)[0]

    wins = np.zeros((simulations, n_teams), dtype=np.int16)
    losses = np.zeros((simulations, n_teams), dtype=np.int16)
    adjusted_total_wins = np.zeros((simulations, n_teams), dtype=np.int16)
    fcs_wins_counted = np.zeros((simulations, n_teams), dtype=np.int16)
    fbs_wins = np.zeros((simulations, n_teams), dtype=np.int16)
    fbs_losses = np.zeros((simulations, n_teams), dtype=np.int16)
    conference_wins = np.zeros((simulations, n_teams), dtype=np.int16)
    conference_losses = np.zeros((simulations, n_teams), dtype=np.int16)
    divisional_wins = np.zeros((simulations, n_teams), dtype=np.int16)
    divisional_losses = np.zeros((simulations, n_teams), dtype=np.int16)
    final_conference_week_losses = np.zeros((simulations, n_teams), dtype=bool)
    expected_wins_base = np.zeros(n_teams, dtype=float)
    expected_games_base = np.zeros(n_teams, dtype=float)
    opponent_lists: list[list[int]] = [[] for _ in range(n_teams)]
    remaining_opponent_lists: list[list[int]] = [[] for _ in range(n_teams)]
    head_to_head_wins_by_sim: list[dict[str, dict[str, int]]] = [
        defaultdict(dict) for _ in range(simulations)
    ]
    conference_game_scores_by_sim: list[list[tuple[int, int, float, float]]] = [
        [] for _ in range(simulations)
    ]

    final_conference_week_by_conference: dict[str, int] = {}
    for _, game in games.iterrows():
        if (
            _is_truthy(game.get("conferencegame"))
            and pd.notna(game.get("homeconference"))
            and game.get("homeconference") == game.get("awayconference")
            and pd.notna(game.get("week"))
        ):
            conference = str(game["homeconference"])
            final_conference_week_by_conference[conference] = max(
                final_conference_week_by_conference.get(conference, 0),
                int(game["week"]),
            )

    completed_as_of = _is_completed_as_of(games, run_date)
    random_draws = rng.random((simulations, len(games)))
    sim_numbers = np.arange(simulations)

    for game_position, (_, game) in enumerate(games.iterrows()):
        home_idx = team_to_idx[game["hometeam"]]
        away_idx = team_to_idx[game["awayteam"]]
        home_team = team_names[home_idx]
        away_team = team_names[away_idx]
        homepoints = pd.to_numeric(game["homepoints"], errors="coerce")
        awaypoints = pd.to_numeric(game["awaypoints"], errors="coerce")
        homewinprob_raw = pd.to_numeric(game["homewinprob"], errors="coerce")
        homewinprob = float(homewinprob_raw) if pd.notna(homewinprob_raw) else 0.5
        homewinprob = float(np.clip(homewinprob, 0.0, 1.0))

        if bool(completed_as_of.loc[game.name]):
            home_win = np.full(simulations, homepoints > awaypoints, dtype=bool)
            expected_home_win = 1.0 if homepoints > awaypoints else 0.0
            remaining = False
        else:
            home_win = random_draws[:, game_position] < homewinprob
            expected_home_win = homewinprob
            remaining = True

        wins[:, home_idx] += home_win
        losses[:, home_idx] += ~home_win
        wins[:, away_idx] += ~home_win
        losses[:, away_idx] += home_win

        home_classification = str(game.get("homeclassification") or "").lower()
        away_classification = str(game.get("awayclassification") or "").lower()
        if home_classification == "fbs" and away_classification == "fbs":
            fbs_wins[:, home_idx] += home_win
            fbs_losses[:, home_idx] += ~home_win
            fbs_wins[:, away_idx] += ~home_win
            fbs_losses[:, away_idx] += home_win

        home_defeated_fcs = home_win & (away_classification == "fcs")
        if np.any(home_defeated_fcs):
            first_counted = home_defeated_fcs & (fcs_wins_counted[:, home_idx] == 0)
            adjusted_total_wins[first_counted, home_idx] += 1
            fcs_wins_counted[home_defeated_fcs, home_idx] += 1
        adjusted_total_wins[home_win & (away_classification != "fcs"), home_idx] += 1

        away_defeated_fcs = (~home_win) & (home_classification == "fcs")
        if np.any(away_defeated_fcs):
            first_counted = away_defeated_fcs & (fcs_wins_counted[:, away_idx] == 0)
            adjusted_total_wins[first_counted, away_idx] += 1
            fcs_wins_counted[away_defeated_fcs, away_idx] += 1
        adjusted_total_wins[(~home_win) & (home_classification != "fcs"), away_idx] += 1

        expected_wins_base[home_idx] += expected_home_win
        expected_wins_base[away_idx] += 1.0 - expected_home_win
        expected_games_base[home_idx] += 1.0
        expected_games_base[away_idx] += 1.0

        opponent_lists[home_idx].append(away_idx)
        opponent_lists[away_idx].append(home_idx)
        if remaining:
            remaining_opponent_lists[home_idx].append(away_idx)
            remaining_opponent_lists[away_idx].append(home_idx)

        is_conference_game = _is_truthy(game.get("conferencegame"))
        same_conference = (
            pd.notna(game.get("homeconference"))
            and pd.notna(game.get("awayconference"))
            and game.get("homeconference") == game.get("awayconference")
        )
        if is_conference_game and same_conference:
            conference_wins[:, home_idx] += home_win
            conference_losses[:, home_idx] += ~home_win
            conference_wins[:, away_idx] += ~home_win
            conference_losses[:, away_idx] += home_win

            conference = str(game["homeconference"])
            if (
                pd.notna(game.get("week"))
                and final_conference_week_by_conference.get(conference) == int(game["week"])
            ):
                final_conference_week_losses[home_win, away_idx] = True
                final_conference_week_losses[~home_win, home_idx] = True

            home_division = sun_belt_division_for_team(home_team)
            away_division = sun_belt_division_for_team(away_team)
            if home_division is not None and home_division == away_division:
                divisional_wins[:, home_idx] += home_win
                divisional_losses[:, home_idx] += ~home_win
                divisional_wins[:, away_idx] += ~home_win
                divisional_losses[:, away_idx] += home_win

            home_scores, away_scores = _simulated_scores(game, home_win)
            for sim, did_home_win in enumerate(home_win.tolist()):
                if did_home_win:
                    head_to_head_wins_by_sim[sim].setdefault(home_team, {})
                    head_to_head_wins_by_sim[sim][home_team][away_team] = (
                        head_to_head_wins_by_sim[sim][home_team].get(away_team, 0) + 1
                    )
                else:
                    head_to_head_wins_by_sim[sim].setdefault(away_team, {})
                    head_to_head_wins_by_sim[sim][away_team][home_team] = (
                        head_to_head_wins_by_sim[sim][away_team].get(home_team, 0) + 1
                    )
                conference_game_scores_by_sim[sim].append(
                    (home_idx, away_idx, float(home_scores[sim]), float(away_scores[sim]))
                )

    with np.errstate(divide="ignore", invalid="ignore"):
        team_strength = np.divide(
            expected_wins_base,
            expected_games_base,
            out=np.full(n_teams, 0.5, dtype=float),
            where=expected_games_base > 0,
        )

    strength_of_schedule = np.array(
        [_safe_mean([team_strength[opp] for opp in opponents]) for opponents in opponent_lists],
        dtype=object,
    )
    remaining_strength_of_schedule = np.array(
        [_safe_mean([team_strength[opp] for opp in opponents]) for opponents in remaining_opponent_lists],
        dtype=object,
    )

    total_wins = wins.copy()
    total_losses = losses.copy()
    championship_game_appearance = np.zeros((simulations, n_teams), dtype=bool)
    conference_champion = np.zeros((simulations, n_teams), dtype=bool)
    add_synthetic_conference_championship = not _has_explicit_fbs_championship_games(games)
    sec_relative_scoring_margin = _build_sec_relative_scoring_margins(
        conference_game_scores_by_sim,
        n_teams,
    )
    sos_for_rank = np.array(
        [float(value) if value is not None else 0.5 for value in strength_of_schedule],
        dtype=float,
    )
    regular_season_resume_score = (
        wins.mean(axis=0) * 100.0
        + conference_wins.mean(axis=0) * 8.0
        + team_strength * 5.0
        + sos_for_rank * 2.0
    )
    projected_cfp_rank_for_tiebreakers = _rank_desc(regular_season_resume_score)

    conference_series = pd.Series(conferences, dtype=object)
    for conference in sorted(value for value in conference_series.dropna().unique() if value not in INDEPENDENT_CONFERENCES):
        conf_indices = np.array(
            [
                idx
                for idx, value in enumerate(conferences)
                if value == conference and fbs_mask[idx]
            ],
            dtype=int,
        )
        if len(conf_indices) < 2:
            continue

        for sim in range(simulations):
            rule_rng = random.Random(int(rng.integers(0, 2**32 - 1)))
            team_records = []
            for idx in conf_indices:
                team_records.append(
                    {
                        "team": team_names[idx],
                        "conference": conference,
                        "division": sun_belt_division_for_team(team_names[idx]),
                        "conference_wins": int(conference_wins[sim, idx]),
                        "conference_losses": int(conference_losses[sim, idx]),
                        "divisional_wins": int(divisional_wins[sim, idx]),
                        "divisional_losses": int(divisional_losses[sim, idx]),
                        "overall_wins": int(wins[sim, idx]),
                        "overall_losses": int(losses[sim, idx]),
                        "fbs_wins": int(fbs_wins[sim, idx]),
                        "fbs_losses": int(fbs_losses[sim, idx]),
                        "adjusted_total_wins": int(adjusted_total_wins[sim, idx]),
                        "team_strength": float(team_strength[idx]),
                        "team_rating_score": float(team_strength[idx]),
                        "computer_composite_score": float(team_strength[idx]),
                        "computer_composite_rank": int(projected_cfp_rank_for_tiebreakers[idx]),
                        "cfp_rank": int(projected_cfp_rank_for_tiebreakers[idx]),
                        "lost_final_conference_week": bool(final_conference_week_losses[sim, idx]),
                        "head_to_head_wins": head_to_head_wins_by_sim[sim],
                        "sec_relative_scoring_margin": float(sec_relative_scoring_margin[sim, idx]),
                    }
                )

            first_team, second_team = select_conference_championship_teams(
                str(conference),
                team_records,
                rule_rng,
            )
            first_idx = team_to_idx[first_team]
            second_idx = team_to_idx[second_team]
            championship_game_appearance[sim, first_idx] = True
            championship_game_appearance[sim, second_idx] = True

            team_metrics = {
                team_names[idx]: {
                    "team_strength": float(team_strength[idx]),
                    "overall_win_pct": float(wins[sim, idx] / max(wins[sim, idx] + losses[sim, idx], 1)),
                    "conference_win_pct": float(
                        conference_wins[sim, idx]
                        / max(conference_wins[sim, idx] + conference_losses[sim, idx], 1)
                    ),
                    "strength_of_schedule": strength_of_schedule[idx],
                }
                for idx in conf_indices
            }
            champion_team = simulate_conference_championship_game(
                first_team,
                second_team,
                team_metrics,
                rule_rng,
            )
            champion_idx = team_to_idx[champion_team]
            runner_up_idx = second_idx if champion_idx == first_idx else first_idx
            conference_champion[sim, champion_idx] = True

            if add_synthetic_conference_championship:
                total_wins[sim, champion_idx] += 1
                total_losses[sim, runner_up_idx] += 1

    playoff = np.zeros((simulations, n_teams), dtype=bool)
    cfp_bye = np.zeros((simulations, n_teams), dtype=bool)
    cfp_at_large = np.zeros((simulations, n_teams), dtype=bool)
    cfp_auto_bid = np.zeros((simulations, n_teams), dtype=bool)
    national_championship_game = np.zeros((simulations, n_teams), dtype=bool)
    national_champion = np.zeros((simulations, n_teams), dtype=bool)

    fbs_indices = output_indices
    final_expected_wins = total_wins.mean(axis=0)
    resume_score = (
        final_expected_wins * 100.0
        + conference_wins.mean(axis=0) * 8.0
        + team_strength * 5.0
        + sos_for_rank * 2.0
    )
    fallback_cfp_ranks = _rank_desc(resume_score)
    base_ranking_scores = np.zeros(n_teams, dtype=float)
    expected_total_wins = np.zeros(n_teams, dtype=float)
    expected_conference_wins = np.zeros(n_teams, dtype=float)
    for idx, team in enumerate(team_names):
        base_ranking_scores[idx] = _ranking_projection_score(
            ranking_projection,
            team,
            fallback_rank=int(fallback_cfp_ranks[idx]),
            fallback_score=float(resume_score[idx] / 10.0),
        )
        expected_total_wins[idx] = float(
            _ranking_projection_value(ranking_projection, team, "projected_wins", final_expected_wins[idx])
            or final_expected_wins[idx]
        )
        expected_conference_wins[idx] = float(
            _ranking_projection_value(
                ranking_projection,
                team,
                "projected_conference_wins",
                conference_wins[:, idx].mean(),
            )
            or conference_wins[:, idx].mean()
        )

    for sim in range(simulations):
        ranking_score = (
            base_ranking_scores
            + (total_wins[sim].astype(float) - expected_total_wins) * 8.5
            + (conference_wins[sim].astype(float) - expected_conference_wins) * 1.5
            + conference_champion[sim].astype(float) * 7.5
            + team_strength * 2.0
            + sos_for_rank * 1.5
            + rng.random(n_teams) * 0.001
        )
        ranked_fbs = fbs_indices[np.argsort(-ranking_score[fbs_indices])].tolist()
        seeded, auto_indices, at_large_indices = _select_cfp_field(
            ranked_fbs=ranked_fbs,
            conference_champion_row=conference_champion[sim],
            conferences=conferences,
            team_names=team_names,
            ranking_scores=ranking_score,
        )
        if auto_indices:
            cfp_auto_bid[sim, auto_indices] = True
        if at_large_indices:
            cfp_at_large[sim, at_large_indices] = True
        if seeded:
            playoff[sim, seeded] = True
            cfp_bye[sim, seeded[:4]] = True

        title_game_indices, champion_idx = _simulate_cfp_bracket(
            seeded,
            ranking_scores=ranking_score,
            team_strength=team_strength,
            total_wins=total_wins,
            sim=sim,
            rng=rng,
        )
        if title_game_indices:
            national_championship_game[sim, title_game_indices] = True
        if champion_idx is not None:
            national_champion[sim, champion_idx] = True

    ap_ranks = _rank_desc(base_ranking_scores + playoff.mean(axis=0) * 5.0)
    cfp_ranks = _rank_desc(base_ranking_scores + playoff.mean(axis=0) * 20.0 + cfp_bye.mean(axis=0) * 10.0)
    resume_ranks = _rank_desc(resume_score)

    records: list[dict[str, Any]] = []
    for idx in output_indices:
        win_counts = np.bincount(
            np.clip(total_wins[:, idx], 0, MAX_WIN_BUCKET),
            minlength=MAX_WIN_BUCKET + 1,
        )
        win_probabilities = win_counts / simulations
        record = {
            "season": int(season),
            "run_date": run_date,
            "run_type": run_type,
            "model_version": _model_version_label(ranking_model_version),
            "team": team_names[idx],
            "conference": conferences[idx],
            "division": None,
            "classification": classification[idx],
            "projected_wins": float(total_wins[:, idx].mean()),
            "projected_losses": float(total_losses[:, idx].mean()),
            "projected_conference_wins": float(conference_wins[:, idx].mean()),
            "projected_conference_losses": float(conference_losses[:, idx].mean()),
            "conference_championship_game_prob": float(championship_game_appearance[:, idx].mean()),
            "conference_champion_prob": float(conference_champion[:, idx].mean()),
            "playoff_prob": float(playoff[:, idx].mean()),
            "cfp_bye_prob": float(cfp_bye[:, idx].mean()),
            "cfp_at_large_prob": float(cfp_at_large[:, idx].mean()),
            "cfp_auto_bid_prob": float(cfp_auto_bid[:, idx].mean()),
            "national_championship_game_prob": float(national_championship_game[:, idx].mean()),
            "national_champion_prob": float(national_champion[:, idx].mean()),
            "bowl_eligible_prob": float((wins[:, idx] >= 6).mean()),
            "projected_ap_ranking": int(ap_ranks[idx]),
            "projected_cfp_ranking": int(cfp_ranks[idx]),
            "resume_ranking": int(resume_ranks[idx]),
            "strength_of_schedule": strength_of_schedule[idx],
            "remaining_strength_of_schedule": remaining_strength_of_schedule[idx],
            "expected_number_of_wins": float(total_wins[:, idx].mean()),
            "simulations": int(simulations),
            "prediction_type": FBS_PREDICTION_TYPE if classification[idx] == "fbs" else FCS_PREDICTION_TYPE,
            "notes": notes,
        }
        for wins_bucket, probability in enumerate(win_probabilities):
            record[f"probability_{wins_bucket}_wins"] = float(probability)
        records.append(record)

    records_df = pd.DataFrame(records)
    records_df = records_df.astype(object).where(pd.notna(records_df), None)
    return records_df.to_dict(orient="records")


def select_ranking_projection_run(
    conn,
    *,
    season: int,
    run_type: str,
    run_date: date,
    explicit_run_id: str | None,
) -> tuple[str, str | None, str]:
    if explicit_run_id:
        with conn.cursor() as cur:
            cur.execute(EXPLICIT_RANKING_PROJECTION_RUN_SQL, (explicit_run_id, season))
            row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                f"No successful ranking_projection_runs row found for {explicit_run_id} and season={season}."
            )
        return str(row[0]), row[1], str(row[2])

    if run_type == "backfill":
        with conn.cursor() as cur:
            cur.execute(LATEST_RANKING_PROJECTION_RUN_BY_DATE_SQL, (season, run_type, run_date))
            row = cur.fetchone()
    else:
        run_types = _comparable_run_types(run_type)
        placeholders = ", ".join(["%s"] * len(run_types))
        sql = LATEST_RANKING_PROJECTION_RUN_SQL.format(run_type_placeholders=placeholders)
        with conn.cursor() as cur:
            cur.execute(sql, (season, *run_types))
            row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"No successful ranking projection snapshot found for season={season}, "
            f"run_type={run_type}, run_date={run_date}."
        )
    return str(row[0]), row[1], str(row[2])


def backfill_run_dates(conn, season: int) -> list[date]:
    with conn.cursor() as cur:
        cur.execute(BACKFILL_RANKING_RUN_DATES_SQL, (season,))
        rows = cur.fetchall()
    return [row[0] for row in rows]


def fetch_ranking_projection_rows(conn, ranking_projection_run_id: str) -> pd.DataFrame:
    return pd.read_sql(RANKING_PROJECTION_ROWS_SQL, conn, params=(ranking_projection_run_id,))


def fetch_game_snapshot(conn, *, season: int, game_prediction_run_id: str, run_date: date) -> pd.DataFrame:
    games = pd.read_sql(GAME_SNAPSHOT_SQL, conn, params=(game_prediction_run_id, season))
    if games.empty:
        raise RuntimeError(f"No regular-season game rows found for season {season}.")
    gamedate = pd.to_datetime(games["gamedate"], errors="coerce").dt.date
    completed_before_run = (
        games["homepoints"].notna()
        & games["awaypoints"].notna()
        & gamedate.lt(run_date)
    )
    missing_predictions = games[~completed_before_run & games["homewinprob"].isna()]
    if not missing_predictions.empty:
        preview = "; ".join(
            f"{row.id}: {row.awayteam} at {row.hometeam} week {row.week}"
            for row in missing_predictions.head(8).itertuples()
        )
        extra = "" if len(missing_predictions) <= 8 else f"; +{len(missing_predictions) - 8} more"
        raise RuntimeError(
            "Ranking projection's game_prediction_run_id does not cover every unplayed regular-season game. "
            f"Missing: {preview}{extra}"
        )
    return games


def process_prediction_snapshot(
    conn,
    *,
    current_season: int,
    run_date: date,
    run_type: str,
    etl_run_id: str,
    simulations: int,
    notes: str | None,
    explicit_ranking_projection_run_id: str | None,
) -> tuple[uuid.UUID, int, str]:
    run_id = uuid.uuid4()
    ranking_projection_run_id, ranking_model_version, game_prediction_run_id = select_ranking_projection_run(
        conn,
        season=current_season,
        run_type=run_type,
        run_date=run_date,
        explicit_run_id=explicit_ranking_projection_run_id,
    )
    model_version = _model_version_label(ranking_model_version)
    print(f"Season prediction run id: {run_id}")
    print(f"Run date/type: {run_date} / {run_type}")
    print(f"Simulations: {simulations}")
    print(f"Using ranking_projection_run_id: {ranking_projection_run_id}")
    print(f"Using game_prediction_run_id: {game_prediction_run_id}")

    preds = fetch_game_snapshot(
        conn,
        season=current_season,
        game_prediction_run_id=game_prediction_run_id,
        run_date=run_date,
    )
    ranking_projection = _ranking_projection_lookup(fetch_ranking_projection_rows(conn, ranking_projection_run_id))
    records = build_season_prediction_records(
        preds,
        ranking_projection=ranking_projection,
        ranking_model_version=ranking_model_version,
        season=current_season,
        run_date=run_date,
        run_type=run_type,
        simulations=simulations,
        notes=notes,
        rng=_rng_from_env(current_season),
    )
    records, prediction_hash = add_prediction_hashes(records)
    print(f"Prepared {len(records)} team-level season predictions for season {current_season}.")
    print(f"Prediction hash: {prediction_hash}")

    create_run(
        conn,
        run_id=run_id,
        season=current_season,
        run_date=run_date,
        run_type=run_type,
        etl_run_id=etl_run_id,
        ranking_projection_run_id=ranking_projection_run_id,
        model_version=model_version,
        row_count=len(records),
        simulations=simulations,
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
                    "Season predictions match latest comparable successful snapshot; "
                    f"marking this run as duplicate of {latest_run_id}."
                )
                write_duplicate_run(
                    conn,
                    run_id=run_id,
                    prediction_hash=prediction_hash,
                    duplicate_of_run_id=latest_run_id,
                )
                print("Finished without inserting duplicate season prediction detail rows.")
                return run_id, 0, "duplicate"

        print("Season prediction set changed; inserting full snapshot rows...")
        for record in records:
            record["season_prediction_run_id"] = run_id
        write_changed_snapshot(
            conn,
            run_id=run_id,
            records=records,
            prediction_hash=prediction_hash,
        )
        print(f"Finished inserting {len(records)} rows into season_predictions_full.")
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
    simulations = _simulations_from_env()
    notes = os.getenv(NOTES_ENV, "").strip() or None
    explicit_ranking_projection_run_id = os.getenv(RANKING_PROJECTION_RUN_ID_ENV, "").strip() or None

    print("Connecting to database...")
    with psycopg.connect(cfg.pg_dsn) as conn:
        print(f"Target season: {current_season}")

        ensure_full_prediction_tables(conn)

        if run_type == "backfill" and not run_date_was_supplied and not explicit_ranking_projection_run_id:
            dates = backfill_run_dates(conn, current_season)
            print(f"Backfilling {len(dates)} season prediction run dates for season {current_season}.")
            total_inserted = 0
            for backfill_date in dates:
                _, inserted_count, _ = process_prediction_snapshot(
                    conn,
                    current_season=current_season,
                    run_date=backfill_date,
                    run_type=run_type,
                    etl_run_id=cfg.run_id,
                    simulations=simulations,
                    notes=notes,
                    explicit_ranking_projection_run_id=None,
                )
                total_inserted += inserted_count
            print(f"Finished season prediction backfill; inserted {total_inserted} detail rows.")
            return

        process_prediction_snapshot(
            conn,
            current_season=current_season,
            run_date=run_date,
            run_type=run_type,
            etl_run_id=cfg.run_id,
            simulations=simulations,
            notes=notes,
            explicit_ranking_projection_run_id=explicit_ranking_projection_run_id,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"update_season_predictions_full.py failed: {e}")
        sys.exit(1)
