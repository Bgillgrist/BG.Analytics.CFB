#!/usr/bin/env python3
"""
Build season-level team prediction snapshots from the game prediction model.

The job follows the same run/detail snapshot pattern as
etl.jobs.update_game_predictions_full. It trains the game model as of a run
date, simulates the remaining regular season, derives season-long team
outcomes, and appends a new full snapshot only when the canonical prediction
payload changes.
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
from etl.jobs.update_game_predictions import (
    FBS_PREDICTION_TYPE,
    FCS_PREDICTION_TYPE,
    INCOMPLETE_MODEL_VERSION,
    LINE_AWARE_MODEL_VERSION,
    build_modeling_table,
    score_current_season,
)
from etl.jobs.update_game_predictions_full import train_models_as_of


RUN_TYPE_ENV = "SEASON_PREDICTION_RUN_TYPE"
RUN_DATE_ENV = "SEASON_PREDICTION_RUN_DATE"
NOTES_ENV = "SEASON_PREDICTION_RUN_NOTES"
SIMULATIONS_ENV = "SEASON_PREDICTION_SIMULATIONS"
RANDOM_SEED_ENV = "SEASON_PREDICTION_RANDOM_SEED"
DEFAULT_SIMULATIONS = 10_000
MAX_WIN_BUCKET = 13
HASH_FLOAT_DECIMAL_PLACES = 4
NORMAL_SNAPSHOT_RUN_TYPES = ("manual", "nightly")
INDEPENDENT_CONFERENCES = {"FBS Independents", "FCS Independents", "Independent"}

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
CREATE INDEX IF NOT EXISTS idx_season_prediction_runs_lookup
  ON public.season_prediction_runs (season, run_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_season_prediction_runs_hash
  ON public.season_prediction_runs (season, run_type, prediction_hash)
  WHERE status = 'success';

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


def _run_date_from_env() -> date:
    raw = os.getenv(RUN_DATE_ENV, "").strip()
    if raw:
        return date.fromisoformat(raw)
    return datetime.now(timezone.utc).date()


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


def _model_version_label() -> str:
    return f"season_sim_2026+{LINE_AWARE_MODEL_VERSION}+{INCOMPLETE_MODEL_VERSION}"


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
                "model_version": _model_version_label(),
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


def build_season_prediction_records(
    preds: pd.DataFrame,
    *,
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
    conference_wins = np.zeros((simulations, n_teams), dtype=np.int16)
    conference_losses = np.zeros((simulations, n_teams), dtype=np.int16)
    expected_wins_base = np.zeros(n_teams, dtype=float)
    expected_games_base = np.zeros(n_teams, dtype=float)
    opponent_lists: list[list[int]] = [[] for _ in range(n_teams)]
    remaining_opponent_lists: list[list[int]] = [[] for _ in range(n_teams)]

    completed_as_of = _is_completed_as_of(games, run_date)
    random_draws = rng.random((simulations, len(games)))

    for game_position, (_, game) in enumerate(games.iterrows()):
        home_idx = team_to_idx[game["hometeam"]]
        away_idx = team_to_idx[game["awayteam"]]
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

        finalist_score = (
            conference_wins[:, conf_indices].astype(float) * 100.0
            + wins[:, conf_indices].astype(float) * 10.0
            + team_strength[conf_indices]
            + rng.random((simulations, len(conf_indices))) * 0.001
        )
        finalist_order = np.argsort(-finalist_score, axis=1)
        top_one = conf_indices[finalist_order[:, 0]]
        top_two = conf_indices[finalist_order[:, 1]]
        row_numbers = np.arange(simulations)
        championship_game_appearance[row_numbers, top_one] = True
        championship_game_appearance[row_numbers, top_two] = True

        strength_delta = team_strength[top_one] - team_strength[top_two]
        top_one_title_prob = 1.0 / (1.0 + np.exp(-5.0 * strength_delta))
        top_one_wins_title = rng.random(simulations) < top_one_title_prob
        champions = np.where(top_one_wins_title, top_one, top_two)
        runners_up = np.where(top_one_wins_title, top_two, top_one)
        conference_champion[row_numbers, champions] = True

        if add_synthetic_conference_championship:
            total_wins[row_numbers, champions] += 1
            total_losses[row_numbers, runners_up] += 1

    playoff = np.zeros((simulations, n_teams), dtype=bool)
    cfp_bye = np.zeros((simulations, n_teams), dtype=bool)
    cfp_at_large = np.zeros((simulations, n_teams), dtype=bool)
    cfp_auto_bid = np.zeros((simulations, n_teams), dtype=bool)
    national_championship_game = np.zeros((simulations, n_teams), dtype=bool)
    national_champion = np.zeros((simulations, n_teams), dtype=bool)

    fbs_indices = output_indices
    sos_for_rank = np.array(
        [float(value) if value is not None else 0.5 for value in strength_of_schedule],
        dtype=float,
    )
    for sim in range(simulations):
        ranking_score = (
            total_wins[sim, fbs_indices].astype(float) * 100.0
            + conference_wins[sim, fbs_indices].astype(float) * 8.0
            + team_strength[fbs_indices] * 5.0
            + sos_for_rank[fbs_indices] * 2.0
            + rng.random(len(fbs_indices)) * 0.001
        )
        ranked_fbs = fbs_indices[np.argsort(-ranking_score)]
        champ_indices = ranked_fbs[conference_champion[sim, ranked_fbs]]
        auto_indices = champ_indices[:5]
        cfp_auto_bid[sim, auto_indices] = True

        remaining_ranked = [idx for idx in ranked_fbs if idx not in set(auto_indices)]
        at_large_indices = np.array(remaining_ranked[: max(0, 12 - len(auto_indices))], dtype=int)
        cfp_at_large[sim, at_large_indices] = True
        playoff_indices = np.concatenate([auto_indices, at_large_indices])
        playoff[sim, playoff_indices] = True

        bye_indices = auto_indices[:4]
        cfp_bye[sim, bye_indices] = True

        playoff_ranked = [idx for idx in ranked_fbs if idx in set(playoff_indices)]
        title_game_indices = np.array(playoff_ranked[:2], dtype=int)
        national_championship_game[sim, title_game_indices] = True
        if playoff_ranked:
            national_champion[sim, playoff_ranked[0]] = True

    final_expected_wins = total_wins.mean(axis=0)
    resume_score = (
        final_expected_wins * 100.0
        + conference_wins.mean(axis=0) * 8.0
        + team_strength * 5.0
        + sos_for_rank * 2.0
    )
    ap_ranks = _rank_desc(resume_score + playoff.mean(axis=0) * 5.0)
    cfp_ranks = _rank_desc(resume_score + playoff.mean(axis=0) * 20.0 + cfp_bye.mean(axis=0) * 10.0)
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
            "model_version": _model_version_label(),
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


def process_prediction_snapshot(
    conn,
    *,
    df: pd.DataFrame,
    current_season: int,
    run_date: date,
    run_type: str,
    etl_run_id: str,
    simulations: int,
    notes: str | None,
) -> tuple[uuid.UUID, int, str]:
    run_id = uuid.uuid4()
    print(f"Season prediction run id: {run_id}")
    print(f"Run date/type: {run_date} / {run_type}")
    print(f"Simulations: {simulations}")

    model_bundle, modeled_df = train_models_as_of(df, current_season, run_date)
    preds = score_current_season(model_bundle, modeled_df, current_season)
    records = build_season_prediction_records(
        preds,
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
        row_count=len(records),
        simulations=simulations,
        notes=notes,
    )

    try:
        latest = get_latest_successful_run(conn, current_season, run_type)
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
    run_type = _run_type_from_env()
    simulations = _simulations_from_env()
    notes = os.getenv(NOTES_ENV, "").strip() or None

    print("Connecting to database...")
    with psycopg.connect(cfg.pg_dsn) as conn:
        print(f"Target season: {current_season}")

        print("Building modeling table...")
        df = build_modeling_table(conn, max_season=current_season)
        print(f"Modeling table rows: {len(df)}")

        ensure_full_prediction_tables(conn)
        process_prediction_snapshot(
            conn,
            df=df,
            current_season=current_season,
            run_date=run_date,
            run_type=run_type,
            etl_run_id=cfg.run_id,
            simulations=simulations,
            notes=notes,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"update_season_predictions_full.py failed: {e}")
        sys.exit(1)
