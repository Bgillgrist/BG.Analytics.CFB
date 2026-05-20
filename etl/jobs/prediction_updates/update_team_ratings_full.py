#!/usr/bin/env python3
"""
Build SRS-style team rating snapshots from completed margins and game predictions.

The job follows the same run/detail snapshot pattern as the other prediction
updates. Each successful team rating run is tied to one successful
game_prediction_run_id, so historical dashboard views can select an as-of run
instead of recomputing one current spot check.
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


RUN_TYPE_ENV = "TEAM_RATING_RUN_TYPE"
RUN_DATE_ENV = "TEAM_RATING_RUN_DATE"
NOTES_ENV = "TEAM_RATING_RUN_NOTES"
GAME_PREDICTION_RUN_ID_ENV = "TEAM_RATING_GAME_PREDICTION_RUN_ID"

HASH_FLOAT_DECIMAL_PLACES = 4
NORMAL_SNAPSHOT_RUN_TYPES = ("manual", "nightly")
WIN_PROBABILITY_SPREAD_SCALE = 14.0
COMPLETED_GAME_WEIGHT = 1.0
PROJECTED_GAME_WEIGHT = 0.45
MAX_MARGIN_SIGNAL = 42.0

DETAIL_HASH_EXCLUDED_COLUMNS = {
    "team_rating_run_id",
    "run_date",
    "run_type",
    "created_at",
    "notes",
    "rating_row_hash",
    "game_prediction_run_id",
}

DETAIL_HASH_COLUMNS = (
    "season",
    "model_version",
    "team",
    "conference",
    "classification",
    "rank",
    "team_rating",
    "power_rating",
    "home_field_advantage",
    "completed_games",
    "projected_games",
    "total_games",
    "average_margin_signal",
    "average_weighted_margin_signal",
    "completed_game_weight",
    "projected_game_weight",
    "max_margin_signal",
    "margin_source",
)

CREATE_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.team_rating_runs (
    team_rating_run_id     UUID PRIMARY KEY,
    season                 INT NOT NULL,
    run_date               DATE NOT NULL,
    run_type               TEXT NOT NULL DEFAULT 'nightly',
    etl_run_id             TEXT,
    game_prediction_run_id UUID REFERENCES public.game_prediction_runs(game_prediction_run_id),
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at           TIMESTAMPTZ,
    status                 TEXT NOT NULL DEFAULT 'running',
    model_version          TEXT NOT NULL,
    rating_hash            TEXT,
    duplicate_of_run_id    UUID REFERENCES public.team_rating_runs(team_rating_run_id),
    row_count              INT NOT NULL DEFAULT 0,
    inserted_row_count     INT NOT NULL DEFAULT 0,
    completed_game_count   INT NOT NULL DEFAULT 0,
    projected_game_count   INT NOT NULL DEFAULT 0,
    dropped_game_count     INT NOT NULL DEFAULT 0,
    home_field_advantage   DOUBLE PRECISION,
    margin_source          TEXT,
    notes                  TEXT,
    error_message          TEXT,
    CONSTRAINT team_rating_runs_status_check
      CHECK (status IN ('running', 'success', 'duplicate', 'failed'))
);
"""

CREATE_DETAILS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.team_ratings (
    team_rating_run_id             UUID NOT NULL
      REFERENCES public.team_rating_runs(team_rating_run_id)
      ON DELETE CASCADE,
    season                         INT NOT NULL,
    run_date                       DATE NOT NULL,
    run_type                       TEXT NOT NULL DEFAULT 'nightly',
    model_version                  TEXT NOT NULL,
    team                           TEXT NOT NULL,
    conference                     TEXT,
    classification                 TEXT,
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    rank                           INT NOT NULL,
    team_rating                    DOUBLE PRECISION NOT NULL,
    power_rating                   DOUBLE PRECISION NOT NULL,
    home_field_advantage           DOUBLE PRECISION NOT NULL,
    completed_games                INT NOT NULL DEFAULT 0,
    projected_games                INT NOT NULL DEFAULT 0,
    total_games                    INT NOT NULL DEFAULT 0,
    average_margin_signal          DOUBLE PRECISION,
    average_weighted_margin_signal DOUBLE PRECISION,
    completed_game_weight          DOUBLE PRECISION NOT NULL,
    projected_game_weight          DOUBLE PRECISION NOT NULL,
    max_margin_signal              DOUBLE PRECISION NOT NULL,
    margin_source                  TEXT NOT NULL,
    game_prediction_run_id         UUID
      REFERENCES public.game_prediction_runs(game_prediction_run_id),
    rating_row_hash                TEXT NOT NULL,
    notes                          TEXT,
    PRIMARY KEY (team_rating_run_id, team)
);
"""

CREATE_INDEXES_SQL = """
ALTER TABLE public.team_rating_runs
ADD COLUMN IF NOT EXISTS game_prediction_run_id UUID;

CREATE INDEX IF NOT EXISTS idx_team_rating_runs_lookup
  ON public.team_rating_runs (season, run_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_team_rating_runs_hash
  ON public.team_rating_runs (season, run_type, rating_hash)
  WHERE status = 'success';

CREATE INDEX IF NOT EXISTS idx_team_rating_runs_game_prediction
  ON public.team_rating_runs (game_prediction_run_id);

CREATE INDEX IF NOT EXISTS idx_team_ratings_team_lookup
  ON public.team_ratings (season, team, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_team_ratings_rank_lookup
  ON public.team_ratings (season, rank);
"""

INSERT_RUN_SQL = """
INSERT INTO public.team_rating_runs (
    team_rating_run_id,
    season,
    run_date,
    run_type,
    etl_run_id,
    game_prediction_run_id,
    status,
    model_version,
    row_count,
    inserted_row_count,
    completed_game_count,
    projected_game_count,
    dropped_game_count,
    home_field_advantage,
    margin_source,
    notes
)
VALUES (
    %(team_rating_run_id)s,
    %(season)s,
    %(run_date)s,
    %(run_type)s,
    %(etl_run_id)s,
    %(game_prediction_run_id)s,
    'running',
    %(model_version)s,
    %(row_count)s,
    0,
    %(completed_game_count)s,
    %(projected_game_count)s,
    %(dropped_game_count)s,
    %(home_field_advantage)s,
    %(margin_source)s,
    %(notes)s
);
"""

MARK_RUN_SUCCESS_SQL = """
UPDATE public.team_rating_runs
SET
    completed_at = NOW(),
    status = 'success',
    rating_hash = %(rating_hash)s,
    inserted_row_count = %(inserted_row_count)s
WHERE team_rating_run_id = %(team_rating_run_id)s;
"""

MARK_RUN_DUPLICATE_SQL = """
UPDATE public.team_rating_runs
SET
    completed_at = NOW(),
    status = 'duplicate',
    rating_hash = %(rating_hash)s,
    duplicate_of_run_id = %(duplicate_of_run_id)s,
    inserted_row_count = 0
WHERE team_rating_run_id = %(team_rating_run_id)s;
"""

MARK_RUN_FAILED_SQL = """
UPDATE public.team_rating_runs
SET
    completed_at = NOW(),
    status = 'failed',
    error_message = %(error_message)s
WHERE team_rating_run_id = %(team_rating_run_id)s;
"""

LATEST_SUCCESSFUL_RUN_SQL = """
SELECT team_rating_run_id, rating_hash
FROM public.team_rating_runs
WHERE season = %s
  AND run_type IN ({run_type_placeholders})
  AND status = 'success'
  AND rating_hash IS NOT NULL
ORDER BY created_at DESC
LIMIT 1;
"""

LATEST_SUCCESSFUL_RUN_BY_DATE_SQL = """
SELECT team_rating_run_id, rating_hash
FROM public.team_rating_runs
WHERE season = %s
  AND run_type = %s
  AND run_date = %s
  AND status = 'success'
  AND rating_hash IS NOT NULL
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

GAME_SNAPSHOT_SQL = """
SELECT
    gd.id::text AS gameid,
    gd.season,
    gd.week,
    CAST(gd.startdate AS date) AS gamedate,
    gd.seasontype,
    gd.hometeam,
    gd.awayteam,
    LOWER(gd.homeclassification) AS homeclassification,
    LOWER(gd.awayclassification) AS awayclassification,
    gd.homeconference,
    gd.awayconference,
    gd.homepoints,
    gd.awaypoints,
    gp.homespread,
    gp.homewinprob,
    gp.model_version AS game_prediction_model_version,
    gp.prediction_type
FROM public.game_data gd
LEFT JOIN public.game_predictions_full gp
  ON gp.gameid = gd.id::text
 AND gp.game_prediction_run_id = %s
WHERE gd.season = %s
  AND gd.hometeam IS NOT NULL
  AND gd.awayteam IS NOT NULL
  AND LOWER(gd.homeclassification) = 'fbs'
  AND LOWER(gd.awayclassification) = 'fbs'
ORDER BY gd.week, gd.startdate, gd.id;
"""

INSERT_DETAIL_SQL = """
INSERT INTO public.team_ratings (
    team_rating_run_id,
    season,
    run_date,
    run_type,
    model_version,
    team,
    conference,
    classification,
    rank,
    team_rating,
    power_rating,
    home_field_advantage,
    completed_games,
    projected_games,
    total_games,
    average_margin_signal,
    average_weighted_margin_signal,
    completed_game_weight,
    projected_game_weight,
    max_margin_signal,
    margin_source,
    game_prediction_run_id,
    rating_row_hash,
    notes
)
VALUES (
    %(team_rating_run_id)s,
    %(season)s,
    %(run_date)s,
    %(run_type)s,
    %(model_version)s,
    %(team)s,
    %(conference)s,
    %(classification)s,
    %(rank)s,
    %(team_rating)s,
    %(power_rating)s,
    %(home_field_advantage)s,
    %(completed_games)s,
    %(projected_games)s,
    %(total_games)s,
    %(average_margin_signal)s,
    %(average_weighted_margin_signal)s,
    %(completed_game_weight)s,
    %(projected_game_weight)s,
    %(max_margin_signal)s,
    %(margin_source)s,
    %(game_prediction_run_id)s,
    %(rating_row_hash)s,
    %(notes)s
);
"""

DETAIL_RECORDS_FOR_RUN_SQL = f"""
SELECT {", ".join(DETAIL_HASH_COLUMNS)}
FROM public.team_ratings
WHERE team_rating_run_id = %s
ORDER BY team;
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


def _model_version_label(game_model_version: str | None) -> str:
    suffix = game_model_version or "unknown_game_model"
    return f"team_rating_srs_2026+{suffix}"


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


def add_rating_hashes(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    enriched: list[dict[str, Any]] = []
    canonical_records = []
    for record in sorted(records, key=lambda item: str(item["team"])):
        canonical = _canonical_record(record)
        row_hash = _hash_payload(canonical)
        enriched_record = dict(record)
        enriched_record["rating_row_hash"] = row_hash
        enriched.append(enriched_record)
        canonical_records.append(canonical)
    return enriched, _hash_payload(canonical_records)


def _rating_hash_from_records(records: list[dict[str, Any]]) -> str:
    canonical_records = [
        _canonical_record(record)
        for record in sorted(records, key=lambda item: str(item["team"]))
    ]
    return _hash_payload(canonical_records)


def ensure_team_rating_tables(conn) -> None:
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


def get_rating_records_for_run(conn, run_id: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(DETAIL_RECORDS_FOR_RUN_SQL, (run_id,))
        rows = cur.fetchall()
    return [dict(zip(DETAIL_HASH_COLUMNS, row)) for row in rows]


def rating_hash_matches_run(
    conn,
    *,
    run_id: str,
    stored_rating_hash: str,
    rating_hash: str,
) -> bool:
    if stored_rating_hash == rating_hash:
        return True
    prior_records = get_rating_records_for_run(conn, run_id)
    return _rating_hash_from_records(prior_records) == rating_hash


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


def fetch_game_snapshot(conn, season: int, game_prediction_run_id: str) -> pd.DataFrame:
    return _read_sql(conn, GAME_SNAPSHOT_SQL, (game_prediction_run_id, season))


def implied_margin_from_probability(probability: pd.Series) -> pd.Series:
    probs = probability.astype(float).clip(0.01, 0.99)
    return WIN_PROBABILITY_SPREAD_SCALE * np.log(probs / (1 - probs))


def _first_non_null(values: pd.Series) -> Any:
    cleaned = values.dropna()
    if cleaned.empty:
        return None
    modes = cleaned.mode(dropna=True)
    if not modes.empty:
        return modes.iloc[0]
    return cleaned.iloc[-1]


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
            conference=("conference", _first_non_null),
            classification=("classification", _first_non_null),
        )
        .sort_values("team")
        .reset_index(drop=True)
    )


def _team_signal_summary(games: pd.DataFrame, teams: list[str]) -> dict[str, dict[str, float | int | None]]:
    summary: dict[str, dict[str, Any]] = {
        team: {
            "completed_games": 0,
            "projected_games": 0,
            "team_margins": [],
            "weighted_team_margins": [],
            "weights": [],
        }
        for team in teams
    }
    for game in games.itertuples(index=False):
        home = game.hometeam
        away = game.awayteam
        margin = float(game.margin_signal)
        weight = float(game.weight)
        completed = bool(game.completed_as_of)
        if home in summary:
            summary[home]["completed_games"] += int(completed)
            summary[home]["projected_games"] += int(not completed)
            summary[home]["team_margins"].append(margin)
            summary[home]["weighted_team_margins"].append(margin * weight)
            summary[home]["weights"].append(weight)
        if away in summary:
            summary[away]["completed_games"] += int(completed)
            summary[away]["projected_games"] += int(not completed)
            summary[away]["team_margins"].append(-margin)
            summary[away]["weighted_team_margins"].append(-margin * weight)
            summary[away]["weights"].append(weight)

    output: dict[str, dict[str, float | int | None]] = {}
    for team, row in summary.items():
        weights = row["weights"]
        output[team] = {
            "completed_games": int(row["completed_games"]),
            "projected_games": int(row["projected_games"]),
            "total_games": int(row["completed_games"] + row["projected_games"]),
            "average_margin_signal": float(np.mean(row["team_margins"])) if row["team_margins"] else None,
            "average_weighted_margin_signal": (
                float(sum(row["weighted_team_margins"]) / sum(weights)) if weights and sum(weights) else None
            ),
        }
    return output


def _margin_source_label(games: pd.DataFrame) -> str:
    sources = set(games["margin_source"].dropna().astype(str))
    if not sources:
        return "none"
    if sources == {"completed"}:
        return "completed"
    projection_sources = sources - {"completed"}
    if len(projection_sources) == 1:
        source = next(iter(projection_sources))
        return source if "completed" not in sources else f"completed+{source}"
    return "mixed"


def build_team_rating_records(
    *,
    season: int,
    run_date: date,
    run_type: str,
    model_version: str,
    game_prediction_run_id: str,
    games: pd.DataFrame,
    notes: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if games.empty:
        raise RuntimeError("No FBS-vs-FBS game_data rows available for team ratings.")

    games = games.copy()
    games["gamedate"] = pd.to_datetime(games["gamedate"], errors="coerce").dt.date
    games["homepoints"] = pd.to_numeric(games["homepoints"], errors="coerce")
    games["awaypoints"] = pd.to_numeric(games["awaypoints"], errors="coerce")
    games["homespread"] = pd.to_numeric(games["homespread"], errors="coerce")
    games["homewinprob"] = pd.to_numeric(games["homewinprob"], errors="coerce")
    games["completed_as_of"] = (
        games["homepoints"].notna()
        & games["awaypoints"].notna()
        & games["gamedate"].lt(run_date)
    )
    games["margin_signal"] = np.nan
    games["margin_source"] = None

    completed_mask = games["completed_as_of"]
    games.loc[completed_mask, "margin_signal"] = (
        games.loc[completed_mask, "homepoints"] - games.loc[completed_mask, "awaypoints"]
    )
    games.loc[completed_mask, "margin_source"] = "completed"

    projected_mask = ~completed_mask
    spread_mask = projected_mask & games["homespread"].notna()
    games.loc[spread_mask, "margin_signal"] = -games.loc[spread_mask, "homespread"]
    games.loc[spread_mask, "margin_source"] = "spread"

    winprob_mask = projected_mask & games["margin_signal"].isna() & games["homewinprob"].notna()
    games.loc[winprob_mask, "margin_signal"] = implied_margin_from_probability(
        games.loc[winprob_mask, "homewinprob"]
    )
    games.loc[winprob_mask, "margin_source"] = "winprob"

    dropped_game_count = int(games["margin_signal"].isna().sum())
    games = games.dropna(subset=["margin_signal", "hometeam", "awayteam"]).copy()
    if games.empty:
        raise RuntimeError("No completed or projected game margin signals available for team ratings.")

    games["margin_signal"] = games["margin_signal"].clip(-MAX_MARGIN_SIGNAL, MAX_MARGIN_SIGNAL)
    games["weight"] = np.where(games["completed_as_of"], COMPLETED_GAME_WEIGHT, PROJECTED_GAME_WEIGHT)

    teams = sorted(pd.concat([games["hometeam"], games["awayteam"]]).dropna().unique())
    if len(teams) < 2:
        raise RuntimeError("At least two teams are required to solve team ratings.")

    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    num_teams = len(teams)
    rows = []
    targets = []
    weights = []

    for game in games.itertuples(index=False):
        row = np.zeros(num_teams + 1)
        row[team_to_idx[game.hometeam]] = 1.0
        row[team_to_idx[game.awayteam]] = -1.0
        row[-1] = 1.0
        rows.append(row)
        targets.append(float(game.margin_signal))
        weights.append(float(game.weight))

    anchor = np.zeros(num_teams + 1)
    anchor[:num_teams] = 1.0
    rows.append(anchor)
    targets.append(0.0)
    weights.append(1000.0)

    matrix = np.vstack(rows)
    target = np.array(targets)
    sqrt_weights = np.sqrt(np.array(weights))
    solution, *_ = np.linalg.lstsq(matrix * sqrt_weights[:, None], target * sqrt_weights, rcond=None)

    ratings = pd.DataFrame({"team": teams, "team_rating": solution[:num_teams]})
    hfa = float(solution[-1])
    ratings = ratings.sort_values(["team_rating", "team"], ascending=[False, True]).reset_index(drop=True)
    ratings["rank"] = np.arange(1, len(ratings) + 1)

    profiles = team_profiles(games).set_index("team").to_dict(orient="index")
    signal_summary = _team_signal_summary(games, teams)
    margin_source = _margin_source_label(games)

    records: list[dict[str, Any]] = []
    for rating in ratings.itertuples(index=False):
        summary = signal_summary[rating.team]
        profile = profiles.get(rating.team, {})
        records.append(
            {
                "season": season,
                "run_date": run_date,
                "run_type": run_type,
                "model_version": model_version,
                "team": rating.team,
                "conference": profile.get("conference"),
                "classification": profile.get("classification"),
                "rank": int(rating.rank),
                "team_rating": float(rating.team_rating),
                "power_rating": float(rating.team_rating),
                "home_field_advantage": hfa,
                "completed_games": summary["completed_games"],
                "projected_games": summary["projected_games"],
                "total_games": summary["total_games"],
                "average_margin_signal": summary["average_margin_signal"],
                "average_weighted_margin_signal": summary["average_weighted_margin_signal"],
                "completed_game_weight": COMPLETED_GAME_WEIGHT,
                "projected_game_weight": PROJECTED_GAME_WEIGHT,
                "max_margin_signal": MAX_MARGIN_SIGNAL,
                "margin_source": margin_source,
                "game_prediction_run_id": game_prediction_run_id,
                "notes": notes,
            }
        )

    records_df = pd.DataFrame(records)
    records_df = records_df.astype(object).where(pd.notna(records_df), None)
    metadata = {
        "completed_game_count": int(games["completed_as_of"].sum()),
        "projected_game_count": int((~games["completed_as_of"]).sum()),
        "dropped_game_count": dropped_game_count,
        "home_field_advantage": hfa,
        "margin_source": margin_source,
    }
    return records_df.to_dict(orient="records"), metadata


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
    completed_game_count: int,
    projected_game_count: int,
    dropped_game_count: int,
    home_field_advantage: float,
    margin_source: str,
    notes: str | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            INSERT_RUN_SQL,
            {
                "team_rating_run_id": run_id,
                "season": season,
                "run_date": run_date,
                "run_type": run_type,
                "etl_run_id": etl_run_id,
                "game_prediction_run_id": game_prediction_run_id,
                "model_version": model_version,
                "row_count": row_count,
                "completed_game_count": completed_game_count,
                "projected_game_count": projected_game_count,
                "dropped_game_count": dropped_game_count,
                "home_field_advantage": home_field_advantage,
                "margin_source": margin_source,
                "notes": notes,
            },
        )
    conn.commit()


def mark_run_failed(conn, run_id: uuid.UUID, error_message: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            MARK_RUN_FAILED_SQL,
            {
                "team_rating_run_id": run_id,
                "error_message": error_message[:2000],
            },
        )
    conn.commit()


def write_changed_snapshot(
    conn,
    *,
    run_id: uuid.UUID,
    records: list[dict[str, Any]],
    rating_hash: str,
) -> None:
    with conn.cursor() as cur:
        cur.executemany(INSERT_DETAIL_SQL, records)
        cur.execute(
            MARK_RUN_SUCCESS_SQL,
            {
                "team_rating_run_id": run_id,
                "rating_hash": rating_hash,
                "inserted_row_count": len(records),
            },
        )
    conn.commit()


def write_duplicate_run(
    conn,
    *,
    run_id: uuid.UUID,
    rating_hash: str,
    duplicate_of_run_id: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            MARK_RUN_DUPLICATE_SQL,
            {
                "team_rating_run_id": run_id,
                "rating_hash": rating_hash,
                "duplicate_of_run_id": duplicate_of_run_id,
            },
        )
    conn.commit()


def process_team_rating_snapshot(
    conn,
    *,
    season: int,
    run_date: date,
    run_type: str,
    etl_run_id: str,
    notes: str | None,
    explicit_game_prediction_run_id: str | None,
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
    print(f"Team rating run id: {run_id}")
    print(f"Run date/type: {run_date} / {run_type}")
    print(f"Using game_prediction_run_id: {game_prediction_run_id}")

    games = fetch_game_snapshot(conn, season, game_prediction_run_id)
    records, metadata = build_team_rating_records(
        season=season,
        run_date=run_date,
        run_type=run_type,
        model_version=model_version,
        game_prediction_run_id=game_prediction_run_id,
        games=games,
        notes=notes,
    )
    records, rating_hash = add_rating_hashes(records)
    print(f"Prepared {len(records)} team ratings for season {season}.")
    print(
        "Signals: "
        f"{metadata['completed_game_count']} completed, "
        f"{metadata['projected_game_count']} projected, "
        f"{metadata['dropped_game_count']} dropped."
    )
    print(f"Solved HFA: {metadata['home_field_advantage']:.4f}")
    print(f"Rating hash: {rating_hash}")

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
        completed_game_count=metadata["completed_game_count"],
        projected_game_count=metadata["projected_game_count"],
        dropped_game_count=metadata["dropped_game_count"],
        home_field_advantage=metadata["home_field_advantage"],
        margin_source=metadata["margin_source"],
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
            if rating_hash_matches_run(
                conn,
                run_id=latest_run_id,
                stored_rating_hash=latest_hash,
                rating_hash=rating_hash,
            ):
                print(
                    "Team ratings match latest comparable successful snapshot; "
                    f"marking this run as duplicate of {latest_run_id}."
                )
                write_duplicate_run(
                    conn,
                    run_id=run_id,
                    rating_hash=rating_hash,
                    duplicate_of_run_id=latest_run_id,
                )
                print("Finished without inserting duplicate team rating detail rows.")
                return run_id, 0, "duplicate"

        print("Team rating set changed; inserting full snapshot rows...")
        for record in records:
            record["team_rating_run_id"] = run_id
        write_changed_snapshot(conn, run_id=run_id, records=records, rating_hash=rating_hash)
        print(f"Finished inserting {len(records)} rows into team_ratings.")
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

    print("Connecting to database...")
    with psycopg.connect(cfg.pg_dsn) as conn:
        print(f"Target season: {season}")
        ensure_team_rating_tables(conn)

        if run_type == "backfill" and not run_date_was_supplied and not explicit_game_prediction_run_id:
            dates = backfill_run_dates(conn, season)
            print(f"Backfilling {len(dates)} team rating run dates for season {season}.")
            total_inserted = 0
            for backfill_date in dates:
                _, inserted_count, _ = process_team_rating_snapshot(
                    conn,
                    season=season,
                    run_date=backfill_date,
                    run_type=run_type,
                    etl_run_id=cfg.run_id,
                    notes=notes,
                    explicit_game_prediction_run_id=None,
                )
                total_inserted += inserted_count
            print(f"Finished team rating backfill; inserted {total_inserted} detail rows.")
            return

        process_team_rating_snapshot(
            conn,
            season=season,
            run_date=run_date,
            run_type=run_type,
            etl_run_id=cfg.run_id,
            notes=notes,
            explicit_game_prediction_run_id=explicit_game_prediction_run_id,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"update_team_ratings_full.py failed: {e}")
        sys.exit(1)
