#!/usr/bin/env python3
"""
Local CSV-only rankings projection table builder.

Reads from CSV snapshots in this folder and writes a projected rankings CSV.
No Neon connection is used.

This is intentionally a framework model:
  - projected_ap_ranking / projected_cfp_ranking estimate the current ranking order
  - projected_end_ap_ranking / projected_end_cfp_ranking estimate the end-of-season order

The scoring formula is designed to be transparent and easy to tune before the
nightly ETL version is added.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_DIR = Path(__file__).resolve().parent
GAME_DATA_CSV = MODEL_DIR / "game_data.csv"
GAME_PREDICTIONS_FULL_CSV = MODEL_DIR / "game_predictions_full.csv"
RANKINGS_CSV = MODEL_DIR / "rankings.csv"
TEAM_ADVANCED_SEASON_STATS_CSV = MODEL_DIR / "team_advanced_season_stats.csv"
TEAM_RECRUITING_RANKINGS_CSV = MODEL_DIR / "team_recruiting_rankings.csv"
TEAM_TALENT_COMPOSITE_CSV = MODEL_DIR / "team_talent_composite.csv"
TEAM_RETURNING_PRODUCTION_CSV = MODEL_DIR / "team_returning_production.csv"
DEFAULT_OUTPUT_CSV = MODEL_DIR / "rankings_projection_full.csv"

AP_POLL_NAMES = {"AP Top 25", "AP Poll"}
COACHES_POLL_NAMES = {"Coaches Poll", "USA Today Coaches Poll"}
CFP_POLL_NAMES = {"Playoff Committee Rankings", "College Football Playoff", "CFP Rankings"}
REGULAR_SEASON_TYPES = {"regular", "regular season", "2"}
HASH_FLOAT_DECIMAL_PLACES = 4

OUTPUT_COLUMNS = [
    "ranking_projection_run_id",
    "season",
    "run_date",
    "run_type",
    "model_version",
    "team",
    "conference",
    "classification",
    "created_at",
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
    "game_prediction_run_id",
    "prediction_hash",
    "prediction_type",
    "notes",
]

HASH_EXCLUDED_COLUMNS = {
    "ranking_projection_run_id",
    "run_date",
    "run_type",
    "created_at",
    "notes",
    "prediction_hash",
}


def read_csv(path: Path, *, required: bool = True) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required CSV: {path}")
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: "" if row.get(col) is None else row.get(col) for col in OUTPUT_COLUMNS})


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    number = parse_float(value)
    if number is None:
        return None
    return int(number)


def is_truthy(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "t", "true", "yes", "y"}


def date_part(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    return text[:10]


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def infer_latest_season(*row_groups: list[dict[str, str]]) -> int:
    seasons: list[int] = []
    for rows in row_groups:
        for row in rows:
            season = parse_int(row.get("season") or row.get("year"))
            if season is not None:
                seasons.append(season)
    if not seasons:
        raise RuntimeError("No usable season values were found in local CSVs.")
    return max(seasons)


def select_game_prediction_run(
    prediction_rows: list[dict[str, str]],
    *,
    season: int,
    game_prediction_run_id: str | None,
) -> tuple[str | None, list[dict[str, str]], str | None]:
    season_rows = [row for row in prediction_rows if parse_int(row.get("season")) == season]
    if not season_rows:
        return None, [], None

    if game_prediction_run_id:
        selected = [row for row in season_rows if row.get("game_prediction_run_id") == game_prediction_run_id]
        if not selected:
            raise RuntimeError(
                f"No rows found for season {season} and game_prediction_run_id={game_prediction_run_id}."
            )
        created_at = max((row.get("created_at") or "" for row in selected), default=None)
        return game_prediction_run_id, selected, date_part(created_at)

    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in season_rows:
        by_run[row.get("game_prediction_run_id", "")].append(row)

    def sort_key(item: tuple[str, list[dict[str, str]]]) -> tuple[str, int, str]:
        run_id, rows = item
        latest_created_at = max((row.get("created_at") or "" for row in rows), default="")
        return latest_created_at, len(rows), run_id

    selected_run_id, selected_rows = max(by_run.items(), key=sort_key)
    created_at = max((row.get("created_at") or "" for row in selected_rows), default=None)
    return selected_run_id or None, selected_rows, date_part(created_at)


def model_version_label(prediction_rows: list[dict[str, str]]) -> str:
    versions = sorted({row.get("model_version", "").strip() for row in prediction_rows if row.get("model_version")})
    suffix = "+".join(versions) if versions else "no_game_prediction_run"
    return f"rankings_projection_local+{suffix}"


def canonical_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, HASH_FLOAT_DECIMAL_PLACES)
    return value


def canonical_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: canonical_value(record.get(key))
        for key in sorted(record)
        if key not in HASH_EXCLUDED_COLUMNS
    }


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add_prediction_hashes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_rows = []
    for row in sorted(rows, key=lambda item: str(item["team"])):
        enriched = dict(row)
        enriched["prediction_hash"] = hash_payload(canonical_record(row))
        output_rows.append(enriched)
    return output_rows


def game_record_from_prediction(pred: dict[str, str], source: dict[str, str]) -> dict[str, Any]:
    home_team = pred.get("home_team") or source.get("hometeam")
    away_team = pred.get("away_team") or source.get("awayteam")
    return {
        "gameid": pred.get("gameid"),
        "season": parse_int(pred.get("season")),
        "week": parse_int(pred.get("week") or source.get("week")),
        "seasontype": str(source.get("seasontype", "regular") or "regular").lower(),
        "startdate": source.get("startdate"),
        "conferencegame": is_truthy(source.get("conferencegame")),
        "hometeam": home_team,
        "awayteam": away_team,
        "homeclassification": str(source.get("homeclassification", "") or "").lower(),
        "awayclassification": str(source.get("awayclassification", "") or "").lower(),
        "homeconference": source.get("homeconference"),
        "awayconference": source.get("awayconference"),
        "homepoints": parse_float(source.get("homepoints")),
        "awaypoints": parse_float(source.get("awaypoints")),
        "pred_homepoints": parse_float(pred.get("homepoints")),
        "pred_awaypoints": parse_float(pred.get("awaypoints")),
        "homewinprob": parse_float(pred.get("homewinprob")),
    }


def game_record_from_actual(source: dict[str, str]) -> dict[str, Any]:
    homepoints = parse_float(source.get("homepoints"))
    awaypoints = parse_float(source.get("awaypoints"))
    return {
        "gameid": source.get("id"),
        "season": parse_int(source.get("season")),
        "week": parse_int(source.get("week")),
        "seasontype": str(source.get("seasontype", "regular") or "regular").lower(),
        "startdate": source.get("startdate"),
        "conferencegame": is_truthy(source.get("conferencegame")),
        "hometeam": source.get("hometeam"),
        "awayteam": source.get("awayteam"),
        "homeclassification": str(source.get("homeclassification", "") or "").lower(),
        "awayclassification": str(source.get("awayclassification", "") or "").lower(),
        "homeconference": source.get("homeconference"),
        "awayconference": source.get("awayconference"),
        "homepoints": homepoints,
        "awaypoints": awaypoints,
        "pred_homepoints": homepoints,
        "pred_awaypoints": awaypoints,
        "homewinprob": None,
    }


def build_game_records(
    *,
    game_data_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    season: int,
    run_date: str,
    allow_missing_game_predictions: bool = False,
) -> list[dict[str, Any]]:
    game_data_by_id = {row["id"]: row for row in game_data_rows if row.get("id")}
    selected_game_ids = set()
    games: list[dict[str, Any]] = []

    for pred in prediction_rows:
        gameid = pred.get("gameid")
        if not gameid:
            continue
        source = game_data_by_id.get(gameid, {})
        selected_game_ids.add(gameid)
        games.append(game_record_from_prediction(pred, source))

    missing_prediction_games: list[str] = []
    using_prediction_snapshot = bool(prediction_rows)
    for source in game_data_rows:
        if parse_int(source.get("season")) != season:
            continue
        gameid = source.get("id")
        if not gameid or gameid in selected_game_ids:
            continue
        if str(source.get("seasontype", "")).lower() != "regular":
            continue
        completed_before_run = (
            date_part(source.get("startdate")) is not None
            and date_part(source.get("startdate")) < run_date
            and parse_float(source.get("homepoints")) is not None
            and parse_float(source.get("awaypoints")) is not None
        )
        if not completed_before_run:
            if using_prediction_snapshot:
                missing_prediction_games.append(
                    f"{gameid}: {source.get('awayteam')} at {source.get('hometeam')} "
                    f"week {source.get('week')}"
                )
            continue
        games.append(game_record_from_actual(source))

    if missing_prediction_games and not allow_missing_game_predictions:
        preview = "; ".join(missing_prediction_games[:8])
        extra = "" if len(missing_prediction_games) <= 8 else f"; +{len(missing_prediction_games) - 8} more"
        raise RuntimeError(
            "Selected game_prediction_run_id does not cover every unplayed regular-season game. "
            "Use one complete prediction snapshot for the rankings run so remaining-season win "
            f"probabilities are all from the same model/date. Missing: {preview}{extra}"
        )

    games = [game for game in games if game["seasontype"] == "regular"]
    if not games:
        raise RuntimeError("No regular-season game rows available for rankings projection.")
    return games


def team_profiles(games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for game in games:
        for side in ("home", "away"):
            team = game.get(f"{side}team")
            if not team:
                continue
            profile = profiles.setdefault(team, {"team": team, "conference": None, "classification": None})
            conference = game.get(f"{side}conference")
            classification = game.get(f"{side}classification")
            if conference:
                profile["conference"] = conference
            if classification:
                profile["classification"] = str(classification).lower()
    return profiles


def rank_desc(values_by_team: dict[str, float], teams: list[str]) -> dict[str, int]:
    ordered = sorted(teams, key=lambda team: (-values_by_team.get(team, 0.0), team))
    return {team: rank + 1 for rank, team in enumerate(ordered)}


def percentile_scores(values_by_team: dict[str, float], teams: list[str], *, lower_is_better: bool = False) -> dict[str, float]:
    usable = {team: values_by_team[team] for team in teams if values_by_team.get(team) is not None}
    if not usable:
        return {team: 50.0 for team in teams}
    ordered = sorted(usable, key=usable.get, reverse=not lower_is_better)
    if len(ordered) == 1:
        return {team: 50.0 for team in teams}
    scores: dict[str, float] = {}
    for index, team in enumerate(ordered):
        scores[team] = 100.0 * (len(ordered) - 1 - index) / (len(ordered) - 1)
    for team in teams:
        scores.setdefault(team, 50.0)
    return scores


def latest_poll_rankings(
    rankings_rows: list[dict[str, str]],
    *,
    season: int,
    poll_names: set[str],
    through_week: int | None = None,
) -> tuple[dict[str, int], dict[str, int], int | None, int | None]:
    season_poll_rows = [
        row
        for row in rankings_rows
        if parse_int(row.get("season")) == season
        and row.get("poll") in poll_names
        and str(row.get("season_type", "")).strip().lower() in REGULAR_SEASON_TYPES
        and (through_week is None or (parse_int(row.get("week")) or 0) <= through_week)
    ]
    if not season_poll_rows:
        return {}, {}, None, None

    weeks = sorted({parse_int(row.get("week")) for row in season_poll_rows if parse_int(row.get("week")) is not None})
    if not weeks:
        return {}, {}, None, None
    current_week = weeks[-1]
    previous_week = weeks[-2] if len(weeks) >= 2 else None

    current = {
        row["school"]: parse_int(row.get("rank"))
        for row in season_poll_rows
        if parse_int(row.get("week")) == current_week and parse_int(row.get("rank")) is not None
    }
    previous = {
        row["school"]: parse_int(row.get("rank"))
        for row in season_poll_rows
        if previous_week is not None
        and parse_int(row.get("week")) == previous_week
        and parse_int(row.get("rank")) is not None
    }
    return current, previous, current_week, previous_week


def best_row_by_season(rows: list[dict[str, str]], *, team: str, season: int) -> tuple[dict[str, str] | None, int | None]:
    candidates = [
        row
        for row in rows
        if row.get("team") == team
        and parse_int(row.get("season") or row.get("year")) is not None
        and parse_int(row.get("season") or row.get("year")) <= season
    ]
    if not candidates:
        return None, None
    selected = max(candidates, key=lambda row: parse_int(row.get("season") or row.get("year")) or 0)
    return selected, parse_int(selected.get("season") or selected.get("year"))


def build_supporting_scores(
    *,
    teams: list[str],
    season: int,
    advanced_rows: list[dict[str, str]],
    recruiting_rows: list[dict[str, str]],
    talent_rows: list[dict[str, str]],
    returning_rows: list[dict[str, str]],
) -> dict[str, dict[str, float | int | None]]:
    raw_efficiency: dict[str, float] = {}
    raw_talent: dict[str, float] = {}
    raw_recruiting: dict[str, float] = {}
    raw_returning: dict[str, float] = {}
    advanced_season: dict[str, int | None] = {}

    for team in teams:
        advanced, advanced_year = best_row_by_season(advanced_rows, team=team, season=season)
        advanced_season[team] = advanced_year
        if advanced:
            offense_ppa = parse_float(advanced.get("offense_ppa")) or 0.0
            defense_ppa = parse_float(advanced.get("defense_ppa")) or 0.0
            offense_success = parse_float(advanced.get("offense_successrate")) or 0.0
            defense_success = parse_float(advanced.get("defense_successrate")) or 0.0
            offense_points = parse_float(advanced.get("offense_pointsperopportunity")) or 0.0
            defense_points = parse_float(advanced.get("defense_pointsperopportunity")) or 0.0
            raw_efficiency[team] = (
                offense_ppa * 45.0
                - defense_ppa * 45.0
                + offense_success * 22.0
                - defense_success * 22.0
                + offense_points * 2.5
                - defense_points * 2.5
            )

        talent, _ = best_row_by_season(talent_rows, team=team, season=season)
        if talent:
            raw_talent[team] = parse_float(talent.get("talent")) or 0.0

        recruiting, _ = best_row_by_season(recruiting_rows, team=team, season=season)
        if recruiting:
            rank = parse_float(recruiting.get("rank"))
            points = parse_float(recruiting.get("points"))
            raw_recruiting[team] = points if points is not None else -(rank or 999.0)

        returning, _ = best_row_by_season(returning_rows, team=team, season=season)
        if returning:
            raw_returning[team] = parse_float(returning.get("percent_ppa")) or parse_float(returning.get("total_ppa")) or 0.0

    efficiency_scores = percentile_scores(raw_efficiency, teams)
    talent_scores = percentile_scores(raw_talent, teams)
    recruiting_scores = percentile_scores(raw_recruiting, teams)
    returning_scores = percentile_scores(raw_returning, teams)

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


def poll_rank_to_score(rank: int | None, *, unranked_score: float = 20.0) -> float:
    if rank is None:
        return unranked_score
    return bounded(105.0 - rank * 3.2)


def build_team_metrics(
    *,
    games: list[dict[str, Any]],
    teams: list[str],
    run_date: str,
) -> dict[str, dict[str, Any]]:
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

    for game in games:
        home = game.get("hometeam")
        away = game.get("awayteam")
        if home not in metrics or away not in metrics:
            continue
        same_conference_game = (
            game.get("conferencegame")
            and game.get("homeconference")
            and game.get("homeconference") == game.get("awayconference")
        )
        homepoints = game.get("homepoints")
        awaypoints = game.get("awaypoints")
        completed = homepoints is not None and awaypoints is not None
        startdate = date_part(game.get("startdate"))
        completed_before_run = completed and (startdate is None or startdate < run_date)

        homewinprob = parse_float(game.get("homewinprob"))
        if homewinprob is None:
            if completed:
                homewinprob = 1.0 if homepoints > awaypoints else 0.0
            else:
                homewinprob = 0.5
        homewinprob = max(0.0, min(1.0, homewinprob))

        metrics[home]["opponents"].append(away)
        metrics[away]["opponents"].append(home)

        if completed_before_run:
            home_won = homepoints > awaypoints
            metrics[home]["current_wins"] += 1.0 if home_won else 0.0
            metrics[home]["current_losses"] += 0.0 if home_won else 1.0
            metrics[away]["current_wins"] += 0.0 if home_won else 1.0
            metrics[away]["current_losses"] += 1.0 if home_won else 0.0
            metrics[home]["points_for"] += homepoints
            metrics[home]["points_against"] += awaypoints
            metrics[away]["points_for"] += awaypoints
            metrics[away]["points_against"] += homepoints
            metrics[home]["games_played"] += 1.0
            metrics[away]["games_played"] += 1.0
            if same_conference_game:
                metrics[home]["current_conference_wins"] += 1.0 if home_won else 0.0
                metrics[home]["current_conference_losses"] += 0.0 if home_won else 1.0
                metrics[away]["current_conference_wins"] += 0.0 if home_won else 1.0
                metrics[away]["current_conference_losses"] += 1.0 if home_won else 0.0

        metrics[home]["projected_wins"] += 1.0 if completed and homepoints > awaypoints else homewinprob
        metrics[home]["projected_losses"] += 0.0 if completed and homepoints > awaypoints else 1.0 - homewinprob
        metrics[away]["projected_wins"] += 0.0 if completed and homepoints > awaypoints else 1.0 - homewinprob
        metrics[away]["projected_losses"] += 1.0 if completed and homepoints > awaypoints else homewinprob

        if same_conference_game:
            metrics[home]["projected_conference_wins"] += 1.0 if completed and homepoints > awaypoints else homewinprob
            metrics[home]["projected_conference_losses"] += 0.0 if completed and homepoints > awaypoints else 1.0 - homewinprob
            metrics[away]["projected_conference_wins"] += 0.0 if completed and homepoints > awaypoints else 1.0 - homewinprob
            metrics[away]["projected_conference_losses"] += 1.0 if completed and homepoints > awaypoints else homewinprob

        if not completed_before_run:
            metrics[home]["remaining_opponents"].append(away)
            metrics[away]["remaining_opponents"].append(home)

    for team, row in metrics.items():
        current_games = row["current_wins"] + row["current_losses"]
        projected_games = row["projected_wins"] + row["projected_losses"]
        margin = safe_ratio(row["points_for"] - row["points_against"], current_games, 0.0)
        current_win_pct = safe_ratio(row["current_wins"], current_games, 0.5)
        projected_win_pct = safe_ratio(row["projected_wins"], projected_games, current_win_pct)
        row["current_win_pct"] = current_win_pct
        row["projected_win_pct"] = projected_win_pct
        row["average_point_margin"] = margin

    team_strength = {
        team: (
            metrics[team]["projected_win_pct"] * 0.70
            + safe_ratio(metrics[team]["average_point_margin"] + 28.0, 56.0, 0.5) * 0.30
        )
        for team in teams
    }
    for team, row in metrics.items():
        row["team_strength"] = team_strength[team]
        row["strength_of_schedule"] = safe_mean([team_strength[opponent] for opponent in row["opponents"]])
        row["remaining_strength_of_schedule"] = safe_mean(
            [team_strength[opponent] for opponent in row["remaining_opponents"]]
        )

    return metrics


def build_ranking_rows(
    *,
    season: int,
    run_date: str,
    run_type: str,
    ranking_projection_run_id: str,
    created_at: str,
    model_version: str,
    game_prediction_run_id: str | None,
    games: list[dict[str, Any]],
    rankings_rows: list[dict[str, str]],
    advanced_rows: list[dict[str, str]],
    recruiting_rows: list[dict[str, str]],
    talent_rows: list[dict[str, str]],
    returning_rows: list[dict[str, str]],
    notes: str | None,
    poll_through_week: int | None = None,
) -> list[dict[str, Any]]:
    profiles = team_profiles(games)
    fbs_teams = sorted(
        team
        for team, profile in profiles.items()
        if profile.get("classification") == "fbs"
    )
    if not fbs_teams:
        raise RuntimeError("No FBS teams found for rankings projection.")

    metrics = build_team_metrics(games=games, teams=fbs_teams, run_date=run_date)
    support = build_supporting_scores(
        teams=fbs_teams,
        season=season,
        advanced_rows=advanced_rows,
        recruiting_rows=recruiting_rows,
        talent_rows=talent_rows,
        returning_rows=returning_rows,
    )
    current_ap, previous_ap, _, _ = latest_poll_rankings(
        rankings_rows,
        season=season,
        poll_names=AP_POLL_NAMES,
        through_week=poll_through_week,
    )
    current_coaches, _, _, _ = latest_poll_rankings(
        rankings_rows,
        season=season,
        poll_names=COACHES_POLL_NAMES,
        through_week=poll_through_week,
    )
    current_cfp, previous_cfp, _, _ = latest_poll_rankings(
        rankings_rows,
        season=season,
        poll_names=CFP_POLL_NAMES,
        through_week=poll_through_week,
    )

    current_win_pct_scores = percentile_scores(
        {team: metrics[team]["current_win_pct"] for team in fbs_teams},
        fbs_teams,
    )
    projected_win_pct_scores = percentile_scores(
        {team: metrics[team]["projected_win_pct"] for team in fbs_teams},
        fbs_teams,
    )
    margin_scores = percentile_scores(
        {team: metrics[team]["average_point_margin"] for team in fbs_teams},
        fbs_teams,
    )
    sos_scores = percentile_scores(
        {team: metrics[team]["strength_of_schedule"] or 0.5 for team in fbs_teams},
        fbs_teams,
    )
    remaining_sos_scores = percentile_scores(
        {team: metrics[team]["remaining_strength_of_schedule"] or 0.5 for team in fbs_teams},
        fbs_teams,
    )
    projected_win_total_scores = percentile_scores(
        {team: metrics[team]["projected_wins"] for team in fbs_teams},
        fbs_teams,
    )
    projected_conference_scores = percentile_scores(
        {team: metrics[team]["projected_conference_wins"] for team in fbs_teams},
        fbs_teams,
    )

    score_rows: dict[str, dict[str, float]] = {}
    for team in fbs_teams:
        power_score = float(support[team]["power_score"] or 50.0)
        poll_inertia_score = (
            poll_rank_to_score(current_ap.get(team))
            + poll_rank_to_score(previous_ap.get(team), unranked_score=18.0)
            + poll_rank_to_score(current_coaches.get(team), unranked_score=18.0)
            + poll_rank_to_score(current_cfp.get(team), unranked_score=18.0)
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

    projected_ap_ranks = rank_desc(
        {team: score_rows[team]["projected_ap_score"] for team in fbs_teams},
        fbs_teams,
    )
    projected_cfp_ranks = rank_desc(
        {team: score_rows[team]["projected_cfp_score"] for team in fbs_teams},
        fbs_teams,
    )
    projected_end_ap_ranks = rank_desc(
        {team: score_rows[team]["projected_end_ap_score"] for team in fbs_teams},
        fbs_teams,
    )
    projected_end_cfp_ranks = rank_desc(
        {team: score_rows[team]["projected_end_cfp_score"] for team in fbs_teams},
        fbs_teams,
    )

    rows: list[dict[str, Any]] = []
    for team in fbs_teams:
        metric = metrics[team]
        scores = score_rows[team]
        row = {
            "ranking_projection_run_id": ranking_projection_run_id,
            "season": season,
            "run_date": run_date,
            "run_type": run_type,
            "model_version": model_version,
            "team": team,
            "conference": profiles[team].get("conference"),
            "classification": profiles[team].get("classification"),
            "created_at": created_at,
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
        rows.append(row)

    return add_prediction_hashes(rows)


def main() -> None:
    game_prediction_rows = read_csv(GAME_PREDICTIONS_FULL_CSV, required=False)
    game_data_rows = read_csv(GAME_DATA_CSV)
    rankings_rows = read_csv(RANKINGS_CSV, required=False)
    advanced_rows = read_csv(TEAM_ADVANCED_SEASON_STATS_CSV, required=False)
    recruiting_rows = read_csv(TEAM_RECRUITING_RANKINGS_CSV, required=False)
    talent_rows = read_csv(TEAM_TALENT_COMPOSITE_CSV, required=False)
    returning_rows = read_csv(TEAM_RETURNING_PRODUCTION_CSV, required=False)
    default_season = infer_latest_season(game_prediction_rows, game_data_rows)

    parser = argparse.ArgumentParser(description="Build local rankings_projection_full CSV from local snapshots.")
    parser.add_argument("--season", type=int, default=default_season)
    parser.add_argument("--game-prediction-run-id", default=None)
    parser.add_argument("--ranking-projection-run-id", default=None)
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--run-type", default="local")
    parser.add_argument("--poll-through-week", type=int, default=None)
    parser.add_argument("--allow-missing-game-predictions", action="store_true")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    game_prediction_run_id, selected_predictions, inferred_run_date = select_game_prediction_run(
        game_prediction_rows,
        season=args.season,
        game_prediction_run_id=args.game_prediction_run_id,
    )
    run_date = args.run_date or inferred_run_date or datetime.now(timezone.utc).date().isoformat()
    ranking_projection_run_id = args.ranking_projection_run_id or str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    games = build_game_records(
        game_data_rows=game_data_rows,
        prediction_rows=selected_predictions,
        season=args.season,
        run_date=run_date,
        allow_missing_game_predictions=args.allow_missing_game_predictions,
    )
    rows = build_ranking_rows(
        season=args.season,
        run_date=run_date,
        run_type=args.run_type,
        ranking_projection_run_id=ranking_projection_run_id,
        created_at=created_at,
        model_version=model_version_label(selected_predictions),
        game_prediction_run_id=game_prediction_run_id,
        games=games,
        rankings_rows=rankings_rows,
        advanced_rows=advanced_rows,
        recruiting_rows=recruiting_rows,
        talent_rows=talent_rows,
        returning_rows=returning_rows,
        notes=args.notes,
        poll_through_week=args.poll_through_week,
    )

    write_csv(args.output, rows)
    print(f"Selected game_prediction_run_id: {game_prediction_run_id or 'none'}")
    print(f"Win probability snapshot rows: {len(selected_predictions)}")
    print(f"Run date: {run_date}")
    if args.poll_through_week is not None:
        print(f"Poll inputs limited through week: {args.poll_through_week}")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
