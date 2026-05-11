#!/usr/bin/env python3
"""
Local CSV-only season prediction table builder.

Reads from CSV snapshots in this folder:
  - game_data.csv
  - game_predictions_full.csv

It writes a local CSV with the same columns as public.season_predictions_full.
No Neon connection is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conference_championship_rules import (
    SEC_RELATIVE_OFFENSE_CAP,
    select_conference_championship_teams,
    simulate_conference_championship_game,
    sun_belt_division_for_team,
)


MODEL_DIR = Path(__file__).resolve().parent
GAME_DATA_CSV = MODEL_DIR / "game_data.csv"
GAME_PREDICTIONS_FULL_CSV = MODEL_DIR / "game_predictions_full.csv"
RANKINGS_PROJECTION_FULL_CSV = MODEL_DIR / "rankings_projection_full.csv"
DEFAULT_OUTPUT_CSV = MODEL_DIR / "season_predictions_full.csv"

DEFAULT_SIMULATIONS = 10_000
MAX_WIN_BUCKET = 13
HASH_FLOAT_DECIMAL_PLACES = 4
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

WIN_PROBABILITY_COLUMNS = [f"probability_{wins}_wins" for wins in range(MAX_WIN_BUCKET + 1)]
OUTPUT_COLUMNS = [
    "season_prediction_run_id",
    "season",
    "run_date",
    "run_type",
    "model_version",
    "team",
    "conference",
    "division",
    "classification",
    "created_at",
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
    "prediction_hash",
    "prediction_type",
    "notes",
]

HASH_EXCLUDED_COLUMNS = {
    "season_prediction_run_id",
    "run_date",
    "run_type",
    "created_at",
    "notes",
    "prediction_hash",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
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


def infer_latest_season(prediction_rows: list[dict[str, str]]) -> int:
    seasons = [parse_int(row.get("season")) for row in prediction_rows]
    seasons = [season for season in seasons if season is not None]
    if not seasons:
        raise RuntimeError("game_predictions_full.csv has no usable season values.")
    return max(seasons)


def select_game_prediction_run(
    prediction_rows: list[dict[str, str]],
    *,
    season: int,
    game_prediction_run_id: str | None,
) -> tuple[str, list[dict[str, str]], str | None]:
    season_rows = [row for row in prediction_rows if parse_int(row.get("season")) == season]
    if not season_rows:
        raise RuntimeError(f"No game prediction rows found for season {season}.")

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
        by_run[row["game_prediction_run_id"]].append(row)

    def sort_key(item: tuple[str, list[dict[str, str]]]) -> tuple[str, int, str]:
        run_id, rows = item
        latest_created_at = max((row.get("created_at") or "" for row in rows), default="")
        return latest_created_at, len(rows), run_id

    selected_run_id, selected_rows = max(by_run.items(), key=sort_key)
    created_at = max((row.get("created_at") or "" for row in selected_rows), default=None)
    return selected_run_id, selected_rows, date_part(created_at)


def model_version_label(prediction_rows: list[dict[str, str]]) -> str:
    versions = sorted({row.get("model_version", "").strip() for row in prediction_rows if row.get("model_version")})
    suffix = "+".join(versions) if versions else "unknown_game_model"
    return f"season_sim_local+{suffix}"


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


def add_prediction_hashes(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    canonical_rows = []
    output_rows = []
    for row in sorted(rows, key=lambda item: str(item["team"])):
        canonical = canonical_record(row)
        row_hash = hash_payload(canonical)
        enriched = dict(row)
        enriched["prediction_hash"] = row_hash
        canonical_rows.append(canonical)
        output_rows.append(enriched)
    return output_rows, hash_payload(canonical_rows)


def select_ranking_projection_rows(
    ranking_rows: list[dict[str, str]],
    *,
    season: int,
    ranking_projection_run_id: str | None,
) -> tuple[str | None, list[dict[str, str]]]:
    season_rows = [row for row in ranking_rows if parse_int(row.get("season")) == season]
    if not season_rows:
        raise RuntimeError(f"No ranking projection rows found for season {season}.")

    if ranking_projection_run_id:
        selected = [
            row
            for row in season_rows
            if row.get("ranking_projection_run_id") == ranking_projection_run_id
        ]
        if not selected:
            raise RuntimeError(
                f"No rows found for season {season} and ranking_projection_run_id={ranking_projection_run_id}."
            )
        return ranking_projection_run_id, selected

    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in season_rows:
        by_run[row.get("ranking_projection_run_id", "")].append(row)

    def sort_key(item: tuple[str, list[dict[str, str]]]) -> tuple[str, int, str]:
        run_id, rows = item
        latest_created_at = max((row.get("created_at") or "" for row in rows), default="")
        return latest_created_at, len(rows), run_id

    selected_run_id, selected_rows = max(by_run.items(), key=sort_key)
    return selected_run_id or None, selected_rows


def ranking_projection_lookup(ranking_rows: list[dict[str, str]]) -> dict[str, dict[str, float | int | None]]:
    lookup: dict[str, dict[str, float | int | None]] = {}
    for row in ranking_rows:
        team = row.get("team")
        if not team:
            continue
        lookup[team] = {
            "projected_ap_ranking": parse_int(row.get("projected_ap_ranking")),
            "projected_end_ap_ranking": parse_int(row.get("projected_end_ap_ranking")),
            "projected_cfp_ranking": parse_int(row.get("projected_cfp_ranking")),
            "projected_end_cfp_ranking": parse_int(row.get("projected_end_cfp_ranking")),
            "projected_end_cfp_score": parse_float(row.get("projected_end_cfp_score")),
            "projected_wins": parse_float(row.get("projected_wins")),
            "projected_conference_wins": parse_float(row.get("projected_conference_wins")),
            "team_strength": parse_float(row.get("team_strength")),
            "strength_of_schedule": parse_float(row.get("strength_of_schedule")),
        }
    return lookup


def ranking_projection_score(
    team: str,
    *,
    ranking_projection: dict[str, dict[str, float | int | None]],
    fallback_rank: int | None,
    fallback_score: float,
) -> float:
    projection = ranking_projection.get(team, {})
    score = projection.get("projected_end_cfp_score")
    if isinstance(score, (float, int)) and not math.isnan(float(score)):
        return float(score)

    rank = projection.get("projected_end_cfp_ranking") or projection.get("projected_cfp_ranking") or fallback_rank
    if isinstance(rank, int) and rank > 0:
        return max(0.0, 130.0 - rank * 2.0)
    return fallback_score


def cfp_selection_score(
    team: str,
    *,
    sim: int,
    total_wins: dict[str, list[int]],
    conference_wins: dict[str, list[int]],
    conference_champions: set[str],
    ranking_projection: dict[str, dict[str, float | int | None]],
    projected_cfp_rank_for_tiebreakers: dict[str, int],
    resume_score: dict[str, float],
    team_strength: dict[str, float],
    sos_for_rank: dict[str, float],
) -> float:
    base_score = ranking_projection_score(
        team,
        ranking_projection=ranking_projection,
        fallback_rank=projected_cfp_rank_for_tiebreakers.get(team),
        fallback_score=resume_score.get(team, 0.0) / 10.0,
    )
    projection = ranking_projection.get(team, {})
    expected_wins = projection.get("projected_wins")
    if not isinstance(expected_wins, (float, int)):
        expected_wins = sum(total_wins[team]) / len(total_wins[team])
    expected_conference_wins = projection.get("projected_conference_wins")
    if not isinstance(expected_conference_wins, (float, int)):
        expected_conference_wins = sum(conference_wins[team]) / len(conference_wins[team])

    win_delta = total_wins[team][sim] - float(expected_wins)
    conference_win_delta = conference_wins[team][sim] - float(expected_conference_wins)
    champion_bonus = 7.5 if team in conference_champions else 0.0
    return (
        base_score
        + win_delta * 8.5
        + conference_win_delta * 1.5
        + champion_bonus
        + team_strength[team] * 2.0
        + sos_for_rank[team] * 1.5
    )


def cfp_game_win_probability(
    team_a: str,
    team_b: str,
    *,
    ranking_scores: dict[str, float],
    team_strength: dict[str, float],
    total_wins: dict[str, list[int]],
    sim: int,
) -> float:
    ranking_delta = (ranking_scores[team_a] - ranking_scores[team_b]) / 18.0
    strength_delta = (team_strength[team_a] - team_strength[team_b]) * 4.0
    win_delta = (total_wins[team_a][sim] - total_wins[team_b][sim]) * 0.12
    probability = 1.0 / (1.0 + math.exp(-(ranking_delta + strength_delta + win_delta)))
    return min(max(probability, 0.05), 0.95)


def simulate_cfp_game(
    team_a: str,
    team_b: str,
    *,
    ranking_scores: dict[str, float],
    team_strength: dict[str, float],
    total_wins: dict[str, list[int]],
    sim: int,
    rng: random.Random,
) -> str:
    probability = cfp_game_win_probability(
        team_a,
        team_b,
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
    )
    return team_a if rng.random() < probability else team_b


def select_cfp_field(
    *,
    ranked_fbs: list[str],
    conference_champions: set[str],
    profiles: dict[str, dict[str, Any]],
    ranking_scores: dict[str, float],
) -> tuple[list[str], list[str], list[str]]:
    rank_order = {team: idx + 1 for idx, team in enumerate(ranked_fbs)}
    champions_by_conference = {
        profiles[team].get("conference"): team
        for team in conference_champions
        if team in rank_order and profiles[team].get("conference")
    }

    auto_bids: list[str] = []
    for conference in sorted(CFP_POWER_AUTO_CONFERENCES):
        champion = champions_by_conference.get(conference)
        if champion:
            auto_bids.append(champion)

    group_candidates = [
        team
        for team in ranked_fbs
        if profiles[team].get("conference") in CFP_GROUP_AUTO_CONFERENCES
    ]
    if group_candidates:
        auto_bids.append(group_candidates[0])

    if NOTRE_DAME in rank_order and rank_order[NOTRE_DAME] <= 12:
        auto_bids.append(NOTRE_DAME)

    auto_bids = list(dict.fromkeys(auto_bids))
    auto_bid_set = set(auto_bids)
    at_large_slots = max(0, 12 - len(auto_bids))
    at_large = [team for team in ranked_fbs if team not in auto_bid_set][:at_large_slots]
    playoff = auto_bids + at_large

    auto_bid_set = set(auto_bids)

    def seed_key(team: str) -> tuple[int, int, float, str]:
        unranked_auto_bid = team in auto_bid_set and rank_order.get(team, 999) > 25
        return (
            1 if unranked_auto_bid else 0,
            rank_order.get(team, 999),
            -ranking_scores.get(team, 0.0),
            team,
        )

    seeded = sorted(playoff, key=seed_key)[:12]
    return seeded, auto_bids, [team for team in seeded if team not in auto_bid_set]


def simulate_cfp_bracket(
    seeded: list[str],
    *,
    ranking_scores: dict[str, float],
    team_strength: dict[str, float],
    total_wins: dict[str, list[int]],
    sim: int,
    rng: random.Random,
) -> tuple[list[str], str]:
    if len(seeded) < 12:
        return [], seeded[0] if seeded else ""

    seed = {idx + 1: team for idx, team in enumerate(seeded)}
    winner_5_12 = simulate_cfp_game(
        seed[5],
        seed[12],
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )
    winner_6_11 = simulate_cfp_game(
        seed[6],
        seed[11],
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )
    winner_7_10 = simulate_cfp_game(
        seed[7],
        seed[10],
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )
    winner_8_9 = simulate_cfp_game(
        seed[8],
        seed[9],
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )

    quarter_1 = simulate_cfp_game(
        seed[1],
        winner_8_9,
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )
    quarter_4 = simulate_cfp_game(
        seed[4],
        winner_5_12,
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )
    quarter_2 = simulate_cfp_game(
        seed[2],
        winner_7_10,
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )
    quarter_3 = simulate_cfp_game(
        seed[3],
        winner_6_11,
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )

    semifinal_1 = simulate_cfp_game(
        quarter_1,
        quarter_4,
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )
    semifinal_2 = simulate_cfp_game(
        quarter_2,
        quarter_3,
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )
    champion = simulate_cfp_game(
        semifinal_1,
        semifinal_2,
        ranking_scores=ranking_scores,
        team_strength=team_strength,
        total_wins=total_wins,
        sim=sim,
        rng=rng,
    )
    return [semifinal_1, semifinal_2], champion


def build_game_records(
    *,
    game_data_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    season: int,
    run_date: str,
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

    for source in game_data_rows:
        if parse_int(source.get("season")) != season:
            continue
        gameid = source.get("id")
        if not gameid or gameid in selected_game_ids:
            continue
        if str(source.get("seasontype", "")).lower() != "regular":
            continue
        if date_part(source.get("startdate")) is None or date_part(source.get("startdate")) >= run_date:
            continue
        if parse_float(source.get("homepoints")) is None or parse_float(source.get("awaypoints")) is None:
            continue
        games.append(game_record_from_actual(source))

    games = [game for game in games if game["seasontype"] == "regular"]
    if not games:
        raise RuntimeError("No regular-season game rows available to simulate.")
    return games


def game_record_from_prediction(pred: dict[str, str], source: dict[str, str]) -> dict[str, Any]:
    home_team = pred.get("home_team") or source.get("hometeam")
    away_team = pred.get("away_team") or source.get("awayteam")
    return {
        "gameid": pred.get("gameid"),
        "season": parse_int(pred.get("season")),
        "week": parse_int(pred.get("week")),
        "seasontype": str(source.get("seasontype", "regular") or "regular").lower(),
        "startdate": source.get("startdate"),
        "conferencegame": is_truthy(source.get("conferencegame")),
        "notes": source.get("notes"),
        "hometeam": home_team,
        "awayteam": away_team,
        "homeclassification": str(source.get("homeclassification", "") or "").lower(),
        "awayclassification": str(source.get("awayclassification", "") or "").lower(),
        "homeconference": source.get("homeconference"),
        "awayconference": source.get("awayconference"),
        "homepoints": parse_float(pred.get("homepoints")),
        "awaypoints": parse_float(pred.get("awaypoints")),
        "homewinprob": parse_float(pred.get("homewinprob")),
        "expected_home_margin": expected_home_margin_from_spread(pred.get("homespread")),
        "totalpred": parse_float(pred.get("totalpred")),
    }


def game_record_from_actual(source: dict[str, str]) -> dict[str, Any]:
    homepoints = parse_float(source.get("homepoints"))
    awaypoints = parse_float(source.get("awaypoints"))
    homewinprob = 1.0 if homepoints is not None and awaypoints is not None and homepoints > awaypoints else 0.0
    return {
        "gameid": source.get("id"),
        "season": parse_int(source.get("season")),
        "week": parse_int(source.get("week")),
        "seasontype": str(source.get("seasontype", "regular") or "regular").lower(),
        "startdate": source.get("startdate"),
        "conferencegame": is_truthy(source.get("conferencegame")),
        "notes": source.get("notes"),
        "hometeam": source.get("hometeam"),
        "awayteam": source.get("awayteam"),
        "homeclassification": str(source.get("homeclassification", "") or "").lower(),
        "awayclassification": str(source.get("awayclassification", "") or "").lower(),
        "homeconference": source.get("homeconference"),
        "awayconference": source.get("awayconference"),
        "homepoints": homepoints,
        "awaypoints": awaypoints,
        "homewinprob": homewinprob,
        "expected_home_margin": homepoints - awaypoints
        if homepoints is not None and awaypoints is not None
        else None,
        "totalpred": homepoints + awaypoints
        if homepoints is not None and awaypoints is not None
        else None,
    }


def team_profiles(games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for game in games:
        for side in ("home", "away"):
            team = game[f"{side}team"]
            if not team:
                continue
            profile = profiles.setdefault(
                team,
                {"team": team, "conference": None, "classification": None},
            )
            conference = game.get(f"{side}conference")
            classification = game.get(f"{side}classification")
            if conference:
                profile["conference"] = conference
            if classification:
                profile["classification"] = str(classification).lower()
    return profiles


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def expected_home_margin_from_spread(homespread: Any) -> float | None:
    spread = parse_float(homespread)
    if spread is None:
        return None
    return -spread


def simulated_score(game: dict[str, Any], home_wins_game: bool) -> tuple[float, float]:
    homepoints = game.get("homepoints")
    awaypoints = game.get("awaypoints")
    if homepoints is not None and awaypoints is not None:
        return float(homepoints), float(awaypoints)

    expected_margin = game.get("expected_home_margin")
    magnitude = abs(float(expected_margin)) if expected_margin is not None else 1.0
    if magnitude == 0.0:
        magnitude = 1.0
    signed_margin = magnitude if home_wins_game else -magnitude
    total = float(game.get("totalpred") or 50.0)
    if total < abs(signed_margin):
        total = abs(signed_margin)
    home_score = max((total + signed_margin) / 2.0, 0.0)
    away_score = max((total - signed_margin) / 2.0, 0.0)
    return home_score, away_score


def relative_scoring_margin(
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

    return offense_pct - defense_pct


def build_sec_relative_scoring_margins(
    conference_game_scores_by_sim: list[list[tuple[str, str, float, float]]],
    teams: list[str],
) -> dict[str, list[float]]:
    margins = {team: [0.0] * len(conference_game_scores_by_sim) for team in teams}
    for sim, scores in enumerate(conference_game_scores_by_sim):
        points_for = {team: 0.0 for team in teams}
        points_allowed = {team: 0.0 for team in teams}
        games_played = {team: 0 for team in teams}

        for home, away, home_score, away_score in scores:
            points_for[home] += home_score
            points_allowed[home] += away_score
            games_played[home] += 1
            points_for[away] += away_score
            points_allowed[away] += home_score
            games_played[away] += 1

        avg_scored = {
            team: points_for[team] / games_played[team] if games_played[team] else 0.0
            for team in teams
        }
        avg_allowed = {
            team: points_allowed[team] / games_played[team] if games_played[team] else 0.0
            for team in teams
        }
        relative_sums = {team: 0.0 for team in teams}
        relative_counts = {team: 0 for team in teams}

        for home, away, home_score, away_score in scores:
            home_margin = relative_scoring_margin(
                points_for=home_score,
                points_allowed=away_score,
                opponent_avg_scored=avg_scored[away],
                opponent_avg_allowed=avg_allowed[away],
            )
            away_margin = relative_scoring_margin(
                points_for=away_score,
                points_allowed=home_score,
                opponent_avg_scored=avg_scored[home],
                opponent_avg_allowed=avg_allowed[home],
            )
            relative_sums[home] += home_margin
            relative_counts[home] += 1
            relative_sums[away] += away_margin
            relative_counts[away] += 1

        for team in teams:
            if relative_counts[team]:
                margins[team][sim] = relative_sums[team] / relative_counts[team]
    return margins


def rank_desc(values_by_team: dict[str, float], teams: list[str]) -> dict[str, int]:
    ordered = sorted(teams, key=lambda team: (-values_by_team.get(team, 0.0), team))
    return {team: rank + 1 for rank, team in enumerate(ordered)}


def has_explicit_fbs_championship_games(games: list[dict[str, Any]]) -> bool:
    for game in games:
        notes = str(game.get("notes") or "").lower()
        if "championship" not in notes:
            continue
        if not game.get("conferencegame"):
            continue
        if game.get("homeconference") != game.get("awayconference"):
            continue
        if game.get("homeclassification") == "fbs" and game.get("awayclassification") == "fbs":
            return True
    return False


def build_season_rows(
    *,
    games: list[dict[str, Any]],
    ranking_projection: dict[str, dict[str, float | int | None]],
    season: int,
    run_date: str,
    run_type: str,
    season_prediction_run_id: str,
    created_at: str,
    model_version: str,
    simulations: int,
    seed: int,
    notes: str | None,
) -> list[dict[str, Any]]:
    profiles = team_profiles(games)
    teams = sorted(profiles)
    fbs_teams = [team for team in teams if profiles[team].get("classification") == "fbs"]
    conferences = sorted(
        {
            profiles[team].get("conference")
            for team in fbs_teams
            if profiles[team].get("conference") and profiles[team].get("conference") not in INDEPENDENT_CONFERENCES
        }
    )

    rng = random.Random(seed)
    wins = {team: [0] * simulations for team in teams}
    losses = {team: [0] * simulations for team in teams}
    adjusted_total_wins = {team: [0] * simulations for team in teams}
    fcs_wins_counted = {team: [0] * simulations for team in teams}
    fbs_wins = {team: [0] * simulations for team in teams}
    fbs_losses = {team: [0] * simulations for team in teams}
    conference_wins = {team: [0] * simulations for team in teams}
    conference_losses = {team: [0] * simulations for team in teams}
    divisional_wins = {team: [0] * simulations for team in teams}
    divisional_losses = {team: [0] * simulations for team in teams}
    expected_wins_base = {team: 0.0 for team in teams}
    expected_games_base = {team: 0.0 for team in teams}
    opponents = {team: [] for team in teams}
    remaining_opponents = {team: [] for team in teams}
    final_conference_week_by_conference: dict[str, int] = {}
    for game in games:
        if (
            game.get("conferencegame")
            and game.get("homeconference")
            and game.get("homeconference") == game.get("awayconference")
            and game.get("week") is not None
        ):
            conference = game["homeconference"]
            final_conference_week_by_conference[conference] = max(
                final_conference_week_by_conference.get(conference, 0),
                int(game["week"]),
            )
    final_conference_week_losses = {team: [0] * simulations for team in teams}
    head_to_head_wins_by_sim: list[dict[str, dict[str, int]]] = [defaultdict(dict) for _ in range(simulations)]
    conference_game_scores_by_sim: list[list[tuple[str, str, float, float]]] = [
        [] for _ in range(simulations)
    ]

    for game in games:
        home = game["hometeam"]
        away = game["awayteam"]
        if home not in wins or away not in wins:
            continue
        homepoints = game.get("homepoints")
        awaypoints = game.get("awaypoints")
        homewinprob = game.get("homewinprob")
        if homewinprob is None:
            homewinprob = 0.5
        homewinprob = min(max(float(homewinprob), 0.0), 1.0)
        completed = homepoints is not None and awaypoints is not None

        expected_home_win = 1.0 if completed and homepoints > awaypoints else homewinprob
        if completed and homepoints <= awaypoints:
            expected_home_win = 0.0

        expected_wins_base[home] += expected_home_win
        expected_wins_base[away] += 1.0 - expected_home_win
        expected_games_base[home] += 1.0
        expected_games_base[away] += 1.0
        opponents[home].append(away)
        opponents[away].append(home)
        if not completed:
            remaining_opponents[home].append(away)
            remaining_opponents[away].append(home)

        same_conference_game = (
            game.get("conferencegame")
            and game.get("homeconference")
            and game.get("homeconference") == game.get("awayconference")
        )

        for sim in range(simulations):
            home_wins_game = homepoints > awaypoints if completed else rng.random() < homewinprob
            home_score, away_score = simulated_score(game, home_wins_game)
            if home_wins_game:
                wins[home][sim] += 1
                losses[away][sim] += 1
                winner = home
                loser = away
                defeated_classification = game.get("awayclassification")
            else:
                wins[away][sim] += 1
                losses[home][sim] += 1
                winner = away
                loser = home
                defeated_classification = game.get("homeclassification")

            if game.get("homeclassification") == "fbs" and game.get("awayclassification") == "fbs":
                fbs_wins[winner][sim] += 1
                fbs_losses[loser][sim] += 1

            if defeated_classification == "fcs":
                if fcs_wins_counted[winner][sim] == 0:
                    adjusted_total_wins[winner][sim] += 1
                fcs_wins_counted[winner][sim] += 1
            else:
                adjusted_total_wins[winner][sim] += 1

            if same_conference_game:
                is_final_conference_week = (
                    game.get("week") is not None
                    and final_conference_week_by_conference.get(game["homeconference"]) == int(game["week"])
                )
                if home_wins_game:
                    conference_wins[home][sim] += 1
                    conference_losses[away][sim] += 1
                    if is_final_conference_week:
                        final_conference_week_losses[away][sim] = 1
                    home_division = sun_belt_division_for_team(home)
                    away_division = sun_belt_division_for_team(away)
                    if home_division is not None and home_division == away_division:
                        divisional_wins[home][sim] += 1
                        divisional_losses[away][sim] += 1
                    head_to_head_wins_by_sim[sim].setdefault(home, {})
                    head_to_head_wins_by_sim[sim][home][away] = head_to_head_wins_by_sim[sim][home].get(away, 0) + 1
                else:
                    conference_wins[away][sim] += 1
                    conference_losses[home][sim] += 1
                    if is_final_conference_week:
                        final_conference_week_losses[home][sim] = 1
                    home_division = sun_belt_division_for_team(home)
                    away_division = sun_belt_division_for_team(away)
                    if home_division is not None and home_division == away_division:
                        divisional_wins[away][sim] += 1
                        divisional_losses[home][sim] += 1
                    head_to_head_wins_by_sim[sim].setdefault(away, {})
                    head_to_head_wins_by_sim[sim][away][home] = head_to_head_wins_by_sim[sim][away].get(home, 0) + 1
                conference_game_scores_by_sim[sim].append((home, away, home_score, away_score))

    team_strength = {
        team: expected_wins_base[team] / expected_games_base[team] if expected_games_base[team] else 0.5
        for team in teams
    }
    strength_of_schedule = {
        team: safe_mean([team_strength[opponent] for opponent in opponents[team]])
        for team in teams
    }
    remaining_strength_of_schedule = {
        team: safe_mean([team_strength[opponent] for opponent in remaining_opponents[team]])
        for team in teams
    }
    sec_relative_scoring_margin = build_sec_relative_scoring_margins(
        conference_game_scores_by_sim,
        teams,
    )
    sos_for_rank = {
        team: strength_of_schedule[team] if strength_of_schedule[team] is not None else 0.5
        for team in teams
    }
    regular_season_projected_wins = {
        team: sum(wins[team]) / simulations
        for team in teams
    }
    regular_season_projected_conference_wins = {
        team: sum(conference_wins[team]) / simulations
        for team in teams
    }
    projected_cfp_rank_for_tiebreakers = rank_desc(
        {
            team: (
                regular_season_projected_wins[team] * 100.0
                + regular_season_projected_conference_wins[team] * 8.0
                + team_strength[team] * 5.0
                + sos_for_rank[team] * 2.0
            )
            for team in fbs_teams
        },
        fbs_teams,
    )

    total_wins = {team: values[:] for team, values in wins.items()}
    total_losses = {team: values[:] for team, values in losses.items()}
    championship_game_counts = {team: 0 for team in teams}
    conference_champion_counts = {team: 0 for team in teams}
    conference_champions_by_sim = [set() for _ in range(simulations)]
    add_synthetic_championship = not has_explicit_fbs_championship_games(games)

    for conference in conferences:
        conference_teams = [
            team
            for team in fbs_teams
            if profiles[team].get("conference") == conference
        ]
        if len(conference_teams) < 2:
            continue

        for sim in range(simulations):
            team_records = []
            for team in conference_teams:
                team_records.append(
                    {
                        "team": team,
                        "conference": conference,
                        "division": sun_belt_division_for_team(team),
                        "conference_wins": conference_wins[team][sim],
                        "conference_losses": conference_losses[team][sim],
                        "divisional_wins": divisional_wins[team][sim],
                        "divisional_losses": divisional_losses[team][sim],
                        "overall_wins": wins[team][sim],
                        "overall_losses": losses[team][sim],
                        "fbs_wins": fbs_wins[team][sim],
                        "fbs_losses": fbs_losses[team][sim],
                        "adjusted_total_wins": adjusted_total_wins[team][sim],
                        "team_strength": team_strength[team],
                        "team_rating_score": team_strength[team],
                        "computer_composite_score": team_strength[team],
                        "computer_composite_rank": projected_cfp_rank_for_tiebreakers.get(team),
                        "cfp_rank": projected_cfp_rank_for_tiebreakers.get(team),
                        "lost_final_conference_week": bool(final_conference_week_losses[team][sim]),
                        "head_to_head_wins": head_to_head_wins_by_sim[sim],
                        "sec_relative_scoring_margin": sec_relative_scoring_margin[team][sim],
                    }
                )
            first, second = select_conference_championship_teams(conference, team_records, rng)
            championship_game_counts[first] += 1
            championship_game_counts[second] += 1

            team_metrics = {
                team: {
                    "team_strength": team_strength[team],
                    "overall_win_pct": wins[team][sim] / max(wins[team][sim] + losses[team][sim], 1),
                    "conference_win_pct": conference_wins[team][sim]
                    / max(conference_wins[team][sim] + conference_losses[team][sim], 1),
                    "strength_of_schedule": strength_of_schedule[team],
                }
                for team in conference_teams
            }
            champion = simulate_conference_championship_game(first, second, team_metrics, rng)
            runner_up = second if champion == first else first
            conference_champion_counts[champion] += 1
            conference_champions_by_sim[sim].add(champion)

            if add_synthetic_championship:
                total_wins[champion][sim] += 1
                total_losses[runner_up][sim] += 1

    playoff_counts = {team: 0 for team in teams}
    cfp_bye_counts = {team: 0 for team in teams}
    cfp_at_large_counts = {team: 0 for team in teams}
    cfp_auto_bid_counts = {team: 0 for team in teams}
    national_championship_game_counts = {team: 0 for team in teams}
    national_champion_counts = {team: 0 for team in teams}

    projected_wins = {team: sum(total_wins[team]) / simulations for team in teams}
    projected_conference_wins = {
        team: sum(conference_wins[team]) / simulations
        for team in teams
    }
    resume_score = {
        team: (
            projected_wins[team] * 100.0
            + projected_conference_wins[team] * 8.0
            + team_strength[team] * 5.0
            + sos_for_rank[team] * 2.0
        )
        for team in teams
    }

    for sim in range(simulations):
        ranking_scores = {
            team: cfp_selection_score(
                team,
                sim=sim,
                total_wins=total_wins,
                conference_wins=conference_wins,
                conference_champions=conference_champions_by_sim[sim],
                ranking_projection=ranking_projection,
                projected_cfp_rank_for_tiebreakers=projected_cfp_rank_for_tiebreakers,
                resume_score=resume_score,
                team_strength=team_strength,
                sos_for_rank=sos_for_rank,
            )
            for team in fbs_teams
        }
        ranked_fbs = sorted(fbs_teams, key=lambda team: (-ranking_scores[team], team))
        seeded, auto_bids, at_large = select_cfp_field(
            ranked_fbs=ranked_fbs,
            conference_champions=conference_champions_by_sim[sim],
            profiles=profiles,
            ranking_scores=ranking_scores,
        )
        playoff = seeded
        bye_teams = seeded[:4]

        for team in auto_bids:
            cfp_auto_bid_counts[team] += 1
        for team in at_large:
            cfp_at_large_counts[team] += 1
        for team in playoff:
            playoff_counts[team] += 1
        for team in bye_teams:
            cfp_bye_counts[team] += 1

        national_championship_teams, national_champion = simulate_cfp_bracket(
            seeded,
            ranking_scores=ranking_scores,
            team_strength=team_strength,
            total_wins=total_wins,
            sim=sim,
            rng=rng,
        )
        for team in national_championship_teams:
            national_championship_game_counts[team] += 1
        if national_champion:
            national_champion_counts[national_champion] += 1

    ap_ranks = rank_desc(
        {
            team: (
                ranking_projection_score(
                    team,
                    ranking_projection=ranking_projection,
                    fallback_rank=projected_cfp_rank_for_tiebreakers.get(team),
                    fallback_score=resume_score[team] / 10.0,
                )
                + playoff_counts[team] / simulations * 5.0
            )
            for team in fbs_teams
        },
        fbs_teams,
    )
    cfp_ranks = rank_desc(
        {
            team: (
                ranking_projection_score(
                    team,
                    ranking_projection=ranking_projection,
                    fallback_rank=projected_cfp_rank_for_tiebreakers.get(team),
                    fallback_score=resume_score[team] / 10.0,
                )
                + playoff_counts[team] / simulations * 20.0
                + cfp_bye_counts[team] / simulations * 10.0
            )
            for team in fbs_teams
        },
        fbs_teams,
    )
    resume_ranks = rank_desc({team: resume_score[team] for team in fbs_teams}, fbs_teams)

    rows: list[dict[str, Any]] = []
    for team in fbs_teams:
        win_buckets = [0] * (MAX_WIN_BUCKET + 1)
        for value in total_wins[team]:
            win_buckets[min(value, MAX_WIN_BUCKET)] += 1

        row = {
            "season_prediction_run_id": season_prediction_run_id,
            "season": season,
            "run_date": run_date,
            "run_type": run_type,
            "model_version": model_version,
            "team": team,
            "conference": profiles[team].get("conference"),
            "division": sun_belt_division_for_team(team),
            "classification": profiles[team].get("classification"),
            "created_at": created_at,
            "projected_wins": projected_wins[team],
            "projected_losses": sum(total_losses[team]) / simulations,
            "projected_conference_wins": projected_conference_wins[team],
            "projected_conference_losses": sum(conference_losses[team]) / simulations,
            "conference_championship_game_prob": championship_game_counts[team] / simulations,
            "conference_champion_prob": conference_champion_counts[team] / simulations,
            "playoff_prob": playoff_counts[team] / simulations,
            "cfp_bye_prob": cfp_bye_counts[team] / simulations,
            "cfp_at_large_prob": cfp_at_large_counts[team] / simulations,
            "cfp_auto_bid_prob": cfp_auto_bid_counts[team] / simulations,
            "national_championship_game_prob": national_championship_game_counts[team] / simulations,
            "national_champion_prob": national_champion_counts[team] / simulations,
            "bowl_eligible_prob": sum(1 for value in wins[team] if value >= 6) / simulations,
            "projected_ap_ranking": ap_ranks[team],
            "projected_cfp_ranking": cfp_ranks[team],
            "resume_ranking": resume_ranks[team],
            "strength_of_schedule": strength_of_schedule[team],
            "remaining_strength_of_schedule": remaining_strength_of_schedule[team],
            "expected_number_of_wins": projected_wins[team],
            "simulations": simulations,
            "prediction_type": "FBS",
            "notes": notes,
        }
        for wins_bucket, count in enumerate(win_buckets):
            row[f"probability_{wins_bucket}_wins"] = count / simulations
        rows.append(row)

    rows, _ = add_prediction_hashes(rows)
    return rows


def main() -> None:
    prediction_rows = read_csv(GAME_PREDICTIONS_FULL_CSV)
    default_season = infer_latest_season(prediction_rows)

    parser = argparse.ArgumentParser(description="Build local season_predictions_full CSV from local snapshots.")
    parser.add_argument("--season", type=int, default=default_season)
    parser.add_argument("--game-prediction-run-id", default=None)
    parser.add_argument("--ranking-projection-run-id", default=None)
    parser.add_argument("--rankings-projection-csv", type=Path, default=RANKINGS_PROJECTION_FULL_CSV)
    parser.add_argument("--season-prediction-run-id", default=None)
    parser.add_argument("--run-date", default=None)
    parser.add_argument("--run-type", default="local")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    if args.simulations <= 0:
        raise ValueError("--simulations must be positive.")

    game_prediction_run_id, selected_predictions, inferred_run_date = select_game_prediction_run(
        prediction_rows,
        season=args.season,
        game_prediction_run_id=args.game_prediction_run_id,
    )
    run_date = args.run_date or inferred_run_date or datetime.now(timezone.utc).date().isoformat()
    season_prediction_run_id = args.season_prediction_run_id or str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    seed = args.seed if args.seed is not None else args.season

    game_data_rows = read_csv(GAME_DATA_CSV)
    ranking_projection_rows = read_csv(args.rankings_projection_csv)
    ranking_projection_run_id, selected_ranking_rows = select_ranking_projection_rows(
        ranking_projection_rows,
        season=args.season,
        ranking_projection_run_id=args.ranking_projection_run_id,
    )
    ranking_projection = ranking_projection_lookup(selected_ranking_rows)
    games = build_game_records(
        game_data_rows=game_data_rows,
        prediction_rows=selected_predictions,
        season=args.season,
        run_date=run_date,
    )
    rows = build_season_rows(
        games=games,
        ranking_projection=ranking_projection,
        season=args.season,
        run_date=run_date,
        run_type=args.run_type,
        season_prediction_run_id=season_prediction_run_id,
        created_at=created_at,
        model_version=model_version_label(selected_predictions),
        simulations=args.simulations,
        seed=seed,
        notes=args.notes,
    )

    write_csv(args.output, rows)
    print(f"Selected game_prediction_run_id: {game_prediction_run_id}")
    print(f"Selected ranking_projection_run_id: {ranking_projection_run_id or 'none'}")
    print(f"Run date: {run_date}")
    print(f"Simulations: {args.simulations}")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
