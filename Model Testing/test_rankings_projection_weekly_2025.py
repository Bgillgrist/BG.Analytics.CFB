#!/usr/bin/env python3
"""
Weekly 2025 rankings projection backtest.

This creates one row per FBS team per ranking week, with:
  - projected AP/CFP rankings from the local rankings framework
  - actual AP/CFP rankings from rankings.csv for that same week

The projection for week N only uses game results through week N - 1 and poll
inertia through week N - 1. Week 1 is therefore a preseason/profile-only
projection.
"""

from __future__ import annotations

import argparse
import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import test_rankings_projection_local as ranking_model


SEASON = 2025
DEFAULT_OUTPUT_CSV = ranking_model.MODEL_DIR / "rankings_projection_weekly_2025.csv"

OUTPUT_COLUMNS = [
    "weekly_ranking_projection_run_id",
    "season",
    "week",
    "projection_as_of_week",
    "team",
    "conference",
    "classification",
    "projected_ap_ranking",
    "actual_ap_ranking",
    "projected_ap_rank_error",
    "projected_cfp_ranking",
    "actual_cfp_ranking",
    "projected_cfp_rank_error",
    "projected_end_ap_ranking",
    "projected_end_cfp_ranking",
    "projected_ap_score",
    "projected_cfp_score",
    "resume_score",
    "projected_resume_score",
    "power_score",
    "poll_inertia_score",
    "current_wins",
    "current_losses",
    "projected_wins",
    "projected_losses",
    "strength_of_schedule",
    "remaining_strength_of_schedule",
    "current_ap_rank_input",
    "current_cfp_rank_input",
    "created_at",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: "" if row.get(col) is None else row.get(col) for col in OUTPUT_COLUMNS})


def actual_rankings_for_week(
    rankings_rows: list[dict[str, str]],
    *,
    season: int,
    week: int,
    poll_names: set[str],
) -> dict[str, int]:
    return {
        row["school"]: ranking_model.parse_int(row.get("rank"))
        for row in rankings_rows
        if ranking_model.parse_int(row.get("season")) == season
        and ranking_model.parse_int(row.get("week")) == week
        and row.get("poll") in poll_names
        and ranking_model.parse_int(row.get("rank")) is not None
    }


def ranking_weeks(rankings_rows: list[dict[str, str]], *, season: int) -> list[int]:
    weeks = {
        ranking_model.parse_int(row.get("week"))
        for row in rankings_rows
        if ranking_model.parse_int(row.get("season")) == season
        and row.get("poll") in ranking_model.AP_POLL_NAMES | ranking_model.CFP_POLL_NAMES
    }
    return sorted(week for week in weeks if week is not None)


def game_as_of_week(row: dict[str, str], *, projection_week: int) -> dict[str, Any]:
    game_week = ranking_model.parse_int(row.get("week"))
    completed_for_projection = (
        game_week is not None
        and game_week < projection_week
        and ranking_model.parse_float(row.get("homepoints")) is not None
        and ranking_model.parse_float(row.get("awaypoints")) is not None
    )
    homepoints = ranking_model.parse_float(row.get("homepoints")) if completed_for_projection else None
    awaypoints = ranking_model.parse_float(row.get("awaypoints")) if completed_for_projection else None
    return {
        "gameid": row.get("id"),
        "season": ranking_model.parse_int(row.get("season")),
        "week": game_week,
        "seasontype": str(row.get("seasontype", "regular") or "regular").lower(),
        "startdate": row.get("startdate"),
        "conferencegame": ranking_model.is_truthy(row.get("conferencegame")),
        "hometeam": row.get("hometeam"),
        "awayteam": row.get("awayteam"),
        "homeclassification": str(row.get("homeclassification", "") or "").lower(),
        "awayclassification": str(row.get("awayclassification", "") or "").lower(),
        "homeconference": row.get("homeconference"),
        "awayconference": row.get("awayconference"),
        "homepoints": homepoints,
        "awaypoints": awaypoints,
        "pred_homepoints": homepoints,
        "pred_awaypoints": awaypoints,
        "homewinprob": None,
    }


def games_for_week(game_data_rows: list[dict[str, str]], *, season: int, projection_week: int) -> list[dict[str, Any]]:
    games = [
        game_as_of_week(row, projection_week=projection_week)
        for row in game_data_rows
        if ranking_model.parse_int(row.get("season")) == season
        and str(row.get("seasontype", "")).lower() == "regular"
        and row.get("hometeam")
        and row.get("awayteam")
    ]
    if not games:
        raise RuntimeError(f"No regular-season games found for season {season}.")
    return games


def rank_error(projected: Any, actual: Any) -> int | None:
    projected_rank = ranking_model.parse_int(projected)
    actual_rank = ranking_model.parse_int(actual)
    if projected_rank is None or actual_rank is None:
        return None
    return projected_rank - actual_rank


def build_weekly_rows(*, season: int, output_run_id: str, created_at: str) -> list[dict[str, Any]]:
    game_data_rows = ranking_model.read_csv(ranking_model.GAME_DATA_CSV)
    rankings_rows = ranking_model.read_csv(ranking_model.RANKINGS_CSV, required=False)
    all_advanced_rows = ranking_model.read_csv(ranking_model.TEAM_ADVANCED_SEASON_STATS_CSV, required=False)
    advanced_rows = [
        row
        for row in all_advanced_rows
        if (ranking_model.parse_int(row.get("season")) or 0) < season
    ]
    recruiting_rows = ranking_model.read_csv(ranking_model.TEAM_RECRUITING_RANKINGS_CSV, required=False)
    talent_rows = ranking_model.read_csv(ranking_model.TEAM_TALENT_COMPOSITE_CSV, required=False)
    returning_rows = ranking_model.read_csv(ranking_model.TEAM_RETURNING_PRODUCTION_CSV, required=False)

    rows: list[dict[str, Any]] = []
    for week in ranking_weeks(rankings_rows, season=season):
        poll_through_week = week - 1
        games = games_for_week(game_data_rows, season=season, projection_week=week)
        projected_rows = ranking_model.build_ranking_rows(
            season=season,
            run_date=f"{season}-12-31",
            run_type="weekly_backtest",
            ranking_projection_run_id=output_run_id,
            created_at=created_at,
            model_version="rankings_projection_weekly_backtest",
            game_prediction_run_id=None,
            games=games,
            rankings_rows=rankings_rows,
            advanced_rows=advanced_rows,
            recruiting_rows=recruiting_rows,
            talent_rows=talent_rows,
            returning_rows=returning_rows,
            notes=f"Projection for ranking week {week}; inputs through week {poll_through_week}.",
            poll_through_week=poll_through_week,
        )
        actual_ap = actual_rankings_for_week(
            rankings_rows,
            season=season,
            week=week,
            poll_names=ranking_model.AP_POLL_NAMES,
        )
        actual_cfp = actual_rankings_for_week(
            rankings_rows,
            season=season,
            week=week,
            poll_names=ranking_model.CFP_POLL_NAMES,
        )

        for row in projected_rows:
            team = row["team"]
            rows.append(
                {
                    "weekly_ranking_projection_run_id": output_run_id,
                    "season": season,
                    "week": week,
                    "projection_as_of_week": poll_through_week,
                    "team": team,
                    "conference": row.get("conference"),
                    "classification": row.get("classification"),
                    "projected_ap_ranking": row.get("projected_ap_ranking"),
                    "actual_ap_ranking": actual_ap.get(team),
                    "projected_ap_rank_error": rank_error(row.get("projected_ap_ranking"), actual_ap.get(team)),
                    "projected_cfp_ranking": row.get("projected_cfp_ranking"),
                    "actual_cfp_ranking": actual_cfp.get(team),
                    "projected_cfp_rank_error": rank_error(row.get("projected_cfp_ranking"), actual_cfp.get(team)),
                    "projected_end_ap_ranking": row.get("projected_end_ap_ranking"),
                    "projected_end_cfp_ranking": row.get("projected_end_cfp_ranking"),
                    "projected_ap_score": row.get("projected_ap_score"),
                    "projected_cfp_score": row.get("projected_cfp_score"),
                    "resume_score": row.get("resume_score"),
                    "projected_resume_score": row.get("projected_resume_score"),
                    "power_score": row.get("power_score"),
                    "poll_inertia_score": row.get("poll_inertia_score"),
                    "current_wins": row.get("current_wins"),
                    "current_losses": row.get("current_losses"),
                    "projected_wins": row.get("projected_wins"),
                    "projected_losses": row.get("projected_losses"),
                    "strength_of_schedule": row.get("strength_of_schedule"),
                    "remaining_strength_of_schedule": row.get("remaining_strength_of_schedule"),
                    "current_ap_rank_input": row.get("current_ap_rank"),
                    "current_cfp_rank_input": row.get("current_cfp_rank"),
                    "created_at": created_at,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weekly 2025 projected-vs-actual rankings CSV.")
    parser.add_argument("--season", type=int, default=SEASON)
    parser.add_argument("--weekly-ranking-projection-run-id", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    output_run_id = args.weekly_ranking_projection_run_id or str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    rows = build_weekly_rows(
        season=args.season,
        output_run_id=output_run_id,
        created_at=created_at,
    )
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
