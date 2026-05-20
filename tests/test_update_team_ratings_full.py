import sys
import types
import unittest
from datetime import date

import pandas as pd


sys.modules.setdefault("psycopg", types.ModuleType("psycopg"))

from etl.jobs.prediction_updates import update_team_ratings_full as job  # noqa: E402


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.cursor_obj = FakeCursor(rows)

    def cursor(self):
        return self.cursor_obj


class TeamRatingsFullTests(unittest.TestCase):
    def test_normal_run_types_share_duplicate_pool(self):
        self.assertEqual(job._comparable_run_types("manual"), ("manual", "nightly"))
        self.assertEqual(job._comparable_run_types("nightly"), ("manual", "nightly"))
        self.assertEqual(job._comparable_run_types("backfill"), ("backfill",))

    def test_latest_successful_run_checks_manual_and_nightly(self):
        conn = FakeConnection([("existing-run-id", "existing-hash")])

        result = job.get_latest_successful_run(conn, 2026, "nightly")

        self.assertEqual(result, ("existing-run-id", "existing-hash"))
        _, params = conn.cursor_obj.executed[0]
        self.assertEqual(params, (2026, "manual", "nightly"))

    def test_hash_excludes_run_metadata_and_upstream_run_id(self):
        record = {
            "season": 2026,
            "run_date": date(2026, 9, 15),
            "run_type": "nightly",
            "model_version": "team_rating_srs_2026+game_model",
            "team": "Example State",
            "conference": "Example",
            "classification": "fbs",
            "rank": 1,
            "team_rating": 12.34561,
            "power_rating": 12.34561,
            "home_field_advantage": 2.25,
            "completed_games": 2,
            "projected_games": 1,
            "total_games": 3,
            "average_margin_signal": 10.0,
            "average_weighted_margin_signal": 8.0,
            "completed_game_weight": 1.0,
            "projected_game_weight": 0.45,
            "max_margin_signal": 42.0,
            "margin_source": "completed+spread",
            "game_prediction_run_id": "run-a",
            "notes": "first",
        }
        changed_metadata = dict(
            record,
            run_date=date(2026, 9, 16),
            run_type="manual",
            game_prediction_run_id="run-b",
            notes="second",
        )

        _, original_hash = job.add_rating_hashes([record])
        _, metadata_hash = job.add_rating_hashes([changed_metadata])

        self.assertEqual(original_hash, metadata_hash)

    def test_build_team_ratings_uses_completed_and_projected_margins(self):
        games = pd.DataFrame(
            [
                {
                    "gameid": "1",
                    "season": 2026,
                    "week": 1,
                    "gamedate": date(2026, 9, 1),
                    "hometeam": "Alpha",
                    "awayteam": "Beta",
                    "homeclassification": "fbs",
                    "awayclassification": "fbs",
                    "homeconference": "A",
                    "awayconference": "B",
                    "homepoints": 30,
                    "awaypoints": 20,
                    "homespread": None,
                    "homewinprob": None,
                },
                {
                    "gameid": "2",
                    "season": 2026,
                    "week": 2,
                    "gamedate": date(2026, 9, 20),
                    "hometeam": "Gamma",
                    "awayteam": "Alpha",
                    "homeclassification": "fbs",
                    "awayclassification": "fbs",
                    "homeconference": "C",
                    "awayconference": "A",
                    "homepoints": None,
                    "awaypoints": None,
                    "homespread": 3.5,
                    "homewinprob": 0.45,
                },
            ]
        )

        records, metadata = job.build_team_rating_records(
            season=2026,
            run_date=date(2026, 9, 10),
            run_type="nightly",
            model_version="team_rating_srs_2026+game_model",
            game_prediction_run_id="game-run",
            games=games,
            notes=None,
        )

        by_team = {record["team"]: record for record in records}
        self.assertEqual(set(by_team), {"Alpha", "Beta", "Gamma"})
        self.assertEqual(by_team["Alpha"]["completed_games"], 1)
        self.assertEqual(by_team["Alpha"]["projected_games"], 1)
        self.assertEqual(metadata["completed_game_count"], 1)
        self.assertEqual(metadata["projected_game_count"], 1)
        self.assertEqual(metadata["dropped_game_count"], 0)
        self.assertEqual(metadata["margin_source"], "completed+spread")
        self.assertTrue(all(record["home_field_advantage"] is not None for record in records))

    def test_same_day_final_score_is_not_completed_as_of_run_date(self):
        games = pd.DataFrame(
            [
                {
                    "gameid": "1",
                    "season": 2026,
                    "week": 1,
                    "gamedate": date(2026, 9, 10),
                    "hometeam": "Alpha",
                    "awayteam": "Beta",
                    "homeclassification": "fbs",
                    "awayclassification": "fbs",
                    "homeconference": "A",
                    "awayconference": "B",
                    "homepoints": 30,
                    "awaypoints": 20,
                    "homespread": -6.0,
                    "homewinprob": 0.7,
                },
            ]
        )

        records, metadata = job.build_team_rating_records(
            season=2026,
            run_date=date(2026, 9, 10),
            run_type="nightly",
            model_version="team_rating_srs_2026+game_model",
            game_prediction_run_id="game-run",
            games=games,
            notes=None,
        )

        self.assertEqual(metadata["completed_game_count"], 0)
        self.assertEqual(metadata["projected_game_count"], 1)
        self.assertEqual(records[0]["margin_source"], "spread")


if __name__ == "__main__":
    unittest.main()
