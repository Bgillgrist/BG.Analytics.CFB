import unittest
from datetime import date

import pandas as pd

from etl.jobs.prediction_updates import update_game_predictions_full as job  # noqa: E402


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


class GamePredictionFullTests(unittest.TestCase):
    def test_hash_excludes_final_scores(self):
        record = {
            "gameid": "401869142",
            "season": 2026,
            "week": 1,
            "home_team": "Louisiana",
            "away_team": "Lamar",
            "homepoints": None,
            "awaypoints": None,
            "homespread": -41.81696448943938,
            "awayspread": 41.81696448943938,
            "totalpred": 56.74207176702493,
            "homewinprob": 0.9923949295190474,
            "awaywinprob": 0.007605070480952603,
            "model_version": "incomplete_2026",
            "prediction_type": "FCS",
        }
        scored_record = dict(record, homepoints=45.0, awaypoints=7.0)

        _, original_hash = job.add_prediction_hashes([record])
        _, scored_hash = job.add_prediction_hashes([scored_record])

        self.assertEqual(original_hash, scored_hash)

    def test_hash_ignores_sub_four_decimal_float_changes(self):
        record = {
            "gameid": "401869142",
            "season": 2026,
            "week": 1,
            "home_team": "Louisiana",
            "away_team": "Lamar",
            "homepoints": None,
            "awaypoints": None,
            "homespread": -41.816941,
            "awayspread": 41.816941,
            "totalpred": 56.742041,
            "homewinprob": 0.992341,
            "awaywinprob": 0.007659,
            "model_version": "incomplete_2026",
            "prediction_type": "FCS",
        }
        tiny_change_record = dict(
            record,
            homespread=-41.816939,
            awayspread=41.816939,
            totalpred=56.742039,
            homewinprob=0.992339,
            awaywinprob=0.007661,
        )

        _, original_hash = job.add_prediction_hashes([record])
        _, tiny_change_hash = job.add_prediction_hashes([tiny_change_record])

        self.assertEqual(original_hash, tiny_change_hash)

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

    def test_normal_snapshot_keeps_games_not_completed_before_run_date(self):
        preds = pd.DataFrame(
            [
                {
                    "id": "prior-final",
                    "gamedate": date(2026, 9, 3),
                    "homepoints": 31,
                    "awaypoints": 17,
                },
                {
                    "id": "prior-missing-score",
                    "gamedate": date(2026, 9, 3),
                    "homepoints": None,
                    "awaypoints": None,
                },
                {
                    "id": "same-day-final",
                    "gamedate": date(2026, 9, 4),
                    "homepoints": 28,
                    "awaypoints": 24,
                },
                {
                    "id": "same-day-unplayed",
                    "gamedate": date(2026, 9, 4),
                    "homepoints": None,
                    "awaypoints": None,
                },
                {
                    "id": "future",
                    "gamedate": date(2026, 9, 5),
                    "homepoints": None,
                    "awaypoints": None,
                },
            ]
        )

        filtered = job.filter_predictions_for_run_type(preds, "nightly", date(2026, 9, 4))

        self.assertEqual(
            filtered["id"].tolist(),
            [
                "prior-missing-score",
                "same-day-final",
                "same-day-unplayed",
                "future",
            ],
        )

    def test_stored_detail_rows_allow_legacy_hash_comparison(self):
        row = (
            "401869142",
            2026,
            1,
            "Louisiana",
            "Lamar",
            None,
            None,
            -41.81696448943938,
            41.81696448943938,
            56.74207176702493,
            0.9923949295190474,
            0.007605070480952603,
            "incomplete_2026",
            "FCS",
        )
        conn = FakeConnection([row])
        record = dict(zip(job.DETAIL_HASH_COLUMNS, row))
        prediction_hash = job._prediction_hash_from_records([record])

        self.assertTrue(
            job.prediction_hash_matches_run(
                conn,
                run_id="existing-run-id",
                stored_prediction_hash="legacy-score-including-hash",
                prediction_hash=prediction_hash,
            )
        )


if __name__ == "__main__":
    unittest.main()
