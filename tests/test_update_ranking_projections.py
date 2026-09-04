import unittest
from datetime import date

import pandas as pd

from etl.jobs.prediction_updates import update_ranking_projections as job  # noqa: E402


class RankingProjectionTests(unittest.TestCase):
    def test_game_snapshot_uses_predictions_for_same_day_games(self):
        game_data = pd.DataFrame(
            [
                {
                    "gameid": "prior-final",
                    "season": 2026,
                    "week": 1,
                    "gamedate": date(2026, 9, 3),
                    "seasontype": "regular",
                    "conferencegame": False,
                    "hometeam": "Alpha",
                    "awayteam": "Beta",
                    "homeclassification": "fbs",
                    "awayclassification": "fbs",
                    "homeconference": "A",
                    "awayconference": "B",
                    "homepoints": 31,
                    "awaypoints": 17,
                },
                {
                    "gameid": "same-day-final",
                    "season": 2026,
                    "week": 1,
                    "gamedate": date(2026, 9, 4),
                    "seasontype": "regular",
                    "conferencegame": False,
                    "hometeam": "Gamma",
                    "awayteam": "Delta",
                    "homeclassification": "fbs",
                    "awayclassification": "fbs",
                    "homeconference": "C",
                    "awayconference": "D",
                    "homepoints": 28,
                    "awaypoints": 24,
                },
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "gameid": "same-day-final",
                    "homewinprob": 0.73,
                },
            ]
        )

        snapshot = job.build_game_snapshot(
            game_data=game_data,
            predictions=predictions,
            run_date=date(2026, 9, 4),
        )

        by_game = snapshot.set_index("gameid")
        self.assertTrue(bool(by_game.loc["prior-final", "completed_before_run"]))
        self.assertEqual(by_game.loc["prior-final", "effective_homewinprob"], 1.0)
        self.assertFalse(bool(by_game.loc["same-day-final", "completed_before_run"]))
        self.assertEqual(by_game.loc["same-day-final", "effective_homewinprob"], 0.73)
        self.assertTrue(pd.isna(by_game.loc["same-day-final", "homepoints"]))
        self.assertTrue(pd.isna(by_game.loc["same-day-final", "awaypoints"]))

    def test_game_snapshot_requires_predictions_for_games_not_completed_before_run(self):
        game_data = pd.DataFrame(
            [
                {
                    "gameid": "same-day-unplayed",
                    "season": 2026,
                    "week": 1,
                    "gamedate": date(2026, 9, 4),
                    "seasontype": "regular",
                    "conferencegame": False,
                    "hometeam": "Alpha",
                    "awayteam": "Beta",
                    "homeclassification": "fbs",
                    "awayclassification": "fbs",
                    "homeconference": "A",
                    "awayconference": "B",
                    "homepoints": None,
                    "awaypoints": None,
                },
            ]
        )
        predictions = pd.DataFrame(columns=["gameid", "homewinprob"])

        with self.assertRaisesRegex(RuntimeError, "same-day-unplayed"):
            job.build_game_snapshot(
                game_data=game_data,
                predictions=predictions,
                run_date=date(2026, 9, 4),
            )


if __name__ == "__main__":
    unittest.main()
