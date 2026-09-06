import sys
import types
import unittest

import numpy as np
import pandas as pd


fake_xgboost = types.ModuleType("xgboost")
fake_xgboost.XGBClassifier = object
fake_xgboost.XGBRegressor = object
sys.modules.setdefault("xgboost", fake_xgboost)

from etl.jobs.prediction_updates import update_game_predictions as job  # noqa: E402


class FakeProbabilityModel:
    def __init__(self, probability):
        self.probability = probability
        self.row_count = 0

    def predict_proba(self, frame):
        self.row_count += len(frame)
        return np.column_stack(
            [
                np.full(len(frame), 1.0 - self.probability),
                np.full(len(frame), self.probability),
            ]
        )


class FakeRegressionModel:
    def __init__(self, value):
        self.value = value
        self.row_count = 0

    def predict(self, frame):
        self.row_count += len(frame)
        return np.full(len(frame), self.value, dtype=float)


def _model_bundle():
    return {
        "fbs_win_with_spread": FakeProbabilityModel(0.71),
        "fbs_win_no_spread": FakeProbabilityModel(0.61),
        "fbs_spread_with_spread": FakeRegressionModel(4.0),
        "fbs_spread_no_spread": FakeRegressionModel(2.0),
        "fbs_total_with_total": FakeRegressionModel(55.0),
        "fbs_total_no_total": FakeRegressionModel(49.0),
        "fcs_win_with_spread": FakeProbabilityModel(0.90),
        "fcs_win_no_spread": FakeProbabilityModel(0.80),
        "fcs_margin_with_spread": FakeRegressionModel(35.0),
        "fcs_margin_no_spread": FakeRegressionModel(21.0),
        "fcs_total_with_total": FakeRegressionModel(62.0),
        "fcs_total_no_total": FakeRegressionModel(44.0),
    }


def _attach_feature_attrs(frame):
    frame.attrs["fbs_spread_features"] = job.FBS_SPREAD_FEATURES
    frame.attrs["fbs_base_features"] = job.FBS_BASE_FEATURES
    frame.attrs["fbs_total_features"] = job.FBS_TOTAL_FEATURES
    frame.attrs["fcs_spread_features"] = job.FCS_SPREAD_FEATURES
    frame.attrs["fcs_base_features"] = job.FCS_BASE_FEATURES
    frame.attrs["fcs_total_features"] = job.FCS_TOTAL_FEATURES
    return frame


class XgbGamePredictionTests(unittest.TestCase):
    def test_score_current_season_selects_fbs_and_fcs_models_by_line_availability(self):
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "season": 2026,
                    "week": 1,
                    "hometeam": "A",
                    "awayteam": "B",
                    "homeclassification": "fbs",
                    "awayclassification": "fbs",
                    "homepoints": None,
                    "awaypoints": None,
                    "avg_spread": -3.5,
                    "avg_over_under": 50.5,
                    "neutralsite": False,
                },
                {
                    "id": 2,
                    "season": 2026,
                    "week": 1,
                    "hometeam": "C",
                    "awayteam": "D",
                    "homeclassification": "fbs",
                    "awayclassification": "fbs",
                    "homepoints": None,
                    "awaypoints": None,
                    "avg_spread": None,
                    "avg_over_under": None,
                    "neutralsite": False,
                },
                {
                    "id": 3,
                    "season": 2026,
                    "week": 1,
                    "hometeam": "E",
                    "awayteam": "FCS One",
                    "homeclassification": "fbs",
                    "awayclassification": "fcs",
                    "homepoints": None,
                    "awaypoints": None,
                    "avg_spread": -28.0,
                    "avg_over_under": 58.0,
                    "neutralsite": False,
                },
                {
                    "id": 4,
                    "season": 2026,
                    "week": 1,
                    "hometeam": "FCS Two",
                    "awayteam": "G",
                    "homeclassification": "fcs",
                    "awayclassification": "fbs",
                    "homepoints": None,
                    "awaypoints": None,
                    "avg_spread": None,
                    "avg_over_under": None,
                    "neutralsite": True,
                },
            ]
        )
        for col in [
            "home_teamrankings_rating",
            "away_teamrankings_rating",
            "home_talent",
            "away_talent",
            "home_recruiting_points",
            "away_recruiting_points",
            "home_returning_total_ppa",
            "away_returning_total_ppa",
            "home_returning_percent_ppa",
            "away_returning_percent_ppa",
        ]:
            df[col] = 1.0

        modeled = _attach_feature_attrs(job.prepare_modeling_dataframe(df))
        models = _model_bundle()

        self.assertNotIn("home_recruiting_points", job.FBS_BASE_FEATURES)
        self.assertNotIn("away_recruiting_points", job.FBS_BASE_FEATURES)
        self.assertNotIn("fbs_recruiting_points", job.FCS_BASE_FEATURES)
        self.assertIn("fbs_recruiting_points", modeled.columns)

        preds = job.score_current_season(models, modeled, 2026).set_index("id")

        self.assertAlmostEqual(preds.loc[1, "homewinprob"], 0.71)
        self.assertAlmostEqual(preds.loc[1, "homespread"], -4.0)
        self.assertAlmostEqual(preds.loc[1, "totalpred"], 55.0)
        self.assertAlmostEqual(preds.loc[2, "homewinprob"], 0.61)
        self.assertAlmostEqual(preds.loc[2, "homespread"], -2.0)
        self.assertAlmostEqual(preds.loc[2, "totalpred"], 49.0)

        self.assertAlmostEqual(preds.loc[3, "homewinprob"], 0.90)
        self.assertAlmostEqual(preds.loc[3, "awaywinprob"], 0.10)
        self.assertAlmostEqual(preds.loc[3, "homespread"], -35.0)
        self.assertAlmostEqual(preds.loc[3, "awayspread"], 35.0)
        self.assertAlmostEqual(preds.loc[3, "totalpred"], 62.0)

        self.assertAlmostEqual(preds.loc[4, "awaywinprob"], 0.80)
        self.assertAlmostEqual(preds.loc[4, "homewinprob"], 0.20)
        self.assertAlmostEqual(preds.loc[4, "awayspread"], -21.0)
        self.assertAlmostEqual(preds.loc[4, "homespread"], 21.0)
        self.assertAlmostEqual(preds.loc[4, "totalpred"], 44.0)

        self.assertEqual(models["fbs_win_with_spread"].row_count, 1)
        self.assertEqual(models["fbs_win_no_spread"].row_count, 1)
        self.assertEqual(models["fcs_win_with_spread"].row_count, 1)
        self.assertEqual(models["fcs_win_no_spread"].row_count, 1)

    def test_prediction_records_label_fbs_and_fcs_model_versions(self):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "season": [2026, 2026, 2026, 2026],
                "week": [1, 1, 1, 1],
                "hometeam": ["A", "C", "E", "FCS Two"],
                "awayteam": ["B", "D", "FCS One", "G"],
                "homepoints": [None, None, None, None],
                "awaypoints": [None, None, None, None],
                "homespread": [-4.0, -2.0, -35.0, 21.0],
                "awayspread": [4.0, 2.0, 35.0, -21.0],
                "totalpred": [55.0, 49.0, 62.0, 44.0],
                "homewinprob": [0.71, 0.61, 0.90, 0.20],
                "awaywinprob": [0.29, 0.39, 0.10, 0.80],
                "has_spread_line": [True, False, True, False],
                "has_total_line": [True, False, True, False],
                "prediction_type": ["FBS", "FBS", "FCS", "FCS"],
            }
        )

        records = job.prediction_records(df)
        versions = {record["gameid"]: record["model_version"] for record in records}

        self.assertEqual(versions["1"], job.XGB_FBS_AWARE_MODEL_VERSION)
        self.assertEqual(versions["2"], job.XGB_FBS_INCOMPLETE_MODEL_VERSION)
        self.assertEqual(versions["3"], job.XGB_FCS_AWARE_MODEL_VERSION)
        self.assertEqual(versions["4"], job.XGB_FCS_INCOMPLETE_MODEL_VERSION)

    def test_fcs_spread_is_converted_to_fbs_perspective(self):
        df = pd.DataFrame(
            [
                {
                    "homeclassification": "fbs",
                    "awayclassification": "fcs",
                    "avg_spread": -31.5,
                },
                {
                    "homeclassification": "fcs",
                    "awayclassification": "fbs",
                    "avg_spread": 24.0,
                },
            ]
        )

        modeled = job.prepare_modeling_dataframe(df)

        self.assertAlmostEqual(modeled.loc[0, "fbs_spread"], -31.5)
        self.assertAlmostEqual(modeled.loc[1, "fbs_spread"], -24.0)

    def test_modeling_sql_uses_strict_prior_teamrankings_pull_date(self):
        captured = {}
        original_read_sql = job.pd.read_sql

        def fake_read_sql(sql, conn, params=None):
            captured["sql"] = sql
            captured["params"] = params
            return pd.DataFrame({"id": [1]})

        try:
            job.pd.read_sql = fake_read_sql
            result = job.build_modeling_table(object(), 2026)
        finally:
            job.pd.read_sql = original_read_sql

        self.assertEqual(len(result), 1)
        self.assertEqual(captured["params"], (2026,))
        self.assertIn("tr.pull_date < g.gamedate", captured["sql"])


if __name__ == "__main__":
    unittest.main()
