"""
Test runner for unit tests using standard library unittest.
"""

import unittest
import numpy as np
import pandas as pd

from src.preprocessor import (
    RUL_CLIP,
    add_rul_labels,
    add_test_rul_labels,
    drop_features,
    fit_scaler,
    identify_low_variance_features,
    transform_features,
)
from src.feature_engineer import (
    add_cumulative_wear_features,
    add_normalized_cycle,
    add_operational_clusters,
    add_rolling_features,
    fit_operational_clusters,
)
from src.evaluator import evaluate_predictions, nasa_score


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.mock_df = pd.DataFrame({
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "s1": [518.67, 518.67, 518.67, 518.67, 518.67],
            "s2": [642.0, 642.5, 643.0, 641.0, 641.5],
        })

    def test_add_rul_labels_unclipped(self):
        df_rul = add_rul_labels(self.mock_df, clip_rul=False)
        self.assertIn("RUL", df_rul.columns)
        self.assertEqual(list(df_rul[df_rul["engine_id"] == 1]["RUL"]), [2, 1, 0])
        self.assertEqual(list(df_rul[df_rul["engine_id"] == 2]["RUL"]), [1, 0])
        self.assertTrue((df_rul["RUL"] >= 0).all())

    def test_add_rul_labels_clipped(self):
        df_long = pd.DataFrame({
            "engine_id": [1] * 200,
            "cycle": list(range(1, 201)),
        })
        df_clipped = add_rul_labels(df_long, clip_rul=True, rul_clip_limit=RUL_CLIP)
        self.assertEqual(df_clipped["RUL"].max(), RUL_CLIP)
        self.assertEqual(df_clipped.loc[0, "RUL"], RUL_CLIP)
        self.assertEqual(df_clipped.loc[199, "RUL"], 0)

    def test_add_test_rul_labels(self):
        test_df = pd.DataFrame({
            "engine_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
        })
        rul_series = pd.Series([10, 20], index=[1, 2])
        test_labeled = add_test_rul_labels(test_df, rul_series, clip_rul=False)
        e1_rul = test_labeled[test_labeled["engine_id"] == 1]["RUL"].tolist()
        self.assertEqual(e1_rul, [11, 10])

    def test_identify_low_variance_features(self):
        low_var = identify_low_variance_features(self.mock_df, ["s1", "s2"])
        self.assertIn("s1", low_var)
        self.assertNotIn("s2", low_var)

    def test_fit_and_transform_scaler(self):
        train_df = self.mock_df.copy()
        test_df = pd.DataFrame({"s2": [642.0]})
        scaler = fit_scaler(train_df, ["s2"])
        scaled_train = transform_features(train_df, ["s2"], scaler)
        scaled_test = transform_features(test_df, ["s2"], scaler)
        self.assertEqual(scaled_train["s2"].min(), 0.0)
        self.assertEqual(scaled_train["s2"].max(), 1.0)
        self.assertTrue(np.isclose(scaled_test.loc[0, "s2"], 0.5))


class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.mock_df = pd.DataFrame({
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
            "op_setting_1": [-0.0001, -0.0002, -0.0001, 0.0020, 0.0021],
            "op_setting_2": [0.0001, 0.0002, 0.0001, 0.0005, 0.0004],
            "op_setting_3": [100.0, 100.0, 100.0, 60.0, 60.0],
            "s2": [10.0, 20.0, 30.0, 100.0, 200.0],
        })

    def test_rolling_isolation(self):
        window = 2
        res = add_rolling_features(self.mock_df, ["s2"], window=window)
        # Engine 1
        self.assertTrue(np.isclose(res.loc[0, f"s2_mean_{window}"], 10.0))
        self.assertTrue(np.isclose(res.loc[1, f"s2_mean_{window}"], 15.0))
        # Engine 2 must NOT bleed from Engine 1
        self.assertTrue(np.isclose(res.loc[3, f"s2_mean_{window}"], 100.0))
        self.assertTrue(np.isclose(res.loc[4, f"s2_mean_{window}"], 150.0))
        self.assertFalse(res[[f"s2_mean_{window}", f"s2_std_{window}"]].isnull().any().any())

    def test_cumulative_wear(self):
        res = add_cumulative_wear_features(self.mock_df, ["s2"])
        cum_col = "s2_cum_change"
        self.assertEqual(res.loc[0, cum_col], 0.0)
        self.assertEqual(res.loc[3, cum_col], 0.0)
        self.assertEqual(res[res["engine_id"] == 1][cum_col].tolist(), [0.0, 10.0, 20.0])

    def test_normalized_cycle(self):
        res = add_normalized_cycle(self.mock_df)
        self.assertTrue(np.isclose(res.loc[2, "cycle_norm"], 1.0))
        self.assertTrue(np.isclose(res.loc[4, "cycle_norm"], 1.0))

    def test_operational_clusters(self):
        kmeans = fit_operational_clusters(self.mock_df, n_clusters=2)
        res = add_operational_clusters(self.mock_df, kmeans)
        self.assertIn("op_cluster", res.columns)


class TestEvaluator(unittest.TestCase):
    def test_nasa_score_asymmetry(self):
        y_true = np.array([50.0, 50.0])
        # Case 1: Early prediction (predicted 40, true 50 -> error d = -10)
        score_early = nasa_score(np.array([50.0]), np.array([40.0]))
        # Case 2: Late prediction (predicted 60, true 50 -> error d = +10)
        score_late = nasa_score(np.array([50.0]), np.array([60.0]))
        # Late prediction penalty should be much higher than early prediction
        self.assertGreater(score_late, score_early)
        self.assertTrue(np.isclose(score_early, np.exp(10.0 / 13.0) - 1.0))
        self.assertTrue(np.isclose(score_late, np.exp(10.0 / 10.0) - 1.0))

    def test_evaluate_predictions(self):
        metrics = evaluate_predictions(np.array([10.0, 20.0]), np.array([10.0, 20.0]))
        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["nasa_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
