"""Tests for the rules that keep training honest and monitoring correct.

Run from the project root (no extra packages needed):

    python -m unittest discover -s tests -v
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # model_utils reads static/model_config.yaml relative to the project root

from utils import model_utils as mu  # noqa: E402


def write_month(folder, prefix, date_str, frame):
    os.makedirs(folder, exist_ok=True)
    frame.to_parquet(folder + prefix + mu.partition_suffix(date_str) + ".parquet", index=False)


class TempStores(unittest.TestCase):
    """Points the label store, feature store and registry at a temporary folder."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp().replace("\\", "/") + "/"
        self.saved = (mu.LABEL_STORE, mu.FEATURE_STORE, mu.REGISTRY)
        mu.LABEL_STORE = self.tmp + "label_store/"
        mu.FEATURE_STORE = self.tmp + "feature_store/"
        mu.REGISTRY = self.tmp + "model_registry.csv"

    def tearDown(self):
        mu.LABEL_STORE, mu.FEATURE_STORE, mu.REGISTRY = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestLabelMaturity(TempStores):
    def setUp(self):
        super().setUp()
        # one loan per label month; each label is observed 6 months after the loan starts
        for i, label_date in enumerate(["2024-04-01", "2024-05-01", "2024-06-01", "2024-07-01"]):
            start = mu.shift_months(label_date, -6)
            customer = f"C{i}"
            write_month(mu.LABEL_STORE, "gold_label_store_", label_date, pd.DataFrame({
                "loan_id": [customer + "_" + start], "Customer_ID": [customer], "loan_start_date": [start],
                "label": [i % 2], "label_def": ["30dpd_6mob"], "snapshot_date": [label_date]}))
            write_month(mu.FEATURE_STORE, "gold_feature_store_", start, pd.DataFrame({
                "Customer_ID": [customer], "snapshot_date": [start], "Age": [30 + i]}))

    def test_no_label_after_training_date(self):
        data = mu.load_training_data("2024-06-01")
        self.assertEqual(sorted(data["label_snapshot_date"]), ["2024-04-01", "2024-05-01", "2024-06-01"])
        self.assertTrue((data["label_snapshot_date"] <= "2024-06-01").all())

    def test_features_come_from_the_application_date(self):
        data = mu.load_training_data("2024-07-01")
        self.assertTrue((data["snapshot_date"] == data["loan_start_date"]).all())


class TestSplit(unittest.TestCase):
    def test_oot_is_the_newest_label_months_only(self):
        months = ["2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]
        data = pd.DataFrame({"label_snapshot_date": np.repeat(months, 50),
                             "label": np.tile([0, 1], 100), "row": range(200)})
        config = {"training": {"oot_label_months": 2, "validation_ratio": 0.2, "random_state": 1}}
        train, val, oot = mu.split_train_val_oot(data, config)

        self.assertEqual(sorted(oot["label_snapshot_date"].unique()), ["2024-05-01", "2024-06-01"])
        for part in (train, val):
            self.assertTrue((part["label_snapshot_date"] < "2024-05-01").all())
        rows = [set(p["row"]) for p in (train, val, oot)]
        self.assertFalse(rows[0] & rows[1] or rows[0] & rows[2] or rows[1] & rows[2])
        self.assertEqual(sum(len(r) for r in rows), 200)


class TestStabilityIndices(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_psi_is_near_zero_for_the_same_distribution(self):
        self.assertLess(mu.psi(self.rng.random(5000), self.rng.random(5000)), 0.01)

    def test_psi_flags_a_shifted_distribution(self):
        self.assertGreater(mu.psi(self.rng.random(5000), self.rng.random(5000) * 0.5), 0.25)

    def test_csi_detects_a_change_in_a_binary_flag(self):
        # decile edges of a 0/1 column collapse to one bin, so this needs the category path
        before = np.array([0] * 900 + [1] * 100)
        after = np.array([0] * 500 + [1] * 500)
        self.assertGreater(mu.csi(before, after), 0.25)
        self.assertLess(mu.csi(before, before.copy()), 0.001)

    def test_csi_detects_a_feature_that_stops_being_collected(self):
        before = pd.Series(self.rng.normal(size=1000))
        after = pd.Series([np.nan] * 1000)
        self.assertGreater(mu.csi(before, after), 0.25)

    def test_csi_compares_categories(self):
        before = pd.Series(["A"] * 50 + ["B"] * 50, dtype=object)
        self.assertLess(mu.csi(before, before.copy()), 0.001)
        self.assertGreater(mu.csi(before, pd.Series(["A"] * 95 + ["C"] * 5, dtype=object)), 0.25)


class TestFeatures(unittest.TestCase):
    def test_clickstream_flag_is_dropped_with_the_clickstream_features(self):
        config = mu.load_config()
        applicants = pd.DataFrame({c: [1.0, 2.0] for group in config["features"].values() for c in group})
        click_means = [c for c in config["features"]["numeric"] if c.startswith("fe_")]
        applicants[click_means] = np.nan  # applicants from 2024-07 have no clickstream
        applicants["has_clickstream"] = 0
        spec, dropped = mu.available_features(applicants, config)
        self.assertIn("has_clickstream", dropped)
        self.assertFalse(set(click_means) & set(spec["numeric"]))
        self.assertNotIn("has_clickstream", spec["numeric"])

    def test_pruning_drops_noise_and_duplicates_and_keeps_log_twins_aligned(self):
        rng = np.random.default_rng(0)
        n = 2000
        signal = rng.normal(size=n)
        label = (signal + rng.normal(scale=0.5, size=n) > 0).astype(int)
        train = pd.DataFrame({
            "signal": signal,
            "signal_copy": signal * 2 + 1,                    # same information, perfectly correlated
            "noise": rng.normal(size=n),                       # unrelated to the label
            "noise_log1p": np.log1p(np.abs(rng.normal(size=n))),
            "flag": (signal > 0).astype(int),                  # informative flag
            "label": label,
        })
        spec = {"numeric": ["signal", "signal_copy", "noise", "noise_log1p"], "flags": ["flag"],
                "categorical": []}
        config = {"training": {"min_information_value": 0.02, "max_correlation": 0.9}}
        pruned, report = mu.prune_features(spec, train, config)

        kept = set(pruned["numeric"] + pruned["flags"])
        self.assertIn("flag", kept)
        self.assertEqual(len(kept & {"signal", "signal_copy"}), 1)  # one of the two duplicates
        self.assertNotIn("noise", kept)
        self.assertNotIn("noise_log1p", kept)  # the twin follows its raw column
        self.assertEqual(report.loc[report["feature"] == "noise", "dropped"].item(), "weak")


class TestModelSelection(unittest.TestCase):
    ORDER = ["logistic_regression", "xgboost"]

    def test_simpler_model_wins_a_near_tie(self):
        # the 2024-01 case: XGBoost behind by 0.004, so logistic regression is kept
        chosen, reason = mu.select_model({"logistic_regression": 0.5976, "xgboost": 0.5938}, self.ORDER, 0.01)
        self.assertEqual(chosen, "logistic_regression")
        chosen, reason = mu.select_model({"logistic_regression": 0.5938, "xgboost": 0.5976}, self.ORDER, 0.01)
        self.assertEqual(chosen, "logistic_regression")
        self.assertIn("simpler model preferred", reason)

    def test_complex_model_wins_a_clear_gap(self):
        chosen, reason = mu.select_model({"logistic_regression": 0.5375, "xgboost": 0.6226}, self.ORDER, 0.01)
        self.assertEqual(chosen, "xgboost")
        self.assertEqual(reason, "highest validation Gini")


class TestChampion(TempStores):
    def setUp(self):
        super().setUp()
        for date, promoted in [("2024-01-01", True), ("2024-07-01", True), ("2025-01-01", False)]:
            mu.write_registry_row({"decision_date": date, "model_version": mu.model_version(date),
                                   "promoted": promoted})

    def test_no_champion_before_the_first_training(self):
        self.assertIsNone(mu.champion_as_of("2023-12-01"))

    def test_never_uses_a_model_trained_after_the_scoring_date(self):
        self.assertEqual(mu.champion_as_of("2024-06-01"), "credit_model_2024_01_01")
        self.assertEqual(mu.champion_as_of("2024-07-01"), "credit_model_2024_07_01")
        self.assertEqual(mu.champion_as_of("2024-07-01", inclusive=False), "credit_model_2024_01_01")

    def test_a_rejected_challenger_never_becomes_champion(self):
        self.assertEqual(mu.champion_as_of("2025-06-01"), "credit_model_2024_07_01")


if __name__ == "__main__":
    unittest.main()
