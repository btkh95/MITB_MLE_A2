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
                             "label": np.tile([0, 1], 100), "row": range(200),
                             "loan_id": [f"L{i:03d}" for i in range(200)]})
        config = {"training": {"oot_label_months": 2, "holdout_ratio": 0.2, "random_state": 1}}
        train, holdout, oot = mu.split_train_holdout_oot(data, config)

        self.assertEqual(sorted(oot["label_snapshot_date"].unique()), ["2024-05-01", "2024-06-01"])
        for part in (train, holdout):
            self.assertTrue((part["label_snapshot_date"] < "2024-05-01").all())
        rows = [set(p["row"]) for p in (train, holdout, oot)]
        self.assertFalse(rows[0] & rows[1] or rows[0] & rows[2] or rows[1] & rows[2])
        self.assertEqual(sum(len(r) for r in rows), 200)


    def test_split_membership_does_not_depend_on_row_order(self):
        rng = np.random.default_rng(3)
        months = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"]
        data = pd.DataFrame({"loan_id": [f"L{i:04d}" for i in range(500)],
                             "label_snapshot_date": np.repeat(months, 100),
                             "label": rng.integers(0, 2, 500)})
        config = {"training": {"oot_label_months": 2, "holdout_ratio": 0.2, "random_state": 55}}
        reference = [set(part["loan_id"]) for part in mu.split_train_holdout_oot(data, config)]
        for order in (data.iloc[::-1], data.sample(frac=1, random_state=7)):
            shuffled = [set(part["loan_id"]) for part in mu.split_train_holdout_oot(order, config)]
            self.assertEqual(shuffled, reference)


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
        self.assertEqual(reason, "highest holdout Gini")


class TestDataChecks(unittest.TestCase):
    def test_training_data_with_one_class_or_duplicates_is_rejected(self):
        good = pd.DataFrame({"loan_id": ["a", "b"], "label": [0, 1]})
        mu.check_training_data(good)
        with self.assertRaises(ValueError):
            mu.check_training_data(good.assign(label=[0, 0]))
        with self.assertRaises(ValueError):
            mu.check_training_data(good.assign(loan_id=["a", "a"]))

    def test_applicants_missing_a_model_column_are_rejected(self):
        applicants = pd.DataFrame({"Customer_ID": ["C1"], "snapshot_date": ["2024-01-01"], "Age": [30]})
        mu.check_applicants(applicants, {"numeric": ["Age"], "flags": [], "categorical": []})
        with self.assertRaises(ValueError):
            mu.check_applicants(applicants, {"numeric": ["Age", "Interest_Rate"], "flags": [], "categorical": []})

    def test_scores_outside_zero_to_one_are_rejected(self):
        mu.check_scores([0.1, 0.9])
        for bad in ([0.1, np.nan], [0.1, 1.2], [-0.1, 0.5]):
            with self.assertRaises(ValueError):
                mu.check_scores(bad)


class TestExplanations(unittest.TestCase):
    def test_reason_codes_name_the_inputs_that_raise_risk(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        X = pd.DataFrame({"risk": rng.normal(size=500), "noise": rng.normal(size=500)})
        y = (X["risk"] + rng.normal(scale=0.3, size=500) > 0).astype(int)
        pipeline = Pipeline([("preprocess", ColumnTransformer([("numeric", StandardScaler(), ["risk", "noise"])])),
                             ("model", LogisticRegression())]).fit(X, y)

        risky = pd.DataFrame({"risk": [3.0], "noise": [0.0]})
        self.assertEqual(mu.reason_codes(pipeline, risky, top_n=1).iloc[0, 0], "risk")
        safe = pd.DataFrame({"risk": [-3.0], "noise": [0.0]})
        self.assertNotEqual(mu.reason_codes(pipeline, safe, top_n=1).iloc[0, 0], "risk")
        self.assertEqual(mu.feature_importance(pipeline, X)["feature"].iloc[0], "risk")


class TestLineage(unittest.TestCase):
    def test_same_loans_give_the_same_fingerprint(self):
        config = {"training": {"random_state": 1}}
        data = pd.DataFrame({"loan_id": ["b", "a"], "label": [1, 0]})
        first = mu.lineage(config, data)["training_data_sha256"]
        self.assertEqual(first, mu.lineage(config, data.iloc[::-1])["training_data_sha256"])
        self.assertNotEqual(first, mu.lineage(config, data.assign(label=[0, 0]))["training_data_sha256"])


class TestRetrainTriggers(TempStores):
    def test_coverage_drop_finds_a_source_that_stopped(self):
        reference = pd.DataFrame({"Age": [30.0, 40.0], "fe_1_mean": [1.0, 2.0]})
        current = pd.DataFrame({"Age": [35.0, 45.0], "fe_1_mean": [np.nan, np.nan]})
        self.assertEqual(mu.coverage_drop(reference, current), (1.0, "fe_1_mean"))
        self.assertEqual(mu.coverage_drop(reference, reference)[0], 0.0)

    def test_calibration_gap_is_predicted_minus_actual(self):
        scored = pd.DataFrame({"model_predictions": [0.2, 0.2, 0.2, 0.2], "label": [1, 0, 0, 0]})
        self.assertAlmostEqual(mu.calibration_gap(scored), -0.05)

    def test_gate_retrains_at_once_when_a_source_stops(self):
        saved = (mu.MODEL_BANK, mu.MONITORING, mu.load_config)
        mu.MODEL_BANK = self.tmp + "model_bank/"
        mu.MONITORING = self.tmp + "monitoring/"
        config = mu.load_config()
        config["training"].update({"retrain_every_months": 6, "coverage_max_missing_rise": 0.5})
        mu.load_config = lambda: config
        try:
            version = mu.model_version("2024-01-01", "xgboost")
            mu.write_registry_row({"decision_date": "2024-01-01", "model_version": version,
                                   "model_type": "xgboost", "promoted": True})
            features = {"numeric": ["Age", "fe_1_mean"], "flags": [], "categorical": []}
            mu.save_artefact({"model_version": version, "features": features,
                              "reference_features": pd.DataFrame({"Age": [30.0, 40.0], "fe_1_mean": [1.0, 2.0]})})
            for date, clicks in [("2024-03-01", [1.0, 2.0]), ("2024-04-01", [np.nan, np.nan])]:
                write_month(mu.FEATURE_STORE, "gold_feature_store_", date, pd.DataFrame({
                    "Customer_ID": ["C1", "C2"], "snapshot_date": [date] * 2, "Age": [31, 41],
                    "fe_1_mean": clicks}))
            self.assertFalse(mu.should_train("2024-03-01"))  # 2 months in, inputs still arriving
            self.assertTrue(mu.should_train("2024-04-01"))   # clickstream stopped: retrain before month 6
        finally:
            mu.MODEL_BANK, mu.MONITORING, mu.load_config = saved


class TestDriftAlertsFollowTheChampion(TempStores):
    def test_only_the_current_champions_alert_can_retrain(self):
        saved = (mu.MODEL_BANK, mu.MONITORING, mu.load_config)
        mu.MODEL_BANK, mu.MONITORING = self.tmp + "model_bank/", self.tmp + "monitoring/"
        config = mu.load_config()
        config["training"].update({"retrain_every_months": 6, "retrain_cooldown_months": 3})
        mu.load_config = lambda: config
        try:
            retired = mu.model_version("2024-01-01", "logistic_regression")
            current = mu.model_version("2024-04-01", "xgboost")
            features = {"numeric": ["Age"], "flags": [], "categorical": []}
            for date, version, mtype in [("2024-01-01", retired, "logistic_regression"),
                                         ("2024-04-01", current, "xgboost")]:
                mu.write_registry_row({"decision_date": date, "model_version": version,
                                       "model_type": mtype, "promoted": True})
                mu.save_artefact({"model_version": version, "features": features,
                                  "reference_features": pd.DataFrame({"Age": [30.0, 40.0]})})
            write_month(mu.FEATURE_STORE, "gold_feature_store_", "2024-08-01", pd.DataFrame({
                "Customer_ID": ["C1", "C2"], "snapshot_date": ["2024-08-01"] * 2, "Age": [31, 41]}))

            def monitoring(rows):
                os.makedirs(mu.MONITORING, exist_ok=True)
                pd.DataFrame(rows).to_parquet(mu.MONITORING + "monitoring_2024_07_01.parquet", index=False)

            # July monitoring: the January cohort's outcomes arrived and flag the retired model
            monitoring([{"metric_type": "performance", "model_version": retired, "retrain_flag": True,
                         "gini_flag": True},
                        {"metric_type": "stability", "model_version": current, "retrain_flag": False,
                         "gini_flag": False}])
            self.assertFalse(mu.should_train("2024-08-01"))
            # the current champion's own alert still retrains (4 months since training > cooldown)
            monitoring([{"metric_type": "performance", "model_version": retired, "retrain_flag": True,
                         "gini_flag": True},
                        {"metric_type": "stability", "model_version": current, "retrain_flag": True,
                         "gini_flag": False}])
            self.assertTrue(mu.should_train("2024-08-01"))
        finally:
            mu.MODEL_BANK, mu.MONITORING, mu.load_config = saved


class TestChampion(TempStores):
    def setUp(self):
        super().setUp()
        for date, model_type, promoted in [("2024-01-01", "logistic_regression", True),
                                           ("2024-07-01", "xgboost", True),
                                           ("2025-01-01", "xgboost", False)]:
            mu.write_registry_row({"decision_date": date, "model_version": mu.model_version(date, model_type),
                                   "model_type": model_type, "promoted": promoted})

    def test_model_names_carry_date_and_family(self):
        self.assertEqual(mu.model_version("2024-01-01", "logistic_regression"), "credit_model_2024_01_01_logreg")
        self.assertEqual(mu.version_date("credit_model_2024_07_01_xgboost"), "2024-07-01")

    def test_shadows_are_the_runner_up_and_later_rejected_challengers(self):
        self.assertEqual(mu.shadows_as_of("2023-12-01"), [])
        self.assertEqual([s[0] for s in mu.shadows_as_of("2024-03-01")],
                         ["credit_model_2024_01_01_xgboost_runner_up"])
        self.assertEqual([s[0] for s in mu.shadows_as_of("2025-01-01")],
                         ["credit_model_2024_07_01_logreg_runner_up", "credit_model_2025_01_01_xgboost"])

    def test_only_the_latest_rejected_challenger_runs_in_shadow(self):
        mu.write_registry_row({"decision_date": "2024-10-01", "model_type": "xgboost", "promoted": False,
                               "model_version": mu.model_version("2024-10-01", "xgboost")})
        self.assertEqual([s[0] for s in mu.shadows_as_of("2024-12-01")],
                         ["credit_model_2024_07_01_logreg_runner_up", "credit_model_2024_10_01_xgboost"])
        self.assertEqual([s[0] for s in mu.shadows_as_of("2025-01-01")],
                         ["credit_model_2024_07_01_logreg_runner_up", "credit_model_2025_01_01_xgboost"])

    def test_no_champion_before_the_first_training(self):
        self.assertIsNone(mu.champion_as_of("2023-12-01"))

    def test_never_uses_a_model_trained_after_the_scoring_date(self):
        self.assertEqual(mu.champion_as_of("2024-06-01"), "credit_model_2024_01_01_logreg")
        self.assertEqual(mu.champion_as_of("2024-07-01"), "credit_model_2024_07_01_xgboost")
        self.assertEqual(mu.champion_as_of("2024-07-01", inclusive=False), "credit_model_2024_01_01_logreg")

    def test_a_rejected_challenger_never_becomes_champion(self):
        self.assertEqual(mu.champion_as_of("2025-06-01"), "credit_model_2024_07_01_xgboost")


if __name__ == "__main__":
    unittest.main()
