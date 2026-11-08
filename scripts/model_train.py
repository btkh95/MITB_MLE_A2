"""Train candidate models on labels known at the snapshot date and save the best to the model bank.

    python scripts/model_train.py --snapshotdate 2024-01-01
"""

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import model_utils as mu


def candidate_features(spec, model_type):
    # logistic regression uses log versions of skewed money columns; trees use raw values
    log_columns = [c for c in spec["numeric"] if c.endswith("_log1p")]
    raw_with_log = [c[:-len("_log1p")] for c in log_columns]
    drop = raw_with_log if model_type == "logistic_regression" else log_columns
    return {**spec, "numeric": [c for c in spec["numeric"] if c not in drop]}


def build_pipeline(features, model_type, random_state):
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    flags = SimpleImputer(strategy="most_frequent")
    if model_type == "logistic_regression":
        numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        model = LogisticRegression(max_iter=5000)
    else:
        numeric = "passthrough"  # XGBoost handles missing values natively
        model = xgb.XGBClassifier(eval_metric="logloss", random_state=random_state, n_jobs=-1)

    preprocess = ColumnTransformer([
        ("numeric", numeric, features["numeric"]),
        ("flags", flags, features["flags"]),
        ("categorical", categorical, features["categorical"]),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def tune(pipeline, model_type, X, y, random_state):
    # hyperparameters chosen by 3-fold CV AUC on the training split only
    if model_type == "logistic_regression":
        search = GridSearchCV(pipeline, {"model__C": [0.01, 0.1, 1.0]}, scoring="roc_auc", cv=3)
    else:
        search = RandomizedSearchCV(pipeline, {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8],
            "model__colsample_bytree": [0.7, 0.8],
            "model__min_child_weight": [5, 10, 20],
            "model__reg_lambda": [1, 10, 20],
        }, n_iter=15, scoring="roc_auc", cv=3, random_state=random_state)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def calibrate(pipeline, features, holdout, oot):
    """Platt scaling fitted on the model-selection holdout; kept only if it lowers the Brier
    score on the out-of-time promotion set (so OOT also decides this).

    Platt scaling is a one-input logistic regression on the model's score. It only stretches
    or shifts the scores, so the order of applicants (and AUC/Gini) stays the same.
    Brier score = mean of (probability - outcome)^2, lower is better.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="sigmoid")
    calibrated.fit(mu.prepare_X(holdout, features), holdout["label"])

    X_oot = mu.prepare_X(oot, features)
    raw = pipeline.predict_proba(X_oot)[:, 1]
    adjusted = calibrated.predict_proba(X_oot)[:, 1]
    report = {
        "method": "platt (sigmoid) on holdout, kept if OOT Brier improves",
        "brier_oot_raw": round(float(brier_score_loss(oot["label"], raw)), 4),
        "brier_oot_calibrated": round(float(brier_score_loss(oot["label"], adjusted)), 4),
        "mean_prediction_oot_raw": round(float(raw.mean()), 4),
        "mean_prediction_oot_calibrated": round(float(adjusted.mean()), 4),
        "bad_rate_oot": round(float(oot["label"].mean()), 4),
    }
    report["applied"] = report["brier_oot_calibrated"] < report["brier_oot_raw"]
    return (calibrated if report["applied"] else pipeline), report


def main(snapshot_date_str):
    config = mu.load_config()
    random_state = config["training"]["random_state"]

    # --- data known at training time ---
    data = mu.load_training_data(snapshot_date_str)
    mu.check_training_data(data)
    # train: fitting (3-fold CV inside it for tuning); holdout: model-selection holdout;
    # oot: out-of-time promotion set (2 newest label months)
    train, holdout, oot = mu.split_train_holdout_oot(data, config)
    for name, rows in [("train", train), ("holdout", holdout), ("oot", oot)]:
        print(f"{name:5} rows: {len(rows):5}  bad rate: {rows['label'].mean():.3f}  "
              f"applications: {rows['loan_start_date'].min()} to {rows['loan_start_date'].max()}")

    # --- feature availability: drop features this month's applicants do not have ---
    applicants = mu.read_features(snapshot_date_str)
    spec, dropped = mu.available_features(applicants, config)
    print("features dropped for low coverage at scoring time:", dropped or "none")

    # --- feature pruning on the training split: weak (low IV) and duplicate (highly correlated) ---
    spec, feature_report = mu.prune_features(spec, train, config)
    pruned = feature_report[feature_report["dropped"] != ""]
    print(f"features pruned on train: {len(pruned)}")
    print(pruned.to_string(index=False) if len(pruned) else "none")

    # --- train and compare candidates ---
    candidates = {}
    for model_type in ["logistic_regression", "xgboost"]:
        features = candidate_features(spec, model_type)
        pipeline = build_pipeline(features, model_type, random_state)
        best, params = tune(pipeline, model_type, mu.prepare_X(train, features), train["label"], random_state)

        scores = {}
        for name, rows in [("train", train), ("holdout", holdout), ("oot", oot)]:
            proba = best.predict_proba(mu.prepare_X(rows, features))[:, 1]
            auc = roc_auc_score(rows["label"], proba)
            scores[f"auc_{name}"] = round(float(auc), 4)
            scores[f"gini_{name}"] = round(float(2 * auc - 1), 4)
        print(model_type, params, scores)
        candidates[model_type] = {"pipeline": best, "features": features, "params": params, "scores": scores}

    # selection rule: highest Gini on the model-selection holdout, with a tie-break that prefers
    # the simpler model when the gap is within select_tie_margin. Each candidate is tuned by CV
    # on train and the winner is picked on the holdout. OOT plays no part in this choice; it is
    # used later for the calibration decision and for promotion, which makes gini_oot (the
    # monitoring baseline) slightly optimistic.
    best_type, selection_reason = mu.select_model(
        {t: c["scores"]["gini_holdout"] for t, c in candidates.items()},
        simplest_first=["logistic_regression", "xgboost"],
        tie_margin=config["training"]["select_tie_margin"])
    best = candidates[best_type]
    print("selected:", best_type, "-", selection_reason)
    runner_up_type = next(t for t in candidates if t != best_type)

    # --- calibration: make the probabilities match observed default rates ---
    final_pipeline, calibration = calibrate(best["pipeline"], best["features"], holdout, oot)
    print("calibration:", calibration)

    # --- explanations: which inputs drive the score, measured on OOT ---
    oot_X = mu.prepare_X(oot, best["features"])
    importance = mu.feature_importance(best["pipeline"], oot_X)
    print("top inputs by mean contribution on OOT:")
    print(importance.head(10).round(4).to_string(index=False))

    # --- save artefact ---
    # every run gets its own name, so a rerun never overwrites the model that scored earlier months
    version = mu.model_version(snapshot_date_str, best_type, mu.new_fit_id())
    # PSI reference = OOT scores: out-of-sample, so an overfit model's spread-out training
    # scores do not show up as drift in the very first month
    # CSI reference = the same OOT rows, as raw model inputs before preprocessing
    reference_features = oot_X.reset_index(drop=True)
    reference_scores = final_pipeline.predict_proba(reference_features)[:, 1]
    artefact = {
        "model_version": version,
        "model_type": best_type,
        "selection_reason": selection_reason,
        # pipeline = what inference uses (calibrated if that helped); base_pipeline = the
        # uncalibrated model, used for explanations because calibration does not change ranking
        "pipeline": final_pipeline,
        "base_pipeline": best["pipeline"],
        "calibration": calibration,
        # the losing model type runs in shadow: it scores applicants but never decides
        "runner_up": {"model_type": runner_up_type, "pipeline": candidates[runner_up_type]["pipeline"],
                      "features": candidates[runner_up_type]["features"]},
        "feature_importance": importance.round(4).to_dict(orient="records"),
        "lineage": mu.lineage(config, data),
        "features": best["features"],
        "dropped_features": dropped,
        # information value of every candidate feature on train, and why any were pruned
        "feature_report": feature_report.round(4).to_dict(orient="records"),
        "hyperparameters": best["params"],
        "metrics": best["scores"],
        "candidates": {t: {"params": c["params"], "scores": c["scores"]} for t, c in candidates.items()},
        "data": {
            "train_date": snapshot_date_str,
            "label_months": [data["label_snapshot_date"].min(), data["label_snapshot_date"].max()],
            "oot_label_months": sorted(oot["label_snapshot_date"].unique().tolist()),
            "rows": {"train": len(train), "holdout": len(holdout), "oot": len(oot)},
            "bad_rate": {"train": round(float(train["label"].mean()), 4),
                         "holdout": round(float(holdout["label"].mean()), 4),
                         "oot": round(float(oot["label"].mean()), 4)},
            # what is needed to rebuild this run exactly: the partitions read and the loans in each split
            "source_partitions": {
                "label_store": sorted(data["label_snapshot_date"].unique().tolist()),
                "feature_store": sorted(data["loan_start_date"].unique().tolist()),
            },
        },
        "split_loan_ids": {"train": sorted(train["loan_id"]), "holdout": sorted(holdout["loan_id"]),
                           "oot": sorted(oot["loan_id"])},
        "reference_scores": reference_scores,
        "reference_features": reference_features,
    }
    directory = mu.save_artefact(artefact)

    # reload check: the saved file must load and give the same OOT scores as the model in memory
    reloaded = mu.load_artefact(version)
    reloaded_scores = reloaded["pipeline"].predict_proba(reference_features)[:, 1]
    if not np.allclose(reloaded_scores, reference_scores):
        raise ValueError(f"reload check failed: {version} scores differently after saving")
    print(f"reload check passed: {version} loads and reproduces its {len(reloaded_scores)} OOT scores")

    # human-readable summary next to the pickle
    summary = {k: v for k, v in artefact.items()
               if k not in ["pipeline", "base_pipeline", "runner_up", "reference_scores", "reference_features",
                            "split_loan_ids"]}
    summary["runner_up"] = runner_up_type
    with open(directory + "metadata.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)
    print("saved to:", directory)

    # hand the exact verified model to promotion; written last, so it only exists for a complete run
    mu.record_verified_fit(snapshot_date_str, version)
    print("verified fit recorded:", version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train candidate models")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
