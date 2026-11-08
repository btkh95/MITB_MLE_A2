"""Train candidate models on labels known at the snapshot date and save the best to the model bank.

    python scripts/model_train.py --snapshotdate 2024-01-01
"""

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def main(snapshot_date_str):
    config = mu.load_config()
    random_state = config["training"]["random_state"]

    # --- data known at training time ---
    data = mu.load_training_data(snapshot_date_str)
    train, val, oot = mu.split_train_val_oot(data, config)
    for name, rows in [("train", train), ("val", val), ("oot", oot)]:
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
        for name, rows in [("train", train), ("val", val), ("oot", oot)]:
            proba = best.predict_proba(mu.prepare_X(rows, features))[:, 1]
            auc = roc_auc_score(rows["label"], proba)
            scores[f"auc_{name}"] = round(float(auc), 4)
            scores[f"gini_{name}"] = round(float(2 * auc - 1), 4)
        print(model_type, params, scores)
        candidates[model_type] = {"pipeline": best, "features": features, "params": params, "scores": scores}

    # selection rule: highest validation Gini, with a tie-break that prefers the simpler model
    # when the gap is within select_tie_margin. Each candidate is tuned by CV on train, the
    # winner is picked on val, and OOT is only reported, so gini_oot stays an unbiased
    # estimate and a fair baseline for monitoring.
    best_type, selection_reason = mu.select_model(
        {t: c["scores"]["gini_val"] for t, c in candidates.items()},
        simplest_first=["logistic_regression", "xgboost"],
        tie_margin=config["training"]["select_tie_margin"])
    best = candidates[best_type]
    print("selected:", best_type, "-", selection_reason)

    # --- save artefact ---
    version = mu.model_version(snapshot_date_str)
    # PSI reference = OOT scores: out-of-sample, so an overfit model's spread-out training
    # scores do not show up as drift in the very first month
    # CSI reference = the same OOT rows, as raw model inputs before preprocessing
    reference_features = mu.prepare_X(oot, best["features"]).reset_index(drop=True)
    reference_scores = best["pipeline"].predict_proba(reference_features)[:, 1]
    artefact = {
        "model_version": version,
        "model_type": best_type,
        "selection_reason": selection_reason,
        "pipeline": best["pipeline"],
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
            "rows": {"train": len(train), "val": len(val), "oot": len(oot)},
            "bad_rate": {"train": round(float(train["label"].mean()), 4),
                         "val": round(float(val["label"].mean()), 4),
                         "oot": round(float(oot["label"].mean()), 4)},
        },
        "reference_scores": reference_scores,
        "reference_features": reference_features,
    }
    directory = mu.save_artefact(artefact)

    # human-readable summary next to the pickle
    summary = {k: v for k, v in artefact.items()
               if k not in ["pipeline", "reference_scores", "reference_features"]}
    with open(directory + "metadata.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=str)
    print("saved to:", directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train candidate models")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
