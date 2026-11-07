"""Monitor stability (this month's scores) and performance (cohort whose labels matured this month).

    python scripts/model_monitor.py --snapshotdate 2024-07-01

Stability: PSI of this month's prediction scores vs the model's OOT scores at training time.
Performance: predictions made label_lag_months ago, joined to the labels observed today.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from utils import model_utils as mu


def read_predictions(date_str):
    path = mu.PREDICTIONS + "predictions_" + mu.partition_suffix(date_str) + ".parquet"
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()


def psi_status(value, config):
    if value >= config["monitoring"]["psi_act"]:
        return "act"
    if value >= config["monitoring"]["psi_watch"]:
        return "watch"
    return "stable"


def stability_row(snapshot_date_str, config):
    predictions = read_predictions(snapshot_date_str)
    if predictions.empty:
        return None
    version = predictions["model_version"].iloc[0]
    artefact = mu.load_artefact(version)
    value = mu.psi(artefact["reference_scores"], predictions["model_predictions"])
    return {
        "metric_type": "stability",
        "cohort_date": snapshot_date_str,
        "model_version": version,
        "n": len(predictions),
        "mean_prediction": float(predictions["model_predictions"].mean()),
        "psi": value,
        "psi_status": psi_status(value, config),
        "retrain_flag": value >= config["monitoring"]["psi_act"],
    }


def performance_row(snapshot_date_str, config):
    cohort_date = mu.shift_months(snapshot_date_str, -config["monitoring"]["label_lag_months"])
    predictions = read_predictions(cohort_date)
    labels = mu.read_labels(snapshot_date_str)
    if predictions.empty or labels.empty:
        return None

    labels = labels[labels["loan_start_date"] == cohort_date]
    scored = predictions.merge(labels[["Customer_ID", "label"]], on="Customer_ID", how="inner")
    if scored["label"].nunique() < 2:
        print("cohort has a single label class; AUC undefined")
        return None

    version = scored["model_version"].iloc[0]
    baseline_gini = mu.load_artefact(version)["metrics"]["gini_oot"]
    auc = roc_auc_score(scored["label"], scored["model_predictions"])
    fpr, tpr, _ = roc_curve(scored["label"], scored["model_predictions"])
    gini = 2 * auc - 1
    gini_ratio = gini / baseline_gini if baseline_gini > 0 else np.nan
    return {
        "metric_type": "performance",
        "cohort_date": cohort_date,
        "model_version": version,
        "n": len(scored),
        "mean_prediction": float(scored["model_predictions"].mean()),
        "bad_rate": float(scored["label"].mean()),
        "auc": float(auc),
        "gini": float(gini),
        "ks": float(np.max(tpr - fpr)),
        "baseline_gini": float(baseline_gini),
        "gini_ratio": float(gini_ratio),
        "retrain_flag": bool(gini_ratio < config["monitoring"]["min_gini_ratio"]),
    }


def main(snapshot_date_str):
    config = mu.load_config()
    rows = [r for r in [stability_row(snapshot_date_str, config),
                        performance_row(snapshot_date_str, config)] if r]
    if not rows:
        print("nothing to monitor on", snapshot_date_str)
        return

    monitoring = pd.DataFrame(rows)
    monitoring.insert(0, "monitor_date", snapshot_date_str)
    os.makedirs(mu.MONITORING, exist_ok=True)
    filepath = mu.MONITORING + "monitoring_" + mu.partition_suffix(snapshot_date_str) + ".parquet"
    monitoring.to_parquet(filepath, index=False)
    print(monitoring.drop(columns=["monitor_date"]).to_string(index=False))
    print("saved to:", filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="monitor model stability and performance")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
