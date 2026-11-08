"""Monitor stability (this month's scores) and performance (cohort whose labels matured this month).

    python scripts/model_monitor.py --snapshotdate 2024-07-01

Stability: PSI of this month's prediction scores vs the model's OOT scores at training time,
and CSI of each input feature vs the same OOT rows (written to model_monitoring/feature_drift/).
Performance: predictions made label_lag_months ago, joined to the labels observed today.
Shadow: the same live Gini for each shadow model, compared with the champion's.
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


def feature_drift(snapshot_date_str, artefact, config):
    """CSI of every model input: this month's applicants vs the model's OOT rows at training."""
    reference = artefact.get("reference_features")
    applicants = mu.read_features(snapshot_date_str)
    if reference is None or applicants.empty:
        return pd.DataFrame()
    current = mu.prepare_X(applicants, artefact["features"])
    rows = []
    for column in reference.columns:
        value = mu.csi(reference[column], current[column])
        rows.append({"feature": column, "csi": value, "csi_status": psi_status(value, config),
                     "missing_share_reference": float(reference[column].isna().mean()),
                     "missing_share_current": float(current[column].isna().mean())})
    drift = pd.DataFrame(rows).sort_values("csi", ascending=False, ignore_index=True)
    drift.insert(0, "model_version", artefact["model_version"])
    drift.insert(0, "monitor_date", snapshot_date_str)
    return drift


def stability_row(snapshot_date_str, config):
    """Score PSI plus a feature drift summary; returns (row, per-feature CSI table)."""
    predictions = read_predictions(snapshot_date_str)
    if predictions.empty:
        return None, pd.DataFrame()
    version = predictions["model_version"].iloc[0]
    artefact = mu.load_artefact(version)
    value = mu.psi(artefact["reference_scores"], predictions["model_predictions"])
    row = {
        "metric_type": "stability",
        "cohort_date": snapshot_date_str,
        "model_version": version,
        "n": len(predictions),
        "mean_prediction": float(predictions["model_predictions"].mean()),
        "psi": value,
        "psi_status": psi_status(value, config),
        # only score drift triggers a retrain; feature drift explains it. With ~70 inputs,
        # one feature crossing a line by chance each month is likely, so CSI is a warning.
        "psi_flag": bool(value >= config["monitoring"]["psi_act"]),
        "retrain_flag": bool(value >= config["monitoring"]["psi_act"]),
    }
    drift = feature_drift(snapshot_date_str, artefact, config)
    if not drift.empty:
        rise = drift["missing_share_current"] - drift["missing_share_reference"]
        row.update({
            "max_csi": float(drift["csi"].iloc[0]),
            "max_csi_feature": drift["feature"].iloc[0],
            "n_features_watch": int((drift["csi_status"] == "watch").sum()),
            "n_features_act": int((drift["csi_status"] == "act").sum()),
            # source coverage: the training gate retrains at once on this; logged here to chart it
            "max_missing_rise": float(rise.max()),
            "max_missing_rise_feature": drift.loc[rise.idxmax(), "feature"],
            "coverage_flag": bool(rise.max() >= config["training"]["coverage_max_missing_rise"]),
        })
    return row, drift


def read_shadow_predictions(date_str):
    path = mu.SHADOW_PREDICTIONS + "shadow_predictions_" + mu.partition_suffix(date_str) + ".parquet"
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()


def scored_cohort(monitor_date_str, config, reader=read_predictions):
    """Predictions made label_lag_months before monitor_date, joined to the labels observed on it."""
    cohort_date = mu.shift_months(monitor_date_str, -config["monitoring"]["label_lag_months"])
    predictions = reader(cohort_date)
    labels = mu.read_labels(monitor_date_str)
    if predictions.empty or labels.empty:
        return pd.DataFrame()
    labels = labels[labels["loan_start_date"] == cohort_date]
    return predictions.merge(labels[["Customer_ID", "label"]], on="Customer_ID", how="inner")


def shadow_rows(snapshot_date_str, champion_row, config):
    """Live Gini of each shadow model vs the champion on the same applicants.

    Uses the same pooled window as the champion's retrain test. A shadow ahead by
    shadow_min_gini_gain over a full window, while that champion is still in production,
    raises shadow_flag: a promotion review in the SOP, not an automatic switch.
    """
    scored = scored_cohort(snapshot_date_str, config, read_shadow_predictions)
    if scored.empty or champion_row is None:
        return []
    window_size = config["monitoring"]["gini_window_months"]
    champion = champion_row["model_version"]
    still_champion = mu.champion_as_of(snapshot_date_str) == champion
    rows = []
    for name, cohort in scored.groupby("shadow_model"):
        if cohort["label"].nunique() < 2:
            continue
        window = [cohort]
        for k in range(1, champion_row["cohorts_pooled"]):
            earlier = scored_cohort(mu.shift_months(snapshot_date_str, -k), config, read_shadow_predictions)
            earlier = earlier[earlier["shadow_model"] == name] if not earlier.empty else earlier
            if earlier.empty:
                break
            window.append(earlier)
        gini_pooled = gini_of(pd.concat(window, ignore_index=True))
        gap = gini_pooled - champion_row["gini_pooled"]
        rows.append({
            "metric_type": "shadow",
            "cohort_date": champion_row["cohort_date"],
            "model_version": champion,
            "shadow_model": name,
            "n": len(cohort),
            "mean_prediction": float(cohort["model_predictions"].mean()),
            "bad_rate": float(cohort["label"].mean()),
            "gini": float(gini_of(cohort)),
            "gini_pooled": float(gini_pooled),
            "cohorts_pooled": len(window),
            "champion_gini_pooled": float(champion_row["gini_pooled"]),
            "shadow_gini_gap": float(gap),
            "shadow_flag": bool(gap >= config["monitoring"]["shadow_min_gini_gain"]
                                and len(window) == window_size and still_champion),
            "retrain_flag": False,
        })
    return rows


def gini_of(scored):
    return 2 * roc_auc_score(scored["label"], scored["model_predictions"]) - 1


def performance_row(snapshot_date_str, config):
    cohort_date = mu.shift_months(snapshot_date_str, -config["monitoring"]["label_lag_months"])
    scored = scored_cohort(snapshot_date_str, config)
    if scored.empty:
        return None
    if scored["label"].nunique() < 2:
        print("cohort has a single label class; AUC undefined")
        return None

    version = scored["model_version"].iloc[0]
    baseline_gini = mu.load_artefact(version)["metrics"]["gini_oot"]
    auc = roc_auc_score(scored["label"], scored["model_predictions"])
    fpr, tpr, _ = roc_curve(scored["label"], scored["model_predictions"])
    gini = 2 * auc - 1

    # The retrain decision uses the latest few cohorts of the same model pooled together.
    # One cohort is ~500 loans, so its Gini moves by about +/-0.05 by chance, which is close
    # to the 10% drop the trigger looks for. Pooling 3 cohorts roughly halves that noise.
    window = [scored]
    for k in range(1, config["monitoring"]["gini_window_months"]):
        earlier = scored_cohort(mu.shift_months(snapshot_date_str, -k), config)
        if earlier.empty or earlier["model_version"].iloc[0] != version:
            break
        window.append(earlier)
    pooled = pd.concat(window, ignore_index=True)
    gini_pooled = gini_of(pooled)
    gini_ratio = gini_pooled / baseline_gini if baseline_gini > 0 else np.nan
    gini_flag = bool(gini_ratio < config["monitoring"]["min_gini_ratio"])
    # calibration trigger: predicted vs actual default rate over the same pooled cohorts;
    # retraining recalibrates the probabilities (Platt scaling is part of training)
    gap = mu.calibration_gap(pooled)
    calibration_flag = bool(abs(gap) >= config["monitoring"]["calibration_max_gap"])
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
        "gini_pooled": float(gini_pooled),
        "cohorts_pooled": len(window),
        "n_pooled": len(pooled),
        "baseline_gini": float(baseline_gini),
        "gini_ratio": float(gini_ratio),
        "calibration_gap_pooled": gap,
        "gini_flag": gini_flag,
        "calibration_flag": calibration_flag,
        "retrain_flag": gini_flag or calibration_flag,
    }


def role_rows(snapshot_date_str, current_rows, config):
    """Shadow comparison by role, so it works however often the models change.

    The per-model shadow flag needs one shadow model to be scored for several months; with monthly
    retraining each shadow is replaced within a month. Here the latest gini_window_months monitoring
    months are pooled by role (the champion's runner-up, or the latest rejected challenger),
    whatever model filled the role, and each month's shadow Gini is compared with the champion's
    on the same applicants. role_flag: the role was ahead in every month and by
    shadow_min_gini_gain on average. A runner-up flag questions the model-family choice; a
    rejected-challenger flag suggests the promotion rule is too strict. Promotion review only.
    """
    window = config["monitoring"]["gini_window_months"]
    months = [pd.DataFrame(current_rows)]
    months += [mu.read_monitoring(mu.shift_months(snapshot_date_str, -k)) for k in range(1, window)]
    gaps = {}
    for table in months:
        if table.empty or "metric_type" not in table:
            continue
        champion = table[table["metric_type"] == "performance"]
        shadows = table[table["metric_type"] == "shadow"]
        if champion.empty or shadows.empty:
            continue
        for _, row in shadows.iterrows():
            role = "runner_up" if row["shadow_model"].endswith("_runner_up") else "rejected_challenger"
            gaps.setdefault(role, []).append(float(row["gini"] - champion["gini"].iloc[0]))
    cohort = next((r["cohort_date"] for r in current_rows if r["metric_type"] == "performance"), None)
    rows = []
    for role, values in gaps.items():
        full = len(values) == window
        rows.append({
            "metric_type": "shadow_role",
            "cohort_date": cohort,
            "shadow_model": role,
            "months_compared": len(values),
            "months_ahead": int(sum(v > 0 for v in values)),
            "mean_gini_gap": float(np.mean(values)),
            "role_flag": bool(full and all(v > 0 for v in values)
                              and np.mean(values) >= config["monitoring"]["shadow_min_gini_gain"]),
            "retrain_flag": False,
        })
    return rows


def main(snapshot_date_str):
    config = mu.load_config()
    stability, drift = stability_row(snapshot_date_str, config)
    performance = performance_row(snapshot_date_str, config)
    rows = [r for r in [stability, performance] if r]
    rows += shadow_rows(snapshot_date_str, performance, config)
    rows += role_rows(snapshot_date_str, rows, config)
    if not rows:
        print("nothing to monitor on", snapshot_date_str)
        return

    monitoring = pd.DataFrame(rows)
    monitoring.insert(0, "monitor_date", snapshot_date_str)
    os.makedirs(mu.MONITORING, exist_ok=True)
    filepath = mu.MONITORING + "monitoring_" + mu.partition_suffix(snapshot_date_str) + ".parquet"
    monitoring.to_parquet(filepath, index=False)
    print(monitoring.drop(columns=["monitor_date"]).T.to_string(header=False))
    print("saved to:", filepath)

    if not drift.empty:
        os.makedirs(mu.FEATURE_DRIFT, exist_ok=True)
        drift_path = mu.FEATURE_DRIFT + "feature_drift_" + mu.partition_suffix(snapshot_date_str) + ".parquet"
        drift.to_parquet(drift_path, index=False)
        print("features by CSI (top 5):")
        print(drift.head(5)[["feature", "csi", "csi_status"]].to_string(index=False))
        print("saved to:", drift_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="monitor model stability and performance")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
