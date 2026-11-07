"""Shared helpers for model training, inference and monitoring.

ML stages read gold tables with pandas: one month of loans is small, and this
keeps the Spark start-up cost out of every model task.
"""

import glob
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta


LABEL_STORE = "datamart/gold/label_store/"
FEATURE_STORE = "datamart/gold/feature_store/joined/"
PREDICTIONS = "datamart/gold/model_predictions/"
MONITORING = "datamart/gold/model_monitoring/"
MODEL_BANK = "model_bank/"
REGISTRY = MODEL_BANK + "model_registry.csv"

KEY_COLUMNS = ["Customer_ID", "snapshot_date"]


def load_config():
    with open("static/model_config.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


def to_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def shift_months(date_str, months):
    return (to_date(date_str) + relativedelta(months=months)).strftime("%Y-%m-%d")


def months_between(earlier_str, later_str):
    delta = relativedelta(to_date(later_str), to_date(earlier_str))
    return delta.years * 12 + delta.months


def partition_suffix(date_str):
    return date_str.replace("-", "_")


def partition_date(path):
    # gold_label_store_2023_07_01.parquet -> 2023-07-01
    stem = os.path.basename(path.rstrip("/")).split(".")[0]
    return "-".join(stem.split("_")[-3:])


#################
# Gold readers  #
#################

def read_features(date_str):
    # applications whose features were captured on date_str; empty if the partition is missing
    path = FEATURE_STORE + "gold_feature_store_" + partition_suffix(date_str) + ".parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y-%m-%d")
    return df


def read_labels(date_str):
    path = LABEL_STORE + "gold_label_store_" + partition_suffix(date_str) + ".parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    for column in ["loan_start_date", "snapshot_date"]:
        df[column] = pd.to_datetime(df[column]).dt.strftime("%Y-%m-%d")
    return df


def load_training_data(train_date_str):
    """Labelled loans known on train_date_str, joined to their application-date features.

    A label is usable only once observed: label snapshot_date <= train date.
    """
    label_paths = sorted(glob.glob(LABEL_STORE + "gold_label_store_*.parquet"))
    label_dates = [partition_date(p) for p in label_paths if partition_date(p) <= train_date_str]
    labels = pd.concat([read_labels(d) for d in label_dates], ignore_index=True)
    labels = labels.rename(columns={"snapshot_date": "label_snapshot_date"})

    feature_dates = sorted(labels["loan_start_date"].unique())
    features = pd.concat([read_features(d) for d in feature_dates], ignore_index=True)

    # point-in-time join: features captured on the application date
    data = labels.merge(features, left_on=["Customer_ID", "loan_start_date"],
                        right_on=KEY_COLUMNS, how="inner")
    print(f"labelled loans known on {train_date_str}: {len(labels)}, with features: {len(data)}")
    return data


def split_train_val_oot(data, config):
    """OOT = newest label months; the earlier loans split into train and validation."""
    from sklearn.model_selection import train_test_split

    label_months = sorted(data["label_snapshot_date"].unique())
    oot_months = label_months[-config["training"]["oot_label_months"]:]
    oot = data[data["label_snapshot_date"].isin(oot_months)]
    earlier = data[~data["label_snapshot_date"].isin(oot_months)]
    train, val = train_test_split(earlier, test_size=config["training"]["validation_ratio"],
                                  random_state=config["training"]["random_state"],
                                  stratify=earlier["label"])
    return train, val, oot


#################
# Features      #
#################

def prepare_X(df, features):
    """Coerce allowlisted columns to model-ready dtypes (same code for training and inference)."""
    X = pd.DataFrame(index=df.index)
    for column in features["numeric"] + features["flags"]:
        X[column] = pd.to_numeric(df[column], errors="coerce").astype(float) if column in df else np.nan
    for column in features["categorical"]:
        X[column] = df[column].astype(object).where(df[column].notna(), np.nan) if column in df else np.nan
    return X


def available_features(applicants, config):
    """Allowlist minus features most of the current applicants lack.

    Returns (feature spec, dropped columns). has_clickstream only has meaning next to the
    clickstream means, so it is dropped with them.
    """
    spec = config["features"]
    max_missing = config["training"]["max_missing_share"]
    dropped = []
    for column in spec["numeric"] + spec["flags"] + spec["categorical"]:
        missing_share = applicants[column].isna().mean() if column in applicants else 1.0
        if missing_share > max_missing:
            dropped.append(column)

    click_means = [c for c in spec["numeric"] if c.startswith("fe_") and c.endswith("_mean")]
    if all(c in dropped for c in click_means) and "has_clickstream" not in dropped:
        dropped.append("has_clickstream")

    kept = {group: [c for c in columns if c not in dropped] for group, columns in spec.items()}
    return kept, dropped


#################
# Model bank    #
#################

def model_version(train_date_str):
    return "credit_model_" + partition_suffix(train_date_str)


def save_artefact(artefact):
    directory = MODEL_BANK + artefact["model_version"] + "/"
    os.makedirs(directory, exist_ok=True)
    with open(directory + "model.pkl", "wb") as file:
        pickle.dump(artefact, file)
    return directory


def load_artefact(version):
    with open(MODEL_BANK + version + "/model.pkl", "rb") as file:
        return pickle.load(file)


def read_registry():
    if not os.path.exists(REGISTRY):
        return pd.DataFrame(columns=["decision_date", "model_version", "promoted"])
    return pd.read_csv(REGISTRY, dtype={"decision_date": str})


def write_registry_row(row):
    # one decision per date, so rerunning a month replaces its row
    registry = read_registry()
    registry = registry[registry["decision_date"] != row["decision_date"]]
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    registry = registry.sort_values("decision_date")
    os.makedirs(MODEL_BANK, exist_ok=True)
    registry.to_csv(REGISTRY, index=False)


def champion_as_of(date_str, inclusive=True):
    """Version of the latest promoted model decided on or before date_str (None if none yet)."""
    registry = read_registry()
    registry = registry[registry["promoted"].astype(str) == "True"]
    if inclusive:
        registry = registry[registry["decision_date"] <= date_str]
    else:
        registry = registry[registry["decision_date"] < date_str]
    if registry.empty:
        return None
    return registry.sort_values("decision_date").iloc[-1]["model_version"]


#################
# Monitoring    #
#################

def psi(reference, current, bins=10):
    """Population stability index of current scores against reference (decile bins of reference)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_share = np.histogram(current, bins=edges)[0] / len(current)
    ref_share = np.clip(ref_share, 1e-4, None)
    cur_share = np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def read_monitoring(date_str):
    path = MONITORING + "monitoring_" + partition_suffix(date_str) + ".parquet"
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()


def should_train(date_str):
    """Training gate: first model, scheduled refresh, or a drift alert after the cooldown."""
    config = load_config()["training"]
    if date_str < config["first_train_date"]:
        print("before first training date")
        return False
    if read_features(date_str).empty:
        print("no applications this month; nothing to train for")
        return False

    registry = read_registry()
    earlier = registry[registry["decision_date"] < date_str]
    if earlier.empty:
        print("no model trained yet -> train")
        return True

    months_since = months_between(earlier["decision_date"].max(), date_str)
    if months_since >= config["retrain_every_months"]:
        print(f"{months_since} months since last training -> scheduled retrain")
        return True

    previous = read_monitoring(shift_months(date_str, -1))
    drift = not previous.empty and previous["retrain_flag"].any()
    if drift and months_since >= config["retrain_cooldown_months"]:
        print("monitoring flagged drift last month -> retrain")
        return True

    print(f"{months_since} months since last training, drift flag: {drift} -> skip")
    return False
