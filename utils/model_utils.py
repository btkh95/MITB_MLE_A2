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
SHADOW_PREDICTIONS = PREDICTIONS + "shadow/"
MONITORING = "datamart/gold/model_monitoring/"
FEATURE_DRIFT = MONITORING + "feature_drift/"
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


def split_train_holdout_oot(data, config):
    """Split labelled loans into train, the model-selection holdout and the out-of-time promotion set.

    oot = the newest label months (out-of-time promotion set: champion vs challenger, and whether
    calibration is kept). The earlier loans split, stratified by label, into train (fitting, with
    3-fold cross-validation inside it for tuning) and holdout (model-selection holdout: picks
    logistic regression or XGBoost, and fits the Platt scaling).
    """
    from sklearn.model_selection import train_test_split

    # train_test_split picks rows by position, so sort by the unique loan_id first: the same loans
    # then always land in the same split, whatever order the label and feature files were read in
    data = data.sort_values("loan_id", kind="mergesort").reset_index(drop=True)
    label_months = sorted(data["label_snapshot_date"].unique())
    oot_months = label_months[-config["training"]["oot_label_months"]:]
    oot = data[data["label_snapshot_date"].isin(oot_months)]
    earlier = data[~data["label_snapshot_date"].isin(oot_months)]
    train, holdout = train_test_split(earlier, test_size=config["training"]["holdout_ratio"],
                                      random_state=config["training"]["random_state"],
                                      stratify=earlier["label"])
    return train, holdout, oot


#################
# Data checks   #
#################
# Each check stops the task with a clear message, so Airflow marks it failed instead of
# training or scoring on broken input.

def check_training_data(data):
    problems = []
    if data.empty:
        problems.append("no labelled loans with features")
    else:
        if not data["label"].isin([0, 1]).all():
            problems.append("labels other than 0/1")
        if data["label"].nunique() < 2:
            problems.append("only one label class")
        if data["loan_id"].duplicated().any():
            problems.append(f"{int(data['loan_id'].duplicated().sum())} duplicate loan_id rows")
    if problems:
        raise ValueError("training data check failed: " + "; ".join(problems))


def check_applicants(applicants, features):
    columns = sum(features.values(), [])
    problems = []
    missing = [c for c in columns if c not in applicants]
    if missing:
        problems.append(f"columns missing from the feature store: {missing}")
    if applicants.duplicated(KEY_COLUMNS).any():
        problems.append("duplicate Customer_ID rows for one snapshot date")
    if problems:
        raise ValueError("applicant data check failed: " + "; ".join(problems))


def check_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("score check failed: probabilities must be finite and between 0 and 1")


#################
# Lineage       #
#################

def lineage(config, data):
    """What produced a model: package versions, the config used and a fingerprint of the data."""
    import hashlib
    import json
    import platform

    import sklearn
    import xgboost

    config_text = json.dumps(config, sort_keys=True)
    rows = data.sort_values("loan_id")[["loan_id", "label"]]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "config_sha256": hashlib.sha256(config_text.encode()).hexdigest()[:16],
        "config": config["training"],
        # same loans and labels -> same fingerprint, so a rerun can be checked against this one
        "training_data_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(rows, index=False).values.tobytes()).hexdigest()[:16],
    }


#################
# Explanations  #
#################

def _readable(name):
    # "numeric__Interest_Rate" -> "Interest_Rate"; "categorical__Credit_Mix_Bad" -> "Credit_Mix_Bad"
    return name.split("__", 1)[-1]


def contributions(pipeline, X):
    """Each input's push on each applicant's score, on the model's log-odds scale.

    Logistic regression: coefficient x standardised value. XGBoost: its built-in SHAP values
    (pred_contribs), which split each score exactly into one part per input plus a baseline.
    Positive = pushes towards default. Returns a DataFrame with one column per model input.
    """
    import xgboost as xgb

    preprocess, model = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    matrix = preprocess.transform(X)
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=float)
    names = [_readable(n) for n in preprocess.get_feature_names_out()]
    if hasattr(model, "coef_"):
        values = matrix * model.coef_[0]
    else:
        values = model.get_booster().predict(xgb.DMatrix(matrix), pred_contribs=True)[:, :-1]
    return pd.DataFrame(values, columns=names, index=X.index)


def reason_codes(pipeline, X, top_n=3):
    """The top_n inputs that pushed each applicant's default risk up the most."""
    contrib = contributions(pipeline, X)
    names = np.array(contrib.columns)
    order = np.argsort(-contrib.values, axis=1)[:, :top_n]
    pushes_up = np.take_along_axis(contrib.values, order, axis=1) > 0
    codes = np.where(pushes_up, names[order], None)
    return pd.DataFrame(codes, columns=[f"reason_{i + 1}" for i in range(top_n)], index=X.index)


def feature_importance(pipeline, X):
    """Average size of each input's push across X (mean |contribution|), largest first."""
    contrib = contributions(pipeline, X).abs().mean()
    return (contrib.sort_values(ascending=False).rename("mean_abs_contribution")
            .rename_axis("feature").reset_index())


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


def _bins(values, bins=10):
    # deciles for numbers with many values; the values themselves otherwise; missing is its own bin
    values = pd.Series(values)
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > bins:
        binned = pd.qcut(values, q=bins, duplicates="drop").astype(str)
    else:
        binned = values.astype(str)
    return binned.where(values.notna(), "missing")


def information_value(values, label, bins=10):
    """Information value (IV): how well one feature separates defaulters from non-defaulters.

    Rule of thumb: < 0.02 not useful, 0.02-0.1 weak, 0.1-0.3 medium, > 0.3 strong.
    Half a loan is added to every bin so an empty bin does not give an infinite value.
    """
    table = pd.crosstab(_bins(values, bins), pd.Series(label).values) + 0.5
    bad = table[1] / table[1].sum()
    good = table[0] / table[0].sum()
    return float(np.sum((good - bad) * np.log(good / bad)))


def select_model(holdout_gini, simplest_first, tie_margin):
    """Pick the model type to keep, using the model-selection holdout Gini only (never OOT).

    holdout_gini: {model_type: holdout Gini}. simplest_first: model types ordered from the
    simplest to the most complex. The highest holdout Gini wins, unless a simpler model
    is within tie_margin of it; then the simplest such model wins, because a gap that small
    is within the noise of a ~500-row holdout and the simpler model is easier to
    explain and less prone to overfit. Returns (model_type, reason).
    """
    top = max(holdout_gini.values())
    best = max(holdout_gini, key=holdout_gini.get)
    for model_type in simplest_first:
        if model_type in holdout_gini and holdout_gini[model_type] >= top - tie_margin:
            if model_type == best:
                return model_type, "highest holdout Gini"
            return model_type, (f"within {tie_margin} of the best holdout Gini "
                                f"({best} {top:.4f}); simpler model preferred")
    return best, "highest holdout Gini"


def prune_features(spec, train, config):
    """Drop weak and duplicate features using the training split only.

    1. weak: information value below training.min_information_value
    2. duplicate: of two numeric inputs with |Spearman correlation| above
       training.max_correlation, keep the one with the higher information value.
    A *_log1p column follows its raw twin, so logistic regression and XGBoost see the same set.
    Returns (pruned spec, table with one row per feature and the reason it was dropped, if any).
    """
    min_iv = config["training"]["min_information_value"]
    max_corr = config["training"]["max_correlation"]
    twins = [c for c in spec["numeric"] if c.endswith("_log1p")]
    columns = [c for group in ["numeric", "flags", "categorical"] for c in spec[group] if c not in twins]

    report = pd.DataFrame({"feature": columns})
    report["information_value"] = [information_value(train[c], train["label"]) if c in train else 0.0
                                   for c in columns]
    report["dropped"] = np.where(report["information_value"] < min_iv, "weak", "")
    iv = report.set_index("feature")["information_value"]

    numeric = [c for c in spec["numeric"] + spec["flags"] if c in columns
               and report.loc[report["feature"] == c, "dropped"].item() == ""]
    corr = train[numeric].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
    removed = set()
    for a in sorted(numeric, key=lambda c: -iv[c]):  # strongest first, so it is the one kept
        if a in removed:
            continue
        for b in numeric:
            if b != a and b not in removed and corr.loc[a, b] > max_corr:
                removed.add(b)
                report.loc[report["feature"] == b, "dropped"] = f"duplicate of {a}"

    dropped = set(report.loc[report["dropped"] != "", "feature"])
    dropped |= {c for c in twins if c[:-len("_log1p")] in dropped}
    pruned = {group: [c for c in cols if c not in dropped] for group, cols in spec.items()}
    return pruned, report.sort_values("information_value", ascending=False, ignore_index=True)


#################
# Model bank    #
#################

# short model family codes used in model names
FAMILY_CODES = {"logistic_regression": "logreg", "xgboost": "xgboost"}


def model_version(train_date_str, model_type):
    """Model name = training date + model family, e.g. credit_model_2024_01_01_logreg."""
    return f"credit_model_{partition_suffix(train_date_str)}_{FAMILY_CODES[model_type]}"


def version_date(version):
    # credit_model_2024_01_01_logreg -> 2024-01-01
    return "-".join(version[len("credit_model_"):].split("_")[:3])


def runner_up_name(champion_version, runner_up_type):
    """Shadow name for the champion's runner-up, e.g. credit_model_2024_01_01_xgboost_runner_up."""
    date = version_date(champion_version)
    return model_version(date, runner_up_type) + "_runner_up"


def version_trained_on(train_date_str):
    """The model saved for a training date (one per date: the winner of that date's comparison)."""
    paths = glob.glob(MODEL_BANK + "credit_model_" + partition_suffix(train_date_str) + "_*/model.pkl")
    if len(paths) != 1:
        raise ValueError(f"expected one model trained on {train_date_str}, found {len(paths)}")
    return os.path.basename(os.path.dirname(paths[0]))


def remove_models_trained_on(train_date_str):
    # retraining a date replaces its model, even if a different family wins this time
    import shutil
    for path in glob.glob(MODEL_BANK + "credit_model_" + partition_suffix(train_date_str) + "_*"):
        shutil.rmtree(path)


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


def shadows_as_of(date_str):
    """Models that score applicants alongside the champion without affecting decisions.

    1. the champion's runner-up: the other model type trained on the same date
    2. challengers trained after the champion that were not promoted
    Returns a list of (shadow name, artefact version, artefact key holding the pipeline).
    """
    champion = champion_as_of(date_str)
    if champion is None:
        return []
    registry = read_registry()
    champion_type = registry.loc[registry["model_version"] == champion, "model_type"]
    other = [t for t in FAMILY_CODES if champion_type.empty or t != champion_type.iloc[0]]
    shadows = [(runner_up_name(champion, other[0]), champion, "runner_up")]
    champion_date = registry.loc[registry["model_version"] == champion, "decision_date"].max()
    rejected = registry[(registry["promoted"].astype(str) != "True") &
                        (registry["decision_date"] > champion_date) &
                        (registry["decision_date"] <= date_str)]
    shadows += [(v, v, "pipeline") for v in rejected["model_version"]]
    return shadows


#################
# Monitoring    #
#################

def _stability_index(ref_share, cur_share):
    # sum over bins of (current% - reference%) * ln(current% / reference%); empty bins floored
    ref_share = np.clip(np.asarray(ref_share, dtype=float), 1e-4, None)
    cur_share = np.clip(np.asarray(cur_share, dtype=float), 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi(reference, current, bins=10):
    """Population stability index of current scores against reference (decile bins of reference)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_share = np.histogram(current, bins=edges)[0] / len(current)
    return _stability_index(ref_share, cur_share)


def csi(reference, current, bins=10):
    """Characteristic stability index: the PSI formula applied to one input feature.

    Numeric features with many values use decile bins of the reference. Flags, categories and
    numbers with few distinct values are compared category by category. Missing values are
    their own bin, so a feature that stops being collected shows up as drift.
    """
    reference = pd.Series(reference)
    current = pd.Series(current)
    numeric = pd.api.types.is_numeric_dtype(reference) and reference.nunique() > bins
    if numeric:
        reference = reference.astype(float)
        current = pd.to_numeric(current, errors="coerce").astype(float)
        edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf

        def shares(values):
            counts = np.histogram(values.dropna(), bins=edges)[0]
            return np.append(counts, values.isna().sum()) / len(values)

        return _stability_index(shares(reference), shares(current))

    ref_cat = reference.astype(object).where(reference.notna(), "missing").astype(str)
    cur_cat = current.astype(object).where(current.notna(), "missing").astype(str)
    categories = sorted(set(ref_cat) | set(cur_cat))
    ref_share = ref_cat.value_counts(normalize=True).reindex(categories, fill_value=0)
    cur_share = cur_cat.value_counts(normalize=True).reindex(categories, fill_value=0)
    return _stability_index(ref_share.values, cur_share.values)


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
