"""Champion/challenger decision for the model trained at the snapshot date.

    python scripts/model_promote.py --snapshotdate 2024-01-01

The challenger is promoted when there is no champion, when the champion needs features
this month's applicants lack, or when it beats the champion's AUC on the challenger's OOT
set by the configured margin. Every decision is appended to model_bank/model_registry.csv.
"""

import argparse

from sklearn.metrics import roc_auc_score

from utils import model_utils as mu


def main(snapshot_date_str):
    config = mu.load_config()
    challenger = mu.load_artefact(mu.model_version(snapshot_date_str))
    champion_version = mu.champion_as_of(snapshot_date_str, inclusive=False)

    row = {
        "decision_date": snapshot_date_str,
        "model_version": challenger["model_version"],
        "model_type": challenger["model_type"],
        "challenger_oot_auc": challenger["metrics"]["auc_oot"],
        "challenger_oot_gini": challenger["metrics"]["gini_oot"],
        "champion_before": champion_version,
        "champion_oot_auc": None,
    }

    if champion_version is None:
        promoted, reason = True, "first_model"
    else:
        champion = mu.load_artefact(champion_version)
        _, dropped = mu.available_features(mu.read_features(snapshot_date_str), config)
        champion_columns = sum(champion["features"].values(), [])
        unavailable = [c for c in champion_columns if c in dropped]

        if unavailable:
            promoted, reason = True, "champion_features_unavailable"
            print("champion needs features missing for current applicants:", unavailable)
        else:
            # score both on the challenger's OOT set (newest mature labels)
            data = mu.load_training_data(snapshot_date_str)
            _, _, oot = mu.split_train_val_oot(data, config)
            proba = champion["pipeline"].predict_proba(mu.prepare_X(oot, champion["features"]))[:, 1]
            row["champion_oot_auc"] = round(float(roc_auc_score(oot["label"], proba)), 4)
            gain = row["challenger_oot_auc"] - row["champion_oot_auc"]
            promoted = gain >= config["promotion"]["min_auc_gain"]
            reason = "auc_gain" if promoted else "insufficient_auc_gain"
            print(f"challenger AUC {row['challenger_oot_auc']} vs champion {row['champion_oot_auc']} "
                  f"(gain {gain:+.4f}, need {config['promotion']['min_auc_gain']})")

    row["promoted"] = promoted
    row["reason"] = reason
    mu.write_registry_row(row)
    print(f"{challenger['model_version']}: promoted={promoted} ({reason}); "
          f"champion now {mu.champion_as_of(snapshot_date_str)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="promote the challenger if it beats the champion")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
