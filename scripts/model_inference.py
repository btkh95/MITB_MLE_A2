"""Score the snapshot date's loan applications with the champion model.

    python scripts/model_inference.py --snapshotdate 2024-01-01
"""

import argparse
import os

from utils import model_utils as mu


def main(snapshot_date_str):
    version = mu.champion_as_of(snapshot_date_str)
    if version is None:
        print("no champion model yet; skipping inference")
        return

    applicants = mu.read_features(snapshot_date_str)
    if applicants.empty:
        print("no applications on", snapshot_date_str, "; skipping inference")
        return

    artefact = mu.load_artefact(version)
    X = mu.prepare_X(applicants, artefact["features"])
    predictions = applicants[mu.KEY_COLUMNS].copy()
    predictions["model_version"] = version
    predictions["model_predictions"] = artefact["pipeline"].predict_proba(X)[:, 1]

    os.makedirs(mu.PREDICTIONS, exist_ok=True)
    filepath = mu.PREDICTIONS + "predictions_" + mu.partition_suffix(snapshot_date_str) + ".parquet"
    predictions.to_parquet(filepath, index=False)
    print(f"scored {len(predictions)} applications with {version}; "
          f"mean default probability {predictions['model_predictions'].mean():.3f}")
    print("saved to:", filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score applications with the champion model")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
