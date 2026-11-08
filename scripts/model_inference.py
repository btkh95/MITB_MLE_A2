"""Score the snapshot date's loan applications with the champion model, and with shadow models.

    python scripts/model_inference.py --snapshotdate 2024-01-01

Champion scores go to datamart/gold/model_predictions/ with the top 3 reasons behind each
score. Shadow models (the champion's runner-up and any rejected challenger) score the same
applicants into model_predictions/shadow/. Their scores are never used for decisions; they
are compared with the champion once the default labels arrive.
"""

import argparse
import os

import pandas as pd

from utils import model_utils as mu


def score(pipeline, features, applicants):
    X = mu.prepare_X(applicants, features)
    scores = pipeline.predict_proba(X)[:, 1]
    mu.check_scores(scores)
    return X, scores


def save(frame, folder, prefix, snapshot_date_str):
    os.makedirs(folder, exist_ok=True)
    filepath = folder + prefix + mu.partition_suffix(snapshot_date_str) + ".parquet"
    frame.to_parquet(filepath, index=False)
    print("saved to:", filepath)


def main(snapshot_date_str):
    version = mu.champion_as_of(snapshot_date_str)
    if version is None:
        print("no champion model yet; skipping inference")
        return

    applicants = mu.read_features(snapshot_date_str)
    if applicants.empty:
        print("no applications on", snapshot_date_str, "; skipping inference")
        return

    # --- champion: the scores the business acts on ---
    artefact = mu.load_artefact(version)
    mu.check_applicants(applicants, artefact["features"])
    X, scores = score(artefact["pipeline"], artefact["features"], applicants)
    predictions = applicants[mu.KEY_COLUMNS].copy()
    predictions["model_version"] = version
    predictions["model_predictions"] = scores
    # reason codes: the inputs that pushed each applicant's risk up the most
    reasons = mu.reason_codes(artefact.get("base_pipeline", artefact["pipeline"]), X)
    predictions = pd.concat([predictions, reasons], axis=1)
    print(f"scored {len(predictions)} applications with {version}; "
          f"mean default probability {scores.mean():.3f}")
    print("most common top reason:", predictions["reason_1"].value_counts().head(3).to_dict())
    save(predictions, mu.PREDICTIONS, "predictions_", snapshot_date_str)

    # --- shadows: same applicants, scores stored but never used ---
    shadow_frames = []
    for name, shadow_version, key in mu.shadows_as_of(snapshot_date_str):
        shadow = mu.load_artefact(shadow_version)
        if key == "runner_up":
            if "runner_up" not in shadow:
                continue
            pipeline, features = shadow["runner_up"]["pipeline"], shadow["runner_up"]["features"]
        else:
            pipeline, features = shadow["pipeline"], shadow["features"]
        _, shadow_scores = score(pipeline, features, applicants)
        frame = applicants[mu.KEY_COLUMNS].copy()
        frame["model_version"] = version  # the champion these shadow scores sit next to
        frame["shadow_model"] = name
        frame["model_predictions"] = shadow_scores
        shadow_frames.append(frame)
        print(f"shadow {name}: mean default probability {shadow_scores.mean():.3f}")
    if shadow_frames:
        save(pd.concat(shadow_frames, ignore_index=True), mu.SHADOW_PREDICTIONS,
             "shadow_predictions_", snapshot_date_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score applications with the champion and shadow models")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
