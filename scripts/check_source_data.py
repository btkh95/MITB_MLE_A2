"""Check that one source file is present and readable before its bronze step runs.

    python scripts/check_source_data.py --source lms --snapshotdate 2023-01-01

Fails (non-zero exit, so the Airflow task fails and nothing downstream runs) when the file
listed in static/dir.yaml is missing, empty, or lacks the key columns. Prints how many rows the
file has for the snapshot date; zero rows is allowed, because some sources have no data in some
months (for example no new applications after 2025-01).
"""

import argparse
import os
import sys

import pandas as pd
import yaml

KEY_COLUMNS = ["Customer_ID", "snapshot_date"]


def main(source, snapshot_date_str):
    with open("static/dir.yaml", encoding="utf-8") as file:
        datasets = yaml.safe_load(file)["bronze"]["datasets"]
    if source not in datasets:
        sys.exit(f"unknown source '{source}'; expected one of {sorted(datasets)}")

    path = os.path.join("data", datasets[source]["filename"])
    if not os.path.isfile(path):
        sys.exit(f"source file missing: {path}")
    if os.path.getsize(path) == 0:
        sys.exit(f"source file is empty: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in KEY_COLUMNS if c not in header]
    if missing:
        sys.exit(f"{path} lacks key columns {missing}")

    dates = pd.read_csv(path, usecols=["snapshot_date"])["snapshot_date"].astype(str)
    if dates.empty:
        sys.exit(f"source file has no rows: {path}")
    rows = int((dates == snapshot_date_str).sum())
    print(f"{path}: {len(dates)} rows, {len(header)} columns; {rows} rows dated {snapshot_date_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="check one source file before its bronze step")
    parser.add_argument("--source", type=str, required=True, help="dataset name in static/dir.yaml")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.source, args.snapshotdate)
