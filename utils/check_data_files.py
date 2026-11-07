import os
import glob
import warnings


def check_data_files(config, data_directory="data"):
    """Print CSV file status and warn about differences from the bronze config."""
    expected_files = {
        dataset["filename"]
        for dataset in config["bronze"]["datasets"].values()
    }

    actual_files = {
        os.path.basename(filepath)
        for filepath in glob.glob(os.path.join(data_directory, "*.csv"))
        if os.path.isfile(filepath)
    }

    missing_files = expected_files - actual_files
    extra_files = actual_files - expected_files

    print(f"CSV files in {data_directory}: {', '.join(sorted(actual_files)) or 'None'}")
    print(f"Missing files: {', '.join(sorted(missing_files)) or 'None'}")
    print(f"Extra files: {', '.join(sorted(extra_files)) or 'None'}")

    if not missing_files:
        print("All expected files found:")
        for filename in sorted(expected_files):
            print(f"  - {filename}")

    if missing_files:
        warnings.warn(
            f"Files listed in dir.yaml but missing from {data_directory}: "
            f"{sorted(missing_files)}",
            stacklevel=2,
        )

    if extra_files:
        warnings.warn(
            f"Files found in {data_directory} but not listed in dir.yaml: "
            f"{sorted(extra_files)}",
            stacklevel=2,
        )
