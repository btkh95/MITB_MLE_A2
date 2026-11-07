"""Run one medallion stage for one snapshot date. Called by the Airflow DAG.

    python scripts/run_data_pipeline.py --stage bronze_lms --snapshotdate 2023-01-01
"""

import argparse
import os

import pyspark
import yaml

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table
from utils.check_data_files import check_data_files


BRONZE = "datamart/bronze/"
SILVER = "datamart/silver/"
GOLD = "datamart/gold/"
GOLD_FEATURES = GOLD + "feature_store/"

# silver stage -> (processing function, bronze subdir, silver subdir)
SILVER_STAGES = {
    "silver_lms": (utils.data_processing_silver_table.process_lms_silver_table, "lms", "loan_daily"),
    "silver_fe_attr": (utils.data_processing_silver_table.process_fe_attr_silver_table, "fe_attr", "fe_attr"),
    "silver_fe_fin": (utils.data_processing_silver_table.process_fe_fin_silver_table, "fe_fin", "fe_fin"),
    "silver_fe_click": (utils.data_processing_silver_table.process_fe_click_silver_table, "fe_click", "fe_click"),
}

# gold feature stage -> (processing function, silver subdir, gold feature subdir)
GOLD_FEATURE_STAGES = {
    "gold_fe_attr": (utils.data_processing_gold_table.process_fe_attr_gold_table, "fe_attr", "fe_attr"),
    "gold_fe_fin": (utils.data_processing_gold_table.process_fe_fin_gold_table, "fe_fin", "fe_fin"),
    "gold_fe_click": (utils.data_processing_gold_table.process_fe_click_gold_table, "fe_click", "fe_click"),
}


def run_stage(stage, date_str, spark):
    # bronze_<dataset>: dataset names come from static/dir.yaml
    if stage.startswith("bronze_"):
        with open("static/dir.yaml") as file:
            config = yaml.safe_load(file)
        check_data_files(config)
        dataset = stage[len("bronze_"):]
        csv_file_path = "data/" + config["bronze"]["datasets"][dataset]["filename"]
        bronze_directory = BRONZE + dataset + "/"
        os.makedirs(bronze_directory, exist_ok=True)
        utils.data_processing_bronze_table.process_bronze_table(date_str, csv_file_path, bronze_directory, spark)

    elif stage in SILVER_STAGES:
        process, bronze_subdir, silver_subdir = SILVER_STAGES[stage]
        os.makedirs(SILVER + silver_subdir + "/", exist_ok=True)
        process(date_str, BRONZE + bronze_subdir + "/", SILVER + silver_subdir + "/", spark)

    elif stage == "gold_label_store":
        os.makedirs(GOLD + "label_store/", exist_ok=True)
        utils.data_processing_gold_table.process_lms_gold_table(
            date_str, SILVER + "loan_daily/", GOLD + "label_store/", spark, dpd=30, mob=6)

    elif stage in GOLD_FEATURE_STAGES:
        process, silver_subdir, gold_subdir = GOLD_FEATURE_STAGES[stage]
        os.makedirs(GOLD_FEATURES + gold_subdir + "/", exist_ok=True)
        process(date_str, SILVER + silver_subdir + "/", GOLD_FEATURES + gold_subdir + "/", spark)

    elif stage == "gold_feature_store":
        os.makedirs(GOLD_FEATURES + "joined/", exist_ok=True)
        utils.data_processing_gold_table.process_features_gold_table(date_str, GOLD_FEATURES, spark)

    else:
        raise ValueError(f"unknown stage: {stage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run one data pipeline stage")
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder \
        .appName(args.stage) \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    run_stage(args.stage, args.snapshotdate, spark)

    spark.stop()
