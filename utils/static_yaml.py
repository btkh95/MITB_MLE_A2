import yaml
import pyspark.sql.functions as F

from pyspark.sql.functions import col


def read_static_yaml(filename):
    # read a config file from the static folder
    with open("static/" + filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_static_yaml(filename, content):
    # write a config file to the static folder
    filepath = "static/" + filename
    with open(filepath, "w", encoding="utf-8") as file:
        yaml.safe_dump(content, file, sort_keys=False)
    print('saved to:', filepath)


def get_loan_types(silver_fe_fin_directory, spark):
    # connect to all silver financial partitions
    df = spark.read.parquet(silver_fe_fin_directory + "silver_fe_fin_*.parquet")

    # split each loan list into one row per loan type; missing lists are dropped
    df = df.select(F.explode(F.split(col("Type_of_Loan"), r",\s*")).alias("loan_type"))
    df = df.select(F.trim(col("loan_type")).alias("loan_type")).distinct()

    return sorted(row.loan_type for row in df.collect())
