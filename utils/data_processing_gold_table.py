import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from utils.static_yaml import read_static_yaml


def _validate_join_keys(df, keys, table_name):
    # Reject ambiguous joins rather than multiplying rows or choosing arbitrary records.
    missing_key = col(keys[0]).isNull()
    for key in keys[1:]:
        missing_key = missing_key | col(key).isNull()
    if df.filter(missing_key).limit(1).count():
        raise ValueError(f"{table_name}: null join keys in {keys}")
    if df.groupBy(*keys).count().filter(col("count") > 1).limit(1).count():
        raise ValueError(f"{table_name}: duplicate join keys in {keys}")


def process_lms_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver LMS table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # filter data: select loans at the target month on book
    df = df.filter(col("mob") == mob)

    # augment data: add default label using the days-past-due threshold
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))

    # augment data: add label definition (for example, 30dpd_6mob)
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # keep loan_start_date so gold features can be matched to the application date
    df = df.select("loan_id", "Customer_ID", "loan_start_date",
                   "label", "label_def", "snapshot_date")

    # save gold LMS labels - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


def process_fe_attr_gold_table(snapshot_date_str, silver_fe_attr_directory, gold_fe_attr_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver attribute table
    partition_name = "silver_fe_attr_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fe_attr_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # augment data: flag missing occupation before filling the category
    df = df.withColumn("occupation_missing", col("Occupation").isNull())

    # transform data: represent missing occupation as an explicit category
    df = df.fillna({"Occupation": "Unknown"})

    # select columns to save: raw audit values remain available in silver
    df = df.drop("Age_raw", "snapshot_date_raw")

    # save gold table - IRL connect to database to write
    partition_name = "gold_fe_attr_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_fe_attr_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_fe_fin_gold_table(snapshot_date_str, silver_fe_fin_directory, gold_fe_fin_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver financial table
    partition_name = "silver_fe_fin_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fe_fin_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # augment data: compare annualized monthly salary with reported annual income
    df = df.withColumn("months12_to_annual_income_ratio",
        F.when(col("Annual_Income") > 0,
               col("Monthly_Inhand_Salary") * 12 / col("Annual_Income")))

    # augment data: calculate the share of monthly salary committed to installments
    df = df.withColumn("EMI_to_Salary_Ratio",
        F.when(col("Monthly_Inhand_Salary") > 0,
               col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")))

    # augment data: calculate outstanding debt relative to annual income
    df = df.withColumn("Debt_to_Income",
        F.when(col("Annual_Income") > 0,
               col("Outstanding_Debt") / col("Annual_Income")))

    # augment data: calculate monthly salary remaining after installments and investments
    df = df.withColumn("Disposable_Income",
        col("Monthly_Inhand_Salary") - col("Total_EMI_per_month") - col("Amount_invested_monthly"))

    # augment data: credit exposure ratios
    # total financial products held
    df = df.withColumn("Num_Fin_Pdts",
        col("Num_Bank_Accounts") + col("Num_Credit_Card") + col("Num_of_Loan"))

    # loans relative to other credit products, a sign of how loans are being funded
    df = df.withColumn("Loans_per_Credit_Item",
        F.when((col("Num_Bank_Accounts") + col("Num_Credit_Card")) > 0,
               col("Num_of_Loan") / (col("Num_Bank_Accounts") + col("Num_Credit_Card"))))

    # average payment delay per loan
    df = df.withColumn("Loan_Extent",
        F.when(col("Num_of_Loan") > 0, col("Delay_from_due_date") / col("Num_of_Loan")))

    # late payments per loan held before applying: the same 12 late payments mean more
    # over 2 loans than over 6. No loans -> null, not a divide-by-zero.
    df = df.withColumn("Delayed_Payments_per_Loan",
        F.when(col("Num_of_Loan") > 0, col("Num_of_Delayed_Payment") / col("Num_of_Loan")))

    # augment data: count each loan type, preserving repeated loans and unknown lists
    # Fixed categories from static/loan_types.yaml keep the schema consistent between monthly partitions.
    loan_types = read_static_yaml("loan_types.yaml")["Type_of_Loan"]
    loans = F.transform(F.split(col("Type_of_Loan"), r",\s*"), lambda loan: F.trim(loan))
    for loan_type in loan_types:
        column_name = "Loan_" + loan_type.replace(" ", "_").replace("-", "_")
        df = df.withColumn(column_name,
            F.when(~col("loan_types_missing"),
                F.size(F.filter(loans, lambda loan: loan == F.lit(loan_type)))))

    # augment data: extract spending behaviour (High or Low)
    df = df.withColumn("Spending_Behaviour", F.split(col("Payment_Behaviour"), "_")[0])

    # augment data: extract payment size (Small, Medium or Large)
    df = df.withColumn("Payments_Size", F.split(col("Payment_Behaviour"), "_")[2])

    # transform data: represent missing categories without learning from future data
    df = df.fillna({column: "Unknown" for column in [
        "Credit_Mix", "Payment_of_Min_Amount", "Spending_Behaviour", "Payments_Size"
    ]})

    # augment data: add log features for nonnegative monetary values
    # Keep original values; negative balances have a null log feature.
    for column in ["Annual_Income", "Monthly_Inhand_Salary", "Outstanding_Debt",
                   "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]:
        df = df.withColumn(column + "_log1p",
            F.when(col(column) >= 0, F.log1p(col(column))))

    # select columns to save: source text and raw audit values remain in silver
    df = df.drop("Num_of_Loan_raw", "snapshot_date_raw", "Type_of_Loan",
                 "Payment_Behaviour", "Credit_History_Age")

    # save gold table - IRL connect to database to write
    partition_name = "gold_fe_fin_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_fe_fin_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_fe_click_gold_table(snapshot_date_str, silver_fe_click_directory, gold_fe_click_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver clickstream history
    filepath = silver_fe_click_directory + "silver_fe_click_*.parquet"
    df = spark.read.parquet(filepath)

    # filter data: use only activity strictly before the feature snapshot date
    # This snapshot can later be joined to a loan starting on the same date.
    df = df.filter(col("snapshot_date") < F.lit(snapshot_date.date()))
    print('loaded from:', filepath, 'historical row count:', df.count())

    # augment data: calculate historical mean for each anonymous feature per customer
    feature_columns = sorted(column for column in df.columns if column.startswith("fe_"))
    aggregations = [F.mean(col(column)).alias(column + "_mean") for column in feature_columns]

    # augment data: count distinct observed activity dates, not individual rows
    aggregations.append(F.countDistinct("snapshot_date").alias("clickstream_days"))
    df = df.groupBy("Customer_ID").agg(*aggregations)

    # augment data: identify customers with prior clickstream activity
    # Customers absent from this table need a zero flag after a downstream left join.
    df = df.withColumn("has_clickstream", F.lit(1).cast(IntegerType()))

    # augment data: record the feature cutoff date for point-in-time joins
    df = df.withColumn("snapshot_date", F.lit(snapshot_date.date()).cast(DateType()))

    # save gold table - IRL connect to database to write
    partition_name = "gold_fe_click_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_fe_click_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_features_gold_table(snapshot_date_str, gold_feature_store_directory, spark):
    # connect to this snapshot's gold feature tables so each Airflow run builds one partition
    date_suffix = snapshot_date_str.replace('-','_') + '.parquet'
    attr_df = spark.read.parquet(gold_feature_store_directory + "fe_attr/gold_fe_attr_" + date_suffix)
    fin_df = spark.read.parquet(gold_feature_store_directory + "fe_fin/gold_fe_fin_" + date_suffix)
    click_df = spark.read.parquet(gold_feature_store_directory + "fe_click/gold_fe_click_" + date_suffix)

    # validate data: each feature source must have one row per customer and feature date
    keys = ["Customer_ID", "snapshot_date"]
    for name, source in [("attributes", attr_df), ("financials", fin_df), ("clickstream", click_df)]:
        _validate_join_keys(source, keys, name)

    # augment data: distinguish a missing source row from missing individual fields
    attr_df = attr_df.withColumn("has_attributes", F.lit(1).cast(IntegerType()))
    fin_df = fin_df.withColumn("has_financials", F.lit(1).cast(IntegerType()))

    # join data: preserve customers present in either attributes or financials
    df = attr_df.join(fin_df, on=keys, how="outer")

    # join data: match historical clickstream at the same feature cutoff date
    df = df.join(click_df, on=keys, how="left")

    # clean data: absent history is zero observations; unknown numeric features -> null
    df = df.fillna({"has_attributes": 0, "has_financials": 0,
                    "has_clickstream": 0, "clickstream_days": 0})

    # save gold feature store - one row per customer and feature date, partitioned by snapshot
    filepath = gold_feature_store_directory + "joined/gold_feature_store_" + date_suffix
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'row count:', df.count())

    return df
