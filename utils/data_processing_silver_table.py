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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType

from utils.static_yaml import read_static_yaml


def process_lms_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed",
        F.when(col("overdue_amt") == 0, F.lit(0))
         .when(col("due_amt") > 0, F.ceil(col("overdue_amt") / col("due_amt")))
         .cast(IntegerType()))
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_fe_click_silver_table(snapshot_date_str, bronze_fe_click_directory, silver_fe_click_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_fe_click_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    input_count = df.count()
    print('loaded from:', filepath, 'row count:', input_count)

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        **{column: IntegerType() for column in df.columns if column.startswith("fe_")},
        "snapshot_date": DateType(),
    }

    for column, data_type in column_type_map.items():
        df = df.withColumn(column, F.expr(
            f"try_cast(`{column}` as {data_type.simpleString()})"))

    # clean data: remove duplicate rows
    df = df.dropDuplicates()
    output_count = df.count()
    print(snapshot_date_str, "clickstream rows:", input_count,
          "duplicates removed:", input_count - output_count)

    # save silver table - IRL connect to database to write
    partition_name = "silver_fe_click_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fe_click_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_fe_attr_silver_table(snapshot_date_str, bronze_fe_attr_directory, silver_fe_attr_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_fe_attr_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # drop PII columns - will never be used in modeling, and should not be stored in silver table
    pii_columns = read_static_yaml("pii.yaml")["pii"]
    df = df.drop(*pii_columns)

    # print out the dropped PII columns and the row count after dropping
    print('dropped PII columns:', pii_columns, 'row count:', df.count())

    # augment data: preserve the original age before cleaning
    df = df.withColumn("Age_raw", col("Age"))

    # clean data: remove trailing underscores and convert age to integer
    df = df.withColumn("Age", F.regexp_replace(F.trim(col("Age")), r"_+$", ""))
    df = df.withColumn("Age", F.expr("try_cast(Age as int)"))

    # augment data: flag missing ages and ages outside 1-100
    df = df.withColumn("age_invalid", col("Age").isNull() | ~col("Age").between(1, 100))

    # clean data: replace invalid ages with null while retaining the customer
    df = df.withColumn("Age", F.when(~col("age_invalid"), col("Age")))

    # clean data: trim occupation and replace blank/underscore placeholders with null
    df = df.withColumn("Occupation", F.trim(col("Occupation")))
    df = df.withColumn("Occupation",
        F.when(col("Occupation").isNull() | (col("Occupation") == "") |
               col("Occupation").rlike(r"^_+$"), F.lit(None))
         .otherwise(col("Occupation")))
    # augment data: preserve the original snapshot date before conversion
    df = df.withColumn("snapshot_date_raw", col("snapshot_date"))

    # clean data: convert snapshot date to date type; invalid values become null
    df = df.withColumn("snapshot_date", F.expr("try_cast(snapshot_date as date)"))

    # report data quality: count invalid ages and missing occupations
    print("attribute quality:", df.select(
        F.count("*").alias("rows"),
        F.sum(col("age_invalid").cast("int")).alias("invalid_age"),
        F.sum(col("Occupation").isNull().cast("int")).alias("missing_occupation")
    ).first().asDict())

    # save silver table - IRL connect to database to write
    partition_name = "silver_fe_attr_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fe_attr_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_fe_fin_silver_table(snapshot_date_str, bronze_fe_fin_directory, silver_fe_fin_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_fe_fin_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # augment data: preserve the original loan count before cleaning
    df = df.withColumn("Num_of_Loan_raw", col("Num_of_Loan"))

    # augment data: preserve the original snapshot date before conversion
    df = df.withColumn("snapshot_date_raw", col("snapshot_date"))

    # clean data: remove surrounding underscores from numeric values; preserve minus signs
    numeric_columns = [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"
    ]
    for column in numeric_columns:
        df = df.withColumn(column, F.regexp_replace(F.trim(col(column)), r"^_+|_+$", ""))

    # clean data: Num_Credit_Inquiries is stored as '4.0'; remove the '.0' so it casts to integer
    df = df.withColumn("Num_Credit_Inquiries",
        F.regexp_replace(col("Num_Credit_Inquiries"), r"\.0$", ""))

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    # Money and ratios retain decimal precision; categories remain strings.
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": DoubleType(),
        "Monthly_Inhand_Salary": DoubleType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": DoubleType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": DoubleType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": DoubleType(),
        "Credit_Utilization_Ratio": DoubleType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": DoubleType(),
        "Amount_invested_monthly": DoubleType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": DoubleType(),
        "snapshot_date": DateType(),
    }
    for column, data_type in column_type_map.items():
        df = df.withColumn(column, F.expr(
            f"try_cast(`{column}` as {data_type.simpleString()})"))

    # clean data: standardize loan-list separators and replace empty lists with null
    df = df.withColumn("Type_of_Loan",
        F.regexp_replace(F.trim(col("Type_of_Loan")), r",\s*and\s+", ", "))
    df = df.withColumn("Type_of_Loan",
        F.when(F.length(col("Type_of_Loan")) > 0, col("Type_of_Loan")))

    # augment data: flag missing loan lists
    df = df.withColumn("loan_types_missing", col("Type_of_Loan").isNull())

    # augment data: count listed loans, including repetitions
    # missing lists -> null
    df = df.withColumn("listed_loan_count",
        F.when(~col("loan_types_missing"),
            F.size(F.filter(F.split(col("Type_of_Loan"), r",\s*"),
                            lambda loan: F.length(F.trim(loan)) > 0))))

    # augment data: classify loan counts as invalid, missing list, match, or mismatch
    df = df.withColumn("loan_count_status",
        F.when(col("Num_of_Loan").isNull() | (col("Num_of_Loan") < 0), "Invalid count")
         .when(col("loan_types_missing"), "Missing loan list")
         .when(col("Num_of_Loan") == col("listed_loan_count"), "Match")
         .otherwise("Mismatch"))

    # augment data: flag invalid/mismatched counts or positive counts without a loan list
    df = df.withColumn("loan_count_suspicious",
        (col("loan_count_status") == "Invalid count") |
        (col("loan_count_status") == "Mismatch") |
        (col("loan_types_missing") & (col("Num_of_Loan") > 0)))

    # clean data: null suspicious loan counts; retain zero counts with missing lists
    df = df.withColumn("Num_of_Loan",
        F.when(~col("loan_count_suspicious"), col("Num_of_Loan")))

    # clean data: replace negative quantities with null
    # Negative delay and credit-limit change can be meaningful, so exclude them.
    # Columns with a valid range in static/valid_ranges.yaml are handled below.
    nonnegative_columns = [
        "Monthly_Inhand_Salary", "Outstanding_Debt", "Amount_invested_monthly"
    ]
    for column in nonnegative_columns:
        df = df.withColumn(column, F.when(col(column) >= 0, col(column)))

    # clean data: replace the Amount_invested_monthly placeholder (__10000__) with null
    df = df.withColumn("Amount_invested_monthly",
        F.when(col("Amount_invested_monthly") != 10000, col("Amount_invested_monthly")))

    # clean data: replace the Monthly_Balance placeholder (about -3.3e26) with null
    df = df.withColumn("Monthly_Balance",
        F.when(col("Monthly_Balance") > -1e9, col("Monthly_Balance")))

    # augment data: flag missing values and values outside the valid ranges
    # clean data: replace flagged values with null
    valid_ranges = read_static_yaml("valid_ranges.yaml")
    for column, (low, high) in valid_ranges["value_ranges"].items():
        flag = column.lower() + "_invalid"
        df = df.withColumn(flag, col(column).isNull() | ~col(column).between(low, high))
        df = df.withColumn(column, F.when(~col(flag), col(column)))

    # augment data: flag values outside the valid ratio to monthly salary
    # clean data: replace flagged values with null
    for column, (low, high) in valid_ranges["salary_ratio_ranges"].items():
        flag = column.lower() + "_invalid"
        df = df.withColumn(flag, col(column).isNull() |
            ~(col(column) / col("Monthly_Inhand_Salary")).between(low, high))
        df = df.withColumn(column, F.when(~col(flag), col(column)))

    # clean data: retain Standard, Good, or Bad credit mix; null other values
    df = df.withColumn("Credit_Mix",
        F.when(F.trim(col("Credit_Mix")).isin("Standard", "Good", "Bad"),
               F.trim(col("Credit_Mix"))))

    # clean data: retain Yes/No minimum-payment responses; null other values
    df = df.withColumn("Payment_of_Min_Amount",
        F.when(F.trim(col("Payment_of_Min_Amount")).isin("Yes", "No"),
               F.trim(col("Payment_of_Min_Amount"))))

    # clean data: retain recognized spending/payment-size categories; null other values
    df = df.withColumn("Payment_Behaviour",
        F.when(col("Payment_Behaviour").rlike(
            r"^(High|Low)_spent_(Small|Medium|Large)_value_payments$"),
            col("Payment_Behaviour")))

    # augment data: convert credit history to total months while keeping the original text
    # Example: '2 Years and 3 Months' becomes 27; unrecognized formats become null.
    df = df.withColumn("Credit_History_Months",
        F.when(col("Credit_History_Age").rlike(r"^\d+ Years? and \d+ Months?$"),
            F.regexp_extract(col("Credit_History_Age"), r"^(\d+)", 1).cast("int") * 12 +
            F.regexp_extract(col("Credit_History_Age"), r"and (\d+)", 1).cast("int")))

    # report data quality: count suspicious loans, missing lists, invalid values and dates
    invalid_columns = [column for column in df.columns if column.endswith("_invalid")]
    print("financial quality:", df.select(
        F.count("*").alias("rows"),
        F.sum(col("loan_count_suspicious").cast("int")).alias("suspicious_loan_counts"),
        F.sum(col("loan_types_missing").cast("int")).alias("missing_loan_lists"),
        *[F.sum(col(column).cast("int")).alias(column) for column in invalid_columns],
        F.sum(col("snapshot_date").isNull().cast("int")).alias("invalid_dates")
    ).first().asDict())

    # save silver table - IRL connect to database to write
    partition_name = "silver_fe_fin_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_fe_fin_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df
