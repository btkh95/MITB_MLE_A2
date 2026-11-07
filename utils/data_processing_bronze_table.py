from datetime import datetime
from pyspark.sql.functions import col
import pyspark.sql.functions as F


def process_bronze_table(snapshot_date_str, csv_file_path, bronze_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=False)

    # remove the stray "x" found in a source snapshot date before partitioning
    df = df.withColumn("snapshot_date", F.regexp_replace(col("snapshot_date"), "x", ""))
    df = df.filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())
    
    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df
