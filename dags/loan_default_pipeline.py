"""Monthly loan-default ML pipeline: datamart -> train -> inference -> monitoring.

Each run processes one snapshot date ({{ ds }}). Backfill runs months in order
(max_active_runs=1) so later months can read earlier partitions.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator


def data_stage(stage):
    return BashOperator(
        task_id=stage,
        bash_command=(
            "cd /opt/airflow && "
            f"python scripts/run_data_pipeline.py --stage {stage} --snapshotdate {{{{ ds }}}}"
        ),
    )


def source_check(source):
    # fails the run before bronze if the source file is missing, empty or lacks key columns
    return BashOperator(
        task_id=f"dep_check_source_{source}",
        bash_command=(
            "cd /opt/airflow && "
            f"python scripts/check_source_data.py --source {source} --snapshotdate {{{{ ds }}}}"
        ),
    )


def model_stage(script, **kwargs):
    return BashOperator(
        task_id=script,
        bash_command=f"cd /opt/airflow && python scripts/{script}.py --snapshotdate {{{{ ds }}}}",
        **kwargs,
    )


def train_gate(ds, **context):
    # imported here so DAG parsing stays light; paths in utils are relative to /opt/airflow
    import os
    os.chdir("/opt/airflow")
    from utils.model_utils import should_train
    return should_train(ds)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "loan_default_pipeline",
    default_args=default_args,
    description="Monthly datamart, model training, inference and monitoring",
    schedule_interval="0 0 1 * *",  # 00:00 on the 1st of each month
    start_date=datetime(2023, 1, 1),
    # last applications are 2025-01; run to 2025-07 so their default labels can be monitored
    end_date=datetime(2025, 7, 1),
    catchup=True,
    max_active_runs=1,
    tags=["cs611", "assignment2"],
) as dag:

    # --- label store ---
    bronze_lms = data_stage("bronze_lms")
    silver_lms = data_stage("silver_lms")
    gold_label_store = data_stage("gold_label_store")
    label_store_completed = EmptyOperator(task_id="label_store_completed")

    source_check("lms") >> bronze_lms >> silver_lms >> gold_label_store >> label_store_completed

    # --- feature store ---
    gold_feature_store = data_stage("gold_feature_store")
    feature_store_completed = EmptyOperator(task_id="feature_store_completed")

    for source in ["fe_attr", "fe_fin", "fe_click"]:
        source_check(source) >> data_stage(f"bronze_{source}") >> data_stage(f"silver_{source}") >> data_stage(f"gold_{source}") >> gold_feature_store

    gold_feature_store >> feature_store_completed

    # --- model training: first date, then every retrain_every_months (1 at launch), or after a drift alert ---
    # skipping training must not skip inference, so only the direct downstream task is skipped
    model_train_gate = ShortCircuitOperator(
        task_id="model_train_gate",
        python_callable=train_gate,
        ignore_downstream_trigger_rules=False,
    )
    model_train = model_stage("model_train")
    model_promote = model_stage("model_promote")

    [label_store_completed, feature_store_completed] >> model_train_gate >> model_train >> model_promote

    # --- model inference: score this month's applications with the champion as of ds ---
    model_inference = model_stage("model_inference", trigger_rule="none_failed")

    [feature_store_completed, model_promote] >> model_inference

    # --- model monitoring: PSI on this month's scores, Gini/KS for the cohort whose labels matured ---
    model_monitor = model_stage("model_monitor")
    model_monitor_charts = model_stage("model_monitor_charts")

    [model_inference, label_store_completed] >> model_monitor >> model_monitor_charts
