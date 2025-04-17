#!/usr/bin/env python3
"""
DAG.py 

This is the main Airflow DAG script for the Homeowner Loss History Prediction project.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable

from tasks.ingestion import ingest_data_from_s3
from tasks.preprocessing import (
    load_data_to_dataframe,
    handle_missing_data,
    detect_outliers_iqr,
    cap_outliers,
    encode_categoricals,
    generate_profile_report
)
from tasks.drift import generate_reference_means, detect_data_drift, self_healing
from tasks.training import manual_override, train_xgboost_hyperopt, compare_and_update_registry
from tasks.notifications import send_to_slack, push_logs_to_s3, archive_data
from tasks.cache import is_cache_valid, update_cache
from tasks.monitoring import record_system_metrics
from agent_actions import handle_function_call

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

S3_BUCKET = os.environ.get("S3_BUCKET", Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket"))
LOCAL_DATA_PATH = "/tmp/homeowner_data.csv"
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.csv"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

@dag(
    default_args=default_args,
    schedule="0 3 * * *",
    catchup=False,
    tags=["homeowner", "loss_history"]
)
def homeowner_dag():

    @task
    def ingest_and_validate():
        s3_key = "raw-data/ut_loss_history_1.csv"
        if not is_cache_valid(S3_BUCKET, s3_key, LOCAL_DATA_PATH):
            update_cache(S3_BUCKET, s3_key, LOCAL_DATA_PATH)
        return LOCAL_DATA_PATH

    @task
    def preprocess_data(file_path: str):
        df = load_data_to_dataframe(file_path)
        df = handle_missing_data(df)
        detect_outliers_iqr(df)
        for col in df.select_dtypes(include=["number"]):
            df = cap_outliers(df, col)
        df = encode_categoricals(df)
        if 'il_total' in df.columns and 'eey' in df.columns:
            df['pure_premium'] = df['il_total'] / df['eey']
            df['sample_weight'] = df['eey']
        else:
            raise ValueError("Missing required columns 'il_total' or 'eey'")
        generate_profile_report(df)
        df.to_csv(LOCAL_PROCESSED_PATH, index=False)
        return LOCAL_PROCESSED_PATH

    @task
    def check_for_drift(processed_path: str):
        if not os.path.exists(REFERENCE_MEANS_PATH):
            logging.warning("Reference means file not found, skipping drift detection.")
            return "train_xgboost_hyperopt__task"
        ref_path = generate_reference_means(processed_path, REFERENCE_MEANS_PATH)
        return detect_data_drift(processed_path, ref_path)

    @task(task_id="perform_self_healing__task")
    def perform_self_healing():
        handle_function_call({
            "function": {
                "name": "notify_slack",
                "arguments": json.dumps({
                    "channel": "#alerts",
                    "title": "⚠️ Drift Detected",
                    "details": "Self-healing was triggered after detecting significant drift.",
                    "urgency": "medium"
                })
            }
        })
        time.sleep(5)
        return "override_done"

    @task(task_id="train_xgboost_hyperopt__task")
    def run_training(path: str):
        return train_xgboost_hyperopt(path, manual_override())

    @task(task_id="compare_and_update_registry__task")
    def run_compare_and_update():
        return compare_and_update_registry()

    @task
    def record_metrics():
        runtime = time.time()
        mem = os.popen("free -m").read()
        record_system_metrics(runtime=runtime, memory_usage=mem)
        return "metrics_logged"

    @task
    def log_and_notify():
        handle_function_call({
            "function": {
                "name": "notify_slack",
                "arguments": json.dumps({
                    "channel": "#alerts",
                    "title": "✅ DAG Complete",
                    "details": "Pipeline run complete and archived.",
                    "urgency": "low"
                })
            }
        })
        return "done"

    @task
    def archive_outputs():
        push_logs_to_s3("/home/ubuntu/airflow/logs/homeowner_dag.log")
        archive_data(LOCAL_PROCESSED_PATH)
        for path in [LOCAL_DATA_PATH, LOCAL_PROCESSED_PATH, REFERENCE_MEANS_PATH]:
            if os.path.exists(path):
                os.remove(path)
        return "archived"

    # --- define dependencies and flow ---
    ingestion_path  = ingest_and_validate()
    processed_file  = preprocess_data(ingestion_path)
    drift_result    = check_for_drift(processed_file)

    healing         = perform_self_healing()
    training        = run_training(processed_file)
    update_registry = run_compare_and_update()
    metrics         = record_metrics()
    notification    = log_and_notify()
    archiving       = archive_outputs()

    branch = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=lambda result: 
            "perform_self_healing__task" if result == "self_healing" 
            else "train_xgboost_hyperopt__task",
        op_args=[drift_result],
    )

    join = EmptyOperator(task_id="join_branches", trigger_rule="one_success")

    # wiring
    ingestion_path >> processed_file >> drift_result >> branch
    branch >> healing >> join
    branch >> training >> join
    join >> update_registry >> metrics >> notification >> archiving

dag = homeowner_dag()
