#!/usr/bin/env python3
"""
homeowner_dag.py

This is the main Airflow DAG for the Homeowner Loss History Prediction project.
It ingests raw data, validates and preprocesses it, snapshots schema, detects drift,
and then either self‐heals or trains all five models in parallel before registering results.
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

from tasks.ingestion import ingest_data_from_s3
from tasks.preprocessing import preprocess_data
from tasks.schema_validation import validate_schema, snapshot_schema
from tasks.drift import generate_reference_means, detect_data_drift, self_healing
from tasks.training import train_xgboost_hyperopt, manual_override, compare_and_update_registry
from tasks.monitoring import record_system_metrics
from utils.slack import post as send_message
from utils.storage import upload as upload_to_s3

# Basic logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Constants
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.csv"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"
MODEL_IDS = ["model1", "model2", "model3", "model4", "model5"]

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="homeowner_loss_history_full_pipeline",
    default_args=default_args,
    schedule_interval="0 10 * * *",  # run daily at 10am
    catchup=False,
    tags=["homeowner", "loss_history"],
)
def homeowner_pipeline():

    @task()
    def ingest_task() -> str:
        """Download raw CSV (with caching) and return local file path."""
        return ingest_data_from_s3()

    @task()
    def preprocess_and_save(raw_path: str) -> str:
        """Preprocess, validate schema, snapshot schema, and save processed CSV."""
        df = preprocess_data(raw_path)
        validate_schema(df)
        snapshot_schema(df)
        df.to_parquet(LOCAL_PROCESSED_PATH, index=False)
        log.info(f"Wrote processed data to {LOCAL_PROCESSED_PATH}")
        return LOCAL_PROCESSED_PATH

    @task()
    def check_for_drift(processed_path: str) -> str:
        """
        If reference means exist, detect drift.
        Returns 'self_healing' or 'train_models'.
        """
        if not os.path.exists(REFERENCE_MEANS_PATH):
            log.warning("No reference means; skipping drift detection.")
            return "train_models"
        ref = generate_reference_means(processed_path, REFERENCE_MEANS_PATH)
        return detect_data_drift(processed_path, ref)

    @task()
    def healing_task() -> str:
        """Notify Slack of drift and simulate self‑healing."""
        send_message(
            channel="#alerts",
            title="⚠️ Drift Detected",
            details="Data drift detected; self‑healing routine executed.",
            urgency="medium"
        )
        time.sleep(5)
        return "override_done"

    @task()
    def train_model_task(processed_path: str, model_id: str) -> float:
        """Train a single XGBoost model with Hyperopt and return test RMSE."""
        return train_xgboost_hyperopt(
            processed_path,
            override_params=manual_override(),
            model_id=model_id
        )

    @task()
    def compare_task() -> str:
        """Compare new models to production and update registry."""
        compare_and_update_registry()
        return "registry_updated"

    @task()
    def record_metrics_task() -> str:
        """Record system metrics (runtime, memory) via monitoring utility."""
        runtime = time.time()
        memory = os.popen("free -m").read()
        record_system_metrics(runtime=runtime, memory_usage=memory)
        return "metrics_logged"

    @task()
    def notify_complete_task() -> str:
        """Notify Slack that pipeline completed successfully."""
        send_message(
            channel="#alerts",
            title="✅ Pipeline Complete",
            details="All models trained and artifacts archived.",
            urgency="low"
        )
        return "notified"

    @task()
    def archive_task() -> str:
        """Upload logs and processed CSV to S3, then clean up."""
        upload_to_s3("/home/airflow/logs/homeowner_dag.log", "logs/homeowner_dag.log")
        upload_to_s3(LOCAL_PROCESSED_PATH, "archive/homeowner_processed.csv")
        for p in [LOCAL_PROCESSED_PATH, REFERENCE_MEANS_PATH]:
            if os.path.exists(p):
                os.remove(p)
        return "archived"

    # DAG flow
    raw_path       = ingest_task()
    processed_path = preprocess_and_save(raw_path)
    drift_decision = check_for_drift(processed_path)

    branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=lambda decision: "healing_task" if decision == "self_healing" else "train_models",
        op_args=[drift_decision],
    )

    heal = healing_task()

    train_tasks = train_model_task.expand(
        processed_path=[processed_path] * len(MODEL_IDS),
        model_id=MODEL_IDS
    )

    join_models = EmptyOperator(task_id="join_models", trigger_rule="one_success")

    compare   = compare_task()
    record    = record_metrics_task()
    notify    = notify_complete_task()
    archive   = archive_task()

    # wiring
    raw_path >> processed_path >> drift_decision >> branch
    branch >> heal >> join_models
    branch >> train_tasks >> join_models

    join_models >> compare >> record >> notify >> archive

dag = homeowner_pipeline()
