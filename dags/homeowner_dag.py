#!/usr/bin/env python3
"""
Home‑owner Loss‑History full pipeline DAG
• calls the unified train_and_compare_fn (RMSE + SHAP + auto‑promotion)
• keeps branch‑logic / drift‑check exactly as before
"""

import os, logging, time
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

from tasks.ingestion import ingest_data_from_s3
from tasks.preprocessing import preprocess_data
from tasks.schema_validation import validate_schema, snapshot_schema
from tasks.drift import (
    generate_reference_means,
    detect_data_drift,
    self_healing as drift_self_heal,
)
from tasks.monitoring import record_system_metrics
from tasks.training import train_and_compare_fn, manual_override
from utils.slack import post as send_message
from utils.storage import upload as upload_to_s3

# ─── Config ────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.parquet"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"
MODEL_IDS = ["model1", "model2", "model3", "model4", "model5"]

def _default_args():
    return dict(
        owner="airflow",
        depends_on_past=False,
        start_date=datetime(2025, 1, 1),
        email_on_failure=False,
        email_on_retry=False,
        retries=1,
        retry_delay=timedelta(minutes=10),
        execution_timeout=timedelta(hours=2),
    )

@dag(
    dag_id="homeowner_loss_history_full_pipeline",
    default_args=_default_args(),
    schedule_interval="0 10 * * *",  # 10 AM daily
    catchup=False,
    tags=["homeowner", "loss_history"],
)
def homeowner_pipeline():
    # 1️⃣ Ingest raw CSV
    raw_path = ingest_data_from_s3()

    # 2️⃣ Preprocess → parquet
    @task()
    def _preprocess(path: str) -> str:
        df = preprocess_data(path)
        validate_schema(df)
        snapshot_schema(df)
        df.to_parquet(LOCAL_PROCESSED_PATH, index=False)
        return LOCAL_PROCESSED_PATH
    processed_path = _preprocess(raw_path)

    # 3️⃣ Drift‑check branch
    @task.branch()
    def _branch(path: str) -> str:
        if os.path.exists(REFERENCE_MEANS_PATH):
            ref = generate_reference_means(path, REFERENCE_MEANS_PATH)
            flag = detect_data_drift(path, ref)
            return "healing_task" if flag == "self_healing" else "join_after_training"
        return "join_after_training"
    drift_decision = _branch(processed_path)

    # 3a Self‑healing
    @task(task_id="healing_task")
    def healing_task():
        send_message(
            channel="#alerts",
            title="⚠️ Drift Detected",
            details="Self‑healing routine executed.",
            urgency="medium",
        )
        drift_self_heal()
    heal = healing_task()

    # 4️⃣ Parallel training
    join_after_training = EmptyOperator(
        task_id="join_after_training",
        trigger_rule=TriggerRule.NONE_FAILED,
    )
    for m in MODEL_IDS:
        PythonOperator(
            task_id=f"train_compare_{m}",
            python_callable=train_and_compare_fn,
            op_kwargs={"model_id": m, "processed_path": LOCAL_PROCESSED_PATH},
        ).set_downstream(join_after_training)

    drift_decision >> heal >> join_after_training
    drift_decision >> join_after_training

    # 5️⃣ Metrics, notify, archive
    @task()
    def record_metrics():
        record_system_metrics(runtime=time.time())

    @task()
    def notify_complete():
        send_message(
            channel="#alerts",
            title="✅ Pipeline Complete",
            details="Training + SHAP logging finished.",
            urgency="low",
        )

    @task()
    def archive():
        upload_to_s3("/home/airflow/logs/homeowner_dag.log", "logs/homeowner_dag.log")
        upload_to_s3(LOCAL_PROCESSED_PATH, "archive/homeowner_processed.parquet")
        for f in (LOCAL_PROCESSED_PATH, REFERENCE_MEANS_PATH):
            try:
                os.remove(f)
            except OSError:
                pass

    join_after_training >> record_metrics() >> notify_complete() >> archive()

dag = homeowner_pipeline()
