#!/usr/bin/env python3
"""
homeowner_dag.py

Home‑owner Loss‑History full pipeline DAG
• Ingest → Preprocess → Drift‑Check → (Self‑Heal ⧸ Manual‑Override → Train)
• Human‑in‑the‑Loop hooks for manual override
• Metrics, Notifications, Archive
"""

import os
import logging
import time

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
    self_healing as drift_self_heal
)
from tasks.monitoring import record_system_metrics
from tasks.training import train_and_compare_fn, manual_override
from utils.slack import post as send_message
from utils.storage import upload as upload_to_s3

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Constants
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.parquet"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"
MODEL_IDS = ["model1", "model2", "model3", "model4", "model5"]


def _default_args():
    return {
        "owner": "airflow",
        "depends_on_past": False,
        "start_date": datetime(2025, 1, 1),
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
        "execution_timeout": timedelta(hours=2),
    }

@dag(
    dag_id="homeowner_loss_history_full_pipeline",
    default_args=_default_args(),
    schedule_interval="0 10 * * *",  # daily at 10 AM
    catchup=False,
    tags=["homeowner", "loss_history"],
)
def homeowner_pipeline():

    # 1️⃣ Ingest raw CSV
    raw_path = ingest_data_from_s3()

    # 2️⃣ Preprocess → Parquet
    @task()
    def _preprocess(path: str) -> str:
        df = preprocess_data(path)
        validate_schema(df)
        snapshot_schema(df)
        df.to_parquet(LOCAL_PROCESSED_PATH, index=False)
        return LOCAL_PROCESSED_PATH
    processed_path = _preprocess(raw_path)

    # 3️⃣ Drift‑check: existing reference?
    @task.branch()
    def _branch(path: str) -> str:
        if os.path.exists(REFERENCE_MEANS_PATH):
            ref = generate_reference_means(path, REFERENCE_MEANS_PATH)
            flag = detect_data_drift(path, ref)
            return "healing_task" if flag == "self_healing" else "override_or_train"
        return "override_or_train"
    branch = _branch(processed_path)

    # 3a Self‑healing path
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

    # 4️⃣ Manual‑Override vs Start‑Training
    @task.branch(task_id="override_or_train")
    def override_branch() -> str:
        return "apply_override" if manual_override() else "start_training"

    @task(task_id="apply_override")
    def apply_override():
        params = manual_override()
        send_message(
            channel="#alerts",
            title="🛠️ Manual Override Applied",
            details=f"Custom hyperparams: {params}",
            urgency="low",
        )
        # training task will consume manual_override internally

    start_training = EmptyOperator(task_id="start_training")

    # 5️⃣ Training per model → join
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

    # wire drift → heal → override/train → training
    branch >> heal >> override_branch()
    branch >> override_branch()
    override_branch() >> apply_override() >> join_after_training
    override_branch() >> start_training >> join_after_training

    # 6️⃣ Metrics, Notify, Archive
    @task()
    def record_metrics_task():
        record_system_metrics(runtime=time.time())

    @task()
    def notify_complete():
        send_message(
            channel="#alerts",
            title="✅ Pipeline Complete",
            details="Training & SHAP logging finished.",
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

    join_after_training >> record_metrics_task() >> notify_complete() >> archive()

    # 7️⃣ Generate Fix Proposals
    @task()
    def generate_fix_proposals():
        send_message(
            channel="#alerts",
            title="🔧 Generating Fix Proposals",
            details="Generating fix proposals based on detected issues.",
            urgency="low",
        )
        # Placeholder for generating fix proposals logic

    # 8️⃣ Run Self-Heal
    @task()
    def run_self_heal():
        send_message(
            channel="#alerts",
            title="🛠️ Running Self-Heal",
            details="Executing self-heal routine.",
            urgency="medium",
        )
        drift_self_heal()

    # 9️⃣ Open Code Console
    @task()
    def open_code_console():
        send_message(
            channel="#alerts",
            title="💻 Opening Code Console",
            details="Opening code console for manual interventions.",
            urgency="low",
        )
        # Placeholder for opening code console logic

    # WebSocket for live updates
    @task()
    def implement_websockets():
        send_message(
            channel="#alerts",
            title="🔄 Implementing WebSockets",
            details="Setting up WebSockets for live updates.",
            urgency="low",
        )
        # Placeholder for WebSocket implementation logic

    # Integrate new UI components and endpoints
    @task()
    def integrate_ui_components():
        send_message(
            channel="#alerts",
            title="🔗 Integrating UI Components",
            details="Integrating new UI components and endpoints.",
            urgency="low",
        )
        # Placeholder for UI components integration logic

    # Define dependencies for new tasks
    join_after_training >> generate_fix_proposals()
    join_after_training >> run_self_heal()
    join_after_training >> open_code_console()
    join_after_training >> implement_websockets()
    join_after_training >> integrate_ui_components()

dag = homeowner_pipeline()
