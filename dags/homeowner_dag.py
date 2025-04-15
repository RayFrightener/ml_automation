#!/usr/bin/env python3
"""
homeowner_dag.py

This is the main Airflow DAG script for the Homeowner Loss History Prediction project.
It integrates tasks from the following modules in the tasks package:
  - ingestion.py          (data ingestion tasks)
  - preprocessing.py      (preprocessing and feature engineering tasks)
  - drift.py              (drift detection and self-healing tasks)
  - training.py           (model training and registry update tasks)
  - notifications.py      (Slack notifications, log pushing, archiving tasks)

Ensure that your tasks package is properly structured and that necessary environment
variables (or Airflow Variables) are set for S3, Slack, MLflow, and other services.
"""

import os
import time
import json
import logging
import warnings
from datetime import datetime, timedelta

# Airflow imports
from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ================================
# Import tasks from our modules
# ================================
from tasks.ingestion import download_csv_from_s3, validate_data_with_ge
from tasks.preprocessing import load_data_to_dataframe, handle_missing_data, detect_outliers_iqr, cap_outliers, encode_categoricals, split_data
from tasks.drift import generate_reference_means, detect_data_drift, self_healing
from tasks.training import manual_override, train_xgboost_hyperopt, compare_and_update_registry
from tasks.notifications import send_to_slack, push_logs_to_s3, archive_data

# ================================
# Global configuration and logging
# ================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Configuration parameters from environment variables or Airflow Variables
S3_BUCKET = os.environ.get("S3_BUCKET", Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket"))
LOCAL_DATA_PATH = "/tmp/homeowner_data.csv"
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.csv"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"  # This file will be generated for drift check
MODEL_ID = os.environ.get("MODEL_ID", Variable.get("MODEL_ID", default_var="model1")).lower().strip()

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# ================================
# Define the DAG using the TaskFlow API
# ================================
@dag(
    default_args=default_args,
    schedule="0 3 * * *",  # Runs daily at 3 AM (adjust as needed)
    catchup=False,
    tags=["homeowner", "loss_history"]
)
def homeowner_dag():
    """
    Homeowner Loss History Prediction DAG:
      1. Ingests raw data from S3 and validates it.
      2. Preprocesses the data (handles missing values, outliers, encodes categoricals).
      3. Generates a reference means file and performs drift detection.
      4. Branches based on drift detection:
            - If drift is detected, enters self-healing.
            - Otherwise, trains the model with hyperparameter tuning.
      5. Joins branch outputs, updates the model registry, sends notifications, and archives outputs.
    """

    # ------------------------------
    # Task 1: Ingestion & Validation
    # ------------------------------
    @task
    def ingest_and_validate() -> str:
        """
        Download the CSV file from S3 and run data validation (e.g., using Great Expectations).
        Returns the local path to the downloaded CSV.
        """
        s3_key = "raw-data/ut_loss_history_1.csv"  # Adjust the S3 key as needed
        logging.info(f"Downloading CSV from S3: Bucket={S3_BUCKET}, Key={s3_key}")
        local_path = download_csv_from_s3(s3_bucket=S3_BUCKET, s3_key=s3_key, local_path=LOCAL_DATA_PATH)
        # Run GE validation (this function may be a no-op if GE is disabled)
        ge_result = validate_data_with_ge(local_csv=local_path, checkpoint_name="homeowner_checkpoint")
        logging.info(f"GE validation result: {ge_result}")
        return local_path

    # ------------------------------
    # Task 2: Preprocessing
    # ------------------------------
    @task
    def preprocess(file_path: str) -> str:
        """
        Preprocess the data:
           - Load CSV into DataFrame.
           - Handle missing values.
           - Detect and cap outliers.
           - Encode categoricals.
           - Save the processed DataFrame.
        Returns the path to the processed CSV.
        """
        logging.info(f"Loading data from {file_path}")
        df = load_data_to_dataframe(file_path)
        logging.info("Handling missing values")
        df = handle_missing_data(df, strategy="mean", missing_threshold=0.4)
        logging.info("Detecting outliers using IQR method")
        outlier_counts = detect_outliers_iqr(df, factor=1.5)
        logging.info(f"Outlier counts: {outlier_counts}")
        # Cap outliers for numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            df = cap_outliers(df, col, factor=1.5)
        logging.info("Encoding categorical variables")
        df = encode_categoricals(df, encoding_map=None)
        df.to_csv(LOCAL_PROCESSED_PATH, index=False)
        logging.info(f"Preprocessed data saved to {LOCAL_PROCESSED_PATH}")
        return LOCAL_PROCESSED_PATH

    # ------------------------------
    # Task 3: Drift Detection
    # ------------------------------
    @task
    def drift_check(processed_path: str) -> str:
        """
        Generate a reference means file from the processed data and then detect data drift by comparing
        current means with the reference means.
        Returns:
            - "self_healing" if drift > 10% is detected in any numeric feature.
            - "train_xgboost_hyperopt" otherwise.
        """
        logging.info("Generating reference means file")
        local_ref = generate_reference_means(processed_path=processed_path, local_ref=REFERENCE_MEANS_PATH)
        logging.info("Running drift detection")
        drift_indicator = detect_data_drift(current_data_path=processed_path, reference_means_path=local_ref)
        logging.info(f"Drift check indicator: {drift_indicator}")
        return drift_indicator

    # ------------------------------
    # Task 4: Self-Healing (if drift detected)
    # ------------------------------
    @task
    def perform_self_healing() -> str:
        """
        Execute self-healing routine when drift is detected.
        This placeholder simulates waiting for manual override or automated fix.
        """
        logging.info("Self-healing initiated due to detected drift. Awaiting manual intervention...")
        time.sleep(5)  # Simulate waiting period
        logging.info("Self-healing routine completed; manual override confirmed.")
        return "override_done"

    # ------------------------------
    # Branching: Decide between self-healing and training
    # ------------------------------
    branch_decider = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=lambda drift: "self_healing" if drift == "self_healing" else "train_xgboost_hyperopt",
        op_args=[drift_check.output],
    )

    # ------------------------------
    # Task 5: Training (if no drift or after self-healing)
    # ------------------------------
    # Get manual override parameters from Airflow Variables (if set)
    override_params = manual_override()

    training_task = train_xgboost_hyperopt(LOCAL_PROCESSED_PATH, override_params)

    # ------------------------------
    # Task 6: Join Branches
    # ------------------------------
    join_operator = EmptyOperator(task_id="join_branches")

    # ------------------------------
    # Task 7: Compare and Update Model Registry
    # ------------------------------
    registry_comparison = compare_and_update_registry()

    # ------------------------------
    # Task 8: Notifications - Slack
    # ------------------------------
    @task
    def notify_completion() -> str:
        """
        Send a Slack notification that the pipeline has completed successfully.
        """
        message = "Homeowner Loss History pipeline completed successfully."
        send_to_slack(channel="#alerts", title="Pipeline Completed", details=message, urgency="medium")
        return "notification_sent"

    # ------------------------------
    # Task 9: Archive Outputs (logs, data)
    # ------------------------------
    @task
    def archive_outputs() -> str:
        """
        Archive logs and processed data by pushing logs to S3 and archiving the processed file.
        """
        push_logs_to_s3(log_file_path="/home/ubuntu/airflow/logs/homeowner_dag.log")
        archive_data(file_path=LOCAL_PROCESSED_PATH)
        logging.info("Archiving outputs completed.")
        return "archive_done"

    # ------------------------------
    # Define DAG Flow Dependencies
    # ------------------------------
    ingestion_path = ingest_and_validate()       # Task 1
    processed_file = preprocess(ingestion_path)    # Task 2
    drift_result = drift_check(processed_file)     # Task 3

    # Branching based on drift detection result
    branch_decider  # (uses drift_result)

    # When branch decision returns "self_healing", perform self_healing task.
    healing_result = perform_self_healing()        # Task 4

    # Otherwise, train the model.
    training_result = training_task                # Task 5

    # Join the branches.
    [healing_result, training_result] >> join_operator  # Task 6

    # After joining, update model registry.
    join_operator >> registry_comparison  # Task 7

    # Then send a notification.
    registry_comparison >> notify_completion()  # Task 8

    # Finally, archive outputs.
    notify_completion() >> archive_outputs()  # Task 9

# Instantiate the DAG
dag = homeowner_dag()
