"""
File: homeowner_loss_history_dag_extended.py

Homeowner Loss History Prediction Project - Extended DAG (Full Example)
-----------------------------------------------------------------------
Includes:
 1) Data Ingestion (S3)
 2) Validation with Great Expectations
 3) Preprocessing (null/outlier handling)
 4) Feature Engineering
 5) Data Drift Detection
 6) Model Training (XGBoost + Hyperopt)
 7) Model Registry Update (MLflow)
 8) Final Evaluation (evaluate multiple models, compare/plot)
 9) Performance Monitoring
10) Notification/Alerting (Slack example)
11) Push Logs to S3
12) Archive Data

Customize placeholders to match your environment:
 - S3 paths
 - Great Expectations details (checkpoint, suite, etc.)
 - MLflow experiment/registry usage
 - Slack token or email settings for notifications
 - etc.
"""

import os
import json
import time
import random
import shutil
import psutil
import numpy as np
import pandas as pd
import xgboost as xgb

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

from hyperparameter_tracker import HyperparamTracker
from model_evaluation import evaluate_model, plot_predictions

# AWS + MLflow + Great Expectations
import boto3
import mlflow
import mlflow.xgboost
import great_expectations as ge

# Hyperopt for Bayesian hyperparameter search
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# If you have a Slack webhook for notifications
SLACK_WEBHOOK = os.environ.get("SLACK_WEBHOOK", None)

# ---------------------------------------------------------------------
# 1. ENVIRONMENT / CONSTANTS
# ---------------------------------------------------------------------

S3_BUCKET = os.environ.get("S3_BUCKET", "grange-seniordesign-bucket")
S3_DATA_FOLDER = "raw-data"
S3_MODELS_FOLDER = "models"
LOCAL_DATA_PATH = "/tmp/homeowner_data.csv"
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.csv"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://3.146.46.179:5000")
MLFLOW_EXPERIMENT_NAME = "Homeowner_Loss_Hist_Proj"

# Example monotonic constraints by model ID (used in train_xgboost_hyperopt)
MODEL_ID = os.environ.get("MODEL_ID", "model1").lower().strip()
MONOTONIC_CONSTRAINTS_MAP = {
    "model1": "(1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1)",
    "model2": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)",
    "model3": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)",
    "model4": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)",
    "model5": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)",
}

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# ---------------------------------------------------------------------
# 2. PYTHON CALLABLES (TASK FUNCTIONS)
# ---------------------------------------------------------------------

def ingest_data_from_s3(**context):
    """
    (1) Data Ingestion
    ------------------
    Download a CSV from S3 to the local filesystem.
    """
    s3 = boto3.client("s3")
    s3_key = f"{S3_DATA_FOLDER}/ut_loss_history_1.csv"  # Adjust as needed
    print(f"Downloading s3://{S3_BUCKET}/{s3_key} -> {LOCAL_DATA_PATH}")
    s3.download_file(S3_BUCKET, s3_key, LOCAL_DATA_PATH)
    print("Data ingestion complete.")


#def validate_data_with_ge(**context):
    """
    (2) Validation with Great Expectations
    --------------------------------------
    Example that runs a checkpoint called 'my_checkpoint'. 
    Make sure your GE project is configured with that checkpoint.
    """
    #ge_context = ge.get_context()
    #batch = ge_context.sources.pandas_default.read_csv(LOCAL_DATA_PATH)
    #print(f"Loaded {LOCAL_DATA_PATH} into Great Expectations batch.")

    #checkpoint_name = "my_checkpoint"  # Adjust
    #results = ge_context.run_checkpoint(checkpoint_name=checkpoint_name)
    #if not results["success"]:
        #raise ValueError("Great Expectations validation failed!")
    #print("Data validation passed via Great Expectations.")

def preprocess_data(**context):
    """
    (3) Preprocessing
    -----------------
    Handle nulls, outliers, basic cleaning. 
    Then save processed data to a new CSV for the next step.
    """
    df = pd.read_csv(LOCAL_DATA_PATH)
    print(f"Read {len(df)} rows from {LOCAL_DATA_PATH}.")

    # Fill nulls
    df.fillna(0, inplace=True)

    # Outlier capping with IQR
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        iqr = Q3 - Q1
        lower_bound = Q1 - 1.5 * iqr
        upper_bound = Q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    df.to_csv(LOCAL_PROCESSED_PATH, index=False)
    print(f"Preprocessing complete. Output: {LOCAL_PROCESSED_PATH}")

def feature_engineering(**context):
    """
    (4) Feature Engineering
    -----------------------
    Create advanced or domain-specific features to improve the model.
    """
    df = pd.read_csv(LOCAL_PROCESSED_PATH)
    if "il_total" in df.columns and "cc_total" in df.columns:
        df["avg_loss_per_claim"] = df.apply(
            lambda row: row["il_total"] / row["cc_total"] if row["cc_total"] > 0 else 0,
            axis=1
        )
        print("Created 'avg_loss_per_claim' feature.")

    df.to_csv(LOCAL_PROCESSED_PATH, index=False)
    print("Feature engineering complete.")

def download_reference_means(**context):
    s3 = boto3.client("s3")
    s3.download_file("grange-seniordesign-bucket", "reference/reference_means.csv", "/tmp/reference_means.csv")

def detect_data_drift(**context):
    """
    (5) Data Drift Detection
    ------------------------
    Reads a reference_means.csv file with columns:
       column_name, mean_value
    Then compares each numeric column in /tmp/homeowner_processed.csv
    to that reference. If the difference is >30%, raises an error.
    Otherwise prints "No significant drift."
    """

    # 1) Read the current processed data
    df_current = pd.read_csv(LOCAL_PROCESSED_PATH)
    print(f"Loaded current data with shape: {df_current.shape}")

    # 2) Read the reference means from a local CSV
    #    Suppose reference_means.csv has columns: column_name, mean_value
    if not os.path.exists(REFERENCE_MEANS_PATH):
        print(f"No reference file found at {REFERENCE_MEANS_PATH}, skipping drift check.")
        return

    df_reference = pd.read_csv(REFERENCE_MEANS_PATH)
    print(f"Loaded reference means with shape: {df_reference.shape}")

    # Convert reference file to a dictionary: {column_name: mean_value}
    reference_dict = {}
    for _, row in df_reference.iterrows():
        col_name = row["column_name"]
        mean_val = row["mean_value"]
        reference_dict[col_name] = mean_val

    # 3) For each numeric column in the current data, compare means
    numeric_cols = df_current.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in reference_dict:
            current_mean = df_current[col].mean()
            ref_mean = reference_dict[col]
            # Avoid zero or negative reference means to prevent divide-by-zero
            if ref_mean > 0:
                drift_ratio = abs(current_mean - ref_mean) / ref_mean
                if drift_ratio > 0.3:  # e.g., 30% difference
                    raise ValueError(
                        f"Data drift detected for column '{col}': "
                        f"current mean={current_mean:.2f}, reference={ref_mean:.2f}, ratio={drift_ratio:.2%}"
                    )
                else:
                    print(f"No significant drift in '{col}' (ratio={drift_ratio:.2%}).")
            else:
                print(f"Reference mean <= 0 for '{col}', skipping ratio check.")
        else:
            print(f"Column '{col}' not in reference file, skipping drift check for that column.")

    print("Data drift check completed. No significant drift found (or no columns to compare).")

def train_xgboost_hyperopt(**context):
    """
    (6) Model Training (XGBoost + Hyperopt)
    ---------------------------------------
    Uses Bayesian hyperparam search, monotonic constraints,
    logs to MLflow, uploads final model to S3.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    constraints = MONOTONIC_CONSTRAINTS_MAP.get(MODEL_ID, "(1,1,1,1)")
    df = pd.read_csv(LOCAL_PROCESSED_PATH)
    target_col = "pure_premium"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    space = {
        "learning_rate": hp.uniform("learning_rate", 0.001, 0.2),
        "max_depth": hp.choice("max_depth", [3, 5, 7, 9]),
        "n_estimators": hp.quniform("n_estimators", 50, 300, 1),
        "reg_alpha": hp.loguniform("reg_alpha", -5, 0),
        "reg_lambda": hp.loguniform("reg_lambda", -5, 0),
    }

    def objective(params):
        params["n_estimators"] = int(params["n_estimators"])
        model = xgb.XGBRegressor(
            monotone_constraints=constraints,
            use_label_encoder=False,
            eval_metric="rmse",
            **params
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        return {"loss": rmse, "status": STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=15, trials=trials)
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = [3, 5, 7, 9][best["max_depth"]]

    # Final model
    final_model = xgb.XGBRegressor(
        monotone_constraints=constraints,
        use_label_encoder=False,
        eval_metric="rmse",
        **best
    )
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)
    final_rmse = mean_squared_error(y_test, preds) ** 0.5

    # Log to MLflow
    with mlflow.start_run(run_name=f"xgboost_{MODEL_ID}_hyperopt") as run:
        mlflow.log_params(best)
        mlflow.log_metric("rmse", final_rmse)
        mlflow.xgboost.log_model(final_model, artifact_path="model")

    # Save & upload
    local_model_path = f"/tmp/xgb_{MODEL_ID}_model.json"
    final_model.save_model(local_model_path)
    s3 = boto3.client("s3")
    s3_key = f"{S3_MODELS_FOLDER}/xgb_{MODEL_ID}_model.json"
    s3.upload_file(local_model_path, S3_BUCKET, s3_key)

    print(f"Training done. Best RMSE={final_rmse:.4f}, model at s3://{S3_BUCKET}/{s3_key}")

def compare_and_update_registry(**context):
    """
    (7) Model Registry Update
    -------------------------
    Compare newly trained model's RMSE to 'Production' model in MLflow.
    If better, promote. Placeholder logic here.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    print("Placeholder: check if new model is better than current Production model, then update stage if so.")

def final_evaluation(**context):
    """
    (8) Extended Final Evaluation
    -----------------------------
    Evaluate multiple models (1–5) from the registry, compare, plot predictions.
    """
    import mlflow
    import pandas as pd
    from model_evaluation import evaluate_model, plot_predictions

    # Load test data from local or S3
    X_test = pd.read_csv("/tmp/X_test.csv")
    Y_test = pd.read_csv("/tmp/Y_test.csv")

    for i in range(1, 6):
        model_name = f"HomeownerLossModel_Model_{i}"
        model_uri = f"models:/{model_name}/Production"
        print(f"Evaluating {model_name} from MLflow registry...")

        try:
            model_pyfunc = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print(f"Skipping {model_name}, no Production version found: {e}")
            continue

        metrics = evaluate_model(
            model_pyfunc,
            X_test,
            Y_test,
            X_test,
            Y_test,
            log_to_mlflow=True,
            model_name=model_name,
            summary_only=False
        )
        print(f"Metrics for {model_name}: {metrics}")

        preds = model_pyfunc.predict(X_test)
        plot_predictions(
            y_true=Y_test,
            y_pred=preds,
            title=f"Actual vs. Predicted for {model_name}",
            plot_type="scatter",
            log_to_mlflow=True,
            filename=f"predictions_{model_name}.png"
        )

    print("Extended final evaluation complete for Models 1–5.")

def performance_monitoring(**context):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def run_inference():
        # your actual inference logic here
        time.sleep(0.2)

    with mlflow.start_run(run_name="performance_monitoring") as run:
        mem_before = psutil.virtual_memory().used / (1024 * 1024)
        start_time = time.time()

        run_inference()

        end_time = time.time()
        mem_after = psutil.virtual_memory().used / (1024 * 1024)

        inference_latency_ms = (end_time - start_time) * 1000
        memory_used_mb = mem_after - mem_before

        mlflow.log_metric("inference_latency_ms", inference_latency_ms)
        mlflow.log_metric("memory_usage_mb", memory_used_mb)

    print(f"Performance monitoring: latency={inference_latency_ms:.2f}ms, memory diff={memory_used_mb:.2f}MB")

def send_notification(**context):
    """
    (10) Notification/Alerting
    --------------------------
    Send Slack (or email) about pipeline results or new model performance.
    """
    message = "Homeowner pipeline run completed successfully!"
    if SLACK_WEBHOOK:
        import requests
        payload = {"text": message}
        response = requests.post(SLACK_WEBHOOK, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"Slack notification failed: {response.text}")
        else:
            print("Slack notification sent.")
    else:
        print("No SLACK_WEBHOOK configured, skipping Slack notification.")

def push_logs_to_s3(**context):
    """
    (11) Push Logs to S3
    --------------------
    If not using Airflow remote logging, we can upload custom logs here.
    """
    log_path = "/tmp/custom_airflow.log"
    with open(log_path, "w") as f:
        f.write("This is an example custom log.\n")

    s3 = boto3.client("s3")
    s3_key = "airflow/logs/custom_airflow.log"
    s3.upload_file(log_path, S3_BUCKET, s3_key)
    print(f"Uploaded logs to s3://{S3_BUCKET}/{s3_key}")

def archive_data(**context):
    """
    (12) Archive Data
    -----------------
    Copy older data from raw-data to an 'archive/' prefix, or Glacier, etc.
    """
    s3 = boto3.client("s3")
    original_key = f"{S3_DATA_FOLDER}/Total_home_loss_hist.csv"
    archive_key = f"archive/Total_home_loss_hist_{int(time.time())}.csv"

    copy_source = {"Bucket": S3_BUCKET, "Key": original_key}
    s3.copy_object(CopySource=copy_source, Bucket=S3_BUCKET, Key=archive_key)
    print(f"Archived s3://{S3_BUCKET}/{original_key} -> s3://{S3_BUCKET}/{archive_key}")

# ---------------------------------------------------------------------
# 3. DAG DEFINITION
# ---------------------------------------------------------------------
with DAG(
    dag_id="homeowner_loss_history_dag_extended",
    default_args=default_args,
    schedule_interval="@hourly",
    catchup=False
) as dag:

    # 1) Ingest
    task_ingest = PythonOperator(
        task_id="ingest_data_from_s3",
        python_callable=ingest_data_from_s3,
    )

    # 2) Validate
    #task_validate = PythonOperator(
        #task_id="validate_data",
        #python_callable=validate_data_with_ge,
    #)

    # 3) Preprocess
    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    # 4) Feature Eng
    task_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    # Download References
    task_download_ref = PythonOperator(
        task_id="download_ref_means",
        python_callable=download_reference_means,
    )

    # 5) Drift Check
    task_drift = PythonOperator(
        task_id="detect_data_drift",
        python_callable=detect_data_drift,
    )

    # 6) Train Model
    task_train = PythonOperator(
        task_id="train_xgboost_hyperopt",
        python_callable=train_xgboost_hyperopt,
    )

    # 7) Registry Update
    task_update_registry = PythonOperator(
        task_id="compare_and_update_registry",
        python_callable=compare_and_update_registry,
    )

    # 8) Final Evaluation
    task_evaluate = PythonOperator(
        task_id="final_evaluation",
        python_callable=final_evaluation,
    )

    # 9) Performance Monitoring
    task_perf_monitor = PythonOperator(
        task_id="performance_monitoring",
        python_callable=performance_monitoring,
    )

    # 10) Notification
    task_notify = PythonOperator(
        task_id="send_notification",
        python_callable=send_notification,
    )

    # 11) Push Logs
    task_push_logs = PythonOperator(
        task_id="push_logs_to_s3",
        python_callable=push_logs_to_s3,
    )

    # 12) Archive
    task_archive = PythonOperator(
        task_id="archive_data",
        python_callable=archive_data,
    )

    # Full pipeline order (linear example)
    (
        task_ingest
        ##>> task_validate
        >> task_preprocess
        >> task_features
        >> task_download_ref
        >> task_drift
        >> task_train
        >> task_update_registry
        >> task_evaluate
        >> task_perf_monitor
        >> task_notify
        >> task_push_logs
        >> task_archive
    )
