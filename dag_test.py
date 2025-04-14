#!/usr/bin/env python3
import os
import json
import time
import random
import shutil
import psutil
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="datetime.datetime.now()")

from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# AWS + MLflow
import boto3
import mlflow
import mlflow.xgboost
# GE is commented out for now to isolate issues
# import great_expectations as ge

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Setup centralized logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Configuration via environment variables or Airflow Variables
S3_BUCKET = os.environ.get("S3_BUCKET", Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket"))
S3_DATA_FOLDER = "raw-data"
S3_MODELS_FOLDER = "models"
LOCAL_DATA_PATH = "/tmp/homeowner_data.csv"
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.csv"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"  # Will be generated and uploaded to S3

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", Variable.get("MLFLOW_TRACKING_URI", default_var="http://3.146.46.179:5000"))
MLFLOW_EXPERIMENT_NAME = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="Homeowner_Loss_Hist_Proj")
MODEL_ID = os.environ.get("MODEL_ID", Variable.get("MODEL_ID", default_var="model1")).lower().strip()

MONOTONIC_CONSTRAINTS_MAP = Variable.get(
    "MONOTONIC_CONSTRAINTS_MAP",
    default_var='{"model1": "(1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1)", "model2": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)", "model3": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)", "model4": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)", "model5": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)"}',
    deserialize_json=True)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

# Create a global boto3 S3 client for reuse
s3_client = boto3.client("s3")

@dag(default_args=default_args, schedule="*/15 * * * *", catchup=False, tags=["homeowner", "loss_history"])
def homeowner_dag():
    """
    Homeowner Loss History Prediction DAG.
    
    Improvements in this DAG:
      - Uses the TaskFlow API.
      - Robust error handling & centralized logging.
      - Expanded hyperparameter search space.
      - Tailored preprocessing (Model1 uses different null/imputation and outlier thresholds than Models2-5).
      - Implements conditional branching using BranchPythonOperator for self-healing/manual override.
      - Auto-generates the reference means file for drift detection and uploads it to S3.
    """

    @task
    def ingest_data_from_s3():
        """Download CSV data from S3 to a local path."""
        try:
            s3_key = f"{S3_DATA_FOLDER}/ut_loss_history_1.csv"
            logging.info(f"Downloading s3://{S3_BUCKET}/{s3_key} to {LOCAL_DATA_PATH}")
            s3_client.download_file(S3_BUCKET, s3_key, LOCAL_DATA_PATH)
            logging.info("Data ingestion complete.")
        except Exception as e:
            logging.error(f"Error in ingest_data_from_s3: {e}")
            raise

    # GE validation is disabled for now
    @task
    def validate_data_with_ge():
        logging.info("Great Expectations validation is disabled for testing.")
        return {"success": True, "message": "GE validation skipped"}

    @task(execution_timeout=timedelta(minutes=15))
    def preprocess_data():
        """
        Process the large CSV in chunks and tailor steps based on MODEL_ID.
        For Model1: fill nulls with 0 and cap outliers at 1.5×IQR.
        For Models2-5: forward-fill nulls, cap outliers at 2.0×IQR, then normalize numeric columns.
        """
        try:
            if not os.path.exists(LOCAL_DATA_PATH):
                raise FileNotFoundError(f"Missing data file: {LOCAL_DATA_PATH}")
            file_size = os.path.getsize(LOCAL_DATA_PATH)
            logging.info(f"{LOCAL_DATA_PATH} exists; size: {file_size} bytes")
            
            chunk_size = 100000
            processed_chunks = []
            for chunk in pd.read_csv(LOCAL_DATA_PATH, chunksize=chunk_size):
                if MODEL_ID == "model1":
                    chunk.fillna(0, inplace=True)
                    logging.info("Model1: Filled null values with 0.")
                else:
                    chunk.fillna(method="ffill", inplace=True)
                    logging.info("Models2-5: Applied forward fill for nulls.")
                
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = chunk[col].quantile(0.25)
                    Q3 = chunk[col].quantile(0.75)
                    iqr = Q3 - Q1
                    if MODEL_ID == "model1":
                        lower_bound = Q1 - 1.5 * iqr
                        upper_bound = Q3 + 1.5 * iqr
                        logging.info(f"Model1: Capping outliers in '{col}' at 1.5x IQR.")
                    else:
                        lower_bound = Q1 - 2.0 * iqr
                        upper_bound = Q3 + 2.0 * iqr
                        logging.info(f"Models2-5: Capping outliers in '{col}' at 2.0x IQR.")
                    chunk[col] = np.clip(chunk[col], lower_bound, upper_bound)
                
                processed_chunks.append(chunk)
            
            df_processed = pd.concat(processed_chunks, ignore_index=True)
            
            if MODEL_ID != "model1":
                for col in df_processed.select_dtypes(include=[np.number]).columns:
                    col_min = df_processed[col].min()
                    col_max = df_processed[col].max()
                    if col_max > col_min:
                        df_processed[col] = (df_processed[col] - col_min) / (col_max - col_min)
                        logging.info(f"Normalized column '{col}' to [0,1].")
            
            df_processed.to_csv(LOCAL_PROCESSED_PATH, index=False)
            logging.info(f"Preprocessing complete. Output saved to {LOCAL_PROCESSED_PATH}")
            return LOCAL_PROCESSED_PATH
        except Exception as e:
            logging.error(f"Error in preprocess_data: {e}")
            raise

    @task
    def feature_engineering(processed_path: str):
        """Apply feature engineering steps (e.g., create avg_loss_per_claim)."""
        try:
            df = pd.read_csv(processed_path)
            if "il_total" in df.columns and "cc_total" in df.columns:
                df["avg_loss_per_claim"] = df.apply(
                    lambda row: row["il_total"] / row["cc_total"] if row["cc_total"] > 0 else 0,
                    axis=1
                )
                logging.info("Created 'avg_loss_per_claim' feature.")
            df.to_csv(processed_path, index=False)
            logging.info("Feature engineering complete.")
            return processed_path
        except Exception as e:
            logging.error(f"Error in feature_engineering: {e}")
            raise

    @task
    def generate_reference_means(processed_path: str):
        """
        Generate the reference means CSV file from processed data, then upload it to S3.
        This ensures detect_data_drift has a valid reference.
        """
        try:
            df = pd.read_csv(processed_path)
            numeric_means = df.select_dtypes(include=[np.number]).mean().reset_index()
            numeric_means.columns = ["column_name", "mean_value"]
            local_ref_path = "/tmp/reference_means.csv"
            numeric_means.to_csv(local_ref_path, index=False)
            s3_key = "reference/reference_means.csv"
            s3_client.upload_file(local_ref_path, S3_BUCKET, s3_key)
            logging.info(f"Generated and uploaded reference means to s3://{S3_BUCKET}/{s3_key}")
            return local_ref_path
        except Exception as e:
            logging.error(f"Error in generate_reference_means: {e}")
            raise

    @task
    def detect_data_drift(processed_path: str):
        """
        Compare current data means with those from the reference means file.
        Return "self_healing" if significant drift is detected (>10%), else "train_model".
        """
        try:
            df_current = pd.read_csv(processed_path)
            if not os.path.exists(REFERENCE_MEANS_PATH):
                logging.warning(f"Reference means file not found at {REFERENCE_MEANS_PATH}. Skipping drift check.")
                return "train_model"
            df_reference = pd.read_csv(REFERENCE_MEANS_PATH)
            reference_dict = {row["column_name"]: row["mean_value"] for _, row in df_reference.iterrows()}
            drift_detected = False
            for col in df_current.select_dtypes(include=[np.number]).columns:
                if col in reference_dict:
                    current_mean = df_current[col].mean()
                    ref_mean = reference_dict[col]
                    if ref_mean > 0:
                        drift_ratio = abs(current_mean - ref_mean) / ref_mean
                        if drift_ratio > 0.1:
                            drift_detected = True
                            logging.error(f"Drift in '{col}': current={current_mean:.2f}, ref={ref_mean:.2f}, ratio={drift_ratio:.2%}")
                    else:
                        logging.warning(f"Non-positive reference mean for '{col}'; skipping check.")
                else:
                    logging.warning(f"No reference for '{col}'; skipping drift check.")
            return "self_healing" if drift_detected else "train_xgboost_hyperopt"
        except Exception as e:
            logging.error(f"Error in detect_data_drift: {e}")
            raise

    def branch_decision_callable(drift_indicator: str):
        logging.info(f"Branch decision: {drift_indicator}")
        return drift_indicator

    # Note: The BranchPythonOperator is defined later in the DAG flow setup
    # using the drift_result variable

    @task
    def self_healing():
        """Simulated self-healing branch that waits for admin override."""
        logging.info("Data drift detected. Executing self-healing routine, awaiting manual override...")
        time.sleep(5)  # Simulate wait time
        logging.info("Manual override complete; proceeding.")
        return "override_done"

    @task
    def manual_override():
        """Check for manual override; return custom hyperparameters if set."""
        try:
            override = Variable.get("MANUAL_OVERRIDE", default_var="False")
            if override.lower() == "true":
                custom_params = json.loads(Variable.get("CUSTOM_HYPERPARAMS", default_var='{}'))
                logging.info("Manual override activated. Using custom hyperparameters.")
                return custom_params
            else:
                logging.info("No manual override; proceeding with automated tuning.")
                return None
        except Exception as e:
            logging.error(f"Error in manual_override: {e}")
            raise

    @task
    def train_xgboost_hyperopt(processed_path: str, override_params):
        """Train an XGBoost model with Hyperopt-based hyperparameter tuning."""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            constraints = MONOTONIC_CONSTRAINTS_MAP.get(MODEL_ID, "(1,1,1,1)")
            df = pd.read_csv(processed_path)
            target_col = "pure_premium"
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in processed data.")
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            # Validate monotonic constraints against feature count
            feature_count = X.shape[1]
            con_list = constraints.strip("()").split(",")
            if len(con_list) != feature_count:
                logging.warning(f"Constraints count ({len(con_list)}) does not match features ({feature_count}). Adjusting...")
                constraints = "(" + ",".join(["1"] * feature_count) + ")"
                logging.info(f"Adjusted constraints: {constraints}")
                updated_map = MONOTONIC_CONSTRAINTS_MAP.copy()
                updated_map[MODEL_ID] = constraints
                Variable.set("MONOTONIC_CONSTRAINTS_MAP", json.dumps(updated_map))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            space = {
                "learning_rate": hp.uniform("learning_rate", 0.001, 0.3),
                "max_depth": hp.choice("max_depth", [3, 5, 7, 9, 11]),
                "n_estimators": hp.quniform("n_estimators", 50, 500, 1),
                "reg_alpha": hp.loguniform("reg_alpha", -5, 0),
                "reg_lambda": hp.loguniform("reg_lambda", -5, 0),
                "subsample": hp.uniform("subsample", 0.5, 1.0),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0)
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
            
            if override_params:
                best = override_params
                logging.info("Using manually overridden hyperparameters.")
            else:
                trials = Trials()
                best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
                best["n_estimators"] = int(best["n_estimators"])
                best["max_depth"] = [3, 5, 7, 9, 11][best["max_depth"]]
                logging.info(f"Best tuned hyperparameters: {best}")
            
            final_model = xgb.XGBRegressor(
                monotone_constraints=constraints,
                use_label_encoder=False,
                eval_metric="rmse",
                **best
            )
            final_model.fit(X_train, y_train)
            rmse = mean_squared_error(y_test, final_model.predict(X_test)) ** 0.5

            with mlflow.start_run(run_name=f"xgboost_{MODEL_ID}_hyperopt"):
                mlflow.log_params(best)
                mlflow.log_metric("rmse", rmse)
                mlflow.xgboost.log_model(final_model, artifact_path="model")
            
            local_model_path = f"/tmp/xgb_{MODEL_ID}_model.json"
            final_model.save_model(local_model_path)
            s3_model_key = f"{S3_MODELS_FOLDER}/xgb_{MODEL_ID}_model.json"
            s3_client.upload_file(local_model_path, S3_BUCKET, s3_model_key)
            logging.info(f"Training complete. RMSE={rmse:.4f}. Model stored at s3://{S3_BUCKET}/{s3_model_key}")
        except Exception as e:
            logging.error(f"Error in train_xgboost_hyperopt: {e}")
            raise

    @task
    def compare_and_update_registry():
        logging.info("Comparing new model with production model (placeholder logic).")

    @task
    def final_evaluation():
        logging.info("Evaluating models from MLflow registry (placeholder logic).")

    @task
    def performance_monitoring():
        logging.info("Monitoring performance metrics (placeholder logic).")

    @task
    def send_notification():
        logging.info("Sending notification (placeholder logic).")

    @task
    def push_logs_to_s3():
        logging.info("Uploading logs to S3 (placeholder logic).")

    @task
    def archive_data():
        logging.info("Archiving old data in S3 (placeholder logic).")

    # DAG flow setup:
    ingestion = ingest_data_from_s3()
    processed = preprocess_data()
    feats = feature_engineering(processed)
    
    # Generate reference means and upload to S3
    gen_ref_means = generate_reference_means(feats)
    
    # Drift detection (after reference means are generated)
    drift_result = detect_data_drift(feats)
    
    override_params = manual_override()

    # Branch decision: use BranchPythonOperator to select the branch
    branch_decision = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=lambda drift: "self_healing" if drift == "self_healing" else "train_xgboost_hyperopt",
        op_args=[drift_result],
    )

    healing = self_healing()
    train_model = train_xgboost_hyperopt(feats, override_params)

    # Use an EmptyOperator to join branches
    join_branches = EmptyOperator(task_id="join_branches")

    registry_update = compare_and_update_registry()
    evaluation_task = final_evaluation()
    performance_task = performance_monitoring()
    notification_task = send_notification()
    logs_task = push_logs_to_s3()
    archive_task = archive_data()

    # Define dependencies:
    ingestion >> processed >> feats
    feats >> gen_ref_means
    feats >> drift_result >> branch_decision
    branch_decision >> healing
    branch_decision >> train_model
    [healing, train_model] >> join_branches
    join_branches >> registry_update >> evaluation_task >> performance_task >> notification_task >> logs_task >> archive_task

dag = homeowner_dag()
