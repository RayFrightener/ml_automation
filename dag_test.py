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
warnings.filterwarnings("ignore", category=DeprecationWarning, message="datetime.datetime.utcnow()")

from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.models import Variable

# AWS + MLflow + Great Expectations
import boto3
import mlflow
import mlflow.xgboost
# Uncomment the following if Great Expectations is configured
import great_expectations as ge

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
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", Variable.get("MLFLOW_TRACKING_URI", default_var="http://3.146.46.179:5000"))
MLFLOW_EXPERIMENT_NAME = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="Homeowner_Loss_Hist_Proj")
MODEL_ID = os.environ.get("MODEL_ID", Variable.get("MODEL_ID", default_var="model1")).lower().strip()

# Monotonic constraints can be configured dynamically via Airflow Variables.
MONOTONIC_CONSTRAINTS_MAP = Variable.get("MONOTONIC_CONSTRAINTS_MAP", default_var='{"model1": "(1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1)", "model2": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)", "model3": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)", "model4": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)", "model5": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)"}', deserialize_json=True)

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
def homeowner_loss_history_dag_extended():
    """
    Homeowner Loss History Prediction DAG.
    
    Improvements:
    1. Uses the TaskFlow API.
    2. Implements robust error handling and centralized logging.
    3. Expands the hyperparameter search space.
    4. Improves code structure and readability.
    5. Optimizes for scalability and performance.
    6. Uses centralized log management.
    7. Includes dynamic workflow branching for self-healing and supports manual override.
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

    @task
    def validate_data_with_ge():
        """
        Runs Great Expectations checkpoint validation on the raw data.
        Assumes that great_expectations.yml is in the working directory or GE_HOME is set.
        """
        try:
            # Hard-coded checkpoint name for quick use; you can change this if needed.
            checkpoint_name = "my_checkpoint"

            # Load GE context - ensures it reads great_expectations.yml from your GE project folder
            ge_context = ge.get_context()

            # Option 1: Trigger the checkpoint directly (if your project defines one)
            results = ge_context.run_checkpoint(checkpoint_name=checkpoint_name)
            
            if not results.get("success", False):
                raise ValueError("Great Expectations validation failed!")
            
            logging.info("Data validated successfully with GE checkpoint '%s'. Results: %s", checkpoint_name, results)
        except Exception as e:
            logging.error("Error in validate_data_with_ge: %s", e)
            raise

    @task
    def preprocess_data():
        """Preprocess data: fill nulls, cap outliers, and save the processed CSV."""
        try:
            df = pd.read_csv(LOCAL_DATA_PATH)
            logging.info(f"Read {len(df)} rows from {LOCAL_DATA_PATH}.")
            df.fillna(0, inplace=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                iqr = Q3 - Q1
                lower_bound = Q1 - 1.5 * iqr
                upper_bound = Q3 + 1.5 * iqr
                df[col] = np.clip(df[col], lower_bound, upper_bound)
            df.to_csv(LOCAL_PROCESSED_PATH, index=False)
            logging.info(f"Preprocessing complete. Output saved to {LOCAL_PROCESSED_PATH}")
            return LOCAL_PROCESSED_PATH
        except Exception as e:
            logging.error(f"Error in preprocess_data: {e}")
            raise

    @task
    def feature_engineering(processed_path: str):
        """Create additional features to aid model performance."""
        try:
            df = pd.read_csv(processed_path)
            if "il_total" in df.columns and "cc_total" in df.columns:
                df["avg_loss_per_claim"] = df.apply(
                    lambda row: row["il_total"] / row["cc_total"] if row["cc_total"] > 0 else 0,
                    axis=1
                )
                logging.info("Created feature 'avg_loss_per_claim'.")
            df.to_csv(processed_path, index=False)
            logging.info("Feature engineering complete.")
            return processed_path
        except Exception as e:
            logging.error(f"Error in feature_engineering: {e}")
            raise

    @task
    def download_reference_means():
        """Download reference means CSV from S3."""
        try:
            s3_client.download_file(S3_BUCKET, "reference/reference_means.csv", REFERENCE_MEANS_PATH)
            logging.info(f"Downloaded reference means to {REFERENCE_MEANS_PATH}")
        except Exception as e:
            logging.error(f"Error in download_reference_means: {e}")
            raise

    @task
    def detect_data_drift(processed_path: str):
        """
        Detect data drift by comparing current data means with reference means.
        Returns a branch indicator: "self_healing_manual_override" if drift is detected,
        else "continue_pipeline".
        """
        try:
            df_current = pd.read_csv(processed_path)
            logging.info(f"Loaded current data with shape: {df_current.shape}")
            if not os.path.exists(REFERENCE_MEANS_PATH):
                logging.warning(f"Reference means file not found at {REFERENCE_MEANS_PATH}. Skipping drift check.")
                return "continue_pipeline"
            df_reference = pd.read_csv(REFERENCE_MEANS_PATH)
            logging.info(f"Loaded reference means with shape: {df_reference.shape}")
            reference_dict = {row["column_name"]: row["mean_value"] for _, row in df_reference.iterrows()}
            drift_detected = False
            for col in df_current.select_dtypes(include=[np.number]).columns:
                if col in reference_dict:
                    current_mean = df_current[col].mean()
                    ref_mean = reference_dict[col]
                    if ref_mean > 0:
                        drift_ratio = abs(current_mean - ref_mean) / ref_mean
                        if drift_ratio > 0.3:
                            drift_detected = True
                            logging.error(f"Data drift detected in column '{col}': current mean={current_mean:.2f}, reference={ref_mean:.2f}, ratio={drift_ratio:.2%}")
                        else:
                            logging.info(f"No significant drift in '{col}' (ratio={drift_ratio:.2%}).")
                    else:
                        logging.warning(f"Non-positive reference mean for '{col}'; skipping ratio check.")
                else:
                    logging.warning(f"Reference for column '{col}' not found; skipping.")
            if drift_detected:
                return "self_healing_manual_override"
            else:
                logging.info("Data drift check passed; no significant drift found.")
                return "continue_pipeline"
        except Exception as e:
            logging.error(f"Error in detect_data_drift: {e}")
            raise

    @task
    def manual_override():
        """
        Check for a manual override. Admins can set the Airflow Variable
        MANUAL_OVERRIDE to 'True' and supply custom hyperparameters (as JSON)
        via CUSTOM_HYPERPARAMS.
        """
        try:
            override = Variable.get("MANUAL_OVERRIDE", default_var="False")
            if override.lower() == "true":
                logging.info("Manual override activated by admin. Using custom hyperparameters.")
                custom_params = json.loads(Variable.get("CUSTOM_HYPERPARAMS", default_var='{}'))
                return custom_params
            else:
                logging.info("No manual override; proceeding with automated tuning.")
                return None
        except Exception as e:
            logging.error(f"Error in manual_override: {e}")
            raise

    @task
    def train_xgboost_hyperopt(processed_path: str, override_params):
        """Train the XGBoost model using Hyperopt for hyperparameter tuning."""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            constraints = MONOTONIC_CONSTRAINTS_MAP.get(MODEL_ID, "(1,1,1,1)")
            df = pd.read_csv(processed_path)
            target_col = "pure_premium"
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found.")
            y = df[target_col]
            X = df.drop(columns=[target_col])
            
            # Validate monotonic constraints match feature count
            # Remove parentheses and count the number of constraints
            constraints_count = len(constraints.strip('()').split(','))
            feature_count = X.shape[1]
            
            # Different models may have different feature sets
            logging.info(f"Model {MODEL_ID} has {feature_count} features and {constraints_count} constraints")
            
            if constraints_count != feature_count:
                logging.warning(f"Monotonic constraints count ({constraints_count}) doesn't match feature count ({feature_count}).")
                logging.warning(f"This may cause training errors. Adjusting constraints to match feature count.")
                
                # Create default constraints matching feature count (all positive)
                constraints = "(" + ",".join(["1"] * feature_count) + ")"
                logging.info(f"Using adjusted constraints: {constraints}")
                
                # Update the constraints map for future runs
                updated_map = MONOTONIC_CONSTRAINTS_MAP.copy()
                updated_map[MODEL_ID] = constraints
                Variable.set("MONOTONIC_CONSTRAINTS_MAP", json.dumps(updated_map))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Expanded hyperparameter search space
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
                params["max_depth"] = params["max_depth"]  # Already chosen from list
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
                logging.info("Using admin-specified hyperparameters.")
            else:
                trials = Trials()
                best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
                best["n_estimators"] = int(best["n_estimators"])
                best["max_depth"] = [3,5,7,9,11][best["max_depth"]]
                logging.info(f"Best hyperparameters from tuning: {best}")

            final_model = xgb.XGBRegressor(
                monotone_constraints=constraints,
                use_label_encoder=False,
                eval_metric="rmse",
                **best
            )
            final_model.fit(X_train, y_train)
            preds = final_model.predict(X_test)
            final_rmse = mean_squared_error(y_test, preds) ** 0.5

            with mlflow.start_run(run_name=f"xgboost_{MODEL_ID}_hyperopt") as run:
                mlflow.log_params(best)
                mlflow.log_metric("rmse", final_rmse)
                mlflow.xgboost.log_model(final_model, artifact_path="model")

            local_model_path = f"/tmp/xgb_{MODEL_ID}_model.json"
            final_model.save_model(local_model_path)
            s3_key = f"{S3_MODELS_FOLDER}/xgb_{MODEL_ID}_model.json"
            s3_client.upload_file(local_model_path, S3_BUCKET, s3_key)
            logging.info(f"Training complete. Best RMSE={final_rmse:.4f}. Model saved at s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            logging.error(f"Error in train_xgboost_hyperopt: {e}")
            raise

    @task
    def compare_and_update_registry():
        """Compare the new model with the current production model and update if better."""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            # Placeholder for registry comparison logic
            logging.info("Comparing new model with production model and updating registry if improved.")
        except Exception as e:
            logging.error(f"Error in compare_and_update_registry: {e}")
            raise

    @task
    def final_evaluation():
        """Evaluate and plot predictions from multiple models in the registry."""
        try:
            X_test = pd.read_csv("/tmp/X_test.csv")
            Y_test = pd.read_csv("/tmp/Y_test.csv")
            for i in range(1, 6):
                model_name = f"HomeownerLossModel_Model_{i}"
                model_uri = f"models:/{model_name}/Production"
                logging.info(f"Evaluating {model_name} from MLflow registry...")
                try:
                    model_pyfunc = mlflow.pyfunc.load_model(model_uri)
                except Exception as e:
                    logging.error(f"Skipping {model_name} (not found): {e}")
                    continue
                preds = model_pyfunc.predict(X_test)
                rmse = mean_squared_error(Y_test, preds) ** 0.5
                logging.info(f"Model {model_name} RMSE: {rmse:.4f}")
            logging.info("Final evaluation complete for all models.")
        except Exception as e:
            logging.error(f"Error in final_evaluation: {e}")
            raise

    @task
    def performance_monitoring():
        """Monitor inference performance (latency and memory usage) and log metrics via MLflow."""
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            start_time = time.time()
            mem_before = psutil.virtual_memory().used / (1024 * 1024)
            time.sleep(0.2)  # Simulate inference
            end_time = time.time()
            mem_after = psutil.virtual_memory().used / (1024 * 1024)
            latency_ms = (end_time - start_time) * 1000
            memory_diff_mb = mem_after - mem_before
            mlflow.log_metric("inference_latency_ms", latency_ms)
            mlflow.log_metric("memory_usage_mb", memory_diff_mb)
            logging.info(f"Performance: latency={latency_ms:.2f}ms, memory change={memory_diff_mb:.2f}MB")
        except Exception as e:
            logging.error(f"Error in performance_monitoring: {e}")
            raise

    @task
    def send_notification():
        """Send a notification (e.g., via Slack) upon pipeline completion."""
        try:
            message = "Homeowner pipeline run completed successfully!"
            slack_webhook = os.environ.get("SLACK_WEBHOOK") or Variable.get("SLACK_WEBHOOK", default_var=None)
            if slack_webhook:
                import requests
                response = requests.post(slack_webhook, data=json.dumps({"text": message}))
                if response.status_code != 200:
                    logging.error(f"Slack notification failed: {response.text}")
                else:
                    logging.info("Slack notification sent.")
            else:
                logging.info("No Slack webhook configured; skipping notification.")
        except Exception as e:
            logging.error(f"Error in send_notification: {e}")
            raise

    @task
    def push_logs_to_s3():
        """Upload custom logs to S3 for centralized log management."""
        try:
            log_path = "/tmp/custom_airflow.log"
            with open(log_path, "w") as f:
                f.write("Centralized log entry for Airflow pipeline.\n")
            s3_key = "airflow/logs/custom_airflow.log"
            s3_client.upload_file(log_path, S3_BUCKET, s3_key)
            logging.info(f"Uploaded logs to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            logging.error(f"Error in push_logs_to_s3: {e}")
            raise

    @task
    def archive_data():
        """Archive old data files in S3."""
        try:
            original_key = f"{S3_DATA_FOLDER}/Total_home_loss_hist.csv"
            archive_key = f"archive/Total_home_loss_hist_{int(time.time())}.csv"
            copy_source = {"Bucket": S3_BUCKET, "Key": original_key}
            s3_client.copy_object(CopySource=copy_source, Bucket=S3_BUCKET, Key=archive_key)
            logging.info(f"Archived s3://{S3_BUCKET}/{original_key} to s3://{S3_BUCKET}/{archive_key}")
        except Exception as e:
            logging.error(f"Error in archive_data: {e}")
            raise

    # Define branch decision based on data drift detection outcome
    @task
    def branch_decision(decision: str):
        if decision == "self_healing_manual_override":
            logging.info("Data drift detected. Routing to self-healing/manual override workflow.")
            return "self_healing"
        else:
            logging.info("No drift detected. Continuing pipeline.")
            return "continue_training"

    @task
    def self_healing():
        """
        Self-healing routine that simulates waiting for a manual override.
        In a real environment, this could pause and await admin input through a UI.
        """
        try:
            logging.info("Executing self-healing routine. Awaiting manual override.")
            time.sleep(5)  # Simulate wait time for admin intervention
            logging.info("Manual override processed; resuming pipeline.")
            return "override_complete"
        except Exception as e:
            logging.error(f"Error in self_healing: {e}")
            raise

    # DAG pipeline ordering using the TaskFlow API
    data_ingest = ingest_data_from_s3()
    data_validation = validate_data_with_ge()
    preprocess = preprocess_data()
    features = feature_engineering(preprocess)
    download_ref = download_reference_means()
    drift_flag = detect_data_drift(features)
    override_params = manual_override()
    branch_path = branch_decision(drift_flag)
    
    # Depending on branch decision, run self-healing or continue directly to training.
    healing = self_healing()
    
    # Using a simple branch merge here: if branch decision is "self_healing", wait for healing,
    # else continue with training. (This example uses a simplified merge; for more complex scenarios,
    # consider using external task sensors or join operators.)
    training_input = features  # In both paths, training uses the processed features.
    train_model = train_xgboost_hyperopt(training_input, override_params)
    
    registry_update = compare_and_update_registry()
    evaluation = final_evaluation()
    performance = performance_monitoring()
    notification = send_notification()
    log_push = push_logs_to_s3()
    archive = archive_data()
    
    # Define overall task dependencies
    data_ingest >> data_validation >> preprocess >> features
    features >> download_ref >> drift_flag >> branch_path
    # If branch decision returns "self_healing", then healing task is executed;
    # Otherwise, training proceeds directly. (Here we call both for illustration.)
    branch_path >> [healing, train_model]
    train_model >> registry_update >> evaluation >> performance >> notification >> log_push >> archive

# Instantiate the DAG
dag = homeowner_loss_history_dag_extended()
