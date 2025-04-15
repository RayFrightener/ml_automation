#!/usr/bin/env python3
"""
training.py

This module defines functions for training the XGBoost model using hyperparameter tuning.
It logs experiments via MLflow and saves the final model to S3.
"""

import os
import json
import logging
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from airflow.models import Variable
import boto3

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Configuration: read from environment or Airflow Variables
S3_BUCKET = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket")
S3_MODELS_FOLDER = "models"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or Variable.get("MLFLOW_TRACKING_URI", default_var="http://3.146.46.179:5000")
MLFLOW_EXPERIMENT_NAME = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="Homeowner_Loss_Hist_Proj")
MODEL_ID = (os.getenv("MODEL_ID") or Variable.get("MODEL_ID", default_var="model1")).lower().strip()
MONOTONIC_CONSTRAINTS_MAP = Variable.get(
    "MONOTONIC_CONSTRAINTS_MAP",
    default_var='{"model1": "(1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1)", "model2": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)", "model3": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)", "model4": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)", "model5": "(1,1,1,1,1,1,1,1,1,1,1,1,1,1)"}',
    deserialize_json=True)
LOCAL_PROCESSED_PATH = "/tmp/homeowner_processed.csv"

# Create global S3 client
s3_client = boto3.client("s3")


def train_xgboost_hyperopt(processed_path: str, override_params):
    """
    Trains an XGBoost model with hyperparameter tuning using Hyperopt.
    Logs parameters and metrics using MLflow and saves the final model to S3.
    
    Args:
        processed_path (str): Path to the preprocessed CSV file.
        override_params (dict or None): Optional manually provided hyperparameters.
    """
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
        
        # Validate monotonic constraints: adjust if feature count mismatches
        feature_count = X.shape[1]
        con_list = constraints.strip("()").split(",")
        if len(con_list) != feature_count:
            logging.warning(f"Constraints count ({len(con_list)}) does not match feature count ({feature_count}). Adjusting...")
            constraints = "(" + ",".join(["1"] * feature_count) + ")"
            logging.info(f"Adjusted constraints: {constraints}")
            updated_map = MONOTONIC_CONSTRAINTS_MAP.copy()
            updated_map[MODEL_ID] = constraints
            Variable.set("MONOTONIC_CONSTRAINTS_MAP", json.dumps(updated_map))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define hyperparameter search space
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

        # Log the experiment using MLflow
        with mlflow.start_run(run_name=f"xgboost_{MODEL_ID}_hyperopt"):
            mlflow.log_params(best)
            mlflow.log_metric("rmse", rmse)
            mlflow.xgboost.log_model(final_model, artifact_path="model")

        # Save the model locally and upload to S3
        local_model_path = f"/tmp/xgb_{MODEL_ID}_model.json"
        final_model.save_model(local_model_path)
        s3_model_key = f"{S3_MODELS_FOLDER}/xgb_{MODEL_ID}_model.json"
        s3_client.upload_file(local_model_path, S3_BUCKET, s3_model_key)
        logging.info(f"Training complete. RMSE={rmse:.4f}. Model stored at s3://{S3_BUCKET}/{s3_model_key}")
    except Exception as e:
        logging.error(f"Error in train_xgboost_hyperopt: {e}")
        raise


def compare_and_update_registry():
    """
    Placeholder for comparing the new model with the existing production model.
    If the new model improves performance, update the registry.
    """
    logging.info("Comparing new model with production model (placeholder logic).")
    return {"status": "success"}


if __name__ == "__main__":
    # For testing purposes, call the training function.
    test_processed_path = LOCAL_PROCESSED_PATH  # Ensure the processed CSV exists for testing.
    test_override_params = None  # Or provide a dict of manual parameters if desired.
    train_xgboost_hyperopt(test_processed_path, test_override_params)
