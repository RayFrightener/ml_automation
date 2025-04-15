#!/usr/bin/env python3
"""
training.py

Handles:
  - Checking for manual override.
  - Hyperparameter tuning via Hyperopt.
  - Model training using XGBoost.
  - Logging metrics and parameters to MLflow.
  - Saving the trained model locally and uploading it to S3.
  - (Placeholder) Comparing and updating the model registry.

Requires:
  - MLflow tracking URI.
  - S3_BUCKET and S3_MODELS_FOLDER.
  - MONOTONIC_CONSTRAINTS_MAP provided as an Airflow Variable (retrieved as MONO_MAP).
"""

import os
import json
import logging
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from airflow.models import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import boto3

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Configuration parameters
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", Variable.get("MLFLOW_TRACKING_URI", default_var="http://3.146.46.179:5000"))
MLFLOW_EXPERIMENT_NAME = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="Homeowner_Loss_Hist_Proj")
S3_BUCKET = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET")
S3_MODELS_FOLDER = "models"
MODEL_ID = os.getenv("MODEL_ID", Variable.get("MODEL_ID", default_var="model1")).strip().lower()
MONO_MAP = Variable.get("MONOTONIC_CONSTRAINTS_MAP", deserialize_json=True)

s3_client = boto3.client("s3")


def manual_override():
    """
    Returns manually overridden hyperparameters from Airflow Variables if present.
    """
    try:
        override = Variable.get("MANUAL_OVERRIDE", default_var="False")
        if override.lower() == "true":
            custom_params = json.loads(Variable.get("CUSTOM_HYPERPARAMS", default_var="{}"))
            logging.info("Manual override activated; using custom hyperparameters.")
            return custom_params
    except Exception as e:
        logging.error(f"Error fetching manual override: {e}")
    return None


def train_xgboost_hyperopt(processed_path, override_params=None):
    """
    Trains an XGBoost model with hyperparameter tuning via Hyperopt.
    
    Args:
        processed_path (str): CSV file path to the preprocessed data.
        override_params (dict): Manually supplied hyperparameters (if any).
        
    Returns:
        float: The final RMSE computed on the test set.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        df = pd.read_csv(processed_path)
        if "pure_premium" not in df.columns:
            raise ValueError("Target column 'pure_premium' missing from the data.")
        y = df["pure_premium"]
        X = df.drop(columns=["pure_premium"])

        # Fetch monotonic constraints based on MODEL_ID
        constraints = MONO_MAP.get(MODEL_ID, "(1,1,1,1)")
        constraint_list = constraints.strip("()").split(",")
        if len(constraint_list) != X.shape[1]:
            # Adjust constraints to match the number of features
            constraints = "(" + ",".join(["1"] * X.shape[1]) + ")"
            MONO_MAP[MODEL_ID] = constraints
            Variable.set("MONOTONIC_CONSTRAINTS_MAP", json.dumps(MONO_MAP))
            logging.warning(f"Adjusted monotonic constraints to match {X.shape[1]} features.")

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
            logging.info("Using manually provided hyperparameters.")
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
        final_rmse = mean_squared_error(y_test, final_model.predict(X_test)) ** 0.5

        with mlflow.start_run(run_name=f"xgboost_{MODEL_ID}_hyperopt"):
            mlflow.log_params(best)
            mlflow.log_metric("rmse", final_rmse)
            mlflow.xgboost.log_model(final_model, artifact_path="model")

        # Save model locally and then upload it to S3
        model_path = f"/tmp/xgb_{MODEL_ID}_model.json"
        final_model.save_model(model_path)
        s3_key = f"{S3_MODELS_FOLDER}/xgb_{MODEL_ID}_model.json"
        s3_client.upload_file(model_path, S3_BUCKET, s3_key)
        logging.info(f"Training complete. RMSE: {final_rmse:.4f}. Model saved at s3://{S3_BUCKET}/{s3_key}")

        return final_rmse

    except Exception as e:
        logging.error(f"Error in train_xgboost_hyperopt: {e}")
        raise


def compare_and_update_registry():
    """
    Placeholder for comparing the newly trained model against the production model.
    In production, this could use the MLflow Model Registry API to promote a model if it
    outperforms the current version.
    """
    logging.info("Comparing new model with production model... (placeholder)")
    return

if __name__ == "__main__":
    # For testing purposes only
    sample_csv = "/tmp/sample_processed.csv"
    if not os.path.exists(sample_csv):
        pd.DataFrame({"pure_premium": [100, 200, 150], "feature1": [1,2,3]}).to_csv(sample_csv, index=False)
    rmse = train_xgboost_hyperopt(sample_csv, manual_override())
    print(f"Final RMSE: {rmse}")
