#!/usr/bin/env python3
"""
tasks/training.py

Train and log XGBoost models with hyperparameter tuning,
then compare and optionally promote new model versions in MLflow.

Refactored to use utils/storage and utils/slack wrappers instead of direct boto3/requests calls.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from airflow.models import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from utils.storage import upload       # wrapper for S3 uploads
from utils.slack import send_message  # wrapper for Slack notifications
from utils.airflow_api import trigger_dag  # wrapper for Airflow API

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Configuration
MLFLOW_TRACKING_URI    = os.getenv(
    "MLFLOW_TRACKING_URI",
    Variable.get("MLFLOW_TRACKING_URI", default_var="http://localhost:5000")
)
MLFLOW_EXPERIMENT_NAME = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="Homeowner_Loss_Hist_Proj")
S3_BUCKET              = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET")
S3_MODELS_FOLDER       = "models"
MONO_MAP               = Variable.get("MONOTONIC_CONSTRAINTS_MAP", deserialize_json=True)


def manual_override() -> Optional[Dict]:
    """
    Fetch custom hyperparameters if manual override is enabled via Airflow Variable.
    """
    try:
        if Variable.get("MANUAL_OVERRIDE", default_var="False").lower() == "true":
            params = json.loads(Variable.get("CUSTOM_HYPERPARAMS", default_var="{}"))
            logging.info("Manual override: using custom hyperparameters.")
            return params
    except Exception as e:
        logging.error(f"Could not fetch manual override: {e}")
    return None


def train_xgboost_hyperopt(
    processed_path: str,
    override_params: Optional[Dict] = None,
    model_id: Optional[str] = None
) -> str:
    """
    Train an XGBoost model with Hyperopt tuning, log metrics/artifacts to MLflow,
    and return the MLflow run ID.
    """
    # Select model_id
    model_id = (
        model_id
        or os.getenv("MODEL_ID", Variable.get("MODEL_ID", default_var="model1"))
    ).strip().lower()
    logging.info(f"Training for MODEL_ID='{model_id}'")

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load data
    df = pd.read_csv(processed_path)
    if "pure_premium" not in df:
        df["pure_premium"]  = df["il_total"] / df["eey"]
        df["sample_weight"] = df["eey"]
        logging.info("Computed pure_premium & sample_weight")

    y = df["pure_premium"]
    w = df["sample_weight"]
    X = df.drop(columns=["pure_premium", "sample_weight"])

    # Monotonic constraints
    constraints = MONO_MAP.get(model_id, [])
    if len(constraints) != X.shape[1]:
        logging.warning("Constraint length mismatch; resetting to zeros.")
        constraints = [0] * X.shape[1]
        MONO_MAP[model_id] = constraints
        Variable.set("MONOTONIC_CONSTRAINTS_MAP", json.dumps(MONO_MAP))
    constraints = tuple(constraints)

    # Train/test split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.3, random_state=42
    )

    # Hyperopt search space
    space = {
        "learning_rate":    hp.uniform("learning_rate", 0.001, 0.3),
        "max_depth":        hp.choice("max_depth", [3,5,7,9,11]),
        "n_estimators":     hp.quniform("n_estimators", 50,500,1),
        "reg_alpha":        hp.loguniform("reg_alpha", -5,0),
        "reg_lambda":       hp.loguniform("reg_lambda", -5,0),
        "subsample":        hp.uniform("subsample", 0.5,1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5,1.0),
    }

    def objective(params):
        params["n_estimators"] = int(params["n_estimators"])
        model = xgb.XGBRegressor(
            tree_method="hist",
            monotone_constraints=constraints,
            use_label_encoder=False,
            eval_metric="rmse",
            **params
        )
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            sample_weight_eval_set=[w_test],
            early_stopping_rounds=20,
            verbose=False
        )
        preds = model.predict(X_test)
        rmse  = mean_squared_error(y_test, preds, sample_weight=w_test)**0.5
        return {"loss": rmse, "status": STATUS_OK}

    # Run tuning
    if override_params:
        best_params = override_params
        logging.info("Using override hyperparameters.")
    else:
        trials = Trials()
        best   = fmin(objective, space, algo=tpe.suggest, max_evals=int(Variable.get("HYPEROPT_MAX_EVALS", 20)), trials=trials)
        best_params = {
            **{k: (int(v) if k=="n_estimators" else v) for k,v in best.items()},
            "max_depth": [3,5,7,9,11][best["max_depth"]]
        }
        logging.info(f"Tuned params for '{model_id}': {best_params}")

    # Final train with best params
    final_model = xgb.XGBRegressor(
        tree_method="gpu_hist",
        monotone_constraints=constraints,
        use_label_encoder=False,
        eval_metric="rmse",
        **best_params
    )
    final_model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[w_test],
        early_stopping_rounds=20,
        verbose=False
    )

    # Evaluation metrics
    preds     = final_model.predict(X_test)
    final_rmse= mean_squared_error(y_test, preds, sample_weight=w_test)**0.5
    final_r2  = r2_score(y_test, preds, sample_weight=w_test)
    thr       = y_test.median()
    final_acc = accuracy_score((y_test>thr).astype(int), (preds>thr).astype(int))
    final_f1  = f1_score((y_test>thr).astype(int), (preds>thr).astype(int))

    # Log charts to MLflow
    df_eval = pd.DataFrame({"actual": y_test, "pred": preds})
    df_eval["decile"] = pd.qcut(df_eval["pred"], 10, labels=False)
    decile_means = df_eval.groupby("decile")[["actual","pred"]].mean()
    fig1, ax1 = plt.subplots()
    ax1.plot(decile_means.index+1, decile_means["actual"], marker="o", label="Actual")
    ax1.plot(decile_means.index+1, decile_means["pred"],   marker="o", label="Predicted")
    ax1.set_title("Actual vs Predicted by Decile")
    ax1.set_xlabel("Decile"); ax1.set_ylabel("Pure Premium"); ax1.legend()
    mlflow.log_figure(fig1, "actual_vs_pred_by_decile.png")
    plt.close(fig1)

    imp_dict = final_model.get_booster().get_score(importance_type="gain")
    imp_df   = pd.DataFrame({"feature":list(imp_dict), "importance":list(imp_dict.values())})
    imp_df.sort_values("importance", inplace=True)
    fig2, ax2 = plt.subplots(figsize=(8, max(4,len(imp_df)*0.3)))
    ax2.barh(imp_df["feature"], imp_df["importance"])  
    ax2.set_title("Feature Importance (gain)")
    ax2.set_xlabel("Gain")
    mlflow.log_figure(fig2, "feature_importance.png")
    plt.close(fig2)

    # Log run to MLflow
    with mlflow.start_run(run_name=f"xgb_{model_id}_hyperopt") as run:
        mlflow.log_params(best_params)
        mlflow.log_metrics({"rmse": final_rmse, "r2": final_r2, "acc": final_acc, "f1": final_f1})
        mlflow.xgboost.log_model(final_model, artifact_path="model")
        run_id = run.info.run_id

    # Slack notification
    send_message(
        channel=Variable.get("SLACK_CHANNEL_DEFAULT", default_var="#alerts"),
        title=f"🔔 Training Summary ({model_id})",
        body=(f"RMSE {final_rmse:.2f}, R2 {final_r2:.2f}, "
              f"Acc {final_acc:.2f}, F1 {final_f1:.2f}"),
        urgency="low"
    )

    # Save & upload model file
    ts    = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"xgb_{model_id}_model_{ts}.json"
    path  = f"/tmp/{fname}"
    final_model.save_model(path)
    upload(path, f"{S3_MODELS_FOLDER}/{fname}")
    logging.info(f"Uploaded model to s3://{S3_BUCKET}/{S3_MODELS_FOLDER}/{fname}")

    return run_id


def compare_and_update_registry(
    model_name: str,
    run_id: str,
    metric_key: str = "rmse",
    stage: str = "Production"
) -> bool:
    """
    Compare the given run's metric against the current Production model. If better,
    transition the new run's model version to `stage`, archiving existing versions.
    Returns True if promotion happened, else False.
    """
    client = MlflowClient()
    # Fetch existing production versions
    prod_versions = client.get_latest_versions(name=model_name, stages=[stage])
    new_metric = client.get_run(run_id).data.metrics.get(metric_key)
    
    if prod_versions:
        current = prod_versions[0]
        prod_run_id = client.get_model_version(name=model_name, version=current.version).run_id
        prod_metric = client.get_run(prod_run_id).data.metrics.get(metric_key)
        logging.info(f"Current {stage} metric={prod_metric:.4f}, new run={new_metric:.4f}")
        # Lower RMSE is better
        if new_metric is not None and prod_metric is not None and new_metric < prod_metric:
            client.transition_model_version_stage(
                name=model_name,
                version=run_id,
                stage=stage,
                archive_existing_versions=True
            )
            logging.info(f"Promoted run {run_id} to {stage}")
            return True
        return False
    else:
        # No existing prod; promote new
        client.transition_model_version_stage(
            name=model_name,
            version=run_id,
            stage=stage
        )
        logging.info(f"Promoted run {run_id} to {stage} (first version)")
        return True
