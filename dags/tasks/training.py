#!/usr/bin/env python3
"""
tasks/training.py  – Unified trainer + explainability
----------------------------------------------------------
 • HyperOpt → finds best hyper‑parameters per model (configurable max‑evals)
 • Automatic fallback from TimeSeriesSplit → train_test_split if not enough chronological rows
 • Logs to MLflow: RMSE, MSE, MAE, R² + SHAP summary + Actual vs. Predicted plot
 • Archives every SHAP plot to s3://<BUCKET>/visuals/shap/
 • Archives actual_vs_predicted plots to s3://<BUCKET>/visuals/avs_pred/
 • If the new RMSE beats the Production model, the model version is promoted
 • Supports manual_override from Airflow Variables for human‑in‑the‑loop
"""

import os
import json
import logging
import tempfile
import time
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import TimeSeriesSplit, train_test_split

import mlflow
import mlflow.xgboost
from airflow.models import Variable

from utils.slack import post as slack_msg
from utils.storage import upload as s3_upload

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

def manual_override() -> Optional[Dict[str, Any]]:
    """
    If the Airflow Variable MANUAL_OVERRIDE == "True",
    return the JSON contained in CUSTOM_HYPERPARAMS; otherwise None.
    """
    try:
        if Variable.get("MANUAL_OVERRIDE", default_var="False").lower() == "true":
            params = json.loads(Variable.get("CUSTOM_HYPERPARAMS", default_var="{}"))
            LOGGER.info("manual_override: using custom hyperparameters %s", params)
            return params
    except Exception as e:
        LOGGER.error("manual_override() error: %s", e)
    return None

# ─── ENV / AIRFLOW VARIABLES ─────────────────────────────────────────────────
BUCKET           = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET")
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI") or Variable.get("MLFLOW_TRACKING_URI")
EXPERIMENT       = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="Homeowner_Loss_Hist_Proj")
MAX_EVALS        = int(os.getenv("HYPEROPT_MAX_EVALS", Variable.get("HYPEROPT_MAX_EVALS", 20)))
SHAP_SAMPLE_ROWS = int(os.getenv("SHAP_SAMPLE_ROWS", 200))

# Monotone constraints map (from Airflow Variable, JSON)
try:
    MONO_MAP: Dict[str, list] = Variable.get("MONOTONIC_CONSTRAINTS_MAP", deserialize_json=True)
except Exception:
    MONO_MAP = {}

# MLflow client setup
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

def _train_val_split(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Try a TimeSeriesSplit first; on failure fallback to random train_test_split.
    """
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, val_idx = next(tscv.split(X))
        LOGGER.info("Using TimeSeriesSplit (n_splits=5)")
        return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
    except Exception as e:
        LOGGER.warning("TimeSeriesSplit failed; falling back to train_test_split: %s", e)
        return train_test_split(X, y, test_size=0.2, random_state=42)

def _shap_summary(model: xgb.XGBRegressor, X_val: pd.DataFrame, model_id: str) -> str:
    """
    Generate a SHAP summary plot, save locally, upload to S3, return S3 key.
    """
    explainer = shap.Explainer(model)
    shap_vals = explainer(X_val[:SHAP_SAMPLE_ROWS])
    shap.summary_plot(shap_vals, X_val[:SHAP_SAMPLE_ROWS], show=False)
    plt.tight_layout()
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=f"_shap_{model_id}.png")
    plt.savefig(tmp_png.name)
    plt.close()
    key = f"visuals/shap/{os.path.basename(tmp_png.name)}"
    s3_upload(tmp_png.name, key)
    return key

def _actual_vs_pred_plot(y_true: np.ndarray, y_pred: np.ndarray, model_id: str) -> str:
    """
    Generate an Actual vs. Predicted scatter plot, save locally, upload to S3, return S3 key.
    """
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual pure_premium")
    plt.ylabel("Predicted pure_premium")
    plt.title(f"Actual vs. Predicted ({model_id})")
    plt.tight_layout()
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=f"_avspred_{model_id}.png")
    plt.savefig(tmp_png.name)
    plt.close()
    key = f"visuals/avs_pred/{os.path.basename(tmp_png.name)}"
    s3_upload(tmp_png.name, key)
    return key

def train_and_compare_fn(model_id: str, processed_path: str) -> None:
    """
    Main entry-point called from the DAG for each model_id.
    Logs metrics, uploads SHAP, actual vs. predicted plot, and auto-promotes if improved.
    """
    start = time.time()

    # Load and prepare
    df = pd.read_parquet(processed_path)
    if "pure_premium" not in df:
        df["pure_premium"] = df["il_total"] / df["eey"]
    y = df["pure_premium"].values
    X = df.drop(columns=["pure_premium"], errors="ignore")

    X_train, X_val, y_train, y_val = _train_val_split(X, y)

    # HyperOpt search space
    space = {
        "learning_rate":    hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
        "max_depth":        hp.choice("max_depth", [3,4,5,6,7,8,9,10]),
        "n_estimators":     hp.quniform("n_estimators", 50, 400, 1),
        "subsample":        hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        hp.loguniform("reg_alpha", -6, 1),
        "reg_lambda":       hp.loguniform("reg_lambda", -6, 1),
    }
    constraints = tuple(MONO_MAP.get(model_id, [0]*X.shape[1]))

    def objective(params):
        params["n_estimators"] = int(params["n_estimators"])
        model = xgb.XGBRegressor(
            tree_method="hist",
            monotone_constraints=constraints,
            objective="reg:squarederror",
            **params,
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return {"loss": rmse, "status": STATUS_OK}

    # Run HyperOpt
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials,
        show_progressbar=False
    )

    # Finalize best_params
    best_params = {k: (int(v) if k=="n_estimators" else v) for k,v in best.items()}
    best_params["max_depth"] = [3,4,5,6,7,8,9,10][best["max_depth"]]

    # Train final model
    final_model = xgb.XGBRegressor(
        tree_method="hist",
        monotone_constraints=constraints,
        objective="reg:squarederror",
        **best_params,
    )
    final_model.fit(X_train, y_train, verbose=False)
    preds = final_model.predict(X_val)

    # Compute all metrics
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, preds)
    r2  = r2_score(y_val, preds)

    # Generate & upload plots
    shap_key   = _shap_summary(final_model, X_val, model_id)
    avsp_key   = _actual_vs_pred_plot(y_val, preds, model_id)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{model_id}_run") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2",  r2)
        mlflow.xgboost.log_model(final_model, "model")
        mlflow.log_artifact(shap_key, artifact_path="shap")
        mlflow.log_artifact(avsp_key, artifact_path="avs_pred")

        run_id = run.info.run_id

    # Compare to Production
    prod_versions = client.get_latest_versions(name=model_id, stages=["Production"])
    prod_rmse = None
    if prod_versions:
        prod_rmse = client.get_run(prod_versions[0].run_id).data.metrics.get("rmse")

    if prod_rmse is None or rmse < prod_rmse:
        version = client.get_model_version_by_run_id(model_id, run_id).version
        client.transition_model_version_stage(
            name=model_id,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        status = "PROMOTED"
    else:
        status = "UNCHANGED"

    # Notify via Slack
    slack_msg(
        channel="#alerts",
        title=f"📊 {model_id} RMSE {rmse:.4f} | {status}",
        details=(
            f"MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}\n"
            f"SHAP: s3://{BUCKET}/{shap_key}\n"
            f"Actual vs Pred: s3://{BUCKET}/{avsp_key}"
        ),
        urgency="low",
    )

    LOGGER.info(
        "%s finished in %.1fs — RMSE %.4f, MSE %.4f, MAE %.4f, R² %.4f (%s)",
        model_id, time.time()-start, rmse, mse, mae, r2, status
    )

    # Integrate new UI components and endpoints
    try:
        handle_function_call({
            "function": {
                "name": "integrate_ui_components",
                "arguments": json.dumps({
                    "channel": "#agent_logs",
                    "title": "🔗 Integrating UI Components",
                    "details": "Integrating new UI components and endpoints.",
                    "urgency": "low"
                })
            }
        })
    except Exception as e:
        LOGGER.warning(f"UI components integration failed: {e}")

# backward‑compatibility alias
train_xgboost_hyperopt = train_and_compare_fn
