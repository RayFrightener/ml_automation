#!/usr/bin/env python3
"""
tasks/training.py  – Unified trainer + explainability
----------------------------------------------------------
 • Baseline check: load last Production model and skip retrain if RMSE ≤ threshold
 • HyperOpt → finds best hyper‑parameters per model (configurable max‑evals)
 • Fallback from TimeSeriesSplit → train_test_split if not enough chronological rows
 • Logs to MLflow: RMSE, MSE, MAE, R² + SHAP summary + Actual vs. Predicted plot
 • Archives SHAP plots to s3://<BUCKET>/visuals/shap/
 • Archives AV‑Pred plots to s3://<BUCKET>/visuals/avs_pred/
 • Auto‑register & promote in MLflow Model Registry
 • Sends Slack notifications on skip/retrain and completion
 • Supports manual_override from Airflow Variables
"""

import os
import json
import logging
import tempfile
import time
import joblib
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
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
from utils.storage import upload as s3_upload, download as s3_download

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def manual_override() -> Optional[Dict[str, Any]]:
    """
    If the Airflow Variable MANUAL_OVERRIDE == "True",
    return custom JSON hyperparameters; otherwise None.
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
BUCKET               = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET")
MLFLOW_URI           = os.getenv("MLFLOW_TRACKING_URI") or Variable.get("MLFLOW_TRACKING_URI")
EXPERIMENT           = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="Homeowner_Loss_Hist_Proj")
MAX_EVALS            = int(os.getenv("HYPEROPT_MAX_EVALS", Variable.get("HYPEROPT_MAX_EVALS", 20)))
SHAP_SAMPLE_ROWS     = int(os.getenv("SHAP_SAMPLE_ROWS", 200))
DRIFT_RMSE_THRESHOLD = float(os.getenv("DRIFT_RMSE_THRESHOLD", Variable.get("DRIFT_RMSE_THRESHOLD", default_var="0.1")))

# Monotone constraints map
try:
    MONO_MAP: Dict[str, list] = Variable.get("MONOTONIC_CONSTRAINTS_MAP", deserialize_json=True)
except Exception:
    MONO_MAP = {}

# MLflow client
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()


def _train_val_split(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Try TimeSeriesSplit; fallback to train_test_split."""
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, val_idx = next(tscv.split(X))
        LOGGER.info("Using TimeSeriesSplit (n_splits=5)")
        return X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
    except Exception as e:
        LOGGER.warning("TimeSeriesSplit failed; fallback train_test_split: %s", e)
        return train_test_split(X, y, test_size=0.2, random_state=42)


def _shap_summary(model: xgb.XGBRegressor, X_val: pd.DataFrame, model_id: str) -> str:
    """Generate SHAP summary plot, upload to S3, return S3 key."""
    explainer = shap.Explainer(model)
    shap_vals = explainer(X_val[:SHAP_SAMPLE_ROWS])
    shap.summary_plot(shap_vals, X_val[:SHAP_SAMPLE_ROWS], show=False)
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_shap_{model_id}.png")
    plt.savefig(tmp.name)
    plt.close()
    key = f"visuals/shap/{os.path.basename(tmp.name)}"
    s3_upload(tmp.name, key)
    return key


def _actual_vs_pred_plot(y_true: np.ndarray, y_pred: np.ndarray, model_id: str) -> str:
    """Generate Actual vs Predicted scatter, upload to S3, return key."""
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual pure_premium")
    plt.ylabel("Predicted pure_premium")
    plt.title(f"Actual vs. Predicted ({model_id})")
    plt.tight_layout()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_avspred_{model_id}.png")
    plt.savefig(tmp.name)
    plt.close()
    key = f"visuals/avs_pred/{os.path.basename(tmp.name)}"
    s3_upload(tmp.name, key)
    return key


def train_and_compare_fn(model_id: str, processed_path: str) -> None:
    """Train pipeline with baseline skip, HyperOpt, MLflow logging, and notifications."""
    start = time.time()

    # Load and prepare data
    df = pd.read_parquet(processed_path)
    if "pure_premium" not in df:
        df["pure_premium"] = df["il_total"] / df["eey"]
    y = df["pure_premium"].values
    X = df.drop(columns=["pure_premium"], errors="ignore")
    X_train, X_val, y_train, y_val = _train_val_split(X, y)

    # Baseline model check
    try:
        prev_file = f"/tmp/{model_id}_prev.joblib"
        s3_download(f"models/{model_id}.joblib", prev_file)
        prev_model = joblib.load(prev_file)
        preds_prev = prev_model.predict(X_val)
        prev_rmse = np.sqrt(mean_squared_error(y_val, preds_prev))
        LOGGER.info(f"Baseline {model_id} RMSE={prev_rmse:.4f}")
        if prev_rmse <= DRIFT_RMSE_THRESHOLD:
            slack_msg(
                channel="#alerts",
                title=f"⏭️ {model_id} retrain skipped",
                details=f"Baseline RMSE {prev_rmse:.4f} ≤ threshold {DRIFT_RMSE_THRESHOLD}",
                urgency="low",
            )
            return
    except Exception as e:
        LOGGER.info(f"Baseline load skipped/error: {e}")

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
        if manual_override():
            params.update(manual_override())
        model = xgb.XGBRegressor(
            tree_method="hist",
            monotone_constraints=constraints,
            objective="reg:squarederror",
            **params,
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_val)
        return {"loss": np.sqrt(mean_squared_error(y_val, preds)), "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials,
        show_progressbar=False
    )

    # Final train
    best_params = {k: (int(v) if k == "n_estimators" else v) for k, v in best.items()}
    best_params["max_depth"] = [3,4,5,6,7,8,9,10][best["max_depth"]]
    final_model = xgb.XGBRegressor(
        tree_method="hist",
        monotone_constraints=constraints,
        objective="reg:squarederror",
        **best_params,
    )
    final_model.fit(X_train, y_train, verbose=False)
    preds = final_model.predict(X_val)

    # Metrics
    mse = mean_squared_error(y_val, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, preds)
    r2  = r2_score(y_val, preds)

    # Plots
    shap_key = _shap_summary(final_model, X_val, model_id)
    avsp_key = _actual_vs_pred_plot(y_val, preds, model_id)

    # MLflow logging
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

    # Auto‑register & create version
    try:
        client.create_registered_model(model_id)
    except RestException as e:
        if "RESOURCE_ALREADY_EXISTS" not in str(e):
            raise
    client.create_model_version(name=model_id, run_id=run_id, artifact_path="model")

    # Promote if improved
    try:
        prod = client.get_latest_versions(name=model_id, stages=["Production"])
    except RestException:
        prod = []
    prod_rmse = None
    if prod:
        prod_rmse = client.get_run(prod[0].run_id).data.metrics.get("rmse")
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

    # Slack notification
    slack_msg(
        channel="#alerts",
        title=f"📊 {model_id} RMSE {rmse:.4f} | {status}",
        details=(
            f"MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}\n"
            f"SHAP:  s3://{BUCKET}/{shap_key}\n"
            f"AVSP:  s3://{BUCKET}/{avsp_key}"
        ),
        urgency="low",
    )

    LOGGER.info(
        "%s finished in %.1fs — RMSE %.4f, MSE %.4f, MAE %.4f, R² %.4f (%s)",
        model_id, time.time() - start, rmse, mse, mae, r2, status
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
