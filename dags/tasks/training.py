#!/usr/bin/env python3
"""
 tasks/training.py  ‑ **Unified trainer + explainability**
 ----------------------------------------------------------
 • HyperOpt → finds best hyper‑parameters *per model* (configurable max‑evals)
 • Automatic fallback from **TimeSeriesSplit → train_test_split** if there are
   not enough chronological rows.
 • Logs to **MLflow** *and* pushes RMSE + SHAP summary to Slack.
 • Archives every SHAP plot to **s3://<BUCKET>/visuals/shap/** so dashboards
   can pick them up.
 • If the new RMSE beats the Production model, the model version is promoted
   automatically.
 """

import os, json, logging, tempfile, time
from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split

import mlflow
import mlflow.xgboost
from airflow.models import Variable

from utils.slack   import post  as slack_msg   # wrapper (retries + webhook)
from utils.storage import upload as s3_upload   # wrapper (retries + boto3)

# ──────────────────────────────────────────────────────────────────────────────
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ─── ENV / AIRFLOW VARIABLES ─────────────────────────────────────────────────
BUCKET              = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET")
MLFLOW_URI          = os.getenv("MLFLOW_TRACKING_URI") or Variable.get("MLFLOW_TRACKING_URI")
EXPERIMENT          = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="Homeowner_Loss_Hist_Proj")
MAX_EVALS           = int(os.getenv("HYPEROPT_MAX_EVALS", Variable.get("HYPEROPT_MAX_EVALS", 20)))
SHAP_SAMPLE_ROWS    = int(os.getenv("SHAP_SAMPLE_ROWS", 200))

# Monotone map (Airflow Variable as JSON)
try:
    MONO_MAP: Dict[str, list] = Variable.get("MONOTONIC_CONSTRAINTS_MAP", deserialize_json=True)
except Exception:
    MONO_MAP = {}

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()

# ─── UTILITIES ────────────────────────────────────────────────────────────────
def manual_override() -> Optional[Dict]:
    """
    If the Airflow Variable MANUAL_OVERRIDE == "True", return the JSON contained
    in CUSTOM_HYPERPARAMS; otherwise return None.  This lets you override the
    Hyperopt parameters from the UI.
    """
    try:
        if Variable.get("MANUAL_OVERRIDE", default_var="False").lower() == "true":
            params = json.loads(Variable.get("CUSTOM_HYPERPARAMS", default_var="{}"))
            logging.info("Manual override enabled – using custom hyperparameters.")
            return params
    except Exception as e:
        logging.error("manual_override(): %s", e)
    return None


def _train_val_split(X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray]:
    """Try TimeSeriesSplit, else fallback to random split."""
    try:
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, val_idx = next(tscv.split(X))  # first split only
        LOGGER.info("Using TimeSeriesSplit (n_splits=5).")
        return (
            X.iloc[train_idx], X.iloc[val_idx],
            y.iloc[train_idx], y.iloc[val_idx]
        )
    except Exception as e:
        LOGGER.warning(f"TimeSeriesSplit failed → fallback to train_test_split. Reason: {e}")
        return train_test_split(X, y, test_size=0.2, random_state=42)


def _shap_summary(model: xgb.XGBRegressor, X_val: pd.DataFrame, model_id: str) -> str:
    explainer      = shap.Explainer(model)
    shap_values    = explainer(X_val[:SHAP_SAMPLE_ROWS])
    shap.summary_plot(shap_values, X_val[:SHAP_SAMPLE_ROWS], show=False)
    plt.tight_layout()
    tmp_png = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{model_id}.png")
    plt.savefig(tmp_png.name)
    plt.close()
    key = f"visuals/shap/{os.path.basename(tmp_png.name)}"
    s3_upload(tmp_png.name, key)
    return key  # e.g. visuals/shap/shap_summary_model1_xxx.png


# ─── MAIN PUBLIC FUNCTION ----------------------------------------------------

def train_and_compare_fn(model_id: str, processed_path: str) -> None:
    """Entry‑point called from the DAG for each model‑ID."""
    start = time.time()

    # ── Load & prepare data ────────────────────────────────────────────────
    df = pd.read_parquet(processed_path)
    if "pure_premium" not in df:
        df["pure_premium"]  = df["il_total"] / df["eey"]
    y = df["pure_premium"]
    X = df.drop(columns=["pure_premium"], errors="ignore")

    X_train, X_val, y_train, y_val = _train_val_split(X, y)

    # ── HyperOpt search space ──────────────────────────────────────────────
    space = {
        "learning_rate":    hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
        "max_depth":        hp.choice("max_depth",  [3,4,5,6,7,8,9,10]),
        "n_estimators":     hp.quniform("n_estimators", 50, 400, 1),
        "subsample":        hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        hp.loguniform("reg_alpha",   -6, 1),
        "reg_lambda":       hp.loguniform("reg_lambda",  -6, 1),
    }

    constraints = tuple(MONO_MAP.get(model_id, [0] * X.shape[1]))

    def objective(params):
        params["n_estimators"] = int(params["n_estimators"])
        model = xgb.XGBRegressor(
            tree_method="hist",
            monotone_constraints=constraints,
            objective="reg:squarederror",
            **params,
        )
        model.fit(X_train, y_train, verbose=False)
        pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, pred, squared=False)
        return {"loss": rmse, "status": STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials, show_progressbar=False)
    best_params = {
        **{k: (int(v) if k == "n_estimators" else v) for k, v in best.items()},
        "max_depth": [3,4,5,6,7,8,9,10][best["max_depth"]],
    }

    # ── Train final model ──────────────────────────────────────────────────
    final_model = xgb.XGBRegressor(
        tree_method="hist",
        monotone_constraints=constraints,
        objective="reg:squarederror",
        **best_params,
    )
    final_model.fit(X_train, y_train, verbose=False)
    pred  = final_model.predict(X_val)
    rmse  = mean_squared_error(y_val, pred, squared=False)

    shap_key = _shap_summary(final_model, X_val, model_id)

    with mlflow.start_run(run_name=f"{model_id}_run") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(final_model, "model")
        mlflow.log_artifact(f"/tmp/{os.path.basename(shap_key)}", artifact_path="shap")
        run_id = run.info.run_id

    # ── Compare with production ───────────────────────────────────────────
    prod = client.get_latest_versions(name=model_id, stages=["Production"])
    prod_rmse = None
    if prod:
        prod_rmse = client.get_run(prod[0].run_id).data.metrics.get("rmse")

    improved = (prod_rmse is None) or (rmse < prod_rmse)
    if improved:
        client.transition_model_version_stage(
            name=model_id,
            version=client.get_model_version_by_run_id(model_id, run_id).version,
            stage="Production",
            archive_existing_versions=True,
        )
        status = "PROMOTED"
    else:
        status = "—"

    # ── Notify Slack ───────────────────────────────────────────────────────
    slack_msg(
        channel="#alerts",
        title=f"📊 {model_id} RMSE {rmse:.4f} | {status}",
        details=f"SHAP summary saved to s3://{BUCKET}/{shap_key}",
        urgency="low",
    )

    LOGGER.info("Model %s finished in %.1fs (RMSE %.4f) — %s", model_id, time.time()-start, rmse, status)

# Backward compatibility export
train_xgboost_hyperopt = train_and_compare_fn
