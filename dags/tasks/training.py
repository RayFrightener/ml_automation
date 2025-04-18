#!/usr/bin/env python3
"""
tasks/training.py

Train and log XGBoost models with hyperparameter tuning.

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

from airflow.models import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from ..utils.storage import upload  # wrapper for S3 uploads
from ..utils.slack import send_message  # wrapper for Slack
from ..utils.airflow_api import trigger_dag  # wrapper for Airflow API

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# configuration
MLFLOW_TRACKING_URI    = os.getenv(
    "MLFLOW_TRACKING_URI",
    Variable.get("MLFLOW_TRACKING_URI", default_var="http://3.146.46.179:5000")
)
MLFLOW_EXPERIMENT_NAME = Variable.get("MLFLOW_EXPERIMENT_NAME", default_var="Homeowner_Loss_Hist_Proj")
S3_BUCKET              = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET")
S3_MODELS_FOLDER       = "models"
MONO_MAP               = Variable.get("MONOTONIC_CONSTRAINTS_MAP", deserialize_json=True)


def manual_override() -> Optional[Dict]:
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
) -> float:
    # select model_id
    model_id = (
        model_id
        or os.getenv("MODEL_ID", Variable.get("MODEL_ID", default_var="model1"))
    ).strip().lower()
    logging.info(f"Training for MODEL_ID='{model_id}'")

    # MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # load data
    df = pd.read_csv(processed_path)
    if "pure_premium" not in df:
        df["pure_premium"]  = df["il_total"] / df["eey"]
        df["sample_weight"] = df["eey"]
        logging.info("Computed pure_premium & sample_weight")

    y = df["pure_premium"]
    w = df["sample_weight"]
    X = df.drop(columns=["pure_premium","sample_weight"])

    # monotonic constraints
    constraints = MONO_MAP.get(model_id)
    if not isinstance(constraints, list):
        raise ValueError(f"Constraints for '{model_id}' must be list.")
    if len(constraints) != X.shape[1]:
        logging.warning("Constraint length mismatch; resetting to zeros.")
        constraints = [0]*X.shape[1]
        MONO_MAP[model_id] = constraints
        Variable.set("MONOTONIC_CONSTRAINTS_MAP", json.dumps(MONO_MAP))
    constraints = tuple(constraints)

    # split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.3, random_state=42
    )

    # hyperopt space
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
            eval_set=[(X_test,y_test)],
            sample_weight_eval_set=[w_test],
            early_stopping_rounds=20,
            verbose=False
        )
        preds = model.predict(X_test)
        rmse  = mean_squared_error(y_test, preds, sample_weight=w_test)**0.5
        return {"loss": rmse, "status": STATUS_OK}

    # choose best params
    if override_params:
        best = override_params
        logging.info("Using override hyperparameters.")
    else:
        trials = Trials()
        best   = fmin(objective, space, algo=tpe.suggest, max_evals=20, trials=trials)
        best["n_estimators"] = int(best["n_estimators"])
        best["max_depth"]    = [3,5,7,9,11][best["max_depth"]]
        logging.info(f"Tuned params for '{model_id}': {best}")

    # final train on GPU
    final_model = xgb.XGBRegressor(
        tree_method="gpu_hist",
        monotone_constraints=constraints,
        use_label_encoder=False,
        eval_metric="rmse",
        **best
    )
    final_model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test,y_test)],
        sample_weight_eval_set=[w_test],
        early_stopping_rounds=20,
        verbose=False
    )

    preds     = final_model.predict(X_test)
    final_rmse= mean_squared_error(y_test, preds, sample_weight=w_test)**0.5
    final_r2  = r2_score(y_test, preds, sample_weight=w_test)

    # classification metrics
    thr       = y_test.median()
    cls_true  = (y_test>thr).astype(int)
    cls_pred  = (preds>thr).astype(int)
    final_acc = accuracy_score(cls_true, cls_pred)
    final_f1  = f1_score(cls_true, cls_pred)

    # decile chart
    df_eval       = pd.DataFrame({"actual":y_test, "pred":preds})
    df_eval["decile"] = pd.qcut(df_eval["pred"],10,labels=False)
    decile_means  = df_eval.groupby("decile")[["actual","pred"]].mean()
    fig1,ax1      = plt.subplots()
    ax1.plot(decile_means.index+1, decile_means["actual"], marker="o", label="Actual")
    ax1.plot(decile_means.index+1, decile_means["pred"],   marker="o", label="Predicted")
    ax1.set_title("Actual vs Predicted by Decile")
    ax1.set_xlabel("Decile"); ax1.set_ylabel("Pure Premium"); ax1.legend()
    mlflow.log_figure(fig1, "actual_vs_pred_by_decile.png")
    plt.close(fig1)

    # feature importance
    booster   = final_model.get_booster()
    imp_dict  = booster.get_score(importance_type="gain")
    imp_df    = pd.DataFrame(
        {"feature":list(imp_dict), "importance":list(imp_dict.values())}
    ).sort_values("importance")
    fig2,ax2  = plt.subplots(figsize=(8, max(4,len(imp_df)*0.3)))
    ax2.barh(imp_df["feature"], imp_df["importance"])
    ax2.set_title("Feature Importance (gain)")
    ax2.set_xlabel("Gain")
    mlflow.log_figure(fig2, "feature_importance.png")
    plt.close(fig2)

    # MLflow logging
    with mlflow.start_run(run_name=f"xgb_{model_id}_hyperopt"):
        mlflow.log_params(best)
        mlflow.log_metrics({
            "rmse": final_rmse,
            "r2":   final_r2,
            "acc":  final_acc,
            "f1":   final_f1
        })
        mlflow.xgboost.log_model(final_model, artifact_path="model")

    # Slack notification
    send_message(
        channel="#agent_logs",
        title=f"Training Summary ({model_id})",
        body=(
            f"RMSE {final_rmse:.2f}, R2 {final_r2:.2f}, "
            f"Acc {final_acc:.2f}, F1 {final_f1:.2f}"
        ),
        urgency="low"
    )

    # save & upload model
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname= f"xgb_{model_id}_model_{ts}.json"
    path = f"/tmp/{fname}"
    final_model.save_model(path)
    upload(path, f"{S3_MODELS_FOLDER}/{fname}")  # uses utils/storage.upload

    logging.info(f"Uploaded model to s3://{S3_BUCKET}/{S3_MODELS_FOLDER}/{fname}")
    return final_rmse
