# utils/config.py

from dotenv import load_dotenv
load_dotenv()

import os
from airflow.models import Variable

# ─── S3 CONFIG ────────────────────────────────────────────────────────────────
S3_BUCKET           = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET")
RAW_DATA_KEY        = "raw-data/ut_loss_history_1.csv"
REFERENCE_KEY       = "reference/reference_means.csv"
REFERENCE_KEY_PREFIX= "reference"
MODELS_FOLDER       = "models"
LOGS_FOLDER         = "logs"
ARCHIVE_FOLDER      = os.getenv("S3_ARCHIVE_FOLDER", "archive")

# ─── MLFLOW CONFIG ────────────────────────────────────────────────────────────
MLFLOW_URI          = (
    os.getenv("MLFLOW_TRACKING_URI")
    or Variable.get("MLFLOW_TRACKING_URI")
)
MLFLOW_EXPERIMENT   = Variable.get(
    "MLFLOW_EXPERIMENT_NAME",
    default_var="Homeowner_Loss_Hist_Proj"
)

# ─── AIRFLOW / DAG CONFIG ────────────────────────────────────────────────────
DEFAULT_START_DATE   = "2025-01-01"
SCHEDULE_CRON        = "0 10 * * *"  # daily at 10 AM
AIRFLOW_DAG_BASE_CONF= {}

# ─── SLACK CONFIG ────────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL    = (
    os.getenv("SLACK_WEBHOOK_URL")
    or Variable.get("SLACK_WEBHOOK_URL", default_var=None)
    or Variable.get("SLACK_WEBHOOK", default_var=None)
)
SLACK_CHANNEL_DEFAULT= (
    os.getenv("SLACK_CHANNEL_DEFAULT")
    or Variable.get("SLACK_CHANNEL_DEFAULT", default_var="#alerts")
)

# ─── HYPEROPT CONFIG ─────────────────────────────────────────────────────────
MAX_EVALS            = int(
    os.getenv("HYPEROPT_MAX_EVALS")
    or Variable.get("HYPEROPT_MAX_EVALS", default_var="20")
)

# ─── THRESHOLDS ────────────────────────────────────────────────────────────────
# Fractional drift threshold for detect_data_drift (default 0.1)
DRIFT_THRESHOLD     = float(
    os.getenv("DRIFT_THRESHOLD")
    or Variable.get("DRIFT_THRESHOLD", default_var="0.1")
)
# RMSE threshold to skip retrain when baseline model is still performant
DRIFT_RMSE_THRESHOLD = float(
    os.getenv("DRIFT_RMSE_THRESHOLD")
    or Variable.get("DRIFT_RMSE_THRESHOLD", default_var="0.1")
)
