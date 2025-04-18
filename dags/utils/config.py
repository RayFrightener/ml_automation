import os
from airflow.models import Variable

# ─── S3 CONFIG ────────────────────────────────────────────────────────────────
# primary bucket
S3_BUCKET       = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET")
# raw data, reference means, models, logs, archive keys
RAW_DATA_KEY       = "raw-data/ut_loss_history_1.csv"
REFERENCE_KEY      = "reference/reference_means.csv"
REFERENCE_KEY_PREFIX = "reference"
MODELS_FOLDER      = "models"
LOGS_FOLDER        = "logs"
ARCHIVE_FOLDER     = os.getenv("S3_ARCHIVE_FOLDER", "archive")

# ─── MLFLOW CONFIG ────────────────────────────────────────────────────────────
MLFLOW_URI        = os.getenv("MLFLOW_TRACKING_URI") \
                    or Variable.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = Variable.get(
    "MLFLOW_EXPERIMENT_NAME",
    default_var="Homeowner_Loss_Hist_Proj"
)

# ─── AIRFLOW / DAG CONFIG ────────────────────────────────────────────────────
# NOTE: these may also be configured via Variables if you prefer
DEFAULT_START_DATE = "2025-01-01"
SCHEDULE_CRON       = "0 10 * * *"

# ─── SLACK CONFIG ────────────────────────────────────────────────────────────
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL") \
                    or Variable.get("SLACK_WEBHOOK_URL", default_var="")

# ─── HYPEROPT CONFIG ─────────────────────────────────────────────────────────
MAX_EVALS = int(os.getenv("HYPEROPT_MAX_EVALS", 20))
