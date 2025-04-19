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
MAX_EVALS           = int(os.getenv("HYPEROPT_MAX_EVALS", 20))

# ─── UI COMPONENTS CONFIG ────────────────────────────────────────────────────
UI_COMPONENTS = {
    "sidebar": {
        "items": [
            "Home",
            "Drift Monitor",
            "Model Metrics",
            "Overrides",
            "Artifacts",
            "Data Quality",
            "Incidents",
            "Settings"
        ],
        "collapsible": True,
        "tooltips": True
    },
    "home_screen": {
        "top_bar": {
            "date_selector": True,
            "environment_toggle": ["Prod", "Dev"],
            "light_dark_mode_switch": True
        },
        "pipeline_health_card": {
            "last_run_timestamp": True,
            "status_icon": ["✅", "⚠️", "❌"],
            "runtime": True
        },
        "error_fix_chat_widget": {
            "enabled": True,
            "position": "bottom_right"
        },
        "quick_action_buttons": [
            "Trigger Retrain",
            "Run Self-Heal",
            "Generate Fix Proposal",
            "Open Code Console"
        ]
    },
    "drift_monitor": {
        "feature_drift_tiles": {
            "grid_format": True,
            "current_drift_percentages": True,
            "thresholds": True,
            "neon_accent_on_warning": True
        },
        "sparklines": True,
        "details_panel": {
            "full_line_chart": True,
            "context_info": True,
            "buttons": ["Self-Heal", "Propose Fix"]
        }
    },
    "model_metrics": {
        "model_selector": ["model1", "model2", "model3", "model4", "model5"],
        "metric_cards": ["RMSE", "MSE", "MAE", "R²"],
        "charts": [
            "RMSE over last N runs",
            "Actual vs. Predicted scatter",
            "SHAP summary"
        ],
        "download_icons": True
    },
    "overrides": {
        "pending_proposals_list": {
            "table_columns": ["problem_summary", "confidence", "timestamp"]
        },
        "proposal_detail_panel": {
            "code_snippets": True,
            "drift_context": True,
            "logs": True
        },
        "actions": ["Approve", "Reject", "Edit then Approve"]
    },
    "artifacts": {
        "s3_tree_view": "visuals/",
        "image_previews": ["SHAP", "AVS"],
        "actions": ["Copy S3 Path", "Download"]
    },
    "data_quality": {
        "schema_report": {
            "table_columns": ["expected", "actual", "type_mismatches"]
        },
        "null_duplicate_stats": True,
        "generate_fix_plan_button": True
    },
    "incidents": {
        "open_tickets_list": {
            "columns": ["status", "severity", "assignee"]
        },
        "create_ticket_form": {
            "fields": ["issue_summary", "severity_dropdown", "assignee_field"]
        },
        "sync_button": True
    },
    "settings": {
        "configuration_options": [
            "drift_thresholds",
            "HyperOpt_settings",
            "MLflow_experiment_name"
        ],
        "channel_mapping": {
            "alert_types": "Slack_channels"
        },
        "user_roles": ["admin", "viewer"]
    },
    "code_interpreter_console": {
        "floating_panel": True,
        "repl_editor": {
            "code_cell": True,
            "run_button": True
        },
        "file_browser": "/tmp",
        "output_panel": {
            "stdout": True,
            "charts": True
        },
        "history_list": True
    },
    "visual_style_ux": {
        "typography": {
            "headings": ["#", "##", "###"],
            "body_text": "sans-serif"
        },
        "colors": {
            "dark_mode": {
                "base": "#131313",
                "accent": "#4ECDC4"
            },
            "light_mode": {
                "base": "#FFFFFF",
                "accent": "#1A535C"
            }
        },
        "components": {
            "rounded_cards": "2xl",
            "soft_shadows": True,
            "spacing": "p-4"
        },
        "feedback": {
            "subtle_animations": True,
            "toast_notifications": True
        }
    },
    "integration_points": {
        "frontend": "React + Shadcn/UI",
        "charts": "Recharts",
        "api": "OpenAI Assistant endpoints",
        "realtime": "WebSockets"
    },
    "security_permissions": {
        "rbac": {
            "roles": ["admin", "viewer"]
        },
        "sandbox": {
            "restricted_directory": "/tmp"
        },
        "audit_trail": True
    }
}
