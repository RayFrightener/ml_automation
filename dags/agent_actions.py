#!/usr/bin/env python3
"""
agent_actions.py

Refactored to:
  - Use utils/slack.post_message instead of direct requests.
  - Use utils/airflow_api.trigger_dag for Airflow API calls.
  - Wrap external calls with tenacity retries.
  - Load magic strings from config.py.
  - Provide a generic function-calling dispatcher.
"""

import json
import logging
from typing import Any, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from utils.slack import post as post_message
from utils.airflow_api import trigger_dag as api_trigger_dag
from utils.config import (
    SLACK_WEBHOOK_URL,  # Using this as a fallback
    AIRFLOW_DAG_BASE_CONF,
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

###############################################
# Slack notification function
###############################################
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def send_to_slack(channel: str, title: str, details: str, urgency: str) -> Dict[str, Any]:
    logging.info(f"Posting to Slack channel {channel}: {title}")
    return post_message(channel=channel, title=title, details=details, urgency=urgency)

###############################################
# Airflow DAG trigger function
###############################################
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def trigger_airflow_dag(dag_id: str, conf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    logging.info(f"Triggering DAG {dag_id} with conf={conf}")
    return api_trigger_dag(dag_id=dag_id, conf=conf or AIRFLOW_DAG_BASE_CONF)

###############################################
# Generic function dispatcher
###############################################
def handle_function_call(payload: Dict[str, Any]) -> Any:
    """
    Generic dispatcher for calling any function defined in this module (or imported here).
    Expects payload like:
      { "function": { "name": "<func_name>", "arguments": "<json-string-of-kwargs>" } }
    """
    func_def = payload.get("function", {})
    name = func_def.get("name")
    args_str = func_def.get("arguments", "{}")
    try:
        kwargs = json.loads(args_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in arguments: {args_str!r}")
    func = globals().get(name)
    if not func or not callable(func):
        raise ValueError(f"Function '{name}' not found")
    logging.info(f"Calling function '{name}' with arguments {kwargs}")
    return func(**kwargs)

###############################################
# Additional Functions for Self‑Healing, Reporting, etc.
###############################################
def propose_fix(
    problem_summary: str,
    proposed_fix: str,
    confidence: float,
    requires_human_approval: bool
) -> Dict[str, Any]:
    result = {
        "problem_summary": problem_summary,
        "proposed_fix": proposed_fix,
        "confidence": confidence,
        "requires_human_approval": requires_human_approval
    }
    logging.info(f"Propose Fix: {result}")
    return result

def override_decision(fix_id: str, approved_by: str, comment: str = "") -> Dict[str, Any]:
    result = {
        "fix_id": fix_id,
        "approved_by": approved_by,
        "comment": comment,
        "status": "approved"
    }
    logging.info(f"Override Decision: {result}")
    return result

def generate_root_cause_report(dag_id: str, execution_date: str) -> Dict[str, Any]:
    report = {
        "dag_id": dag_id,
        "execution_date": execution_date,
        "root_cause": "Data drift in feature 'example_feature' and increased RMSE.",
        "details": "Detected a 15% drift in 'example_feature' and RMSE increased by 0.5 units."
    }
    logging.info(f"Root Cause Report: {report}")
    return report

def suggest_hyperparam_improvement(
    model_id: str,
    current_rmse: float,
    previous_best_rmse: float
) -> Dict[str, Any]:
    suggestion = {
        "model_id": model_id,
        "suggestion": "Increase 'n_estimators' and decrease 'learning_rate'.",
        "current_rmse": current_rmse,
        "previous_best_rmse": previous_best_rmse,
        "confidence": 0.8
    }
    logging.info(f"Hyperparameter Improvement Suggestion: {suggestion}")
    return suggestion

def validate_data_integrity(dataset_path: str) -> Dict[str, Any]:
    import pandas as pd
    report = {"dataset_path": dataset_path, "valid": True, "issues": None, "row_count": None}
    try:
        df = pd.read_csv(dataset_path)
        report["row_count"] = len(df)
    except Exception as e:
        report["valid"] = False
        report["issues"] = str(e)
    logging.info(f"Data Integrity Report: {report}")
    return report

def describe_fix_plan(issue_type: str, solution_summary: str) -> Dict[str, Any]:
    plan = {
        "issue_type": issue_type,
        "solution_summary": solution_summary,
        "expected_outcome": "Improved model performance and reduced drift."
    }
    logging.info(f"Fix Plan Description: {plan}")
    return plan

def fetch_airflow_logs(dag_id: str, run_id: str) -> Dict[str, Any]:
    logs = f"Simulated logs for DAG {dag_id}, run {run_id}."
    logging.info(logs)
    return {"logs": logs}

def update_airflow_variable(key: str, value: str) -> Dict[str, Any]:
    logging.info(f"Updating Airflow Variable {key} to {value}")
    return {"status": "success", "variable": key, "new_value": value}

def list_recent_failures(lookback_hours: int) -> Dict[str, Any]:
    failures = [{"dag_id": "homeowner_dag", "task_id": "train_xgboost_hyperopt", "failure_time": "2025-04-14T05:00:00Z"}]
    logging.info(f"Recent failures (last {lookback_hours}h): {failures}")
    return {"failures": failures}

def escalate_issue(issue_summary: str, contact_method: str, severity: str) -> Dict[str, Any]:
    escalation_message = f"Escalation [{severity}]: {issue_summary} via {contact_method}."
    logging.info(escalation_message)
    return {"status": "escalated", "message": escalation_message}
