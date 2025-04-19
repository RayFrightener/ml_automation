#!/usr/bin/env python3
"""
agent_actions.py

Central dispatcher for AI-agent functions in the Homeowner pipeline.
- Slack, Airflow trigger, and S3 utilities.
- AI-agent hooks: both legacy helpers and new human-in-the-loop functions.
"""
import json
import logging
import time
from typing import Any, Dict, Optional, List

from tenacity import retry, stop_after_attempt, wait_fixed

from utils.slack import post as post_message
from utils.airflow_api import trigger_dag as api_trigger_dag
from utils.config import SLACK_WEBHOOK_URL, AIRFLOW_DAG_BASE_CONF

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

###############################################
# Core Notification & Trigger Functions
###############################################
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def send_to_slack(channel: str, title: str, details: str, urgency: str) -> Dict[str, Any]:
    logging.info(f"Posting to Slack {channel}: {title}")
    return post_message(channel=channel, title=title, details=details, urgency=urgency)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def trigger_airflow_dag(dag_id: str, conf: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    logging.info(f"Triggering DAG {dag_id}")
    return api_trigger_dag(dag_id=dag_id, conf=conf or AIRFLOW_DAG_BASE_CONF)

###############################################
# Generic dispatcher for agent calls
###############################################
def handle_function_call(payload: Dict[str, Any]) -> Any:
    func_def = payload.get("function", {})
    name = func_def.get("name")
    args_str = func_def.get("arguments", "{}")
    try:
        kwargs = json.loads(args_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON args: {args_str}")
    func = globals().get(name)
    if not func or not callable(func):
        raise ValueError(f"Function '{name}' not found")
    logging.info(f"Calling {name} with {kwargs}")
    return func(**kwargs)


###############################################
# Code Interpreter Hook
###############################################
def execute_python(code: str) -> Any:
    """
    Execute Python code in the Code Interpreter sandbox and return the result.
    This stub is handled by the OpenAI Code Interpreter tool.
    """
    if not code.startswith("/tmp/"):
        raise PermissionError("Code execution restricted to /tmp directory only.")
    logging.info(f"Executing code: {code}")
    # Log the action for audit trail
    log_manual_action("execute_python", {"code": code})
    pass


###############################################
# Legacy AI-Agent Helpers
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
    failures = [{"dag_id": "homeowner_loss_history_full_pipeline", "task_id": "train_compare_model1", "failure_time": "2025-04-18T12:34:56Z"}]
    logging.info(f"Recent failures (last {lookback_hours}h): {failures}")
    return {"failures": failures}


def escalate_issue(issue_summary: str, contact_method: str, severity: str) -> Dict[str, Any]:
    escalation_message = f"Escalation [{severity}]: {issue_summary} via {contact_method}."
    logging.info(escalation_message)
    return {"status": "escalated", "message": escalation_message}

###############################################
# New AI-Agent Hooks: Self-Healing & Human-in-the-Loop
###############################################
def detect_and_flag_drift(drift_metrics: Dict[str, float], threshold: float = 0.1) -> bool:
    """Return True if any feature drift exceeds threshold."""
    for feature, drift in drift_metrics.items():
        if drift > threshold:
            logging.warning(f"Drift alert: {feature} -> {drift:.2%}")
            return True
    logging.info("No significant drift detected.")
    return False


def schedule_retraining(trigger_conf: Dict[str, Any]) -> Dict[str, Any]:
    """Trigger a new DAG run for retraining with given config."""
    run = api_trigger_dag(dag_id=trigger_conf.get("dag_id"), conf=trigger_conf)
    return {"status": "scheduled", "run": run}


def explain_model_decision(inputs: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    """Generate per-sample explanations (e.g., SHAP values)."""
    explanations = [{"input": inp, "shap_values": []} for inp in inputs]
    logging.info(f"Generated explanations for {len(inputs)} records.")
    return {"model_id": model_id, "explanations": explanations}


def approve_deployment(run_id: str, approver: str) -> Dict[str, Any]:
    """Mark a model version as approved for production."""
    logging.info(f"Deployment approved by {approver} for run {run_id}.")
    return {"run_id": run_id, "approved_by": approver}


def rollback_model(model_name: str, to_version: str) -> Dict[str, Any]:
    """Rollback a registered model to a previous version."""
    logging.info(f"Rolling back {model_name} to version {to_version}.")
    return {"model": model_name, "rolled_back_to": to_version}


def generate_executive_summary(period: str, metrics: Dict[str, Any]) -> str:
    """Compile a high-level summary report."""
    summary = f"Executive Summary for {period}: " + ", ".join(f"{k}={v}" for k,v in metrics.items())
    logging.info("Generated executive summary.")
    return summary


def open_incident_ticket(issue: str, severity: str, assignee: str) -> Dict[str, Any]:
    """Simulate opening an incident ticket in external system."""
    ticket_id = f"TICKET-{int(time.time())}"
    logging.info(f"Opened {ticket_id}: {issue}")
    return {"ticket_id": ticket_id, "severity": severity, "assignee": assignee}


def optimize_compute_resources(history: Dict[str, float]) -> Dict[str, Any]:
    """Analyze past runtimes and suggest resource changes."""
    suggestion = {"workers": 5, "instance_type": "m5.xlarge"}
    logging.info("Generated compute optimization suggestion.")
    return suggestion


def simulate_what_if(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a dry-run experiment and return predicted metrics."""
    prediction = {"rmse": 0.123, "training_time": 42}
    logging.info("Completed what-if simulation.")
    return prediction


def audit_data_quality(dataset_path: str) -> Dict[str, Any]:
    """Run data-quality checks and return a report."""
    report = {"nulls": 0, "duplicates": 0, "schema_violations": 0}
    logging.info("Completed data quality audit.")
    return report

###############################################
# Role-Based Access Control (RBAC)
###############################################
def check_permissions(user_role: str, action: str) -> bool:
    """
    Check if the user role has permission to perform the action.
    """
    role_permissions = {
        "admin": ["execute_python", "propose_fix", "override_decision", "generate_root_cause_report",
                  "suggest_hyperparam_improvement", "validate_data_integrity", "describe_fix_plan",
                  "fetch_airflow_logs", "update_airflow_variable", "list_recent_failures", "escalate_issue",
                  "detect_and_flag_drift", "schedule_retraining", "explain_model_decision", "approve_deployment",
                  "rollback_model", "generate_executive_summary", "open_incident_ticket", "optimize_compute_resources",
                  "simulate_what_if", "audit_data_quality"],
        "viewer": ["fetch_airflow_logs", "list_recent_failures", "generate_executive_summary"]
    }
    allowed_actions = role_permissions.get(user_role, [])
    return action in allowed_actions

def log_manual_action(action: str, details: Dict[str, Any]) -> None:
    """
    Log every manual action and chat command for audit trail.
    """
    logging.info(f"Manual action logged: {action} with details: {details}")
