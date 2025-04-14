#!/usr/bin/env python3
"""
agent_actions.py

This module defines helper functions for an AI assistant that:
  - Sends Slack notifications.
  - Triggers Airflow DAG runs.
  - Proposes fixes with explanation.
  - Logs manual override decisions.
  - Generates root cause reports.
  - Suggests hyperparameter improvements.
  - Validates data integrity.
  - Describes a fix plan.

Ensure the following environment variables are set, either via a .env file or Airflow Variables:
  - SLACK_WEBHOOK_URL: The Slack incoming webhook URL.
  - AIRFLOW_API_URL: Base URL of your Airflow instance (e.g., "http://your-airflow-host:8080").
  - AIRFLOW_USERNAME: Airflow API username.
  - AIRFLOW_PASSWORD: Airflow API password.
  - OPENAI_API_KEY: Your OpenAI API key.
"""

from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/ubuntu/airflow/.env")

import os
import json
import requests
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Read configuration from environment variables
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://3.146.46.179:8081")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "utoledo")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "power2do")

###############################################
# Slack notification function
###############################################
def send_to_slack(channel, title, details, urgency):
    """
    Send a formatted message to Slack.
    
    Args:
        channel (str): Slack channel (e.g., "#alerts").
        title (str): A short headline (e.g., "🚨 Data Drift Detected").
        details (str): Detailed message explaining the issue and proposed fix.
        urgency (str): Urgency level (e.g., "low", "medium", "high", "critical").
        
    Returns:
        dict: A dictionary with the Slack response details.
    """
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL is not set in the environment.")
    
    msg = f"*{title}*\nChannel: {channel}\nUrgency: `{urgency}`\n\n{details}"
    payload = {"text": msg}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)
    logging.info(f"Slack response: {response.status_code}, {response.text}")
    return {"status": response.status_code, "detail": response.text}

###############################################
# Airflow DAG trigger function
###############################################
def trigger_airflow_dag(dag_id, conf=None):
    """
    Trigger an Airflow DAG run using the Airflow REST API.
    
    Args:
        dag_id (str): The DAG ID to trigger.
        conf (dict, optional): Any configuration parameters to pass to the DAG run.
    
    Returns:
        dict: A dictionary with the Airflow API response.
    """
    endpoint = f"{AIRFLOW_API_URL}/api/v1/dags/{dag_id}/dagRuns"
    headers = {"Content-Type": "application/json"}
    auth = (AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
    payload = {"conf": conf or {}}
    
    response = requests.post(endpoint, json=payload, headers=headers, auth=auth)
    logging.info(f"Triggered DAG '{dag_id}' with response: {response.status_code}, {response.text}")
    return {"status": response.status_code, "detail": response.text}

###############################################
# Additional Functions for Self-Healing and Explainability
###############################################
def propose_fix(problem_summary, proposed_fix, confidence, requires_human_approval):
    """
    Propose a fix for a pipeline or model issue.
    
    Args:
        problem_summary (str): Brief summary of the issue.
        proposed_fix (str): Explanation of the proposed solution.
        confidence (float): Confidence score between 0 and 1.
        requires_human_approval (bool): Whether the fix requires manual override.
    
    Returns:
        dict: The fix proposal details.
    """
    result = {
        "problem_summary": problem_summary,
        "proposed_fix": proposed_fix,
        "confidence": confidence,
        "requires_human_approval": requires_human_approval
    }
    logging.info(f"Propose Fix: {result}")
    return result

def override_decision(fix_id, approved_by, comment=""):
    """
    Log and approve a manual override decision.
    
    Args:
        fix_id (str): Identifier of the fix proposal.
        approved_by (str): Name of the admin approving the fix.
        comment (str, optional): Optional comment.
    
    Returns:
        dict: Details of the override decision.
    """
    result = {
        "fix_id": fix_id,
        "approved_by": approved_by,
        "comment": comment,
        "status": "approved"
    }
    logging.info(f"Override Decision: {result}")
    return result

def generate_root_cause_report(dag_id, execution_date):
    """
    Analyze logs/metrics to determine the root cause of an issue.
    
    Args:
        dag_id (str): The DAG ID for which to generate the report.
        execution_date (str): ISO timestamp of the failed run.
    
    Returns:
        dict: A report detailing the likely root cause.
    """
    # Placeholder logic: In production you could parse logs or query metrics.
    report = {
        "dag_id": dag_id,
        "execution_date": execution_date,
        "root_cause": "Data drift in feature 'example_feature' and increased RMSE.",
        "details": "Detected a 15% drift in 'example_feature' and RMSE increased by 0.5 units compared to the previous best."
    }
    logging.info(f"Root Cause Report: {report}")
    return report

def suggest_hyperparam_improvement(model_id, current_rmse, previous_best_rmse):
    """
    Suggest new hyperparameter configurations based on RMSE degradation.
    
    Args:
        model_id (str): Identifier of the model.
        current_rmse (float): Current RMSE.
        previous_best_rmse (float): The best historical RMSE.
    
    Returns:
        dict: Suggested hyperparameter adjustments.
    """
    suggestion = {
        "model_id": model_id,
        "suggestion": "Increase 'n_estimators' and decrease 'learning_rate'",
        "current_rmse": current_rmse,
        "previous_best_rmse": previous_best_rmse,
        "confidence": 0.8
    }
    logging.info(f"Hyperparameter Improvement Suggestion: {suggestion}")
    return suggestion

def validate_data_integrity(dataset_path):
    """
    Validate the quality and integrity of the dataset.
    
    Args:
        dataset_path (str): Path to the dataset file.
    
    Returns:
        dict: A validation report.
    """
    # Placeholder: In production, add schema checking, missing value analysis etc.
    report = {
        "dataset_path": dataset_path,
        "valid": True,
        "issues": None,
        "row_count": None
    }
    try:
        df = pd.read_csv(dataset_path)
        report["row_count"] = len(df)
    except Exception as e:
        report["valid"] = False
        report["issues"] = str(e)
    
    logging.info(f"Data Integrity Report: {report}")
    return report

def describe_fix_plan(issue_type, solution_summary):
    """
    Generate a summary of the fix plan with reasoning for transparency.
    
    Args:
        issue_type (str): Type of issue detected.
        solution_summary (str): Summary of the proposed solution.
    
    Returns:
        dict: A description of the fix plan.
    """
    plan = {
        "issue_type": issue_type,
        "solution_summary": solution_summary,
        "expected_outcome": "Improved model performance and reduced drift."
    }
    logging.info(f"Fix Plan Description: {plan}")
    return plan

###############################################
# Function to handle tool calls from the OpenAI Assistant
###############################################
def handle_function_call(tool_call):
    """
    Handle a tool call from the OpenAI Assistant and dispatch to the appropriate function.
    
    Args:
        tool_call (dict): A dictionary containing the function call details.
    
    Returns:
        dict: The result from the dispatched function.
    """
    try:
        func_name = tool_call.get("function", {}).get("name")
        arguments = tool_call.get("function", {}).get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        
        logging.info(f"Handling function call: {func_name} with arguments: {arguments}")
        
        if func_name == "notify_slack":
            return send_to_slack(
                channel=arguments.get("channel", "#general"),
                title=arguments.get("title", "Notification"),
                details=arguments.get("details", ""),
                urgency=arguments.get("urgency", "medium")
            )
        elif func_name == "airflow_trigger_dag":
            return trigger_airflow_dag(
                dag_id=arguments["dag_id"],
                conf=arguments.get("conf", {})
            )
        elif func_name == "propose_fix":
            return propose_fix(
                problem_summary=arguments["problem_summary"],
                proposed_fix=arguments["proposed_fix"],
                confidence=arguments["confidence"],
                requires_human_approval=arguments["requires_human_approval"]
            )
        elif func_name == "override_decision":
            return override_decision(
                fix_id=arguments["fix_id"],
                approved_by=arguments["approved_by"],
                comment=arguments.get("comment", "")
            )
        elif func_name == "generate_root_cause_report":
            return generate_root_cause_report(
                dag_id=arguments["dag_id"],
                execution_date=arguments["execution_date"]
            )
        elif func_name == "suggest_hyperparam_improvement":
            return suggest_hyperparam_improvement(
                model_id=arguments["model_id"],
                current_rmse=arguments["current_rmse"],
                previous_best_rmse=arguments["previous_best_rmse"]
            )
        elif func_name == "validate_data_integrity":
            return validate_data_integrity(
                dataset_path=arguments["dataset_path"]
            )
        elif func_name == "describe_fix_plan":
            return describe_fix_plan(
                issue_type=arguments["issue_type"],
                solution_summary=arguments["solution_summary"]
            )
        else:
            raise ValueError(f"Function {func_name} not recognized.")
    except Exception as ex:
        logging.error(f"Error handling function call: {ex}")
        return {"error": str(ex)}

# For testing purposes: Uncomment the following block to simulate a tool call.
if __name__ == "__main__":
     # Test notify_slack
     test_tool_call = {
         "function": {
             "name": "notify_slack",
             "arguments": json.dumps({
                 "channel": "#alerts",
                 "title": "🚨 Test Alert",
                 "details": "This is a test notification.",
                 "urgency": "high"
             })
         }
     }
     result = handle_function_call(test_tool_call)
     print("Test Result:", result)

# For testing purposes: Uncomment the following block to simulate a tool call.
# if __name__ == "__main__":
#     # Test notify_slack
#     test_tool_call = {
#         "function": {
#             "name": "notify_slack",
#             "arguments": json.dumps({
#                 "channel": "#alerts",
#                 "title": "🚨 Test Alert",
#                 "details": "This is a test notification.",
#                 "urgency": "high"
#             })
#         }
#     }
#     result = handle_function_call(test_tool_call)
#     print("Test Result:", result)