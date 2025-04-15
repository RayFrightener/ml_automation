#!/usr/bin/env python3
"""
notifications.py

This module defines functions for:
  - Sending Slack notifications using agent_actions.
  - Uploading logs to S3.
  - Archiving data to S3.

Ensure the following environment variables are set via .env or Airflow Variables:
  - SLACK_WEBHOOK_URL
  - S3_BUCKET
  - S3_ARCHIVE_FOLDER (optional, defaults to 'archive')
"""

from dotenv import load_dotenv
load_dotenv()  # Assumes .env is in the project root

import os
import logging
import boto3
from airflow.models import Variable
from dags.agent_actions import send_to_slack, escalate_issue

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Load configuration
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL") or Variable.get("SLACK_WEBHOOK_URL", default_var="")
S3_BUCKET = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket")
S3_ARCHIVE_FOLDER = os.getenv("S3_ARCHIVE_FOLDER", "archive")

# S3 client
s3_client = boto3.client("s3")


def push_logs_to_s3(log_file_path):
    """
    Uploads a log file to the S3 bucket under 'logs/'.

    Args:
        log_file_path (str): Local log file path

    Returns:
        dict: Status and S3 path
    """
    try:
        s3_key = f"logs/{os.path.basename(log_file_path)}"
        s3_client.upload_file(log_file_path, S3_BUCKET, s3_key)
        logging.info(f"Uploaded {log_file_path} to s3://{S3_BUCKET}/{s3_key}")
        return {"status": "success", "s3_path": f"s3://{S3_BUCKET}/{s3_key}"}
    except Exception as e:
        logging.error(f"Failed to upload log: {e}")
        return {"status": "error", "detail": str(e)}


def archive_data(file_path):
    """
    Archives a data file to the 'archive/' folder in S3.

    Args:
        file_path (str): Local file to upload

    Returns:
        dict: Status and S3 archive path
    """
    try:
        s3_key = f"{S3_ARCHIVE_FOLDER}/{os.path.basename(file_path)}"
        s3_client.upload_file(file_path, S3_BUCKET, s3_key)
        logging.info(f"Archived file to s3://{S3_BUCKET}/{s3_key}")
        return {"status": "success", "s3_path": f"s3://{S3_BUCKET}/{s3_key}"}
    except Exception as e:
        logging.error(f"Failed to archive file: {e}")
        return {"status": "error", "detail": str(e)}


def notify_success(channel="#alerts"):
    """
    Sends a simple success Slack notification.

    Args:
        channel (str): Slack channel

    Returns:
        dict: Slack API response
    """
    return send_to_slack(
        channel=channel,
        title="✅ Pipeline Completed",
        details="The homeowner DAG ran successfully.",
        urgency="low"
    )
