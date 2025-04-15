# notifications.py
import os
import json
import logging
import requests
import time
import boto3
from airflow.models import Variable

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
S3_BUCKET = os.environ.get("S3_BUCKET", Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket"))

s3_client = boto3.client("s3")

def send_notification():
    """
    Sends a placeholder notification. (Extend this as needed.)
    
    Returns:
        str: A success message.
    """
    logging.info("Sending notification (placeholder logic).")
    # In a real implementation, you might build a rich message to Slack.
    return "Notification sent."

def push_logs_to_s3():
    """
    Uploads the custom Airflow log file to S3.
    
    Returns:
        str: S3 key where the log file was stored.
    """
    try:
        log_path = "/tmp/custom_airflow.log"
        with open(log_path, "w") as f:
            f.write("Centralized log entry for Airflow pipeline.\n")
        s3_key = "airflow/logs/custom_airflow.log"
        s3_client.upload_file(log_path, S3_BUCKET, s3_key)
        logging.info(f"Uploaded logs to s3://{S3_BUCKET}/{s3_key}")
        return s3_key
    except Exception as e:
        logging.error(f"Error in push_logs_to_s3: {e}")
        raise

def archive_data():
    """
    Archives the old total home loss history CSV file from S3.
    
    Returns:
        str: S3 key of the archived file.
    """
    try:
        S3_DATA_FOLDER = "raw-data"
        original_key = f"{S3_DATA_FOLDER}/Total_home_loss_hist.csv"
        archive_key = f"archive/Total_home_loss_hist_{int(time.time())}.csv"
        copy_source = {"Bucket": S3_BUCKET, "Key": original_key}
        s3_client.copy_object(CopySource=copy_source, Bucket=S3_BUCKET, Key=archive_key)
        logging.info(f"Archived s3://{S3_BUCKET}/{original_key} to s3://{S3_BUCKET}/{archive_key}")
        return archive_key
    except Exception as e:
        logging.error(f"Error in archive_data: {e}")
        raise
#!/usr/bin/env python3
"""
notifications.py

This module defines functions for notifications and log-related operations:
  - Sending Slack notifications.
  - Pushing log files to S3.
  - Archiving data files to S3.

Ensure the following environment variables are set either via a .env file or Airflow Variables:
  - SLACK_WEBHOOK_URL: The Slack incoming webhook URL.
  - S3_BUCKET: The S3 bucket name.
  - S3_ARCHIVE_FOLDER: The S3 folder for archived files (default: "archive").
"""

from dotenv import load_dotenv
load_dotenv()  # Assumes .env is in the current working directory or specify the path if needed

import os
import json
import logging
import requests
import boto3
from airflow.models import Variable

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Read configuration from environment variables or Airflow Variables
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL") or Variable.get("SLACK_WEBHOOK_URL", default_var="")
S3_BUCKET = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET", default_var="")
S3_ARCHIVE_FOLDER = os.getenv("S3_ARCHIVE_FOLDER") or Variable.get("S3_ARCHIVE_FOLDER", default_var="archive")

# Create global S3 client
s3_client = boto3.client("s3")


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


def push_logs_to_s3(log_file_path):
    """
    Upload the specified log file to the S3 bucket under a 'logs/' prefix.
    
    Args:
        log_file_path (str): Local path to the log file.
    
    Returns:
        dict: A dictionary with the S3 path of the uploaded log.
    """
    s3_key = f"logs/{os.path.basename(log_file_path)}"
    s3_client.upload_file(log_file_path, S3_BUCKET, s3_key)
    logging.info(f"Uploaded log file {log_file_path} to s3://{S3_BUCKET}/{s3_key}")
    return {"status": "success", "s3_path": f"s3://{S3_BUCKET}/{s3_key}"}


def archive_data(file_path):
    """
    Archive a data file by uploading it to an archive folder in S3.
    
    Args:
        file_path (str): Local path to the file that should be archived.
    
    Returns:
        dict: A dictionary with the S3 path of the archived file.
    """
    s3_key = f"{S3_ARCHIVE_FOLDER}/{os.path.basename(file_path)}"
    s3_client.upload_file(file_path, S3_BUCKET, s3_key)
    logging.info(f"Archived file {file_path} to s3://{S3_BUCKET}/{s3_key}")
    return {"status": "success", "s3_path": f"s3://{S3_BUCKET}/{s3_key}"}
