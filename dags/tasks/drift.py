#!/usr/bin/env python3
"""
drift.py

This module defines functions for detecting data drift and triggering self-healing.
It compares current processed data against a reference means file (generated earlier)
and determines if corrective actions (self-healing) are needed.
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
import boto3
from airflow.models import Variable
from dags.tasks.cache import is_cache_valid, update_cache

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Configuration: read from environment or Airflow Variables
S3_BUCKET = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket")
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"  # Local copy of the reference means
s3_client = boto3.client("s3")

def detect_data_drift(processed_path: str):
    """
    Compare current data means with those from the reference means file.
    Returns "self_healing" if significant drift is detected (>10%), else "train_xgboost_hyperopt".
    
    Args:
        processed_path (str): Path to the processed CSV file.
        
    Returns:
        str: "self_healing" if drift is detected; otherwise, "train_xgboost_hyperopt".
    """
    try:
        df_current = pd.read_csv(processed_path)
        
        # Check if we need to download the reference means file from S3
        s3_key = "reference/reference_means.csv"
        if not is_cache_valid(S3_BUCKET, s3_key, REFERENCE_MEANS_PATH):
            update_cache(S3_BUCKET, s3_key, REFERENCE_MEANS_PATH)
            logging.info(f"Downloaded and cached reference means: {s3_key}")
        else:
            logging.info(f"Cache hit. Using cached reference means file.")
            
        if not os.path.exists(REFERENCE_MEANS_PATH):
            logging.warning(f"Reference means file not found at {REFERENCE_MEANS_PATH}. Skipping drift check.")
            return "train_xgboost_hyperopt"
            
        df_reference = pd.read_csv(REFERENCE_MEANS_PATH)
        reference_dict = {row["column_name"]: row["mean_value"] for _, row in df_reference.iterrows()}
        drift_detected = False
        for col in df_current.select_dtypes(include=[np.number]).columns:
            if col in reference_dict:
                current_mean = df_current[col].mean()
                ref_mean = reference_dict[col]
                if ref_mean > 0:
                    drift_ratio = abs(current_mean - ref_mean) / ref_mean
                    if drift_ratio > 0.10:
                        drift_detected = True
                        logging.error(f"Data drift detected in '{col}': current mean={current_mean:.2f}, ref mean={ref_mean:.2f}, drift ratio={drift_ratio:.2%}")
                else:
                    logging.warning(f"Non-positive reference mean for '{col}'; skipping drift check.")
            else:
                logging.warning(f"No reference value for '{col}'; skipping drift check.")
        return "self_healing" if drift_detected else "train_xgboost_hyperopt"
    except Exception as e:
        logging.error(f"Error in detect_data_drift: {e}")
        raise


def self_healing():
    """
    Simulated self-healing branch that waits for a manual override.
    In a production system, this might trigger further actions such as re-ingesting data
    or retraining the model.
    
    Returns:
        str: A status message indicating that the override has been applied.
    """
    try:
        logging.info("Data drift detected. Executing self-healing routine and awaiting manual override...")
        time.sleep(5)  # Simulate waiting time for manual intervention
        logging.info("Manual override processed; proceeding with fix application.")
        return "override_done"
    except Exception as e:
        logging.error(f"Error in self_healing: {e}")
        raise


def list_recent_failures(lookback_hours):
    """
    List recent DAG or task failures from the past N hours.
    (Placeholder: In production, query Airflow’s metadata database or log storage.)
    
    Args:
        lookback_hours (int): Number of hours to look back.
        
    Returns:
        list: A list of failure dictionaries.
    """
    failures = [
        {"dag_id": "homeowner_dag", "task_id": "train_xgboost_hyperopt", "failure_time": "2025-04-14T05:00:00Z"}
    ]
    logging.info(f"Recent failures in the past {lookback_hours} hours: {failures}")
    return failures


if __name__ == "__main__":
    # For testing the drift detection:
    test_processed_path = "/tmp/homeowner_processed.csv"  # Ensure this file exists for testing.
    result = detect_data_drift(test_processed_path)
    print("Drift detection result:", result)
