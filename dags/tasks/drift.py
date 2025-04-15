#!/usr/bin/env python3
"""
drift.py

Handles:
  - Generation of reference means from the processed data.
  - Data drift detection by comparing current data with reference means.
  - A self-healing routine (simulated) when drift is detected.

Requires:
  - S3_BUCKET and REFERENCE_MEANS_PATH configuration.
  - boto3 for S3 operations.
"""

import os
import logging
import pandas as pd
import numpy as np
import time
from airflow.models import Variable
import boto3

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket"))
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"  # Local path to store generated reference means
s3_client = boto3.client("s3")

def generate_reference_means(processed_path, local_ref=REFERENCE_MEANS_PATH):
    """
    Generates a reference means CSV file from the processed data and uploads it to S3.

    Args:
        processed_path (str): Path to the processed CSV.
        local_ref (str): Local path to save the reference means CSV.

    Returns:
        str: Local path to the generated reference means CSV.
    """
    try:
        df = pd.read_csv(processed_path)
        # Calculate means for all numeric columns
        means = df.select_dtypes(include=[np.number]).mean().reset_index()
        means.columns = ["column_name", "mean_value"]
        means.to_csv(local_ref, index=False)
        s3_key = "reference/reference_means.csv"
        s3_client.upload_file(local_ref, S3_BUCKET, s3_key)
        logging.info(f"Generated and uploaded reference means to s3://{S3_BUCKET}/{s3_key}")
        return local_ref
    except Exception as e:
        logging.error(f"Error generating reference means: {e}")
        raise

def detect_data_drift(current_data_path, reference_means_path, threshold=0.1):
    """
    Detects data drift by comparing current data’s mean values against the reference means.

    Args:
        current_data_path (str): Path to the current processed data CSV.
        reference_means_path (str): Path to the reference means CSV.
        threshold (float): Proportional drift threshold (default 0.1 for 10%).

    Returns:
        str: "self_healing" if drift is detected; "train_xgboost_hyperopt" otherwise.
    """
    try:
        df_current = pd.read_csv(current_data_path)
        df_ref = pd.read_csv(reference_means_path)
        ref_dict = {row["column_name"]: row["mean_value"] for _, row in df_ref.iterrows()}
        drift_detected = False
        for col in df_current.select_dtypes(include=[np.number]).columns:
            if col in ref_dict:
                current_mean = df_current[col].mean()
                ref_mean = ref_dict[col]
                if ref_mean != 0:
                    drift_ratio = abs(current_mean - ref_mean) / abs(ref_mean)
                    if drift_ratio > threshold:
                        drift_detected = True
                        logging.error(f"Drift detected in {col}: current={current_mean:.2f}, ref={ref_mean:.2f}, ratio={drift_ratio:.2%}")
                else:
                    logging.warning(f"Reference mean for {col} is 0; skipping drift check.")
            else:
                logging.warning(f"No reference for column {col}; skipping.")
        return "self_healing" if drift_detected else "train_xgboost_hyperopt"
    except Exception as e:
        logging.error(f"Error in detect_data_drift: {e}")
        raise

def self_healing():
    """
    Simulates a self-healing routine when data drift is detected.
    In a production environment, this could trigger revalidation or data cleanup routines.
    
    Returns:
        str: A message indicating that self-healing (and manual override) is complete.
    """
    try:
        logging.info("Drift detected. Initiating self-healing routine; awaiting manual override...")
        time.sleep(5)  # Simulate wait period for human intervention
        logging.info("Self-healing routine completed; manual override confirmed.")
        return "override_done"
    except Exception as e:
        logging.error(f"Error in self_healing: {e}")
        raise

if __name__ == "__main__":
    # For testing purposes only
    test_processed_path = "/tmp/sample_processed.csv"
    if not os.path.exists(test_processed_path):
        pd.DataFrame({"pure_premium": [100, 200, 150], "feature1": [1, 2, 3]}).to_csv(test_processed_path, index=False)
    ref_path = generate_reference_means(test_processed_path)
    drift_status = detect_data_drift(test_processed_path, ref_path)
    print("Drift detection result:", drift_status)
