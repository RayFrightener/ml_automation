#!/usr/bin/env python3
"""
tasks/drift.py

Handles:
  - Generation of reference means from the processed data.
  - Data drift detection by comparing current data with versioned reference means.
  - A self‑healing routine when drift is detected.

Refactored to:
  - Use utils/storage.upload instead of boto3 directly.
  - Version the reference means file on upload under a timestamped prefix.
  - Wrap S3 operations with tenacity retries.
  - Load drift threshold from Airflow Variable "DRIFT_THRESHOLD".
"""

import os
import logging
import time
import pandas as pd
import numpy as np
from airflow.models import Variable
from tenacity import retry, stop_after_attempt, wait_fixed

from utils.storage import upload
from utils.config import S3_BUCKET, REFERENCE_KEY_PREFIX

# Setup logging
t=logging.getLogger()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Local path for the reference means CSV
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"

# Drift threshold loaded from Airflow Variable or default 0.1
try:
    DRIFT_THRESHOLD = float(Variable.get("DRIFT_THRESHOLD", default_var="0.1"))
except Exception:
    DRIFT_THRESHOLD = 0.1
    logging.warning(f"Using default drift threshold {DRIFT_THRESHOLD}")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def generate_reference_means(
    processed_path: str,
    local_ref: str = REFERENCE_MEANS_PATH
) -> str:
    """
    Generates a timestamped reference means CSV and uploads it to S3 under the configured prefix.

    Args:
        processed_path: Path to the latest processed data parquet.
        local_ref: Local path to write the reference means.

    Returns:
        The local path to the generated reference means CSV.
    """
    df = pd.read_parquet(processed_path)
    means = (
        df.select_dtypes(include=[np.number])
          .mean()
          .reset_index()
          .rename(columns={"index": "column_name", 0: "mean_value"})
    )
    means.to_csv(local_ref, index=False)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    s3_key = f"{REFERENCE_KEY_PREFIX}/reference_means_{ts}.csv"
    upload(local_ref, s3_key)
    logging.info(f"Uploaded reference means to s3://{S3_BUCKET}/{s3_key}")
    return local_ref


def detect_data_drift(
    current_data_path: str,
    reference_means_path: str,
    threshold: float = None
) -> str:
    """
    Compares current data means vs. the reference means to detect drift.

    Args:
        current_data_path: Path to the current processed parquet.
        reference_means_path: Path to a versioned reference means CSV.
        threshold: Fractional threshold for drift; if None uses DRIFT_THRESHOLD.

    Returns:
        "self_healing" if drift detected, else "train_xgboost_hyperopt".
    """
    if threshold is None:
        threshold = DRIFT_THRESHOLD

    df_current = pd.read_parquet(current_data_path)
    df_ref     = pd.read_csv(reference_means_path)
    ref_map    = dict(zip(df_ref["column_name"], df_ref["mean_value"]))

    drifted = False
    for col in df_current.select_dtypes(include=[np.number]).columns:
        if col not in ref_map:
            logging.warning(f"No reference for '{col}', skipping.")
            continue
        ref_val = ref_map[col]
        curr_val = df_current[col].mean()
        if ref_val != 0:
            ratio = abs(curr_val - ref_val) / abs(ref_val)
            if ratio > threshold:
                logging.error(
                    f"Drift in '{col}': current={curr_val:.2f}, ref={ref_val:.2f}, drift={ratio:.2%}"
                )
                drifted = True
        else:
            logging.warning(f"Reference mean for '{col}' is zero, skipping drift check.")

    return "self_healing" if drifted else "train_xgboost_hyperopt"


def self_healing() -> str:
    """
    Simulates a self‑healing routine when drift is detected.
    In production, this could trigger remediation workflows.

    Returns:
        A marker string once self‑healing completes.
    """
    logging.info("Drift detected: starting self‑healing (await manual approval)...")
    time.sleep(5)
    logging.info("Self‑healing complete; manual override confirmed.")
    return "override_done"

def update_drift_detection_with_ui_components():
    """
    Placeholder function to update the drift detection process with new UI components and endpoints.
    """
    logging.info("Updating drift detection process with new UI components and endpoints.")
    # Placeholder for actual implementation
