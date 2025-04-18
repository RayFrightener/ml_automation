#!/usr/bin/env python3
"""
preprocessing.py

Handles:
  - Data loading from CSV
  - Handling missing values
  - Outlier detection and capping
  - Categorical encoding
  - Generating pure_premium & sample_weight
  - Pandera schema validation (after pure_premium exists)
  - Selecting the correct loss-history features per MODEL_ID
  - Data profiling report + Slack notification via agent_actions
"""

import os
import json
import logging
import time
from typing import Dict

import pandas as pd
from ydata_profiling import ProfileReport
from airflow.models import Variable
from tasks.schema_validation import validate_schema
from agent_actions import handle_function_call

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

PROFILE_REPORT_PATH = "/tmp/homeowner_profile_report.html"

# Prefix groups for loss-history features
RAW_PREFIXES = ["num_loss_3yr_", "num_loss_yrs45_", "num_loss_free_yrs_"]
DECAY_PREFIXES = {
    "model2": ["lhdwc_5y_1d_"],  # equal
    "model3": ["lhdwc_5y_2d_"],  # linear decay
    "model4": ["lhdwc_5y_3d_"],  # fast decay
    "model5": ["lhdwc_5y_4d_"],  # slow decay
}


def load_data_to_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Read CSV from local path into DataFrame.
    """
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded data from {csv_path}, shape={df.shape}")
    return df


def handle_missing_data(
    df: pd.DataFrame,
    strategy: str = "mean",
    missing_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Drop columns with > threshold missing, then impute remaining missing.
    """
    null_ratio = df.isnull().mean()
    drop_cols = null_ratio[null_ratio > missing_threshold].index.tolist()
    df.drop(columns=drop_cols, inplace=True)
    logging.info(f"Dropped columns >{missing_threshold*100:.0f}% missing: {drop_cols}")

    if strategy == "mean":
        df.fillna(df.mean(numeric_only=True), inplace=True)
    elif strategy == "zero":
        df.fillna(0, inplace=True)
    elif strategy == "ffill":
        df.fillna(method="ffill", inplace=True)
    else:
        raise ValueError(f"Unknown missing data strategy '{strategy}'")

    logging.info(f"Applied missing data strategy: {strategy}")
    return df


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, int]:
    """
    Count outliers per numeric column using the IQR method.
    """
    counts: Dict[str, int] = {}
    for col in df.select_dtypes(include="number"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = (df[col] < q1 - factor * iqr) | (df[col] > q3 + factor * iqr)
        counts[col] = int(mask.sum())
    return counts


def cap_outliers(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Clip numeric column values to within [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[col] = df[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One‑hot encode all object/categorical columns.
    """
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    logging.info(f"One‑hot encoded columns: {obj_cols}")
    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the loss‑history features needed for current MODEL_ID,
    plus all non–loss‑history columns.
    """
    model_id = (
        os.getenv("MODEL_ID")
        or Variable.get("MODEL_ID", default_var="model1")
    ).strip().lower()
    logging.info(f"Selecting features for MODEL_ID='{model_id}'")

    if model_id == "model1":
        keep_prefixes = RAW_PREFIXES
    elif model_id in DECAY_PREFIXES:
        keep_prefixes = DECAY_PREFIXES[model_id]
    else:
        raise ValueError(f"Unknown MODEL_ID '{model_id}'")

    all_loss_prefixes = RAW_PREFIXES + sum(DECAY_PREFIXES.values(), [])
    def is_loss_col(col: str) -> bool:
        return any(col.startswith(p) for p in all_loss_prefixes)

    keep_cols = [
        col for col in df.columns
        if (any(col.startswith(p) for p in keep_prefixes) or not is_loss_col(col))
    ]
    removed = [col for col in df.columns if col not in keep_cols]
    logging.info(f"Dropped loss‑history cols for {model_id}: {removed}")
    return df[keep_cols]


def generate_profile_report(df: pd.DataFrame, output_path: str = PROFILE_REPORT_PATH) -> None:
    """
    Generate a minimal profiling report, write to HTML, and notify via agent_actions.
    """
    profile = ProfileReport(df, title="Homeowner Data Profile", minimal=True)
    profile.to_file(output_path)
    logging.info(f"Saved profiling report to {output_path}")

    try:
        handle_function_call({
            "function": {
                "name": "send_to_slack",
                "arguments": json.dumps({
                    "channel": "#agent_logs",
                    "title": "📊 Profiling Summary",
                    "details": f"Profile saved to {output_path}",
                    "urgency": "low"
                })
            }
        })
    except Exception as e:
        logging.warning(f"Slack notification failed: {e}")


def preprocess_data(
    csv_path: str,
    strategy: str = "mean",
    missing_threshold: float = 0.3
) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Load CSV
      2. Handle missing values
      3. Detect & cap outliers
      4. Categorical encoding
      5. Compute pure_premium & sample_weight
      6. Pandera schema validation
      7. Model‑specific feature selection
      8. Profiling & Slack notification
    """
    # 1) Load CSV
    df = load_data_to_dataframe(csv_path)

    # 2) Missing data
    df = handle_missing_data(df, strategy, missing_threshold)

    # 3) Detect & cap outliers
    _ = detect_outliers_iqr(df)
    for col in df.select_dtypes(include="number"):
        df = cap_outliers(df, col)

    # 4) Encode categoricals
    df = encode_categoricals(df)

    # 5) Compute target & weight
    if "pure_premium" not in df.columns:
        if {"il_total", "eey"}.issubset(df.columns):
            df["pure_premium"]  = df["il_total"] / df["eey"]
            df["sample_weight"] = df["eey"]
            logging.info("Computed pure_premium & sample_weight")
        else:
            raise ValueError("Columns 'il_total' and 'eey' required to compute pure_premium")

    # 6) Now that pure_premium exists, validate against schema
    df = validate_schema(df)

    # 7) Select only the features needed for this MODEL_ID
    df = select_model_features(df)

    # 8) Profile & notify
    generate_profile_report(df)

    return df
