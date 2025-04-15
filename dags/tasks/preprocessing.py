#!/usr/bin/env python3
"""
preprocessing.py

Handles:
  - Data loading from CSV
  - Handling missing values
  - Outlier detection and capping
  - Categorical encoding
  - Data profiling (YData Profiling only, no SDK)
  - Agent Slack notification for profiling summary

This version avoids ydata-sdk and uses ydata-profiling standalone (formerly pandas-profiling).
"""

import os
import json
import logging
import pandas as pd
from ydata_profiling import ProfileReport
from airflow.models import Variable
from agent_actions import handle_function_call

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Default configuration
PROFILE_REPORT_PATH = "/tmp/homeowner_profile_report.html"


def load_data_to_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded data from {csv_path} with shape {df.shape}")
    return df


def handle_missing_data(df, strategy="mean", missing_threshold=0.3):
    null_ratio = df.isnull().mean()
    drop_cols = null_ratio[null_ratio > missing_threshold].index
    df.drop(columns=drop_cols, inplace=True)
    logging.info(f"Dropped columns due to missing values > {missing_threshold}: {list(drop_cols)}")

    if strategy == "mean":
        df.fillna(df.mean(numeric_only=True), inplace=True)
    elif strategy == "zero":
        df.fillna(0, inplace=True)
    elif strategy == "ffill":
        df.fillna(method="ffill", inplace=True)
    logging.info(f"Applied missing data strategy: {strategy}")
    return df


def detect_outliers_iqr(df, factor=1.5):
    outlier_counts = {}
    for col in df.select_dtypes(include=["number"]):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - factor * IQR) | (df[col] > Q3 + factor * IQR)]
        outlier_counts[col] = outliers.shape[0]
    return outlier_counts


def cap_outliers(df, col, factor=1.5):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    df[col] = df[col].clip(lower, upper)
    return df


def encode_categoricals(df, encoding_map=None):
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    logging.info(f"Encoded categorical columns: {list(obj_cols)}")
    return df


def generate_profile_report(df, output_path=PROFILE_REPORT_PATH):
    profile = ProfileReport(df, title="Homeowner Data Profile", minimal=True)
    profile.to_file(output_path)
    logging.info(f"Data profiling report saved to {output_path}")

    # Notify agent
    try:
        handle_function_call({
            "function": {
                "name": "notify_slack",
                "arguments": json.dumps({
                    "channel": "#agent_logs",
                    "title": "📊 Profiling Summary",
                    "details": f"Profiling complete. Output saved to {output_path}",
                    "urgency": "low"
                })
            }
        })
        logging.info("Agent notified with profiling result.")
    except Exception as e:
        logging.warning(f"Agent notification failed: {e}")
