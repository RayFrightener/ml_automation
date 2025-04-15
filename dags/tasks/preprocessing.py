#!/usr/bin/env python3
"""
preprocessing.py

This module defines data preprocessing functions including:
  - Loading data into a DataFrame.
  - Handling missing values.
  - Detecting and capping outliers (using the IQR method).
  - Encoding categorical variables.
  - Splitting the data into training, validation, and test sets.
  
These functions are designed to be modular and can be integrated into your Airflow DAG.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import boto3
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from airflow.models import Variable
from dags.tasks.cache import is_cache_valid, update_cache

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Create S3 client
s3_client = boto3.client("s3")
S3_BUCKET = os.getenv("S3_BUCKET") or Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket")


def load_data_to_dataframe(local_path: str, s3_key: str = None) -> pd.DataFrame:
    """
    Reads a CSV file from the specified local path into a pandas DataFrame.
    If s3_key is provided, it will check the cache and download from S3 if needed.
    
    Args:
        local_path (str): The local file path to the CSV file.
        s3_key (str, optional): The S3 key for the file if it should be downloaded from S3.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        # If s3_key is provided, check cache and download if needed
        if s3_key:
            if not is_cache_valid(S3_BUCKET, s3_key, local_path):
                update_cache(S3_BUCKET, s3_key, local_path)
                logging.info(f"Downloaded and cached: {s3_key}")
            else:
                logging.info(f"Cache hit. Using cached file: {local_path}")
        
        df = pd.read_csv(local_path)
        logging.info(f"Loaded data from {local_path} with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {local_path}: {e}")
        raise


def handle_missing_data(df: pd.DataFrame, strategy: str = "mean", missing_threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle missing values by dropping columns with a missing fraction above the threshold,
    and imputing remaining missing values using the specified strategy.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str, optional): Imputation strategy: "mean", "median", or "mode". Default is "mean".
        missing_threshold (float, optional): Threshold for dropping columns. Default is 0.5.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Drop columns where missing fraction exceeds the threshold
    cols_to_drop = [col for col in df.columns if df[col].isna().mean() > missing_threshold]
    if cols_to_drop:
        logging.info(f"Dropping columns due to high missing fraction: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Impute remaining missing values
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                impute_value = df[col].mean()
            elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                impute_value = df[col].median()
            elif strategy == "mode":
                impute_value = df[col].mode()[0]
            else:
                impute_value = 0
            df[col] = df[col].fillna(impute_value)
            logging.info(f"Imputed missing values in column '{col}' using {strategy} strategy.")
    return df


def detect_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Dict[str, int]:
    """
    Identify the number of outliers in each numeric column using the IQR method.
    
    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        factor (float, optional): The IQR multiplier (default is 1.5).
    
    Returns:
        dict: A dictionary mapping each numeric column to the number of detected outliers.
    """
    outlier_counts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        iqr = Q3 - Q1
        lower_bound = Q1 - factor * iqr
        upper_bound = Q3 + factor * iqr
        count = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_counts[col] = count
        logging.info(f"Detected {count} outliers in column '{col}' with factor {factor}.")
    return outlier_counts


def cap_outliers(df: pd.DataFrame, col: str, factor: float = 1.5) -> pd.DataFrame:
    """
    Cap outlier values in a given numeric column at the lower and upper bounds defined by the IQR method.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the column to process.
        col (str): The column name.
        factor (float, optional): The IQR multiplier (default is 1.5).
    
    Returns:
        pd.DataFrame: The DataFrame with outliers in the specified column capped.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    iqr = Q3 - Q1
    lower_bound = Q1 - factor * iqr
    upper_bound = Q3 + factor * iqr
    df[col] = np.clip(df[col], lower_bound, upper_bound)
    logging.info(f"Capped outliers in '{col}' at lower bound {lower_bound} and upper bound {upper_bound}.")
    return df


def encode_categoricals(df: pd.DataFrame, encoding_map: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Encode categorical columns in the DataFrame.
    If an encoding map is provided, it will use that mapping; otherwise, performs one-hot encoding.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        encoding_map (dict, optional): A mapping for encoding categorical values.
    
    Returns:
        pd.DataFrame: The DataFrame with categorical variables encoded.
    """
    if encoding_map:
        logging.info("Using custom encoding map for categoricals.")
        for col, mapping in encoding_map.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
    else:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        logging.info(f"One-hot encoding the following categorical columns: {list(categorical_cols)}")
        df = pd.get_dummies(df, columns=categorical_cols)
    return df


def split_data(df: pd.DataFrame, target_col: str, split_ratios: tuple = (0.7, 0.15, 0.15), random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Split the DataFrame into training, validation, and test sets based on provided ratios.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The column name of the target variable.
        split_ratios (tuple, optional): Ratios for training, validation, and test sets (default: 0.7, 0.15, 0.15).
        random_state (int, optional): Random state for reproducibility (default: 42).
    
    Returns:
        dict: A dictionary with keys "train_x", "train_y", "val_x", "val_y", "test_x", "test_y".
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    train_ratio, val_ratio, test_ratio = split_ratios
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), random_state=random_state)
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_size), random_state=random_state)
    logging.info(f"Split data: {len(y_train)} training, {len(y_val)} validation, and {len(y_test)} test samples.")
    return {"train_x": X_train, "train_y": y_train, "val_x": X_val, "val_y": y_val, "test_x": X_test, "test_y": y_test}


if __name__ == "__main__":
    # For testing: Replace 'sample_data.csv' and 'loss_amount' with real file and target column names.
    sample_path = "sample_data.csv"
    try:
        df = load_data_to_dataframe(sample_path)
        df = handle_missing_data(df, strategy="mean", missing_threshold=0.3)
        outlier_counts = detect_outliers_iqr(df)
        logging.info(f"Outlier counts: {outlier_counts}")
        # Optionally, cap outliers for one column (adjust 'some_numeric_col' as needed)
        if "some_numeric_col" in df.columns:
            df = cap_outliers(df, "some_numeric_col")
        df = encode_categoricals(df)
        splits = split_data(df, target_col="loss_amount", split_ratios=(0.7, 0.15, 0.15))
        logging.info(f"Data splits: {splits.keys()}")
    except Exception as e:
        logging.error(f"Preprocessing test failed: {e}")
