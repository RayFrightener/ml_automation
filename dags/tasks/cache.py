#!/usr/bin/env python3
"""
cache.py

This module provides caching utilities for the Homeowner Loss History Prediction project.
It supports:
  - Checking if a locally cached file is up-to-date compared to its S3 version.
  - Updating (downloading) the file from S3 if it has changed.
  - Clearing (removing) the cache.
  - A helper function to cache the output of a function to a local file.

Usage:
  Import the module in other tasks (e.g., ingestion.py, preprocessing.py) to avoid redundant processing.
  
Dependencies:
  - boto3 must be installed and configured.
  - Logging is used for error handling and tracing.
"""

import os
import logging
from datetime import datetime
import boto3
from botocore.exceptions import ClientError

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Create a global boto3 S3 client for reuse
s3_client = boto3.client("s3")

def get_s3_last_modified(bucket: str, key: str) -> datetime:
    """
    Retrieve the LastModified timestamp for an S3 object.
    
    Args:
        bucket (str): S3 bucket name.
        key (str): Object key in the bucket.
        
    Returns:
        datetime: The last modified timestamp.
        
    Raises:
        ClientError: If the object does not exist or if there is an access issue.
    """
    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
        last_modified = response["LastModified"]
        logging.info(f"S3 object s3://{bucket}/{key} last modified on {last_modified}")
        return last_modified
    except ClientError as e:
        logging.error(f"Failed to get S3 object head for s3://{bucket}/{key}: {e}")
        raise

def get_local_last_modified(local_path: str) -> datetime:
    """
    Get the last modification time of a local file.
    
    Args:
        local_path (str): Local file path.
    
    Returns:
        datetime: Last modified time of the file, or None if file does not exist.
    """
    if not os.path.exists(local_path):
        logging.info(f"Local file {local_path} does not exist.")
        return None
    timestamp = os.path.getmtime(local_path)
    last_modified = datetime.fromtimestamp(timestamp)
    logging.info(f"Local file {local_path} last modified on {last_modified}")
    return last_modified

def is_cache_valid(s3_bucket: str, s3_key: str, local_path: str) -> bool:
    """
    Check whether the local file is up-to-date relative to its S3 counterpart.
    
    Args:
        s3_bucket (str): S3 bucket name.
        s3_key (str): Object key on S3.
        local_path (str): Path to the local cache file.
        
    Returns:
        bool: True if the cached file exists and its last modified date is later than or equal to the S3 file.
    """
    try:
        s3_last_modified = get_s3_last_modified(s3_bucket, s3_key)
    except Exception as e:
        logging.error(f"Could not determine S3 last modified time: {e}")
        return False

    local_last_modified = get_local_last_modified(local_path)
    if local_last_modified is None:
        return False

    # Consider the cache valid if the local file was modified after (or at the same time as) the S3 file.
    if local_last_modified >= s3_last_modified:
        logging.info("Cache is valid: local file is up-to-date with S3.")
        return True
    else:
        logging.info("Cache is stale: local file is older than S3 version.")
        return False

def update_cache(s3_bucket: str, s3_key: str, local_path: str) -> bool:
    """
    Update the local cache by downloading the file from S3.
    
    Args:
        s3_bucket (str): S3 bucket name.
        s3_key (str): Object key on S3.
        local_path (str): Local file path to download the file to.
        
    Returns:
        bool: True if update succeeded.
        
    Raises:
        Exception: If the download fails.
    """
    try:
        logging.info(f"Downloading s3://{s3_bucket}/{s3_key} to {local_path}.")
        s3_client.download_file(s3_bucket, s3_key, local_path)
        logging.info("Cache updated successfully.")
        return True
    except Exception as e:
        logging.error(f"Error updating cache from S3: {e}")
        raise

def clear_cache(local_path: str) -> bool:
    """
    Remove the local cached file.
    
    Args:
        local_path (str): Path to the cached file.
    
    Returns:
        bool: True if the file was removed, False if no file existed.
        
    Raises:
        Exception: If deletion fails.
    """
    try:
        if os.path.exists(local_path):
            os.remove(local_path)
            logging.info(f"Cache cleared: {local_path} deleted.")
            return True
        else:
            logging.info("No cache file to remove.")
            return False
    except Exception as e:
        logging.error(f"Error clearing cache file {local_path}: {e}")
        raise

def cache_function_output(cache_file: str, func, *args, **kwargs):
    """
    Cache the output of a function to a local file. If the cache file exists, return its contents;
    otherwise, execute the function, write its output to the file, and return the output.
    
    Note: This simple implementation assumes the function returns a string.
    
    Args:
        cache_file (str): Path to the cache file.
        func (callable): The function whose output is to be cached.
        *args, **kwargs: Arguments to pass to the function.
    
    Returns:
        str: The function's output, either read from cache or freshly computed.
    """
    if os.path.exists(cache_file):
        logging.info(f"Cache hit: reading from {cache_file}.")
        with open(cache_file, "r") as f:
            return f.read()
    else:
        logging.info("Cache miss. Executing function to generate output.")
        output = func(*args, **kwargs)
        with open(cache_file, "w") as f:
            f.write(output)
        logging.info(f"Output cached to {cache_file}.")
        return output

if __name__ == "__main__":
    # Example usage (for testing only)
    test_bucket = "grange-seniordesign-bucket"
    test_key = "raw-data/ut_loss_history_1.csv"
    test_local = "/tmp/homeowner_data.csv"

    # Check cache validity
    valid = is_cache_valid(test_bucket, test_key, test_local)
    print(f"Cache valid: {valid}")

    # If not valid, update cache
    if not valid:
        update_cache(test_bucket, test_key, test_local)
        print("Cache updated.")
    
    # Optionally clear cache (uncomment for testing cleanup)
    # clear_cache(test_local)
    # print("Cache cleared.")
