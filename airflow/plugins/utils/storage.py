#!/usr/bin/env python3
"""
utils/storage.py

Retrying wrappers around common S3 operations.
"""

import boto3
from tenacity import retry, wait_exponential, stop_after_attempt
from .config import DATA_BUCKET
from typing import Optional

# Reuse a single boto3 client
_s3_client = boto3.client("s3")

@retry(wait=wait_exponential(multiplier=1, min=2, max=10),
       stop=stop_after_attempt(3))
def download(key: str, local_path: str, bucket: Optional[str] = None) -> None:
    """
    Download an object from S3 to a local file, retrying on failure.

    Args:
        key (str): The S3 object key (path within the bucket).
        local_path (str): Local filesystem path to write the file to.
        bucket (Optional[str]): S3 bucket name; defaults to DATA_BUCKET if None.
    """
    _s3_client.download_file(bucket or DATA_BUCKET, key, local_path)

@retry(wait=wait_exponential(multiplier=1, min=2, max=10),
       stop=stop_after_attempt(3))
def upload(local_path: str, key: str, bucket: Optional[str] = None) -> None:
    """
    Upload a local file to S3, retrying on failure.

    Args:
        local_path (str): Path of the local file to upload.
        key (str): The S3 object key under which to store the file.
        bucket (Optional[str]): S3 bucket name; defaults to DATA_BUCKET if None.
    """
    _s3_client.upload_file(local_path, bucket or DATA_BUCKET, key)

def update_storage_process_with_ui_components():
    """
    Placeholder function to update the storage process with new UI components and endpoints.
    """
    logging.info("Updating storage process with new UI components and endpoints.")
    # Placeholder for actual implementation
