#!/usr/bin/env python3
"""
utils/storage.py

Retrying wrappers around common S3 operations.
"""

import boto3
from tenacity import retry, wait_exponential, stop_after_attempt
from utils.config import S3_BUCKET

# Reuse a single boto3 client
_s3_client = boto3.client("s3")

@retry(wait=wait_exponential(multiplier=1, min=2, max=10),
       stop=stop_after_attempt(3))
def download(key: str, local_path: str) -> None:
    """
    Download an object from S3 to a local file, retrying on failure.

    Args:
        key (str): The S3 object key (path within the bucket).
        local_path (str): Local filesystem path to write the file to.
    """
    _s3_client.download_file(S3_BUCKET, key, local_path)

@retry(wait=wait_exponential(multiplier=1, min=2, max=10),
       stop=stop_after_attempt(3))
def upload(local_path: str, key: str) -> None:
    """
    Upload a local file to S3, retrying on failure.

    Args:
        local_path (str): Path of the local file to upload.
        key (str): The S3 object key under which to store the file.
    """
    _s3_client.upload_file(local_path, S3_BUCKET, key)
