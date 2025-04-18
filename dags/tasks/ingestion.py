# ingestion.py
import os
import logging
import boto3
from airflow.models import Variable
from tasks.cache import is_cache_valid, update_cache

S3_BUCKET = os.environ.get("S3_BUCKET", Variable.get("S3_BUCKET", default_var="grange-seniordesign-bucket"))
S3_DATA_FOLDER = "raw-data"
LOCAL_DATA_PATH = "/tmp/homeowner_data.csv"

s3_client = boto3.client("s3")

def ingest_data_from_s3():
    """
    Downloads the raw CSV file from S3.
    Uses caching to avoid redundant downloads.
    
    Returns:
        str: Local path to the downloaded file.
    """
    try:
        s3_key = f"{S3_DATA_FOLDER}/ut_loss_history_1.csv"
        if not is_cache_valid(S3_BUCKET, s3_key, LOCAL_DATA_PATH):
            update_cache(S3_BUCKET, s3_key, LOCAL_DATA_PATH)
            logging.info(f"Downloaded and cached: {s3_key}")
        else:
            logging.info(f"Cache hit. Skipping download for: {s3_key}")
        return LOCAL_DATA_PATH
    except Exception as e:
        logging.error(f"Error in ingest_data_from_s3: {e}")
        raise
