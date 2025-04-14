#!/usr/bin/env python3

import boto3
import logging
import time

# Hard-coded S3 bucket and prefixes for GE artifacts
BUCKET_NAME = "grange-seniordesign-bucket"
PREFIXES = {
    "expectations": "great_expectations/expectations",
    "validations": "great_expectations/validations",
    "checkpoints": "great_expectations/checkpoints",
    "data_docs": "great_expectations/data_docs"
}

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Create a global S3 client so that it isn’t recreated in every function.
s3 = boto3.client('s3')


def list_prefix_objects(bucket_name, prefix, retries=3, delay=2):
    """
    Lists the objects under a given S3 prefix with retries.
    
    Args:
        bucket_name (str): The S3 bucket name.
        prefix (str): The S3 prefix to list.
        retries (int): Number of retry attempts.
        delay (int): Delay in seconds between retries.
        
    Returns:
        dict: The S3 list_objects_v2 response or None if it fails.
    """
    attempt = 0
    while attempt < retries:
        try:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            logging.info("List objects response for prefix '%s': %s", prefix, response)
            return response
        except Exception as e:
            logging.error("Error listing objects for prefix '%s' (attempt %d): %s", prefix, attempt + 1, e)
            attempt += 1
            time.sleep(delay)
    return None


def create_s3_prefix_if_not_exists(bucket_name, prefix, retries=3, delay=2):
    """
    Checks if a given S3 prefix exists in the bucket. If it does not, creates a zero-byte object
    to mimic the folder. Includes retry logic.
    
    Args:
        bucket_name (str): The S3 bucket name.
        prefix (str): The S3 prefix to check/create.
        retries (int): Number of retry attempts.
        delay (int): Delay in seconds between retries.
    """
    attempt = 0
    while attempt < retries:
        try:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)
            logging.info("List objects response for prefix '%s': %s", prefix, response)
            if 'Contents' not in response:
                s3.put_object(Bucket=bucket_name, Key=f"{prefix.rstrip('/')}/")
                logging.info("Created S3 prefix '%s' in bucket '%s'.", prefix, bucket_name)
            else:
                logging.info("S3 prefix '%s' already exists in bucket '%s'.", prefix, bucket_name)
            return
        except Exception as e:
            logging.error("Error creating/verifying prefix '%s' (attempt %d): %s", prefix, attempt + 1, e)
            attempt += 1
            time.sleep(delay)
    raise Exception(f"Failed to create or verify S3 prefix '{prefix}' in bucket '{bucket_name}' after {retries} attempts.")


def verify_prefixes(bucket_name, prefixes):
    """
    Verifies that the specified prefixes exist in the bucket by listing their contents.
    
    Args:
        bucket_name (str): The S3 bucket name.
        prefixes (dict): A dictionary of prefix names to S3 prefixes.
    """
    logging.info("Verifying prefixes in bucket '%s'...", bucket_name)
    all_verified = True
    for key, prefix in prefixes.items():
        response = list_prefix_objects(bucket_name, prefix)
        if response is None or 'Contents' not in response:
            logging.error("Prefix '%s' (%s) not found or is empty.", key, prefix)
            all_verified = False
        else:
            logging.info("Prefix '%s' (%s) verification successful. Contents: %s", key, prefix, response.get("Contents"))
    if all_verified:
        logging.info("All defined prefixes verified successfully.")
    else:
        logging.error("Some prefixes were not verified. Please check the errors above.")


def setup_ge_s3():
    """
    Sets up the required S3 folder structure for Great Expectations.
    Hard-coded to use the bucket "grange-seniordesign-bucket" and create/verify
    the following prefixes:
      - great_expectations/expectations
      - great_expectations/validations
      - great_expectations/checkpoints
      - great_expectations/data_docs
    """
    for key, prefix in PREFIXES.items():
        logging.info("Ensuring S3 prefix exists for '%s': %s", key, prefix)
        create_s3_prefix_if_not_exists(BUCKET_NAME, prefix)
    
    logging.info("All prefixes ensured. Now verifying prefixes...")
    verify_prefixes(BUCKET_NAME, PREFIXES)
    logging.info("S3 setup for Great Expectations completed successfully.")


if __name__ == "__main__":
    setup_ge_s3()
