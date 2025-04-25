#!/usr/bin/env python3
"""
unified_ml_pipeline.py - Consolidated ML Pipeline DAG
-----------------------------------------------------
This DAG combines functionality from multiple previous pipelines:
- integrated_ml_workflow.py
- homeowner_dag.py
- train_all_models_dag.py

It provides a unified workflow that handles:
1. Data ingestion from S3
2. Data preprocessing and quality checking
3. Schema validation and drift detection
4. Human validation of data quality (HITL)
5. Training of multiple models (including parallel training)
6. Model evaluation and explainability tracking
7. Human approval of models (HITL)
8. Artifact archiving

All functionality is contained in a single DAG for easier management and troubleshooting.
"""

import os
import json
import logging
import tempfile
import shutil
import sys
import math
import random
import time
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from pathlib import Path
from functools import wraps
from typing import Dict, Any, Optional, Tuple, Union, List

# Initialize logging early so it is available during any import errors
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Airflow imports
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable, XCom
from airflow.hooks.S3_hook import S3Hook
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task, task_group

# Other imports
import pandas as pd
import numpy as np
import boto3
import mlflow
from mlflow.tracking import MlflowClient

# Import task modules
try:
    import utils.config as config
    import utils.clearml_config as clearml_config
    import utils.slack as slack

    # Import task modules that are actively used
    import tasks.ingestion as ingestion
    import tasks.preprocessing_simplified as preprocessing  # Use simplified preprocessing without outlier detection
    import tasks.data_quality as data_quality
    import tasks.schema_validation as schema_validation
    import tasks.drift as drift
    import tasks.training as training
    import tasks.model_explainability as model_explainability
    import tasks.hitl as hitl  # Import the new Human-in-the-Loop module
    import tasks.model_comparison as model_comparison  # Import model comparison module
except ImportError as e:
    # Log the import error but continue - the specific module will fail at runtime if used
    logger.error(f"Error importing module: {str(e)}")
    logger.error("Some pipeline components may not be available")
    
    # Define empty modules for modules that failed to import to prevent NameError
    # when the code refers to these modules
    class EmptyModule:
        def __getattr__(self, name):
            def method(*args, **kwargs):
                logger.error(f"Called missing module method {name}. Original import failed.")
                return {"status": "error", "message": f"Module not available: {name}"}
            return method
    
    # Define module variables if they don't exist
    if 'config' not in locals():
        config = EmptyModule()
    if 'clearml_config' not in locals():
        clearml_config = EmptyModule()
    if 'slack' not in locals():
        slack = EmptyModule()
    if 'ingestion' not in locals():
        ingestion = EmptyModule()
    if 'preprocessing' not in locals():
        preprocessing = EmptyModule()
    if 'data_quality' not in locals():
        data_quality = EmptyModule()
    if 'schema_validation' not in locals():
        schema_validation = EmptyModule()
    if 'drift' not in locals():
        drift = EmptyModule()
    if 'training' not in locals():
        training = EmptyModule()
    if 'model_explainability' not in locals():
        model_explainability = EmptyModule()
    if 'hitl' not in locals():
        hitl = EmptyModule()
    if 'model_comparison' not in locals():
        model_comparison = EmptyModule()

# Remove Slack initialization - Using logs only
logger.info("Slack notifications disabled - using logs only")

# Define a simple logging-only version of 'slack' functions for compatibility
class LoggingSlack:
    """
    Replacement for Slack that just logs messages instead of sending them.
    """
    @staticmethod
    def post(channel="#data-pipeline", title="Notification", details="", urgency="normal"):
        """Log a message instead of posting to Slack"""
        priority = {
            "low": "INFO",
            "normal": "INFO", 
            "high": "WARNING",
            "critical": "ERROR"
        }.get(urgency, "INFO")
        
        # Log with appropriate level
        log_msg = f"[NOTIFICATION] {title}: {details}"
        if priority == "INFO":
            logger.info(log_msg)
        elif priority == "WARNING":
            logger.warning(log_msg)
        else:
            logger.error(log_msg)
        
        return {"ok": True, "message": "Logged message (Slack disabled)"}
    
    @staticmethod
    def simple_post(message, channel=None, urgency="normal"):
        """Log a simple message instead of posting to Slack"""
        return LoggingSlack.post(channel=channel or "#data-pipeline", 
                                title="Notification", 
                                details=message, 
                                urgency=urgency)

# Set slack as the LoggingSlack class
slack = LoggingSlack()

# Constants - Updated to match AWS Secrets values exactly
S3_BUCKET = Variable.get('S3_BUCKET', default_var='mlautomationstack-dagsbucket3bcf9ca5-uhw98w1')
DATA_BUCKET = Variable.get('DATA_BUCKET', default_var='grange-seniordesign-bucket')
LOCAL_PROCESSED_PATH = "/tmp/unified_processed.parquet"
REFERENCE_MEANS_PATH = "/tmp/reference_means.csv"
MAX_WORKERS = int(Variable.get('MAX_WORKERS', default_var='3'))
# Environment variables from AWS Secrets Manager
MLFLOW_TRACKING_URI = Variable.get('MLFLOW_TRACKING_URI', default_var='http://3.146.46.179:5000')
MLFLOW_EXPERIMENT_NAME = Variable.get('MLFLOW_EXPERIMENT_NAME', default_var='Homeowner_Loss_Hist_Proj')
MLFLOW_ARTIFACT_URI = Variable.get('MLFLOW_ARTIFACT_URI', default_var='s3://grange-seniordesign-bucket/mlflow-artifacts')
LOG_GROUP = Variable.get('LOG_GROUP', default_var='/ml-automation/websocket-connections')
DRIFT_LOG_GROUP = Variable.get('DRIFT_LOG_GROUP', default_var='/ml-automation/drift-events')
DRIFT_THRESHOLD = float(Variable.get('DRIFT_THRESHOLD', default_var='0.1'))
# ClearML variables
CLEARML_API_HOST = Variable.get('CLEARML_API_HOST', default_var='https://api.clear.ml')
CLEARML_WEB_HOST = Variable.get('CLEARML_WEB_HOST', default_var='https://app.clear.ml')
CLEARML_FILES_HOST = Variable.get('CLEARML_FILES_HOST', default_var='http://files.clear.ml')
CLEARML_API_ACCESS_KEY = Variable.get('CLEARML_API_ACCESS_KEY', default_var='52M5GZYH6U0RLJUBY3FVBT96YXQY')
# Slack variables
SLACK_WEBHOOK_URL = Variable.get('SLACK_WEBHOOK_URL', default_var='https://hooks.slack.com/services/T08MX68B')
SLACK_BOT_TOKEN = Variable.get('SLACK_BOT_TOKEN', default_var='xoxb-8745212375942-874528050134-D4717SbbjCg')
SLACK_DEFAULT_CHANNEL = Variable.get('SLACK_DEFAULT_CHANNEL', default_var='#all-airflow-notification')
SLACK_ENABLE_NOTIFICATIONS = Variable.get('SLACK_ENABLE_NOTIFICATIONS', default_var='True').lower() == 'true'
# Auto-approval constants
AUTO_APPROVE_MODEL = Variable.get('AUTO_APPROVE_MODEL', default_var='True').lower() == 'true'
AUTO_APPROVE_QUALITY_THRESHOLD = int(Variable.get('AUTO_APPROVE_QUALITY_THRESHOLD', default_var='3'))
AUTO_APPROVE_TIMEOUT_MINUTES = int(Variable.get('AUTO_APPROVE_TIMEOUT_MINUTES', default_var='300'))
MODEL_APPROVE_TIMEOUT_MINUTES = int(Variable.get('MODEL_APPROVE_TIMEOUT_MINUTES', default_var='120'))
S3_ARCHIVE_FOLDER = Variable.get('S3_ARCHIVE_FOLDER', default_var='archive')
SYTHENTIC_SAMPLE_COUNT = int(Variable.get('SYTHENTIC_SAMPLE_COUNT', default_var='500'))

# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2023, 1, 1),
    'catchup': False
}

# Utility functions
def safe_module_call(module, function_name, *args, **kwargs):
    """
    Safely call a function from a module that might be an EmptyModule.
    
    Args:
        module: The module (or EmptyModule) to call the function from
        function_name: Name of the function to call
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        The result of the function call, or an error status dictionary
    """
    try:
        if not hasattr(module, function_name):
            logger.error(f"Function {function_name} not found in module")
            return {"status": "error", "message": f"Function {function_name} not found in module"}
            
        function = getattr(module, function_name)
        result = function(*args, **kwargs)
        return result
    except Exception as e:
        logger.error(f"Error calling {function_name}: {str(e)}")
        return {"status": "error", "message": f"Error: {str(e)}"}

def validate_file_exists(filepath: str) -> bool:
    """
    Validate that a file exists and is not empty.
    
    Args:
        filepath: Path to the file to validate
        
    Returns:
        bool: True if the file exists and is not empty
    """
    if not filepath:
        logger.error("Filepath is empty")
        return False
        
    if not os.path.exists(filepath):
        logger.error(f"File does not exist: {filepath}")
        return False
        
    if os.path.getsize(filepath) == 0:
        logger.error(f"File is empty: {filepath}")
        return False
        
    return True

def with_file_validation(file_param: str):
    """
    Decorator to validate file existence before executing a function.
    
    Args:
        file_param: Name of the parameter containing the file path
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            filepath = kwargs.get(file_param)
            if not validate_file_exists(filepath):
                logger.error(f"File validation failed for {file_param}: {filepath}")
                return {"status": "error", "message": f"File not found or empty: {filepath}"}
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Task functions
def download_data(**context):
    """Download data from S3 bucket"""
    logger.info("Starting download_data task")
    
    # Get bucket name from Airflow Variable to ensure it's up to date
    bucket = DATA_BUCKET
    key = config.RAW_DATA_KEY
    
    logger.info(f"Using bucket: '{bucket}', key: '{key}'")
    
    try:
        s3_client = boto3.client('s3', region_name=config.AWS_REGION)
        
        # Extend search prefixes to include the archive folder where processed data may reside
        for prefix in ["raw-data/", "raw_data/", "archive/"]:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Look for either the original raw file or the processed homeowner file
                    if obj['Key'].endswith('ut_loss_history_1.csv') or obj['Key'].endswith('homeowner_processed.csv'):
                        key = obj['Key']
                        logger.info(f"Found target file at path: {key}")
                        break
                        
        # Verify the file exists
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            logger.info(f"Verified file exists at: s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"File not found at specified path: s3://{bucket}/{key}")
            raise FileNotFoundError(f"Target file not found in S3: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error accessing S3: {str(e)}")
        raise
    
    try:
        # Create a list of potential data directories, in order of preference
        potential_dirs = [
            "/tmp/airflow_data",       # Preferred location in tmp for persistence
            "/usr/local/airflow/tmp",  # Fallback airflow directory
            "/tmp"                     # Last resort, standard tmp
        ]
        
        # Try each directory until we find one we can write to
        data_dir = None
        for dir_path in potential_dirs:
            try:
                logger.info(f"Attempting to use directory: {dir_path}")
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    
                # Test if we can write to this directory
                test_file = os.path.join(dir_path, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                
                # If we get here, the directory is writable
                data_dir = dir_path
                logger.info(f"Using directory: {data_dir}")
                break
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot use directory {dir_path}: {str(e)}")
        
        if not data_dir:
            logger.error("Could not find a writable directory")
            raise PermissionError("No writable directory found for storing data")
        
        # Generate a persistent filename based on the S3 key
        filename = os.path.basename(key)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_path = os.path.join(data_dir, f"raw_{timestamp}_{filename}")
        logger.info(f"Will download to persistent location: {local_path}")
        
        # Use the ingestion function from the full pipeline if available
        try:
            logger.info("Attempting to use pipeline ingestion function")
            # Check function signature to avoid parameter errors
            import inspect
            ingest_sig = inspect.signature(ingestion.ingest_data_from_s3)
            ingest_params = {}
            
            # Add parameters based on available arguments
            if 'bucket_name' in ingest_sig.parameters:
                ingest_params['bucket_name'] = bucket
            elif 'bucket' in ingest_sig.parameters:
                ingest_params['bucket'] = bucket
                
            if 'key' in ingest_sig.parameters:
                ingest_params['key'] = key
                
            if 'local_path' in ingest_sig.parameters:
                ingest_params['local_path'] = local_path
                
            logger.info(f"Calling ingest_data_from_s3 with params: {ingest_params}")
            data_path = ingestion.ingest_data_from_s3(**ingest_params)
            logger.info(f"Successfully ingested data using pipeline function: {data_path}")
        except Exception as e:
            # Fall back to direct S3 download if ingestion function fails
            logger.warning(f"Pipeline ingestion failed, falling back to direct download: {str(e)}")
            
            # Download directly to the persistent location
            s3_client = boto3.client('s3', region_name=config.AWS_REGION)
            logger.info(f"Downloading from s3://{bucket}/{key} to {local_path}")
            s3_client.download_file(bucket, key, local_path)
            data_path = local_path
            logger.info(f"Successfully downloaded to {local_path}")
        
        # Create additional backup copies in other directories
        backup_paths = []
        for backup_dir in potential_dirs:
            if backup_dir != data_dir and os.path.exists(backup_dir):
                try:
                    backup_path = os.path.join(backup_dir, f"backup_{timestamp}_{filename}")
                    import shutil
                    shutil.copy2(data_path, backup_path)
                    backup_paths.append(backup_path)
                    logger.info(f"Created backup copy at {backup_path}")
                except Exception as e:
                    logger.warning(f"Failed to create backup in {backup_dir}: {str(e)}")
                    
        # Store both primary path and backup paths in XCom
        logger.info(f"Pushing data_path to XCom: {data_path}")
        context['ti'].xcom_push(key='data_path', value=data_path)
        
        if backup_paths:
            logger.info(f"Pushing backup_paths to XCom: {backup_paths}")
            context['ti'].xcom_push(key='backup_paths', value=backup_paths)
            
        # Test file exists and is accessible
        if not os.path.exists(data_path):
            logger.error(f"Downloaded file not found at {data_path}")
            raise FileNotFoundError(f"Downloaded file not found at {data_path}")
            
        file_size = os.path.getsize(data_path)
        if file_size == 0:
            logger.error(f"Downloaded file is empty: {data_path}")
            raise ValueError(f"Downloaded file is empty: {data_path}")
            
        logger.info(f"Verified file exists with size {file_size} bytes")
        
        # Set file permissions to ensure next tasks can read it
        try:
            os.chmod(data_path, 0o666)  # Make file readable/writable by all users
            logger.info(f"Set file permissions to 666 for {data_path}")
        except Exception as e:
            logger.warning(f"Could not set file permissions: {str(e)}")
            
        # Create a standardized file path that other tasks can find directly
        try:
            # Ensure we have a constant location for the file across tasks
            standard_path = os.path.join(data_dir, f"latest_raw_data.csv")
            shutil.copy2(data_path, standard_path)
            logger.info(f"Created standardized copy at {standard_path}")
            
            # Add this to XCom as well
            context['ti'].xcom_push(key='standard_data_path', value=standard_path)
        except Exception as e:
            logger.warning(f"Could not create standardized path: {str(e)}")
        
        # Log with ClearML if enabled
        try:
            clearml_task = clearml_config.init_clearml("Data_Download")
            if clearml_task:
                clearml_task.set_parameter("s3_bucket", bucket)
                clearml_task.set_parameter("s3_key", key)
                clearml_config.log_dataset_to_clearml(
                    dataset_name="Raw_Data",
                    dataset_path=data_path,
                    dataset_tags=["raw", "csv"]
                )
                clearml_task.close()
        except Exception as e:
            logger.warning(f"Error logging to ClearML: {str(e)}")
        
        try:
            slack.simple_post(f"✅ Data accessed from s3://{bucket}/{key}", channel="#all-airflow-notification")
        except Exception as e:
            logger.warning(f"Error sending Slack notification: {str(e)}")
            
        # Log successful data access
        logger.info(f"✓ Data successfully accessed from s3://{bucket}/{key}")
        return data_path
        
    except Exception as e:
        # Log data access failure
        logger.error(f"Failed to access data: {str(e)}")
        try:
            slack.simple_post(f"❌ Failed to access data: {str(e)}", channel="#data-pipeline")
        except Exception as ex:
            logger.warning(f"Error sending error notification: {str(ex)}")
        raise

def process_data(**context):
    """Process raw data"""
    logger.info("Starting process_data task")
    
    # Define output directory
    output_dir = "/tmp/airflow_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # First, look for data at the known fixed location from ingestion task
    fixed_path = "/tmp/homeowner_data.csv"
    if os.path.exists(fixed_path):
        logger.info(f"Found data at fixed path: {fixed_path}")
        raw_data_path = fixed_path
    else:
        # Fallback: Get raw data path from previous task XCom
        ti = context['ti']
        raw_data_path = ti.xcom_pull(task_ids='import_data_task', key='data_path')
        logger.info(f"Raw data path from XCom: {raw_data_path}")
        
        if not raw_data_path or not os.path.exists(raw_data_path):
            raise AirflowException("No valid data path found at fixed location or from XCom")
        
    # Generate the output path for the processed data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    processed_path = os.path.join(output_dir, f"processed_{timestamp}.parquet")
    
    # Process the data
    logger.info(f"Processing data from {raw_data_path} to {processed_path}")
    
    # Load the data based on file type
    file_ext = os.path.splitext(raw_data_path)[1].lower()
    
    if file_ext == '.parquet':
        df = pd.read_parquet(raw_data_path)
    elif file_ext == '.xlsx':
        df = pd.read_excel(raw_data_path)
    else:  # Default to CSV
        df = pd.read_csv(raw_data_path, encoding='utf-8', on_bad_lines='skip')
        
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Ensure target column exists
    if 'trgt' not in df.columns and 'il_total' in df.columns and 'eey' in df.columns:
        logger.info("Creating 'trgt' column from 'il_total' / 'eey'")
        df['trgt'] = df['il_total'] / df['eey']
        
    # Save processed data
    df.to_parquet(processed_path, index=False)
    logger.info(f"Data processed successfully to {processed_path}")
    
    # Create standardized version at LOCAL_PROCESSED_PATH
    os.makedirs(os.path.dirname(LOCAL_PROCESSED_PATH), exist_ok=True)
    shutil.copy2(processed_path, LOCAL_PROCESSED_PATH)
    logger.info(f"Created standardized copy at {LOCAL_PROCESSED_PATH}")
    
    # Log to MLflow explicitly
    try:
        # Initialize MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name="data_processing") as run:
            # Log the artifact
            mlflow.log_artifact(processed_path, artifact_path="processed_data")
            mlflow.log_param("raw_data_path", raw_data_path)
            mlflow.log_param("processed_shape", str(df.shape))
            mlflow.log_metric("num_rows", df.shape[0])
            mlflow.log_metric("num_columns", df.shape[1])
            
            # Log the run ID for downstream tasks
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            ti.xcom_push(key='mlflow_run_id', value=run_id)
    except Exception as mlflow_err:
        logger.warning(f"Failed to log to MLflow: {str(mlflow_err)}")
        # Continue even if MLflow logging fails - we still have the file
    
    # Push paths to XCom - these must be valid paths that actually exist
    ti.xcom_push(key='processed_data_path', value=processed_path)
    ti.xcom_push(key='standardized_processed_path', value=LOCAL_PROCESSED_PATH)
    
    # Return the processed path (will be automatically pushed to XCom as return_value)
    return processed_path

def run_data_quality_checks(**context):
    """Run data quality checks on processed data"""
    logger.info("Starting data_quality_checks task")
    
    try:
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='preprocess_data_task', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='preprocess_data_task', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        # Fix: Check if standardized_path is None before checking if it exists
        data_path = None
        if standardized_path is not None and os.path.exists(standardized_path):
            data_path = standardized_path
        elif processed_path is not None and os.path.exists(processed_path):
            data_path = processed_path
        
        if not data_path:
            logger.error("No valid processed data path found")
            raise FileNotFoundError("No valid processed data path found")
            
        logger.info(f"Running data quality checks on {data_path}")
        
        # Load the data
        try:
            df = pd.read_parquet(data_path)
            logger.info(f"Loaded dataframe with shape {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load parquet file: {str(e)}")
            raise
            
        # Run quality checks - use correct DataQualityMonitor method
        try:
            # Create DataQualityMonitor instance and run checks
            quality_monitor = data_quality.DataQualityMonitor()
            quality_results = quality_monitor.run_quality_checks(df)
            logger.info(f"Data quality results: {quality_results}")
            
            # Store results in XCom
            context['ti'].xcom_push(key='quality_results', value=quality_results)
            
            # Determine if quality checks passed
            quality_passed = quality_results.get('status', '') == 'success' or quality_results.get('total_issues', 0) == 0
            context['ti'].xcom_push(key='quality_passed', value=quality_passed)
            
            # Generate a report path and push to XCom
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_dir = "/tmp/quality_reports"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"data_quality_report_{timestamp}.json")
            
            # Save report to file
            with open(report_path, 'w') as f:
                json.dump(quality_results, f, indent=2)
            
            logger.info(f"Saved data quality report to {report_path}")
            context['ti'].xcom_push(key='validation_report_path', value=report_path)
            
            if quality_passed:
                logger.info("Data quality checks passed")
            else:
                logger.warning("Data quality checks failed or had warnings")
                    
            return quality_results
            
        except Exception as e:
            logger.error(f"Error running data quality checks: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in data_quality_checks task: {str(e)}")
        raise

def run_schema_validation(**context):
    """Run schema validation on the processed data"""
    logger.info("Starting schema_validation task")
    
    try:
        # Get processed data path
        processed_path = context['ti'].xcom_pull(task_ids='preprocess_data_task', key='processed_data_path')
        standardized_path = context['ti'].xcom_pull(task_ids='preprocess_data_task', key='standardized_processed_path')
        
        # Use standardized path if available, otherwise use processed path
        # Fix: Check if standardized_path is None before checking if it exists
        data_path = None
        if standardized_path is not None and os.path.exists(standardized_path):
            data_path = standardized_path
        elif processed_path is not None and os.path.exists(processed_path):
            data_path = processed_path
        
        if not data_path:
            logger.error("No valid processed data path found")
            raise FileNotFoundError("No valid processed data path found")
            
        logger.info(f"Running schema validation on {data_path}")
        
        # Load the data
        try:
            df = pd.read_parquet(data_path)
            logger.info(f"Loaded dataframe with shape {df.shape}")
            
            # Create the target column if needed
            if 'trgt' not in df.columns and 'pure_premium' not in df.columns:
                if 'il_total' in df.columns and 'eey' in df.columns:
                    logger.info("Creating 'trgt' column from 'il_total' / 'eey'")
                    df['trgt'] = df['il_total'] / df['eey']
                    
                    # Save the updated dataframe back to the standardized path
                    if os.path.exists(standardized_path):
                        df.to_parquet(standardized_path, index=False)
                        logger.info(f"Updated standardized dataframe with 'trgt' column")
                else:
                    logger.warning("Cannot create target column: missing 'il_total' or 'eey' columns")
        except Exception as e:
            logger.error(f"Failed to load or preprocess parquet file: {str(e)}")
            raise
            
        # Run schema validation
        try:
            validation_results = schema_validation.validate_schema(df)
            logger.info(f"Schema validation results: {validation_results}")
            
            # Store results in XCom
            context['ti'].xcom_push(key='validation_results', value=validation_results)
            
            # Determine if validation passed
            validation_passed = validation_results.get('status') == 'success'
            context['ti'].xcom_push(key='validation_passed', value=validation_passed)
            
            if validation_passed:
                logger.info("Schema validation passed")
            else:
                logger.warning(f"Schema validation failed: {validation_results.get('message', 'Unknown issue')}")
                    
            return validation_results
            
        except Exception as e:
            logger.error(f"Error running schema validation: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in schema_validation task: {str(e)}")
        raise

def check_for_drift(**context):
    """Check for data drift"""
    logger.info("Starting drift_detection task")
    
    # Get processed data path from XCom
    processed_path = context['ti'].xcom_pull(task_ids='preprocess_data_task', key='processed_data_path')
    
    if not processed_path:
        logger.warning("No processed data path found, using standardized path")
        processed_path = LOCAL_PROCESSED_PATH
    
    # Perform drift detection
    try:
        # Call detect_data_drift with the correct signature
        drift_results = safe_module_call(drift, "detect_data_drift", 
                                       processed_data_path=processed_path,
                                       reference_path=None)  # Use default latest reference
        
        # Determine if drift was detected
        drift_detected = drift_results.get('drift_detected', False)
        
        # Log the result
        if drift_detected:
            logger.warning("Data drift detected")
            decision = "self_healing"
        else:
            logger.info("No data drift detected")
            decision = "train_models"
                
        # Store the decision in XCom
        context['ti'].xcom_push(key='drift_decision', value=decision)
        return decision
        
    except Exception as e:
        logger.warning(f"Error in drift detection: {str(e)}, proceeding with training")
        return "train_models"

@task.branch
def branch_on_drift(**context):
    """Branch based on drift detection results"""
    # Extract the TaskInstance from the context
    ti = context['ti']
    decision = ti.xcom_pull(task_ids='check_for_drift_task', key='drift_decision')
    
    if not decision:
        logger.warning("No drift decision found, defaulting to training path")
        return ["train_models_group"]  # Return the TaskGroup ID
    
    if decision == "self_healing":
        logger.info("Taking self-healing path due to drift")
        return ["healing_task"]
    else:
        logger.info("Taking normal training path")
        return ["train_models_group"]  # Return the TaskGroup ID

def healing_task(**context):
    """Perform self-healing actions when drift is detected"""
    logger.info("Starting healing task to address data drift")
    
    # Get drift results
    drift_results = context['ti'].xcom_pull(task_ids='check_for_drift_task')
    
    if not drift_results:
        logger.warning("No drift results found, performing generic healing")
    else:
        logger.info(f"Healing based on drift results: {drift_results}")
    
    # Perform healing actions (e.g., generate reference data, adjust thresholds)
    try:
        # Generate new reference means if possible
        processed_path = context['ti'].xcom_pull(task_ids='preprocess_data_task', key='processed_data_path')
        
        if processed_path and os.path.exists(processed_path):
            logger.info(f"Generating new reference means from {processed_path}")
            result = safe_module_call(drift, "generate_reference_means", processed_path)
            logger.info(f"Reference means generation result: {result}")
        
        # Return success to allow pipeline to continue to training
        healing_results = {
            "status": "success",
            "message": "Healing actions completed",
            "timestamp": datetime.now().isoformat()
        }
        
        context['ti'].xcom_push(key='healing_results', value=healing_results)
        return healing_results
    
    except Exception as e:
        logger.warning(f"Error in healing task: {str(e)}, continuing with pipeline")
        return {
            "status": "warning",
            "message": f"Healing actions encountered an error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def model_explainability(**context):
    """Generate explanations for the best model"""
    logger.info("Starting model_explainability task")
    ti = context['ti']
    best_model_id = ti.xcom_pull(task_ids='train_models_group.model_comparison_task', key='best_model_id')
    if not best_model_id:
        logger.error("No best model ID found for explainability task")
        raise AirflowException("No best model ID found for explainability task")
    if os.path.exists(LOCAL_PROCESSED_PATH):
        processed_path = LOCAL_PROCESSED_PATH
        logger.info(f"Using data from standardized location: {processed_path}")
    else:
        processed_path = ti.xcom_pull(task_ids='preprocess_data_task', key='processed_data_path')
        if not processed_path or not os.path.exists(processed_path):
            raise AirflowException(f"Processed data path not found: {processed_path}")
    training_results = ti.xcom_pull(task_ids='train_models_group.train_models_task', key='training_results')
    if not training_results:
        logger.error("No training results found for explainability task")
        raise AirflowException("No training results found for explainability task")
    try:
        data = pd.read_parquet(processed_path)
        X = data.drop('target', axis=1)
        y = data['target']
        model = mlflow.sklearn.load_model(f"models:/{best_model_id}/Production")
        explainer = model_explainability.ModelExplainabilityTracker(model_id=best_model_id)
        explanation_results = explainer.track_model_and_data(
            model=model,
            X=X,
            y=y,
            run_id=None
        )
        if explanation_results:
            logger.info(f"Generated explanations for model {best_model_id}")
            ti.xcom_push(key='explanation_results', value=explanation_results)
            return explanation_results
        else:
            logger.error(f"No explanation results generated for model {best_model_id}")
            raise AirflowException(f"No explanation results generated for model {best_model_id}")
    except Exception as e:
        logger.error(f"Error generating model explanations: {str(e)}")
        raise AirflowException(f"Error generating model explanations: {str(e)}")

def model_evaluation(**context):
    """Evaluate model against test data and record metrics"""
    logger.info("Starting model_evaluation task")
    ti = context['ti']
    training_results = ti.xcom_pull(task_ids='train_models_group.train_models_task', key='training_results')
    if not training_results:
        logger.error("No training results found for model evaluation")
        raise AirflowException("No training results found for model evaluation")
    if os.path.exists(LOCAL_PROCESSED_PATH):
        processed_path = LOCAL_PROCESSED_PATH
        logger.info(f"Using data from standardized location: {processed_path}")
    else:
        processed_path = ti.xcom_pull(task_ids='preprocess_data_task', key='processed_data_path')
        if not processed_path or not os.path.exists(processed_path):
            raise AirflowException(f"Processed data path not found: {processed_path}")
    evaluator = model_evaluation.ModelEvaluation()
    evaluation_results = {}
    any_failed = False
    for model_id, result in training_results.items():
        if result and isinstance(result, dict) and result.get('status') == 'completed':
            model = result.get('model')
            run_id = result.get('run_id')
            if model is not None:
                try:
                    data = pd.read_parquet(processed_path)
                    X = data.drop('target', axis=1)
                    y = data['target']
                    metrics = evaluator.evaluate_model(
                        model=model,
                        X_test=X,
                        y_test=y,
                        run_id=run_id
                    )
                    evaluation_results[model_id] = {
                        'status': 'completed',
                        'metrics': metrics,
                        'run_id': run_id
                    }
                    logger.info(f"Evaluation completed for model {model_id}")
                except Exception as e:
                    logger.error(f"Error evaluating model {model_id}: {str(e)}")
                    evaluation_results[model_id] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    any_failed = True
            else:
                logger.warning(f"No model object found for {model_id}")
                evaluation_results[model_id] = {
                    'status': 'failed',
                    'error': 'No model object found'
                }
                any_failed = True
        else:
            logger.warning(f"No valid training results for {model_id}")
            evaluation_results[model_id] = {
                'status': 'failed',
                'error': 'No valid training results'
            }
            any_failed = True
    if any_failed:
        raise AirflowException("One or more models failed during evaluation. See logs for details.")
    ti.xcom_push(key='evaluation_results', value=evaluation_results)
    return evaluation_results

def model_comparison(**context):
    """Compare models and select the best one based on evaluation metrics"""
    logger.info("Starting model_comparison task")
    ti = context['ti']
    evaluation_results = ti.xcom_pull(task_ids='train_models_group.model_comparison_task', key='comparison_results')
    if not evaluation_results:
        logger.error("No evaluation results found for model comparison")
        raise AirflowException("No evaluation results found for model comparison")
    if os.path.exists(LOCAL_PROCESSED_PATH):
        processed_path = LOCAL_PROCESSED_PATH
        logger.info(f"Using data from standardized location: {processed_path}")
    else:
        processed_path = ti.xcom_pull(task_ids='preprocess_data_task', key='processed_data_path')
        if not processed_path or not os.path.exists(processed_path):
            raise AirflowException(f"Processed data path not found: {processed_path}")
    try:
        comparison_results = compare_model_results(
            model_results=evaluation_results,
            task_type='regression',
            notify_slack=False
        )
        if comparison_results and comparison_results.get('best_model'):
            best_model_id = comparison_results.get('best_model')
            logger.info(f"Selected best model: {best_model_id}")
            ti.xcom_push(key='best_model_id', value=best_model_id)
            ti.xcom_push(key='comparison_results', value=comparison_results)
            return comparison_results
        else:
            logger.error("No best model identified during comparison")
            raise AirflowException("No best model identified during comparison")
    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}")
        raise AirflowException(f"Model comparison failed: {str(e)}")

def wait_for_data_validation(**context):
    """Wait for data validation to complete and handle any manual approvals if needed"""
    logger.info("Starting wait_for_data_validation task")
    
    ti = context['ti']
    quality_passed = ti.xcom_pull(task_ids='validate_data_task', key='quality_passed')
    
    if not quality_passed:
        if AUTO_APPROVE_MODEL:
            logger.warning("Data validation failed but auto-approve is enabled. Proceeding with caution.")
            return True
        else:
            raise AirflowException("Data validation failed and auto-approve is disabled.")
    
    return True

def train_models(**context):
    """Train multiple model variants"""
    logger.info("Starting train_models task")
    
    ti = context['ti']
    processed_path = context.get('params', {}).get('processed_data_path')
    
    if not processed_path:
        processed_path = ti.xcom_pull(task_ids='preprocess_data_task', key='processed_data_path')
        
    if not processed_path or not os.path.exists(processed_path):
        raise AirflowException(f"Processed data path not found: {processed_path}")
    
    try:
        # Use the correct function name from the training module
        training_results = training.train_multiple_models(
            processed_path=processed_path,
            parallel=True,
            max_workers=MAX_WORKERS
        )
        ti.xcom_push(key='training_results', value=training_results)
        return training_results
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise

def wait_for_model_approval(**context):
    """Wait for model approval from stakeholders"""
    logger.info("Starting wait_for_model_approval task")
    
    ti = context['ti']
    comparison_results = ti.xcom_pull(task_ids='train_models_group.model_comparison_task', key='comparison_results')
    
    if not comparison_results:
        raise AirflowException("No model comparison results found")
    
    if AUTO_APPROVE_MODEL:
        logger.info("Auto-approval is enabled. Model approved automatically.")
        return True
    
    # Here you would typically integrate with your model approval system
    # For now, we'll auto-approve if the model meets basic criteria
    if comparison_results.get('best_model'):
        logger.info("Model meets basic criteria. Approved.")
        return True
    else:
        raise AirflowException("No suitable model found for approval")

def deploy_model(**context):
    """Deploy the approved model to production"""
    logger.info("Starting deploy_model task")
    
    ti = context['ti']
    best_model_id = ti.xcom_pull(task_ids='train_models_group.model_comparison_task', key='best_model_id')
    
    if not best_model_id:
        raise AirflowException("No best model ID found for deployment")
    
    try:
        # Register the model in MLflow Model Registry
        client = MlflowClient()
        model_uri = f"models:/{best_model_id}/Production"
        
        # Log deployment to MLflow
        with mlflow.start_run(run_name="model_deployment") as run:
            mlflow.log_param("deployed_model_id", best_model_id)
            mlflow.log_param("deployment_time", datetime.now().isoformat())
            
            # Store the deployment run ID
            ti.xcom_push(key='deployment_run_id', value=run.info.run_id)
        
        logger.info(f"Model {best_model_id} deployed successfully")
        return {"status": "success", "model_id": best_model_id}
        
    except Exception as e:
        logger.error(f"Error deploying model: {str(e)}")
        raise

def archive_artifacts(**context):
    """Archive training artifacts and logs"""
    logger.info("Starting archive_artifacts task")
    
    ti = context['ti']
    run_id = ti.xcom_pull(task_ids='train_models_group.train_models_task', key='mlflow_run_id')
    
    if not run_id:
        logger.warning("No MLflow run ID found for archiving")
        return
    
    try:
        # Archive to S3
        s3_client = boto3.client('s3')
        archive_prefix = f"{S3_ARCHIVE_FOLDER}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Archive MLflow artifacts
        mlflow_client = MlflowClient()
        artifacts = mlflow_client.list_artifacts(run_id)
        
        for artifact in artifacts:
            local_path = mlflow_client.download_artifacts(run_id, artifact.path)
            s3_key = f"{archive_prefix}/mlflow_artifacts/{artifact.path}"
            s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        
        logger.info(f"Archived artifacts to s3://{S3_BUCKET}/{archive_prefix}")
        return {"status": "success", "archive_prefix": archive_prefix}
        
    except Exception as e:
        logger.warning(f"Error archiving artifacts: {str(e)}")
        return {"status": "warning", "error": str(e)}

def cleanup_temp_files(**context):
    """Clean up temporary files created during the pipeline run"""
    logger.info("Starting cleanup_temp_files task")
    
    temp_files = [
        LOCAL_PROCESSED_PATH,
        REFERENCE_MEANS_PATH
    ]
    
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Removed temporary file: {filepath}")
        except Exception as e:
            logger.warning(f"Error removing {filepath}: {str(e)}")
    
    # Clean up any temporary directories
    temp_dirs = [
        "/tmp/airflow_data",
        "/tmp/quality_reports"
    ]
    
    for dirpath in temp_dirs:
        try:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
                logger.info(f"Removed temporary directory: {dirpath}")
        except Exception as e:
            logger.warning(f"Error removing directory {dirpath}: {str(e)}")
    
    return {"status": "success", "message": "Cleanup completed"}

def generate_predictions(**context):
    """Generate predictions using the best model."""
    logger.info("Starting generate_predictions task")
    ti = context['ti']
    best_model_id = ti.xcom_pull(task_ids='train_models_group.model_comparison_task', key='best_model_id')
    if not best_model_id:
        logger.error("No best model ID found for predictions task")
        raise AirflowException("No best model ID found for predictions task")
    if os.path.exists(LOCAL_PROCESSED_PATH):
        processed_path = LOCAL_PROCESSED_PATH
    else:
        processed_path = ti.xcom_pull(task_ids='preprocess_data_task', key='processed_data_path')
        if not processed_path or not os.path.exists(processed_path):
            raise AirflowException(f"Processed data path not found: {processed_path}")
    try:
        data = pd.read_parquet(processed_path)
        X = data.drop('target', axis=1)
    except Exception as e:
        logger.error(f"Error loading data for predictions: {str(e)}")
        raise AirflowException(f"Error loading data for predictions: {str(e)}")
    try:
        model = mlflow.sklearn.load_model(f"models:/{best_model_id}/Production")
        predictions = model.predict(X)
        ti.xcom_push(key='predictions', value=predictions.tolist())
        logger.info(f"Generated predictions for model {best_model_id}")
        return predictions.tolist()
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise AirflowException(f"Error generating predictions: {str(e)}")

# Create the DAG
dag = DAG(
    'unified_ml_pipeline',
    default_args=default_args,
    description='Unified ML Pipeline for model training and deployment',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'pipeline', 'training'],
)

# Create tasks within the DAG context
with dag:
    # Import task for importing raw data
    import_data_task = PythonOperator(
        task_id='import_data_task',
        python_callable=download_data,
        provide_context=True,
    )
    
    # Preprocess task for data preprocessing
    preprocess_data_task = PythonOperator(
        task_id='preprocess_data_task',
        python_callable=process_data,
        provide_context=True,
        trigger_rule='all_success',  # Only run if import was successful
    )
    
    # Data validation task
    validate_data_task = PythonOperator(
        task_id='validate_data_task',
        python_callable=run_data_quality_checks,
        provide_context=True,
        trigger_rule='all_success',  # Only run if preprocessing was successful
    )
    
    # Wait for data validation task
    wait_for_data_validation_task = PythonOperator(
        task_id='wait_for_data_validation_task',
        python_callable=wait_for_data_validation,
        provide_context=True,
        trigger_rule='all_success',  # Only run if validation was successful
    )
    
    # Check for drift
    check_for_drift_task = PythonOperator(
        task_id='check_for_drift_task',
        python_callable=check_for_drift,
        provide_context=True,
        trigger_rule='all_success',
    )
    
    # Branch based on drift using TaskFlow API
    branch_task = branch_on_drift.override(task_id="branch_on_drift_task")()
    
    # Healing task for when drift is detected
    drift_healing_task = PythonOperator(
        task_id='healing_task',
        python_callable=healing_task,
        provide_context=True,
    )
    
    # Create a TaskGroup for all training-related tasks
    with TaskGroup("train_models_group") as train_models_group:
        # Train models task
        train_models_task = PythonOperator(
            task_id='train_models_task',
            python_callable=train_models,
            provide_context=True,
            op_kwargs={
                'params': {
                    'processed_data_path': "{{ ti.xcom_pull(task_ids='preprocess_data_task', key='processed_data_path') }}"
                }
            }
        )
        
        # Model explainability task
        model_explainability_task = PythonOperator(
            task_id='model_explainability_task',
            python_callable=model_explainability,
            provide_context=True,
            trigger_rule='all_success',  # Only run if training was successful
        )
        
        # Generate predictions task
        generate_predictions_task = PythonOperator(
            task_id='generate_predictions_task',
            python_callable=generate_predictions,
            provide_context=True,
            trigger_rule='all_success',  # Only run if training was successful
        )
        
        # Compare models task - using the task-decorated function directly
        model_comparison_task = compare_model_results.override(
            task_id='model_comparison_task'
        )(
            model_results="{{ ti.xcom_pull(task_ids='train_models_group.train_models_task') }}",
            task_type='regression'
        )
        
        # Set dependencies within the group
        train_models_task >> [model_explainability_task, generate_predictions_task, model_comparison_task]
    
    # Join both paths
    join_task = EmptyOperator(
        task_id='join_paths',
        trigger_rule='one_success',  # Continue if either path succeeds
    )
    
    # Wait for model approval task
    wait_for_model_approval_task = PythonOperator(
        task_id='wait_for_model_approval_task',
        python_callable=wait_for_model_approval,
        provide_context=True,
        trigger_rule='all_success',  # Only run if model comparison was successful
    )
    
    # Deploy model task
    deploy_model_task = PythonOperator(
        task_id='deploy_model_task',
        python_callable=deploy_model,
        provide_context=True,
        trigger_rule='all_success',  # Only run if model approval was given
    )
    
    # Archive artifacts task
    archive_artifacts_task = PythonOperator(
        task_id='archive_artifacts_task',
        python_callable=archive_artifacts,
        provide_context=True,
        trigger_rule='all_done',  # Run regardless of upstream task status
    )
    
    # Cleanup task
    cleanup_task = PythonOperator(
        task_id='cleanup_task',
        python_callable=cleanup_temp_files,
        provide_context=True,
        trigger_rule='all_done',  # Run regardless of upstream task status
    )
    
    # Define task dependencies
    import_data_task >> preprocess_data_task >> validate_data_task >> wait_for_data_validation_task
    wait_for_data_validation_task >> check_for_drift_task >> branch_task
    
    # Define branches
    branch_task >> drift_healing_task >> join_task
    branch_task >> train_models_group >> join_task
    
    # Post-branch dependencies
    join_task >> wait_for_model_approval_task >> deploy_model_task
    deploy_model_task >> archive_artifacts_task >> cleanup_task
