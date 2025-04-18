#!/usr/bin/env python3
"""
utils/airflow_api.py

Helper for triggering Airflow DAG runs via the Airflow REST API,
with retry logic for resilience against transient failures.
"""

from dotenv import load_dotenv
load_dotenv()   # <-- so AIRFLOW_API_URL, AIRFLOW_USERNAME, AIRFLOW_PASSWORD get picked up

import os
import requests
from tenacity import retry, wait_exponential, stop_after_attempt

AIRFLOW_API_URL  = os.getenv("AIRFLOW_API_URL", "http://localhost:8080")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "")

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(5),
)
def trigger_dag(dag_id: str, conf: dict = None) -> dict:
    """
    Trigger an Airflow DAG run via the REST API.

    Args:
        dag_id (str): The DAG ID to trigger.
        conf (dict, optional): A dict of configuration parameters to pass to the DAG run.

    Returns:
        dict: The JSON response from the Airflow API.

    Raises:
        requests.HTTPError: If the HTTP request fails even after retries.
    """
    endpoint = f"{AIRFLOW_API_URL}/api/v1/dags/{dag_id}/dagRuns"
    payload = {"conf": conf or {}}
    resp = requests.post(
        endpoint,
        json=payload,
        auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()
