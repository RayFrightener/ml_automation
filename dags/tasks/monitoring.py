# tasks/monitoring.py

"""
monitoring.py

This module provides utility functions to log basic system performance metrics
during DAG runs, such as runtime and memory usage. Logs are stored via Airflow's logger.

Dependencies:
    - psutil (install with `pip install psutil`)
"""

import logging
import time
import psutil


def record_system_metrics(runtime=None, memory_usage=None):
    """
    Record and log system metrics.

    Args:
        runtime (float, optional): A timestamp or runtime duration.
        memory_usage (str, optional): A memory usage string (e.g., from `free -m`).
    """
    if runtime:
        logging.info(f"[System Monitoring] Runtime Timestamp: {runtime}")
    else:
        logging.info(f"[System Monitoring] Runtime: {time.time()}")

    if memory_usage:
        logging.info(f"[System Monitoring] Custom Memory Snapshot:\n{memory_usage}")
    else:
        virtual_mem = psutil.virtual_memory()
        logging.info(f"[System Monitoring] Available RAM: {virtual_mem.available / (1024 * 1024):.2f} MB")
        logging.info(f"[System Monitoring] Used RAM: {virtual_mem.used / (1024 * 1024):.2f} MB")
        logging.info(f"[System Monitoring] Total RAM: {virtual_mem.total / (1024 * 1024):.2f} MB")
        logging.info(f"[System Monitoring] RAM Usage Percentage: {virtual_mem.percent}%")