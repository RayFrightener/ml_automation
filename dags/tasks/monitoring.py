#!/usr/bin/env python3
"""
tasks/monitoring.py

Monitoring utilities with Prometheus instrumentation:
  - Records DAG runtime
  - Records memory metrics
  - Exposes /metrics on port 8000
"""

import logging
import time
import psutil
from prometheus_client import start_http_server, Gauge

# Attempt to start Prometheus metrics server once
try:
    start_http_server(8000)
    logging.info("Prometheus metrics server started on port 8000")
except OSError as e:
    logging.warning(f"Prometheus metrics server already running: {e}")

# Define Prometheus gauges
dag_runtime_gauge = Gauge(
    "homeowner_dag_runtime_seconds",
    "Duration of the DAG run in seconds"
)
memory_available_gauge = Gauge(
    "homeowner_memory_available_mb",
    "Available system memory in megabytes"
)
memory_used_gauge = Gauge(
    "homeowner_memory_used_mb",
    "Used system memory in megabytes"
)
memory_total_gauge = Gauge(
    "homeowner_memory_total_mb",
    "Total system memory in megabytes"
)
memory_percent_gauge = Gauge(
    "homeowner_memory_usage_percent",
    "Percentage of system memory in use"
)

def record_system_metrics(runtime: float = None, memory_usage: str = None) -> None:
    """
    Record and export system metrics to Prometheus and log them.

    Args:
        runtime (float, optional): The runtime duration in seconds.
        memory_usage (str, optional): Ignored (string snapshots aren't Prometheus-friendly).
    """
    # Record DAG runtime
    if runtime is not None:
        dag_runtime_gauge.set(runtime)
        logging.info(f"[Monitoring] DAG runtime set to {runtime:.2f}s")
    else:
        now = time.time()
        dag_runtime_gauge.set(now)
        logging.info(f"[Monitoring] DAG runtime timestamp set to {now:.2f}")

    # Record memory stats
    vm = psutil.virtual_memory()
    avail_mb = vm.available / (1024 * 1024)
    used_mb  = vm.used      / (1024 * 1024)
    total_mb = vm.total     / (1024 * 1024)

    memory_available_gauge.set(avail_mb)
    memory_used_gauge.set(used_mb)
    memory_total_gauge.set(total_mb)
    memory_percent_gauge.set(vm.percent)

    logging.info(
        "[Monitoring] Memory — "
        f"Avail: {avail_mb:.2f}MB, "
        f"Used: {used_mb:.2f}MB, "
        f"Total: {total_mb:.2f}MB, "
        f"Usage: {vm.percent:.1f}%"
    )
