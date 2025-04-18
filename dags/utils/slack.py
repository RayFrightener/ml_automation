#!/usr/bin/env python3
"""
utils/slack.py

Helper to send Slack notifications via an incoming webhook,
with built‑in retry logic to guard against transient failures.
"""

import os
import requests
from tenacity import retry, wait_fixed, stop_after_attempt
from utils.config import SLACK_WEBHOOK_URL

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def post(channel: str, title: str, details: str, urgency: str) -> None:
    """
    Send a formatted message to Slack.

    Args:
        channel (str): Slack channel name (e.g. "#alerts").
        title (str): Short headline for the message.
        details (str): Detailed body text.
        urgency (str): Urgency level ("low", "medium", "high", etc.).

    Raises:
        requests.HTTPError: if Slack returns an error status.
    """
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL is not configured.")
    text = f"*{title}*\nChannel: {channel}\nUrgency: `{urgency}`\n\n{details}"
    payload = {"text": text}
    resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
    resp.raise_for_status()
