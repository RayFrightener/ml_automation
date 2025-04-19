#!/usr/bin/env python3
"""
utils/slack.py

Send Slack messages via Incoming Webhook, with retries.
"""

import requests
from tenacity import retry, wait_fixed, stop_after_attempt
from utils.config import SLACK_WEBHOOK_URL, SLACK_CHANNEL_DEFAULT

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def post(
    channel: str = SLACK_CHANNEL_DEFAULT,
    title:   str = "",
    details: str = "",
    urgency: str = "low"
) -> None:
    """
    Send a notification to Slack.

    Args:
      channel: Slack channel name (e.g. "#alerts").
      title:   Short headline.
      details: Longer message body.
      urgency: "low", "medium" or "high".
    """
    if not SLACK_WEBHOOK_URL:
        raise ValueError("SLACK_WEBHOOK_URL is not configured!")

    text = f"*{title}*\nChannel: {channel}\nUrgency: `{urgency}`\n\n{details}"
    resp = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
    resp.raise_for_status()

def update_slack_notification_process_with_ui_components():
    """
    Placeholder function to update the Slack notification process with new UI components and endpoints.
    """
    logging.info("Updating Slack notification process with new UI components and endpoints.")
    # Placeholder for actual implementation

# backwards‐compatible aliases
send_message = post
post_message = post
