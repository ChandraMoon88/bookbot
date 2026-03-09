"""
services/notification_service/push_sender.py
----------------------------------------------
Sends Web Push notifications via the Web Push Protocol (RFC 8030).
Uses the `pywebpush` library.
Subscription data comes from the guest's browser registration stored in Supabase.
"""

import os
import json
import logging
from typing import Optional

log = logging.getLogger(__name__)

VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "")
VAPID_CLAIMS      = {"sub": f"mailto:{os.environ.get('NOTIFY_FROM_EMAIL', 'push@bookhotel.ai')}"}


def send(
    subscription_info: dict,
    title:             str,
    body:              str,
    url:               Optional[str] = None,
    icon:              Optional[str] = None,
) -> bool:
    """
    Args:
        subscription_info: The browser's PushSubscription JSON
                           {"endpoint": ..., "keys": {"p256dh": ..., "auth": ...}}
        title:             Notification title
        body:              Notification body text
        url:               Optional click URL
        icon:              Optional icon image URL
    """
    if not VAPID_PRIVATE_KEY:
        log.warning("VAPID_PRIVATE_KEY not set; push notification skipped")
        return False

    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        log.error("pywebpush is required: pip install pywebpush")
        return False

    payload = {"title": title, "body": body}
    if url:
        payload["url"] = url
    if icon:
        payload["icon"] = icon

    try:
        webpush(
            subscription_info=subscription_info,
            data=json.dumps(payload),
            vapid_private_key=VAPID_PRIVATE_KEY,
            vapid_claims=VAPID_CLAIMS,
        )
        log.info("Push notification sent: %s", title)
        return True
    except Exception as exc:
        log.error("Push notification failed: %s", exc)
        return False
