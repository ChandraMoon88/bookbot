"""
services/notification_service/whatsapp_sender.py
--------------------------------------------------
Sends WhatsApp messages via the Meta WhatsApp Business Cloud API (Graph API v18+).
Uses pre-approved message templates for transactional messages.
"""

import os
import json
import logging
import urllib.request

log = logging.getLogger(__name__)

WA_PHONE_ID   = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
WA_TOKEN      = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
META_API_BASE = "https://graph.facebook.com/v18.0"


def send_template(
    to:            str,
    template_name: str,
    language:      str = "en_US",
    components:    list = None,
) -> bool:
    """
    Sends a WhatsApp template message.

    Args:
        to:             E.164 phone number ("+14155551234")
        template_name:  Pre-approved template name
        language:       BCP-47 locale code
        components:     Template button/body parameter list
    """
    payload: dict = {
        "messaging_product": "whatsapp",
        "to":                to,
        "type":              "template",
        "template": {
            "name":     template_name,
            "language": {"code": language},
        },
    }
    if components:
        payload["template"]["components"] = components

    url  = f"{META_API_BASE}/{WA_PHONE_ID}/messages"
    data = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type":  "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            log.info("WA template %s → %s (status=%s)", template_name, to, resp.status)
            return resp.status == 200
    except Exception as exc:
        log.error("WhatsApp send failed: %s", exc)
        return False


def send_text(to: str, text: str) -> bool:
    """Sends a plain text WhatsApp message (only valid within 24-h window)."""
    payload = {
        "messaging_product": "whatsapp",
        "to":                to,
        "type":              "text",
        "text":              {"body": text},
    }
    url  = f"{META_API_BASE}/{WA_PHONE_ID}/messages"
    data = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {WA_TOKEN}",
        "Content-Type":  "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status == 200
    except Exception as exc:
        log.error("WhatsApp text send failed: %s", exc)
        return False
