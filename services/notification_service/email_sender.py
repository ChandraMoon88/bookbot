"""
services/notification_service/email_sender.py
-----------------------------------------------
Sends transactional e-mails via the SendGrid API.
Uses Jinja2 to render HTML templates stored in templates/.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

import urllib.request

log = logging.getLogger(__name__)

SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
FROM_EMAIL       = os.environ.get("NOTIFY_FROM_EMAIL", "no-reply@bookhotel.ai")
FROM_NAME        = os.environ.get("NOTIFY_FROM_NAME",  "BookHotel")
TEMPLATE_DIR     = Path(__file__).resolve().parents[2] / "templates"


def _render(template_name: str, context: dict) -> str:
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
        env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        tpl = env.get_template(template_name)
        return tpl.render(**context)
    except Exception as exc:
        log.warning("Template render failed (%s), using plain text fallback", exc)
        return str(context)


def send(
    to_email:   str,
    subject:    str,
    template:   str,
    context:    dict,
    to_name:    Optional[str] = None,
) -> bool:
    """
    Renders `template` (filename inside templates/) with `context`
    and sends via SendGrid.  Returns True on success.
    """
    html_body = _render(template, context)

    payload = {
        "personalizations": [
            {
                "to": [{"email": to_email, "name": to_name or to_email}],
                "subject": subject,
            }
        ],
        "from": {"email": FROM_EMAIL, "name": FROM_NAME},
        "content": [{"type": "text/html", "value": html_body}],
    }

    data = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type":  "application/json",
    }
    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            log.info("Email sent to %s (status=%s)", to_email, resp.status)
            return resp.status in (200, 202)
    except Exception as exc:
        log.error("SendGrid send failed: %s", exc)
        return False
