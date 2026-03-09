"""
services/corporate_service/approval_engine.py
-----------------------------------------------
Routes high-value corporate bookings through an approval workflow.

Rules:
  amount < APPROVAL_THRESHOLD  → auto-approve
  amount >= APPROVAL_THRESHOLD → create pending approval request,
                                  email approver, return approval_id
"""

import os
import json
import logging
from datetime import datetime, timezone
import urllib.request

log = logging.getLogger(__name__)

SUPABASE_URL        = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY        = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
APPROVAL_THRESHOLD  = float(os.environ.get("CORPORATE_APPROVAL_THRESHOLD_USD", "1000"))
NOTIFY_URL          = os.environ.get("NOTIFICATION_SERVICE_URL", "http://localhost:8007")


def _headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def _post(table: str, payload: dict) -> dict:
    url  = f"{SUPABASE_URL}/rest/v1/{table}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result[0] if isinstance(result, list) else result


def evaluate(
    corporate_account_id: str,
    booking_id:           str,
    amount_usd:           float,
    approver_email:       str,
) -> dict:
    """
    Returns {status, approval_id, message}.
    status: "approved" | "pending_approval"
    """
    if amount_usd < APPROVAL_THRESHOLD:
        return {"status": "approved", "approval_id": None,
                "message": "Auto-approved (below threshold)"}

    record = _post("approval_requests", {
        "corporate_account_id": corporate_account_id,
        "booking_id":           booking_id,
        "amount_usd":           amount_usd,
        "approver_email":       approver_email,
        "status":               "pending",
        "requested_at":         datetime.now(timezone.utc).isoformat(),
    })
    approval_id = record.get("id")

    # Notify approver via notification service
    try:
        payload = json.dumps({
            "channel":        "email",
            "to_email":       approver_email,
            "email_subject":  f"Approval Required: Corporate Booking ${amount_usd:.0f}",
            "email_template": "approval_request.html",
            "template_ctx": {
                "booking_id":   booking_id,
                "amount_usd":   amount_usd,
                "approval_id":  approval_id,
                "approve_url":  f"https://bookhotel.ai/approve/{approval_id}",
            },
        }).encode()
        notify_req = urllib.request.Request(
            f"{NOTIFY_URL}/notify",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(notify_req)
    except Exception as exc:
        log.warning("Could not send approval email: %s", exc)

    return {
        "status":      "pending_approval",
        "approval_id": approval_id,
        "message":     f"Approval required (≥${APPROVAL_THRESHOLD:.0f}). Email sent to {approver_email}.",
    }
