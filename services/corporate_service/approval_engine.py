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

DATABASE_URL        = os.environ.get("DATABASE_URL", "")
APPROVAL_THRESHOLD  = float(os.environ.get("CORPORATE_APPROVAL_THRESHOLD_USD", "1000"))
NOTIFY_URL          = os.environ.get("NOTIFICATION_SERVICE_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def _insert_approval(payload: dict) -> dict:
    cols = list(payload.keys())
    phs  = ", ".join(f"%({c})s" for c in cols)
    sql  = f"INSERT INTO approval_requests ({', '.join(cols)}) VALUES ({phs}) RETURNING *"
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, payload)
            result = dict(cur.fetchone())
            conn.commit()
    return result


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

    record      = _insert_approval({
        "corporate_account_id": corporate_account_id,
        "booking_id":           booking_id,
        "amount_usd":           amount_usd,
        "approver_email":       approver_email,
        "status":               "pending",
        "requested_at":         datetime.now(timezone.utc),
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
