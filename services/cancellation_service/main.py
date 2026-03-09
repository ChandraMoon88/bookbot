"""
services/cancellation_service/main.py
---------------------------------------
FastAPI cancellation + refund microservice.
"""

import os
import json
import logging
from datetime import datetime, timezone
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .refund_engine import calculate
from .stripe_refund import issue_refund

log = logging.getLogger(__name__)
app = FastAPI(title="Cancellation Service")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def _get_booking(booking_id: str) -> dict:
    url  = f"{SUPABASE_URL}/rest/v1/bookings?id=eq.{booking_id}&select=*"
    req  = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req) as resp:
        rows = json.loads(resp.read())
    if not rows:
        raise ValueError(f"Booking {booking_id} not found")
    return rows[0]


def _cancel_booking(booking_id: str, refund_usd: float) -> dict:
    url  = f"{SUPABASE_URL}/rest/v1/bookings?id=eq.{booking_id}"
    data = json.dumps({
        "status":        "cancelled",
        "cancelled_at":  datetime.now(timezone.utc).isoformat(),
        "refund_amount": refund_usd,
        "updated_at":    datetime.now(timezone.utc).isoformat(),
    }).encode()
    req = urllib.request.Request(url, data=data, headers=_headers(), method="PATCH")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


class CancelRequest(BaseModel):
    booking_id: str
    reason:     Optional[str] = None


@app.post("/cancel")
def cancel(req: CancelRequest):
    booking = _get_booking(req.booking_id)

    if booking.get("status") == "cancelled":
        raise HTTPException(status_code=409, detail="Booking already cancelled")

    refund_info = calculate(
        check_in_str=booking["check_in"],
        amount_paid=float(booking.get("total_amount", 0)),
        reason=req.reason,
    )

    # Attempt Stripe refund if a payment_intent_id is recorded
    stripe_result = None
    pi_id = booking.get("payment_intent_id")
    if pi_id and refund_info["refund_usd"] > 0:
        try:
            stripe_result = issue_refund(
                payment_intent_id=pi_id,
                amount_usd=refund_info["refund_usd"],
            )
        except Exception as exc:
            log.warning("Stripe refund failed: %s (booking will still be cancelled)", exc)

    # Mark cancelled in DB
    _cancel_booking(req.booking_id, refund_info["refund_usd"])

    return {
        "booking_id": req.booking_id,
        "status":     "cancelled",
        "refund":     refund_info,
        "stripe":     stripe_result,
    }


@app.post("/policy")
def policy(check_in: str, amount_paid: float, reason: Optional[str] = None):
    return calculate(check_in, amount_paid, reason)


@app.get("/health")
def health():
    return {"status": "ok", "service": "cancellation_service"}
