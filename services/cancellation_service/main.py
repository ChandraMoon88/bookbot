"""
services/cancellation_service/main.py
---------------------------------------
FastAPI cancellation + refund microservice.
"""

import os
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .refund_engine import calculate
from .stripe_refund import issue_refund

log = logging.getLogger(__name__)
app = FastAPI(title="Cancellation Service")

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def _get_booking(booking_id: str) -> dict:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM bookings WHERE id = %s", (booking_id,))
            row = cur.fetchone()
    if not row:
        raise ValueError(f"Booking {booking_id} not found")
    return dict(row)


def _cancel_booking(booking_id: str, refund_usd: float) -> dict:
    now = datetime.now(timezone.utc)
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE bookings SET status='cancelled', cancelled_at=%s, "
                "refund_amount=%s, updated_at=%s WHERE id=%s RETURNING *",
                (now, refund_usd, now, booking_id),
            )
            result = cur.fetchone()
            conn.commit()
    return dict(result) if result else {}


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
