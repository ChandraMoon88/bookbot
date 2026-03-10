"""
hf_space/routers/notification.py
----------------------------------
Notification dispatcher — thin wrapper around services/notification_service/

Handles outbound notifications:
  - Email (confirmation, cancellation, modification, check-in reminder)
  - Push (if guest enrolled in push notifications)
  - WhatsApp (via Twilio, if TWILIO_* env vars set)

Also provides:
  POST /api/notify/booking_confirmed
  POST /api/notify/check_in_reminder
  POST /api/notify/cancellation
  POST /api/notify/modification
  POST /api/notify/review_request

Called by other routers and Celery scheduled tasks.
"""

from __future__ import annotations

import os
from datetime import datetime

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Request models ─────────────────────────────────────────────────────────────

class BookingNotifyRequest(BaseModel):
    booking_id:   str
    notification_type: str  # confirmed | cancelled | modified | reminder


class ReviewNotifyRequest(BaseModel):
    booking_id: str
    psid:       str


# ── API endpoints ──────────────────────────────────────────────────────────────

@router.post("/booking")
async def notify_booking(req: BookingNotifyRequest) -> dict:
    """
    Trigger booking notification email + optional WhatsApp.
    Called internally after booking state changes.
    """
    from hf_space.db.supabase import get_supabase
    sb = get_supabase()

    try:
        res = await sb.table("bookings").select("*").eq("id", req.booking_id).single().execute()
        booking = res.data or {}
    except Exception as e:
        return {"success": False, "error": f"Booking not found: {e}"}

    if not booking:
        return {"success": False, "error": "Booking not found"}

    guest_data = booking.get("guest_data", {})
    email      = guest_data.get("email", "")
    phone      = guest_data.get("phone", "")

    results: dict = {}

    # Email
    if email:
        try:
            from services.notification_service.email_sender import send_booking_email
            await send_booking_email(booking, req.notification_type)
            results["email"] = "sent"
        except Exception as e:
            log.error("email_notify_failed", error=str(e), type=req.notification_type)
            results["email"] = f"failed: {e}"

    # WhatsApp (optional)
    whatsapp_token = os.environ.get("TWILIO_ACCOUNT_SID", "")
    if phone and whatsapp_token:
        try:
            from services.notification_service.whatsapp_sender import send_whatsapp
            await send_whatsapp(phone, booking, req.notification_type)
            results["whatsapp"] = "sent"
        except Exception as e:
            log.warning("whatsapp_notify_failed", error=str(e))
            results["whatsapp"] = f"failed: {e}"

    return {"success": True, "results": results}


@router.post("/review_request")
async def notify_review_request(req: ReviewNotifyRequest) -> dict:
    """
    Send a proactive Messenger message asking for a review.
    Called by Celery 2h after check-out.
    """
    try:
        # Build review trigger message and send via HF Space process_message endpoint
        import httpx
        hf_url = os.environ.get("HF_SPACE_URL", "http://localhost:7860")
        payload = {
            "psid":         req.psid,
            "message_type": "postback",
            "text":         f"REVIEW_TRIGGER_{req.booking_id}",
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(f"{hf_url}/api/process_message", json=payload)
        return {"success": resp.status_code == 200}
    except Exception as e:
        log.error("review_notify_failed", error=str(e))
        return {"success": False, "error": str(e)}


@router.post("/check_in_reminder")
async def notify_check_in_reminder(booking_id: str) -> dict:
    """Send check-in day reminder (called by scheduled Celery task)."""
    return await notify_booking(BookingNotifyRequest(
        booking_id=booking_id,
        notification_type="reminder",
    ))
