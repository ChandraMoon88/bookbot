"""
hf_space/routers/cancellation.py
----------------------------------
Module 9 — Booking Cancellation & Refund

Policy:
  >72h before check-in  → full refund
  24–72h                → 50% refund
  <24h                  → no refund
  Force-majeure         → full refund (passed to force_majeure_service)

Stripe refund issued via services/cancellation_service/stripe_refund.py

Flow:
  CANCEL_BOOKING postback → list bookings → select → confirm policy →
  confirm cancellation → Stripe refund → update DB → notify guest
"""

from __future__ import annotations

import hashlib
import json

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_cancellation_start(psid: str, lang: str) -> tuple[list[dict], str]:
    """Show active bookings to select for cancellation."""
    mb = MessengerResponse(psid)
    sb = get_supabase()
    psid_hash = hashlib.sha256(psid.encode()).hexdigest()

    try:
        res = await sb.table("bookings") \
            .select("id,reference,hotel_data:hotels(name),check_in,check_out,grand_total_usd,payment_intent_id") \
            .eq("psid_hash", psid_hash) \
            .in_("status", ["confirmed", "modified"]) \
            .order("check_in", desc=False) \
            .limit(5).execute()
        bookings = res.data or []
    except Exception as e:
        log.error("fetch_bookings_failed", error=str(e))
        bookings = []

    if not bookings:
        return [mb.text("You have no active bookings to cancel.")], "greeting"

    options = [
        {"title": b.get("reference", "")[:20], "payload": f"CANCEL_SELECT_{b['id']}"}
        for b in bookings
    ]

    return [mb.quick_replies(
        "Which booking would you like to cancel?",
        options + [{"title": "← Never mind", "payload": "MENU_MAIN"}],
    )], "cancelling"


async def handle_cancellation_input(psid: str, text: str, lang: str) -> tuple[list[dict], str]:
    """Dispatch within cancelling state."""
    mb = MessengerResponse(psid)

    if text.startswith("CANCEL_SELECT_"):
        booking_id = text.replace("CANCEL_SELECT_", "")
        return await _show_cancellation_policy(psid, booking_id, lang)

    if text == "CANCEL_CONFIRM":
        return await _process_cancellation(psid, lang)

    if text == "CANCEL_ABORT":
        await set_user_state(psid, "greeting")
        return [mb.text("Cancellation abandoned. Your booking is safe. ✅")], "greeting"

    if text == "CANCEL_FORCE_MAJEURE":
        return await _force_majeure_request(psid, lang)

    return [mb.text("Would you like to cancel your booking?"),
            mb.quick_replies("Please confirm:", [
                {"title": "Yes, cancel",    "payload": "CANCEL_CONFIRM"},
                {"title": "No, keep it ✅", "payload": "CANCEL_ABORT"},
            ])], "cancelling"


async def _show_cancellation_policy(psid: str, booking_id: str, lang: str) -> tuple[list[dict], str]:
    """Retrieve booking and show refund policy."""
    mb = MessengerResponse(psid)
    sb = get_supabase()

    try:
        res = await sb.table("bookings").select("*").eq("id", booking_id).single().execute()
        booking = res.data or {}
    except Exception:
        return [mb.text("Booking not found.")], "greeting"

    r = get_redis()
    await r.set(f"user:{psid}:cancel_booking_id", booking_id, ex=3600)
    await r.set(f"user:{psid}:cancel_booking", json.dumps(booking), ex=3600)

    try:
        from services.cancellation_service.refund_engine import calculate_refund
        refund_info = await calculate_refund(booking)
    except Exception:
        from datetime import datetime, date
        check_in = booking.get("check_in", "")
        try:
            delta = (datetime.strptime(check_in, "%Y-%m-%d").date() - date.today()).days
        except Exception:
            delta = 999
        if delta > 72 / 24:
            refund_pct = 100
        elif delta > 24 / 24:
            refund_pct = 50
        else:
            refund_pct = 0
        total = booking.get("grand_total_usd", 0)
        refund_info = {"refund_pct": refund_pct, "refund_amount": total * refund_pct / 100}

    refund_pct    = refund_info.get("refund_pct", 0)
    refund_amount = refund_info.get("refund_amount", 0)
    total         = booking.get("grand_total_usd", 0)

    quick_opts = [
        {"title": f"Confirm Cancel ({refund_pct}% refund)", "payload": "CANCEL_CONFIRM"},
        {"title": "Keep my booking ✅",                       "payload": "CANCEL_ABORT"},
    ]

    news_monitor = False
    try:
        from services.force_majeure_service.news_monitor import check_force_majeure
        news_monitor = await check_force_majeure(booking.get("hotel_id", ""), booking.get("check_in", ""))
    except Exception:
        pass

    if news_monitor:
        quick_opts.append({"title": "⚠️ Force majeure", "payload": "CANCEL_FORCE_MAJEURE"})

    return [
        mb.text(
            f"📋 Cancellation Policy\n\n"
            f"Booking: {booking.get('reference','')}\n"
            f"Total paid: ${total:.0f}\n\n"
            f"Refund: ${refund_amount:.0f} ({refund_pct}%)\n"
            f"{'Note: No refund due to late notice.' if refund_pct == 0 else ''}"
        ),
        mb.quick_replies("Proceed with cancellation?", quick_opts),
    ], "cancelling"


async def _process_cancellation(psid: str, lang: str) -> tuple[list[dict], str]:
    """Actually cancel the booking and initiate refund."""
    mb = MessengerResponse(psid)
    sb = get_supabase()
    r  = get_redis()

    booking_id = str(await r.get(f"user:{psid}:cancel_booking_id") or "")
    raw        = await r.get(f"user:{psid}:cancel_booking")
    booking    = json.loads(raw) if raw else {}

    if not booking_id:
        return [mb.text("Session expired. Please start again.")], "greeting"

    # Cancel in DB
    try:
        await sb.table("bookings").update({"status": "cancelled"}).eq("id", booking_id).execute()
    except Exception as e:
        log.error("cancel_booking_db_failed", error=str(e))
        return [mb.text("Cancellation failed. Please contact support.")], "cancelling"

    # Issue Stripe refund
    refund_msg = ""
    intent_id = booking.get("payment_intent_id", "")
    if intent_id:
        try:
            from services.cancellation_service.refund_engine import calculate_refund
            refund_info = await calculate_refund(booking)
            refund_amount = refund_info.get("refund_amount", 0)
            if refund_amount > 0:
                from services.cancellation_service.stripe_refund import issue_stripe_refund
                await issue_stripe_refund(intent_id, int(refund_amount * 100))
                refund_msg = f"\n💰 Refund of ${refund_amount:.0f} initiated — usually 5–10 business days."
        except Exception as e:
            log.error("refund_failed", error=str(e))
            refund_msg = "\n⚠️ Refund will be processed manually. Our team will contact you."

    # Clean up
    for key in ["cancel_booking_id", "cancel_booking"]:
        await r.delete(f"user:{psid}:{key}")

    await set_user_state(psid, "greeting")

    # Trigger cancellation email
    try:
        from services.notification_service.email_sender import send_cancellation_email
        import asyncio
        asyncio.create_task(send_cancellation_email(booking))
    except Exception:
        pass

    return [
        mb.text(f"✅ Booking {booking.get('reference','')} has been cancelled.{refund_msg}"),
        mb.quick_replies("Anything else?", [
            {"title": "🔍 Search new hotels", "payload": "SEARCH_START"},
            {"title": "🏠 Main menu",          "payload": "MENU_MAIN"},
        ]),
    ], "greeting"


async def _force_majeure_request(psid: str, lang: str) -> tuple[list[dict], str]:
    """Handle force majeure claim for full refund."""
    mb = MessengerResponse(psid)
    r  = get_redis()

    booking_id = str(await r.get(f"user:{psid}:cancel_booking_id") or "")
    raw        = await r.get(f"user:{psid}:cancel_booking")
    booking    = json.loads(raw) if raw else {}

    try:
        from services.force_majeure_service.main import submit_force_majeure_claim
        result = await submit_force_majeure_claim(booking_id, booking)
        return [mb.text(
            "⚠️ Force majeure claim submitted.\n\n"
            f"Reference: {result.get('claim_id', 'N/A')}\n"
            "Our team will review and issue a full refund within 48 hours."
        )], "greeting"
    except Exception as e:
        return [mb.text("Unable to submit claim automatically. Please contact support.")], "greeting"


# ── API endpoints ──────────────────────────────────────────────────────────────

class CancelRequest(BaseModel):
    booking_id: str
    reason: str = "guest_request"


@router.post("/cancel")
async def cancel_booking(req: CancelRequest) -> dict:
    sb = get_supabase()
    try:
        await sb.table("bookings").update({"status": "cancelled", "cancel_reason": req.reason}).eq("id", req.booking_id).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/policy/{booking_id}")
async def get_cancel_policy(booking_id: str) -> dict:
    sb = get_supabase()
    try:
        res = await sb.table("bookings").select("*").eq("id", booking_id).single().execute()
        booking = res.data or {}
        from services.cancellation_service.refund_engine import calculate_refund
        return await calculate_refund(booking)
    except Exception as e:
        return {"error": str(e)}
