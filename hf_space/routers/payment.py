"""
hf_space/routers/payment.py
----------------------------
Module 6 — Payment Processing & Booking Confirmation

Flow:
  reviewing_booking → paying → booking_confirmed

Stripe: card data NEVER touches server — Stripe.js inside Messenger Webview.
The server only:
  1. Creates a PaymentIntent server-side
  2. Returns client_secret to webview
  3. Webhook from Stripe confirms payment → triggers confirmation

State: user:{psid}:payment_intent = {intent_id, amount, currency}
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid

import stripe
import structlog
from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile, get_booking_draft, set_booking_draft
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_booking_review_start(psid: str, lang: str) -> tuple[list[dict], str]:
    """
    Show the final booking summary card before payment.
    Called from addons router when user proceeds.
    """
    mb = MessengerResponse(psid)
    draft   = await get_booking_draft(psid) or {}
    profile = await get_user_profile(psid) or {}

    # Build booking summary
    hotel    = draft.get("hotel", {})
    room     = draft.get("room", {})
    guest    = draft.get("guest", {})
    rate     = draft.get("rate_plan", "room_only")
    currency = profile.get("currency", "USD")

    # Tally addons
    r    = get_redis()
    raw_cart = await r.get(f"user:{psid}:addon_cart")
    cart: list = json.loads(raw_cart) if raw_cart else []
    addon_total = sum(a.get("price", 0) for a in cart)

    nights = draft.get("nights", 1)
    room_price_per_night = float(draft.get("room_price_per_night", 0))
    rate_multipliers = {"room_only": 1.0, "breakfast": 1.15, "half_board": 1.35, "full_board": 1.60}
    room_total = room_price_per_night * rate_multipliers.get(rate, 1.0) * nights
    grand_total = room_total + addon_total

    addon_lines = "\n".join(f"  • {a['name']} — ${a['price']:.0f}" for a in cart) if cart else "  (none)"

    check_in  = draft.get("check_in", "")
    check_out = draft.get("check_out", "")

    summary_text = (
        f"📋 **Booking Summary**\n\n"
        f"🏨 {hotel.get('name', '')}\n"
        f"🛏️ {room.get('name', '')} × {nights} night(s)\n"
        f"📅 {check_in} → {check_out}\n"
        f"👤 {guest.get('name', '')}\n\n"
        f"Room ({rate.replace('_', ' ')}): ${room_total:.0f}\n"
        f"Add-ons:\n{addon_lines}\n"
        f"{'─'*20}\n"
        f"**Total: ${grand_total:.0f} {currency}**"
    )

    booking_card = {
        "recipient": {"id": psid},
        "message": {
            "attachment": {
                "type": "template",
                "payload": {
                    "template_type": "generic",
                    "elements": [{
                        "title": f"{hotel.get('name', 'Your Hotel')} — ${grand_total:.0f} {currency}",
                        "subtitle": f"{check_in} → {check_out} · {room.get('name','')}",
                        "image_url": hotel.get("image_url", "https://via.placeholder.com/400x200"),
                        "buttons": [
                            {
                                "type": "postback",
                                "title": "Confirm & Pay 💳",
                                "payload": "PAYMENT_START",
                            },
                            {
                                "type": "postback",
                                "title": "Change Booking ✏️",
                                "payload": "BOOKING_MODIFY_INIT",
                            },
                            {
                                "type": "postback",
                                "title": "Cancel ❌",
                                "payload": "BOOKING_CANCEL_INIT",
                            },
                        ],
                    }],
                },
            }
        },
    }

    # Persist totals to draft
    draft["grand_total_usd"] = grand_total
    draft["room_total_usd"]  = room_total
    draft["addon_total_usd"] = addon_total
    await set_booking_draft(psid, draft)
    await set_user_state(psid, "reviewing_booking")

    return mb.send_sequence([mb.text(summary_text), booking_card]), "reviewing_booking"


async def handle_payment_start(psid: str, lang: str) -> tuple[list[dict], str]:
    """
    Create Stripe PaymentIntent and serve webview URL.
    """
    mb = MessengerResponse(psid)
    draft    = await get_booking_draft(psid) or {}
    profile  = await get_user_profile(psid) or {}
    currency = profile.get("currency", "usd").lower()

    amount_usd = draft.get("grand_total_usd", 0)
    if amount_usd <= 0:
        return [mb.text("Something went wrong with your booking total. Let me restart. 🔄")], "searching"

    amount_cents = int(amount_usd * 100)

    try:
        intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency=currency,
            metadata={"psid_hash": hashlib.sha256(psid.encode()).hexdigest()[:16], "hotel_id": draft.get("hotel_id", "")},
            automatic_payment_methods={"enabled": True},
        )
    except stripe.error.StripeError as e:
        log.error("stripe_create_intent_failed", error=str(e))
        return [mb.text("Payment system unavailable right now. Please try again shortly.")], "reviewing_booking"

    # Store intent
    r = get_redis()
    await r.set(f"user:{psid}:payment_intent", json.dumps({
        "intent_id": intent.id,
        "amount": amount_usd,
        "currency": currency,
    }), ex=3600)

    draft["payment_intent_id"] = intent.id
    await set_booking_draft(psid, draft)

    # Build Messenger Webview URL to pay
    hf_url = os.environ.get("HF_SPACE_URL", "")
    pay_url = f"{hf_url}/pay_webview?psid={psid}&client_secret={intent.client_secret}&amount={amount_usd:.2f}&currency={currency.upper()}"

    await set_user_state(psid, "paying")

    return mb.send_sequence([
        mb.text(f"💳 Ready to pay ${amount_usd:.2f} {currency.upper()}? Tap the button below to open our secure payment page."),
        mb.webview_button("", "Pay Now — Secure 🔒", pay_url),
    ]), "paying"


async def handle_payment_postback(psid: str, lang: str) -> tuple[list[dict], str]:
    """Check payment status after user returns from webview."""
    mb = MessengerResponse(psid)
    r  = get_redis()
    raw = await r.get(f"user:{psid}:payment_intent")
    if not raw:
        return [mb.text("I can't find your payment session. Let me show your booking again.")], "reviewing_booking"

    intent_data = json.loads(raw)
    intent_id   = intent_data.get("intent_id", "")

    try:
        intent = stripe.PaymentIntent.retrieve(intent_id)
    except stripe.error.StripeError:
        return [mb.text("Unable to verify payment. Please try again.")], "reviewing_booking"

    if intent.status == "succeeded":
        return await _confirm_booking(psid, intent, lang)

    return [mb.quick_replies(
        "Your payment hasn't been received yet. What would you like to do?",
        [
            {"title": "Try paying again 💳", "payload": "PAYMENT_START"},
            {"title": "Contact support 🆘",  "payload": "HANDOFF_REQUEST"},
        ],
    )], "paying"


async def _confirm_booking(psid: str, intent, lang: str) -> tuple[list[dict], str]:
    """Write booking to Supabase and send confirmation."""
    mb    = MessengerResponse(psid)
    sb    = get_supabase()
    draft = await get_booking_draft(psid) or {}
    r     = get_redis()

    booking_ref = f"HBK-{uuid.uuid4().hex[:8].upper()}"

    # Read cart
    raw_cart = await r.get(f"user:{psid}:addon_cart")
    cart: list = json.loads(raw_cart) if raw_cart else []

    booking_row = {
        "reference":        booking_ref,
        "psid_hash":        hashlib.sha256(psid.encode()).hexdigest(),
        "hotel_id":         draft.get("hotel_id", ""),
        "room_id":          draft.get("room_id", ""),
        "rate_plan":        draft.get("rate_plan", "room_only"),
        "check_in":         draft.get("check_in", ""),
        "check_out":        draft.get("check_out", ""),
        "nights":           draft.get("nights", 1),
        "guests":           draft.get("guests", 1),
        "guest_data":       draft.get("guest", {}),
        "addons":           cart,
        "room_total_usd":   draft.get("room_total_usd", 0),
        "addon_total_usd":  draft.get("addon_total_usd", 0),
        "grand_total_usd":  draft.get("grand_total_usd", 0),
        "payment_intent_id":intent.id,
        "status":           "confirmed",
        "created_at":       "now()",
    }

    try:
        res = await sb.table("bookings").insert(booking_row).execute()
        booking_id = res.data[0]["id"] if res.data else None
    except Exception as e:
        log.error("booking_insert_failed", error=str(e))
        return [mb.text("Your payment went through but we had a database error. A team member will contact you shortly.")], "booking_confirmed"

    # Release soft lock
    try:
        from services.room_service.soft_lock import release_lock
        await release_lock(draft.get("room_id", ""), draft.get("check_in", ""), draft.get("check_out", ""))
    except Exception:
        pass

    # Clear draft and cart
    await r.delete(f"user:{psid}:addon_cart")
    await r.delete(f"user:{psid}:booking_draft")
    await r.delete(f"user:{psid}:payment_intent")

    # Update state
    await set_user_state(psid, "booking_confirmed")

    hotel = draft.get("hotel", {})
    room  = draft.get("room", {})

    confirm_text = (
        f"🎉 **Booking Confirmed!**\n\n"
        f"Reference: **{booking_ref}**\n"
        f"🏨 {hotel.get('name', '')}\n"
        f"🛏️ {room.get('name', '')}\n"
        f"📅 {draft.get('check_in','')} → {draft.get('check_out','')}\n\n"
        f"A confirmation email has been sent. See you soon! 😊"
    )

    msgs = mb.send_sequence([mb.text(confirm_text)])
    msgs += [mb.quick_replies(
        "What's next?",
        [
            {"title": "📧 Download confirmation", "payload": f"BOOKING_PDF_{booking_ref}"},
            {"title": "✏️ Modify booking",         "payload": "MODIFY_BOOKING"},
            {"title": "🏠 Back to menu",            "payload": "MENU_MAIN"},
        ],
    )]

    # Trigger async confirmation email (fire-and-forget)
    try:
        from services.confirmation_service.main import send_confirmation
        import asyncio
        asyncio.create_task(send_confirmation(booking_id))
    except Exception:
        pass

    # Award loyalty points
    try:
        from hf_space.routers.loyalty import award_booking_points
        import asyncio
        asyncio.create_task(award_booking_points(psid, booking_id, draft.get("grand_total_usd", 0)))
    except Exception:
        pass

    return msgs, "booking_confirmed"


# ── Stripe Webhook ─────────────────────────────────────────────────────────────

@router.post("/stripe_webhook")
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)) -> dict:
    """
    Stripe sends payment events here. We only trust events that pass signature verification.
    """
    payload = await request.body()
    sig     = stripe_signature or ""
    secret  = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, secret)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid webhook: {e}")

    if event["type"] == "payment_intent.succeeded":
        intent = event["data"]["object"]
        psid_hash = intent.get("metadata", {}).get("psid_hash", "")
        log.info("payment_succeeded", psid_hash=psid_hash, intent_id=intent["id"])
        # Note: full confirm happens via handle_payment_postback triggered when user returns to chat.

    elif event["type"] == "payment_intent.payment_failed":
        intent = event["data"]["object"]
        log.warning("payment_failed", intent_id=intent["id"])

    return {"received": True}


# ── API endpoints ──────────────────────────────────────────────────────────────

@router.get("/intent")
async def create_intent(psid: str) -> dict:
    """
    Called by pay_webview.html to get the Stripe client_secret.
    Returns amount + currency for display.
    """
    r = get_redis()
    raw = await r.get(f"user:{psid}:payment_intent")
    if not raw:
        raise HTTPException(status_code=404, detail="No active payment session")
    return json.loads(raw)


class PayStatusRequest(BaseModel):
    psid: str


@router.post("/status")
async def payment_status(req: PayStatusRequest) -> dict:
    r   = get_redis()
    raw = await r.get(f"user:{req.psid}:payment_intent")
    if not raw:
        return {"status": "no_session"}
    data = json.loads(raw)
    try:
        intent = stripe.PaymentIntent.retrieve(data["intent_id"])
        return {"status": intent.status, "amount": data["amount"], "currency": data["currency"]}
    except stripe.error.StripeError:
        return {"status": "error"}
