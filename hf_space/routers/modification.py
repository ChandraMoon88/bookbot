"""
hf_space/routers/modification.py
----------------------------------
Module 8 — Booking Modification

Allows guests to change dates, room type, rate plan or guest details
on an existing confirmed booking.

Policy rules (from services/modification_service/policy_engine.py):
  - Date changes free >48h before check-in
  - Room upgrade always allowed (pay difference)
  - Downgrade allowed with fee
  - In-stay modifications: only add-ons / guest notes

Flow:
  MODIFY_BOOKING postback → list bookings → select booking → choose field →
  apply change → confirm → write to DB
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile, get_booking_draft, set_booking_draft
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_modification_start(psid: str, lang: str) -> tuple[list[dict], str]:
    """Show existing confirmed bookings to select for modification."""
    mb = MessengerResponse(psid)
    sb = get_supabase()
    import hashlib
    psid_hash = hashlib.sha256(psid.encode()).hexdigest()

    try:
        res = await sb.table("bookings") \
            .select("id,reference,hotel_data:hotels(name),check_in,check_out,status") \
            .eq("psid_hash", psid_hash) \
            .in_("status", ["confirmed", "modified"]) \
            .order("check_in", desc=True) \
            .limit(5).execute()
        bookings = res.data or []
    except Exception as e:
        log.error("fetch_bookings_failed", error=str(e))
        bookings = []

    if not bookings:
        return [mb.text("You don't have any upcoming bookings to modify.")], "greeting"

    options = []
    for b in bookings:
        hotel_name = (b.get("hotel_data") or {}).get("name", "Hotel")
        options.append({
            "title": b["reference"][:20],
            "payload": f"MODIFY_SELECT_{b['id']}",
        })

    return [mb.quick_replies(
        "Which booking would you like to modify?",
        options + [{"title": "Never mind ←", "payload": "MENU_MAIN"}],
    )], "modifying"


async def handle_modification_input(psid: str, text: str, lang: str) -> tuple[list[dict], str]:
    """Dispatch within modifying state."""
    mb = MessengerResponse(psid)

    if text.startswith("MODIFY_SELECT_"):
        booking_id = text.replace("MODIFY_SELECT_", "")
        return await _select_booking(psid, booking_id, lang)

    if text.startswith("MODIFY_FIELD_"):
        field = text.replace("MODIFY_FIELD_", "")
        return await _change_field(psid, field, lang)

    if text.startswith("MODIFY_DATE_"):
        return await _handle_date_change(psid, text, lang)

    if text == "MODIFY_CONFIRM":
        return await _apply_modification(psid, lang)

    if text == "MODIFY_CANCEL":
        await set_user_state(psid, "greeting")
        return [mb.text("No changes made. 👍")], "greeting"

    return [mb.quick_replies(
        "What would you like to modify?",
        [
            {"title": "📅 Change dates",          "payload": "MODIFY_FIELD_dates"},
            {"title": "🛏️ Change room",            "payload": "MODIFY_FIELD_room"},
            {"title": "👤 Update guest details",   "payload": "MODIFY_FIELD_guest"},
            {"title": "🍽 Change meal plan",        "payload": "MODIFY_FIELD_rate"},
            {"title": "← Back",                    "payload": "MENU_MAIN"},
        ],
    )], "modifying"


async def _select_booking(psid: str, booking_id: str, lang: str) -> tuple[list[dict], str]:
    mb = MessengerResponse(psid)
    sb = get_supabase()

    try:
        res = await sb.table("bookings").select("*").eq("id", booking_id).single().execute()
        booking = res.data or {}
    except Exception:
        return [mb.text("Booking not found.")], "greeting"

    check_in = booking.get("check_in", "")
    r = get_redis()
    await r.set(f"user:{psid}:mod_booking_id", booking_id, ex=3600)
    await r.set(f"user:{psid}:mod_booking", json.dumps(booking), ex=3600)

    # Check if within free-change window
    try:
        from services.modification_service.policy_engine import check_modification_policy
        policy = await check_modification_policy(booking)
    except Exception:
        policy = {"free_change": True, "fee": 0, "reason": ""}

    fee_text = ""
    if not policy.get("free_change"):
        fee_text = f"\n⚠️ A change fee of ${policy.get('fee', 0):.0f} may apply."

    hotel_name = (booking.get("hotel_data") or {}).get("name", "your hotel")
    return [
        mb.text(f"📋 Booking {booking.get('reference','')}\n{hotel_name}\n{check_in} → {booking.get('check_out','')}{fee_text}"),
        mb.quick_replies("What would you like to change?", [
            {"title": "📅 Change dates",        "payload": "MODIFY_FIELD_dates"},
            {"title": "🛏️ Change room",          "payload": "MODIFY_FIELD_room"},
            {"title": "👤 Update guest info",    "payload": "MODIFY_FIELD_guest"},
            {"title": "🍽 Change meal plan",      "payload": "MODIFY_FIELD_rate"},
            {"title": "← Back",                  "payload": "MENU_MAIN"},
        ]),
    ], "modifying"


async def _change_field(psid: str, field: str, lang: str) -> tuple[list[dict], str]:
    mb = MessengerResponse(psid)
    r  = get_redis()
    await r.set(f"user:{psid}:mod_field", field, ex=3600)

    if field == "dates":
        return [mb.text("Please send your new check-in date (YYYY-MM-DD):")], "modifying"

    if field == "rate":
        return [mb.quick_replies(
            "Select your new meal plan:",
            [
                {"title": "🛏 Room only",        "payload": "MODIFY_DATE_rate_room_only"},
                {"title": "🍳 Breakfast",        "payload": "MODIFY_DATE_rate_breakfast"},
                {"title": "🍽 Half board",        "payload": "MODIFY_DATE_rate_half_board"},
                {"title": "🍴 Full board",        "payload": "MODIFY_DATE_rate_full_board"},
            ],
        )], "modifying"

    if field == "room":
        r2 = get_redis()
        raw = await r2.get(f"user:{psid}:mod_booking")
        booking = json.loads(raw) if raw else {}
        hotel_id = booking.get("hotel_id", "")
        from hf_space.routers.rooms import handle_hotel_selected
        return await handle_hotel_selected(psid, hotel_id, lang)

    if field == "guest":
        return [mb.text("Please enter the guest name:")], "modifying"

    return [mb.text("Not supported.")], "modifying"


async def _handle_date_change(psid: str, text: str, lang: str) -> tuple[list[dict], str]:
    """Handle new date input or rate change via postback."""
    mb = MessengerResponse(psid)
    r  = get_redis()
    field = (await r.get(f"user:{psid}:mod_field") or b"").decode() if hasattr(await r.get(f"user:{psid}:mod_field") or b"", "decode") else str(await r.get(f"user:{psid}:mod_field") or "")

    if text.startswith("MODIFY_DATE_rate_"):
        new_rate = text.replace("MODIFY_DATE_rate_", "")
        await r.set(f"user:{psid}:mod_new_value", new_rate, ex=3600)
        return [mb.quick_replies(
            f"Change meal plan to **{new_rate.replace('_',' ')}**?  Confirm?",
            [
                {"title": "Confirm ✅", "payload": "MODIFY_CONFIRM"},
                {"title": "Cancel ❌",  "payload": "MODIFY_CANCEL"},
            ],
        )], "modifying"

    # Date text — simple parse
    import re
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if date_match:
        await r.set(f"user:{psid}:mod_new_value", date_match.group(1), ex=3600)
        return [mb.quick_replies(
            f"Change to new check-in: **{date_match.group(1)}**?",
            [
                {"title": "Confirm ✅", "payload": "MODIFY_CONFIRM"},
                {"title": "Cancel ❌",  "payload": "MODIFY_CANCEL"},
            ],
        )], "modifying"

    return [mb.text("Please enter the date in YYYY-MM-DD format.")], "modifying"


async def _apply_modification(psid: str, lang: str) -> tuple[list[dict], str]:
    """Apply the stored modification to the booking."""
    mb = MessengerResponse(psid)
    sb = get_supabase()
    r  = get_redis()

    booking_id = (await r.get(f"user:{psid}:mod_booking_id") or b"").decode() if hasattr(await r.get(f"user:{psid}:mod_booking_id") or b"", "decode") else str(await r.get(f"user:{psid}:mod_booking_id") or "")
    raw        = await r.get(f"user:{psid}:mod_booking")
    booking    = json.loads(raw) if raw else {}
    field      = str(await r.get(f"user:{psid}:mod_field") or "")
    new_val    = str(await r.get(f"user:{psid}:mod_new_value") or "")

    update_data: dict[str, Any] = {"status": "modified"}

    if field == "dates":
        update_data["check_in"] = new_val
    elif field == "rate":
        update_data["rate_plan"] = new_val
    elif field == "room":
        update_data["room_id"] = new_val
    elif field == "guest":
        guest = booking.get("guest_data", {})
        guest["name"] = new_val
        update_data["guest_data"] = guest

    try:
        await sb.table("bookings").update(update_data).eq("id", booking_id).execute()
    except Exception as e:
        log.error("booking_update_failed", error=str(e))
        return [mb.text("Modification failed. Please contact support.")], "modifying"

    # Clean up Redis keys
    for key in ["mod_booking_id", "mod_booking", "mod_field", "mod_new_value"]:
        await r.delete(f"user:{psid}:{key}")

    await set_user_state(psid, "booking_confirmed")
    return [
        mb.text("✅ Your booking has been updated! You'll receive a confirmation shortly."),
        mb.quick_replies("Anything else?", [
            {"title": "🏠 Main menu",      "payload": "MENU_MAIN"},
            {"title": "✏️ Modify again",   "payload": "MODIFY_BOOKING"},
        ]),
    ], "booking_confirmed"


# ── API endpoint ───────────────────────────────────────────────────────────────

class ModifyRequest(BaseModel):
    booking_id: str
    field: str
    new_value: str


@router.post("/apply")
async def apply_modification(req: ModifyRequest) -> dict:
    sb = get_supabase()
    update: dict[str, Any] = {"status": "modified"}
    if req.field == "check_in":
        update["check_in"] = req.new_value
    elif req.field == "rate_plan":
        update["rate_plan"] = req.new_value
    elif req.field == "room_id":
        update["room_id"] = req.new_value
    try:
        await sb.table("bookings").update(update).eq("id", req.booking_id).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/policy/{booking_id}")
async def get_mod_policy(booking_id: str) -> dict:
    sb = get_supabase()
    try:
        res = await sb.table("bookings").select("*").eq("id", booking_id).single().execute()
        booking = res.data or {}
        from services.modification_service.policy_engine import check_modification_policy
        return await check_modification_policy(booking)
    except Exception as e:
        return {"error": str(e)}
