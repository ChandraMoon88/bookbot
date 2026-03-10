"""
hf_space/routers/rooms.py
--------------------------
Module 3 — Room Selection with smart cards.

Triggered when user clicks "✅ Select Hotel" (postback: HOTEL_SELECT_{id}).

Flow:
  1. Fetch available rooms for hotel + dates from Supabase
  2. Show room card carousel (top 3)
  3. ON "See Photos"  → send up to 5 room images
  4. ON "Full Details"→ list template with specs
  5. ON "Choose Room" → rate plan quick replies
  6. Acquire soft lock immediately after rate selection
  7. Show price summary before proceeding to guest form

Backend:
  GET  /api/rooms/{hotel_id}
  POST /api/rooms/soft_lock
  POST /api/rooms/refresh_lock
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile, get_booking_draft, set_booking_draft
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse
from services.room_service.soft_lock import acquire_lock, release_lock

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Request models ─────────────────────────────────────────────────────────────

class SoftLockRequest(BaseModel):
    room_type_id: str
    check_in:     str
    check_out:    str
    psid:         str
    lock_minutes: int = 15


class RefreshLockRequest(BaseModel):
    lock_id: str
    psid:    str


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_hotel_selected(
    psid: str, hotel_id: str, lang: str
) -> tuple[list[dict], str]:
    """
    User tapped "✅ Select Hotel" from a hotel card.
    Fetch rooms and show carousel.
    """
    mb = MessengerResponse(psid)

    # Store selected hotel in draft
    r = get_redis()
    draft = await get_booking_draft(psid) or {}
    draft["hotel_id"] = hotel_id
    await set_booking_draft(psid, draft)

    # Fetch hotel details
    sb = get_supabase()
    try:
        h_res = await sb.table("hotels").select("name, star_rating").eq("id", hotel_id).single().execute()
        hotel = h_res.data or {}
    except Exception:
        hotel = {}

    hotel_name = hotel.get("name", "your selected hotel")

    # Fetch available rooms
    rooms = await _fetch_rooms(hotel_id, draft.get("check_in", ""), draft.get("check_out", ""), draft.get("num_guests", 1))

    if not rooms:
        return (
            mb.send_sequence([
                mb.text(f"Sorry, no rooms are available at {hotel_name} for those dates. 😔"),
                mb.quick_replies(
                    "What would you like to do?",
                    [
                        {"title": "← Back to hotels", "payload": "SEARCH_SHOW_MORE"},
                        {"title": "🔄 Change dates",   "payload": "SEARCH_CHANGE_DATES"},
                    ],
                ),
            ]),
            "viewing_hotels",
        )

    intro_msgs = [
        mb.text(f"Great choice! Let me check available rooms at {hotel_name} 🏨"),
    ]
    card_msg = mb.room_cards(rooms[:3])
    nav_qr = mb.quick_replies(
        f"Found {len(rooms)} room type(s):",
        [
            {"title": "Show all rooms",   "payload": f"ROOMS_ALL_{hotel_id}"},
            {"title": "← Back to hotels", "payload": "SEARCH_SHOW_MORE"},
        ],
    )

    await set_user_state(psid, "selecting_room")
    msgs = intro_msgs + mb.send_sequence([card_msg]) + [nav_qr]
    return msgs, "selecting_room"


async def handle_room_selected(
    psid: str, room_id: str, lang: str
) -> tuple[list[dict], str]:
    """
    User tapped "✅ Choose Room". Show rate plan quick replies.
    """
    mb = MessengerResponse(psid)

    # Fetch room + rate plans
    sb = get_supabase()
    try:
        r_res = await sb.table("room_types").select("*").eq("id", room_id).single().execute()
        room = r_res.data or {}
    except Exception:
        room = {}

    if not room:
        return [mb.text("Room not found. Please select another.")], "selecting_room"

    # Store room in booking draft
    draft = await get_booking_draft(psid) or {}
    draft["room_type_id"] = room_id
    draft["room_name"]    = room.get("name", "Selected Room")
    await set_booking_draft(psid, draft)

    # Build rate plan quick replies from room data
    profile  = await get_user_profile(psid)
    currency = (profile or {}).get("currency", "USD")
    base_price = room.get("price_usd", 100)

    rate_plans = [
        {"title": f"🛏 Room Only  {currency} {_fmt_price(base_price, currency, 1.0)}", "payload": f"RATE_room_only_{room_id}"},
        {"title": f"☕ Breakfast  {currency} {_fmt_price(base_price, currency, 1.15)}", "payload": f"RATE_breakfast_{room_id}"},
        {"title": f"🌮 Half Board {currency} {_fmt_price(base_price, currency, 1.35)}", "payload": f"RATE_half_board_{room_id}"},
        {"title": f"🍽 Full Board {currency} {_fmt_price(base_price, currency, 1.60)}", "payload": f"RATE_full_board_{room_id}"},
    ]

    await set_user_state(psid, "choosing_rate")
    return [mb.quick_replies("Which rate plan suits you? 🍽️", rate_plans)], "choosing_rate"


async def handle_rate_selected(
    psid: str, text: str, lang: str
) -> tuple[list[dict], str]:
    """
    User selected a rate plan. Acquire soft lock → show price summary.
    """
    mb = MessengerResponse(psid)
    draft = await get_booking_draft(psid) or {}

    # Parse rate payload: RATE_{plan}_{room_id}
    if not text.startswith("RATE_"):
        return [mb.quick_replies("Please select a rate plan:", [
            {"title": "🛏 Room Only", "payload": f"RATE_room_only_{draft.get('room_type_id','')}"},
            {"title": "☕ Breakfast", "payload": f"RATE_breakfast_{draft.get('room_type_id','')}"},
        ])], "choosing_rate"

    parts = text.split("_", 2)  # ["RATE", plan, room_id]
    rate_plan = parts[1] if len(parts) > 1 else "room_only"
    draft["rate_plan"] = rate_plan

    # Acquire soft lock
    lock_result = acquire_lock(
        room_type_id=draft.get("room_type_id", ""),
        check_in=draft.get("check_in", ""),
        check_out=draft.get("check_out", ""),
        session_id=psid,
        lock_minutes=15,
    )

    if not lock_result.get("locked"):
        # Race condition — find next available
        return mb.send_sequence([
            mb.text("Room just got reserved by someone else! Let me find the next best option… 😅"),
        ]) + [mb.text("Searching for next available room…")], "selecting_room"

    draft["lock_id"]    = lock_result.get("lock_id", "")
    draft["lock_expires"] = lock_result.get("expires_at", "")

    # Calculate total
    profile   = await get_user_profile(psid)
    currency  = (profile or {}).get("currency", "USD")
    multiplier = {"room_only": 1.0, "breakfast": 1.15, "half_board": 1.35, "full_board": 1.60}.get(rate_plan, 1.0)

    sb = get_supabase()
    try:
        r_res = await sb.table("room_types").select("price_usd, name").eq("id", draft["room_type_id"]).single().execute()
        room = r_res.data or {}
    except Exception:
        room = {}

    price_per_night_usd = room.get("price_usd", 100) * multiplier
    nights = _count_nights(draft.get("check_in", ""), draft.get("check_out", ""))
    total_usd = price_per_night_usd * max(nights, 1)

    from hf_space.routers.search import _get_fx_rates
    rates = await _get_fx_rates(currency)
    total_local = round(total_usd * rates.get(currency, 1))
    total_usd_rounded = round(total_usd)

    draft["total_usd"]   = total_usd_rounded
    draft["total_local"] = total_local
    draft["currency"]    = currency
    await set_booking_draft(psid, draft)

    # Fetch hotel name
    hotel_name = draft.get("hotel_name", "Hotel")
    try:
        h_res = await sb.table("hotels").select("name, star_rating").eq("id", draft.get("hotel_id", "")).single().execute()
        hd = h_res.data or {}
        hotel_name = hd.get("name", hotel_name)
        draft["hotel_name"] = hotel_name
        draft["hotel_stars"] = hd.get("star_rating", 5)
        await set_booking_draft(psid, draft)
    except Exception:
        pass

    rate_label = {"room_only": "Room Only", "breakfast": "Breakfast Incl.", "half_board": "Half Board", "full_board": "Full Board"}.get(rate_plan, rate_plan)
    summary_items = [
        {"title": f"🏨 Hotel",     "subtitle": f"{hotel_name} {'⭐'*int(draft.get('hotel_stars',5))}"},
        {"title": f"🛏 Room",      "subtitle": draft.get("room_name", "Selected Room")},
        {"title": f"🍽 Plan",      "subtitle": rate_label},
        {"title": f"📅 Dates",     "subtitle": f"{draft.get('check_in','')} → {draft.get('check_out','')} ({nights} night(s))"},
    ]
    summary_list = mb.list_template(
        f"💰 Total: {currency} {total_local:,} (~${total_usd_rounded} USD)",
        summary_items,
    )

    await set_user_state(psid, "reviewing_booking")
    msgs = mb.send_sequence([
        mb.text("📋 Your selection:"),
        summary_list,
    ]) + [
        mb.quick_replies(
            f"Total: {currency} {total_local:,} (~${total_usd_rounded} USD)",
            [
                {"title": "✅ Looks Good!",  "payload": "BOOKING_PROCEED_GUEST"},
                {"title": "✏️ Change Room", "payload": f"HOTEL_SELECT_{draft.get('hotel_id','')}"},
                {"title": "❌ Start Over",  "payload": "SEARCH_NEW"},
            ],
        )
    ]
    return msgs, "reviewing_booking"


async def handle_room_text_input(
    psid: str, text: str, lang: str
) -> tuple[list[dict], str]:
    """Handle free text while in room selection state."""
    mb = MessengerResponse(psid)
    draft = await get_booking_draft(psid) or {}
    hotel_id = draft.get("hotel_id", "")
    if hotel_id:
        return await handle_hotel_selected(psid, hotel_id, lang)
    return [mb.text("Please select a hotel first.")], "viewing_hotels"


async def handle_room_photos(psid: str, room_id: str) -> tuple[list[dict], str]:
    """Send up to 5 room photos."""
    mb = MessengerResponse(psid)
    sb = get_supabase()
    try:
        r_res = await sb.table("room_photos").select("photo_url").eq("room_type_id", room_id).limit(5).execute()
        photos = [p["photo_url"] for p in (r_res.data or [])]
    except Exception:
        photos = []

    if not photos:
        return [mb.text("No photos available for this room.")], "selecting_room"

    msgs: list[dict] = []
    for url in photos:
        msgs.extend(mb.send_sequence([mb.image(url)]))

    msgs.append(mb.quick_replies(
        "How does this room look?",
        [
            {"title": "✅ Choose This Room", "payload": f"ROOM_SELECT_{room_id}"},
            {"title": "← Other Rooms",       "payload": "ROOMS_BACK"},
        ],
    ))
    return msgs, "selecting_room"


async def handle_room_details(psid: str, room_id: str, lang: str) -> tuple[list[dict], str]:
    """Show full room details as a list template."""
    mb = MessengerResponse(psid)
    sb = get_supabase()
    try:
        r_res = await sb.table("room_types").select("*").eq("id", room_id).single().execute()
        room = r_res.data or {}
    except Exception:
        room = {}

    if not room:
        return [mb.text("Room details not available.")], "selecting_room"

    amenities = ", ".join((room.get("amenities") or [])[:8])
    items = [
        {"title": "Size",          "subtitle": f"{room.get('size_m2','?')} m²"},
        {"title": "Bed type",      "subtitle": room.get("bed_type", "?")},
        {"title": "Max guests",    "subtitle": str(room.get("max_occupancy", "?"))},
        {"title": "Amenities",     "subtitle": amenities or "—"},
    ]
    msgs = mb.send_sequence([
        mb.list_template(room.get("name", "Room"), items),
    ]) + [
        mb.quick_replies(
            "Ready to choose this room?",
            [
                {"title": "✅ Choose Room",  "payload": f"ROOM_SELECT_{room_id}"},
                {"title": "← Back to Rooms","payload": "ROOMS_BACK"},
            ],
        )
    ]
    return msgs, "selecting_room"


# ── API endpoints ──────────────────────────────────────────────────────────────

@router.get("/{hotel_id}")
async def get_rooms(
    hotel_id: str,
    check_in:  str = "",
    check_out: str = "",
    guests:    int = 1,
) -> dict:
    """GET /api/rooms/{hotel_id} — return available rooms for given dates."""
    rooms = await _fetch_rooms(hotel_id, check_in, check_out, guests)
    return {"rooms": rooms}


@router.post("/soft_lock")
async def soft_lock_room(req: SoftLockRequest) -> dict:
    """POST /api/rooms/soft_lock — atomic Redis lock for all nights."""
    result = acquire_lock(
        room_type_id=req.room_type_id,
        check_in=req.check_in,
        check_out=req.check_out,
        session_id=req.psid,
        lock_minutes=req.lock_minutes,
    )
    return result


@router.post("/refresh_lock")
async def refresh_lock(req: RefreshLockRequest) -> dict:
    """POST /api/rooms/refresh_lock — extend lock TTL during payment."""
    r = get_redis()
    key = f"soft_lock_id:{req.lock_id}"
    lock_data_raw = await r.get(key)
    if not lock_data_raw:
        return {"refreshed": False, "message": "Lock not found or expired"}

    import json as _json
    lock_data = _json.loads(lock_data_raw)
    room_type_id = lock_data.get("room_type_id", "")
    check_in     = lock_data.get("check_in", "")
    check_out    = lock_data.get("check_out", "")

    result = acquire_lock(
        room_type_id=room_type_id,
        check_in=check_in,
        check_out=check_out,
        session_id=req.psid,
        lock_minutes=15,
    )
    return {"refreshed": result.get("locked", False)}


# ── Internal helpers ───────────────────────────────────────────────────────────

async def _fetch_rooms(hotel_id: str, check_in: str, check_out: str, guests: int) -> list[dict]:
    """Fetch available room types from Supabase for given hotel + dates."""
    sb = get_supabase()
    try:
        result = await sb.table("room_types") \
            .select("*") \
            .eq("hotel_id", hotel_id) \
            .gte("max_occupancy", guests) \
            .eq("active", True) \
            .execute()
        rooms = result.data or []
    except Exception as exc:
        log.error("Fetch rooms failed for hotel %s: %s", hotel_id, exc)
        rooms = []

    # Normalise to card format
    formatted = []
    for r in rooms:
        formatted.append({
            "room_id":       r.get("id", ""),
            "name":          r.get("name", "Room"),
            "size_m2":       r.get("size_m2", 0),
            "bed_type":      r.get("bed_type", ""),
            "price_from":    r.get("price_usd", 0),
            "price_usd":     r.get("price_usd", 0),
            "currency":      "USD",
            "thumbnail_url": r.get("thumbnail_url", "https://via.placeholder.com/400x200"),
            "features":      r.get("features") or [],
            "max_occupancy": r.get("max_occupancy", 2),
            "amenities":     r.get("amenities") or [],
        })
    return formatted


def _count_nights(check_in: str, check_out: str) -> int:
    try:
        ci = datetime.strptime(check_in, "%Y-%m-%d").date()
        co = datetime.strptime(check_out, "%Y-%m-%d").date()
        return max((co - ci).days, 1)
    except ValueError:
        return 1


def _fmt_price(base_usd: float, currency: str, multiplier: float) -> str:
    """Quick price format for rate plan labels (truncated for button title)."""
    price = round(base_usd * multiplier)
    return f"{price:,}"
