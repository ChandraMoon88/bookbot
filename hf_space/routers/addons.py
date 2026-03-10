"""
hf_space/routers/addons.py
---------------------------
Module 5 — Add-Ons & Upsell, personalised and button-driven.

Triggered after guest_complete state.

Smart opener personalised by trip_purpose:
  honeymoon → champagne, rose petals, couples spa
  family    → extra bed, kids club, airport transfer, babysitting
  business  → late checkout, meeting room, airport sedan, laundry
  default   → top-rated add-ons for the hotel

Cart stored in Redis: user:{psid}:addon_cart (JSON list)

Backend:
  GET  /api/addons/recommend
  POST /api/addons/cart/add
  POST /api/addons/cart/remove
  GET  /api/addons/cart/{psid}
"""

from __future__ import annotations

import json
import logging

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile, get_booking_draft, set_booking_draft
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Addon category names ───────────────────────────────────────────────────────
_ADDON_CATEGORIES = [
    {"title": "🧖 Spa & Wellness",    "subtitle": "Massages, facials, treatments", "payload": "ADDON_CAT_spa"},
    {"title": "🍽️ Dining & Drinks",   "subtitle": "Welcome drinks, dining packages", "payload": "ADDON_CAT_dining"},
    {"title": "🚗 Transport",         "subtitle": "Airport transfers, car hire", "payload": "ADDON_CAT_transport"},
    {"title": "🎯 Activities",        "subtitle": "Tours, experiences, tickets", "payload": "ADDON_CAT_activities"},
    {"title": "🛏️ Room Extras",       "subtitle": "Upgrade, extra beds, décor", "payload": "ADDON_CAT_room"},
]


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_addons_start(psid: str, lang: str) -> tuple[list[dict], str]:
    """
    Entry point for add-ons. Personalised opener by trip_purpose.
    """
    mb = MessengerResponse(psid)
    draft   = await get_booking_draft(psid) or {}
    profile = await get_user_profile(psid) or {}

    guest        = draft.get("guest", {})
    trip_purpose = guest.get("trip_purpose", "leisure")
    hotel_id     = draft.get("hotel_id", "")
    currency     = profile.get("currency", "USD")

    # Fetch personalised add-ons
    addons = await _fetch_recommended_addons(hotel_id, trip_purpose)

    if not addons:
        # No add-ons available — skip to payment
        await set_user_state(psid, "reviewing_booking")
        return await _proceed_to_review(psid, lang)

    # Build personalised opener text
    opener = _build_opener_text(trip_purpose)

    # Build generic template cards (up to 3)
    addon_cards = _build_addon_cards(mb, addons[:3], currency)

    after_qr = mb.quick_replies(
        "What would you like to do?",
        [
            {"title": "See all add-ons 📋", "payload": "ADDON_SHOW_ALL"},
            {"title": "Skip add-ons →",     "payload": "ADDON_SKIP"},
            {"title": "Add all 3! 🎁",      "payload": "ADDON_ADD_ALL"},
        ],
    )

    await set_user_state(psid, "selecting_addons")
    return mb.send_sequence([mb.text(opener), addon_cards]) + [after_qr], "selecting_addons"


async def handle_addon_input(psid: str, text: str, lang: str) -> tuple[list[dict], str]:
    """Handle add-on postbacks and text within selecting_addons state."""
    mb = MessengerResponse(psid)

    if text == "ADDON_SKIP":
        return await _proceed_to_review(psid, lang)

    if text == "ADDON_ADD_ALL":
        return await _add_all_recommended(psid, lang)

    if text == "ADDON_SHOW_ALL":
        return await _show_addon_categories(psid, lang)

    if text.startswith("ADDON_CAT_"):
        category = text.replace("ADDON_CAT_", "")
        return await _show_category_addons(psid, category, lang)

    if text.startswith("ADDON_ADD_"):
        addon_id = text.replace("ADDON_ADD_", "")
        return await _add_addon_to_cart(psid, addon_id, lang)

    if text.startswith("ADDON_REMOVE_"):
        addon_id = text.replace("ADDON_REMOVE_", "")
        return await _remove_addon_from_cart(psid, addon_id, lang)

    if text.startswith("ADDON_INFO_"):
        addon_id = text.replace("ADDON_INFO_", "")
        return await _show_addon_info(psid, addon_id, lang)

    if text == "ADDON_CART_VIEW":
        return await _show_cart(psid, lang)

    if text == "ADDON_PROCEED":
        return await _proceed_to_review(psid, lang)

    # Unrecognised input
    return [mb.quick_replies(
        "What would you like to do?",
        [
            {"title": "See all add-ons 📋", "payload": "ADDON_SHOW_ALL"},
            {"title": "Skip add-ons →",     "payload": "ADDON_SKIP"},
            {"title": "🛍 View Cart",        "payload": "ADDON_CART_VIEW"},
        ],
    )], "selecting_addons"


async def _add_addon_to_cart(psid: str, addon_id: str, lang: str) -> tuple[list[dict], str]:
    mb = MessengerResponse(psid)
    r  = get_redis()

    # Fetch addon details
    sb = get_supabase()
    try:
        res = await sb.table("addons").select("*").eq("id", addon_id).single().execute()
        addon = res.data or {}
    except Exception:
        addon = {}

    if not addon:
        return [mb.text("Add-on not found.")], "selecting_addons"

    # Get/update cart
    raw_cart = await r.get(f"user:{psid}:addon_cart")
    cart: list = json.loads(raw_cart) if raw_cart else []
    if not any(a["addon_id"] == addon_id for a in cart):
        cart.append({
            "addon_id": addon_id,
            "name":     addon.get("name", ""),
            "price":    addon.get("price_usd", 0),
            "currency": "USD",
        })
        await r.set(f"user:{psid}:addon_cart", json.dumps(cart), ex=172800)

    subtotal = sum(a["price"] for a in cart)
    cart_items = [{"title": a["name"], "subtitle": f"${a['price']:.0f}"} for a in cart[-4:]]

    cart_list = mb.list_template(
        f"🛍 Your extras ({len(cart)} item(s) · ${subtotal:.0f} total)",
        cart_items,
    )
    msgs = mb.send_sequence([
        mb.text(f"✅ Added **{addon.get('name', '')}**!"),
        cart_list,
    ]) + [
        mb.quick_replies(
            "Want to add more?",
            [
                {"title": "Continue adding 🛍️", "payload": "ADDON_SHOW_ALL"},
                {"title": "Proceed to payment →","payload": "ADDON_PROCEED"},
            ],
        )
    ]
    return msgs, "selecting_addons"


async def _remove_addon_from_cart(psid: str, addon_id: str, lang: str) -> tuple[list[dict], str]:
    r = get_redis()
    raw_cart = await r.get(f"user:{psid}:addon_cart")
    cart: list = json.loads(raw_cart) if raw_cart else []
    cart = [a for a in cart if a["addon_id"] != addon_id]
    await r.set(f"user:{psid}:addon_cart", json.dumps(cart), ex=172800)
    mb = MessengerResponse(psid)
    return [mb.text("Removed from cart. ✅")], "selecting_addons"


async def _add_all_recommended(psid: str, lang: str) -> tuple[list[dict], str]:
    """Add top 3 recommended add-ons to cart."""
    draft = await get_booking_draft(psid) or {}
    guest = draft.get("guest", {})
    trip_purpose = guest.get("trip_purpose", "leisure")
    hotel_id     = draft.get("hotel_id", "")
    addons = await _fetch_recommended_addons(hotel_id, trip_purpose)
    for addon in addons[:3]:
        await _add_addon_to_cart.__wrapped__(psid, addon["id"], lang) if hasattr(_add_addon_to_cart, "__wrapped__") else None
        r = get_redis()
        raw_cart = await r.get(f"user:{psid}:addon_cart")
        cart: list = json.loads(raw_cart) if raw_cart else []
        aId = addon.get("id", "")
        if not any(a["addon_id"] == aId for a in cart):
            cart.append({"addon_id": aId, "name": addon.get("name",""), "price": addon.get("price_usd",0), "currency":"USD"})
            await r.set(f"user:{psid}:addon_cart", json.dumps(cart), ex=172800)

    mb = MessengerResponse(psid)
    return mb.send_sequence([mb.text("🎁 All 3 extras added to your booking!")]) + [
        mb.quick_replies("Ready to pay?", [
            {"title": "Proceed to payment →", "payload": "ADDON_PROCEED"},
            {"title": "🛍 View Cart",          "payload": "ADDON_CART_VIEW"},
        ])
    ], "selecting_addons"


async def _show_addon_categories(psid: str, lang: str) -> tuple[list[dict], str]:
    mb = MessengerResponse(psid)
    cat_list = mb.list_template(
        "Browse add-on categories:",
        [{"title": c["title"], "subtitle": c["subtitle"], "payload": c["payload"]} for c in _ADDON_CATEGORIES],
    )
    return mb.send_sequence([cat_list]), "selecting_addons"


async def _show_category_addons(psid: str, category: str, lang: str) -> tuple[list[dict], str]:
    mb = MessengerResponse(psid)
    draft = await get_booking_draft(psid) or {}
    hotel_id = draft.get("hotel_id", "")

    sb = get_supabase()
    try:
        res = await sb.table("addons") \
            .select("*") \
            .eq("hotel_id", hotel_id) \
            .contains("tags", [category]) \
            .eq("active", True) \
            .limit(6).execute()
        addons = res.data or []
    except Exception:
        addons = []

    if not addons:
        return [mb.text(f"No add-ons available in this category for your hotel.")], "selecting_addons"

    profile  = await get_user_profile(psid) or {}
    currency = profile.get("currency", "USD")

    cards = _build_addon_cards(mb, addons[:5], currency)
    return mb.send_sequence([cards]) + [
        mb.quick_replies("Add another category?", [
            {"title": c["title"][:20], "payload": c["payload"]} for c in _ADDON_CATEGORIES[:3]
        ] + [{"title": "Proceed to payment →", "payload": "ADDON_PROCEED"}])
    ], "selecting_addons"


async def _show_addon_info(psid: str, addon_id: str, lang: str) -> tuple[list[dict], str]:
    sb = get_supabase()
    mb = MessengerResponse(psid)
    try:
        res = await sb.table("addons").select("*").eq("id", addon_id).single().execute()
        addon = res.data or {}
    except Exception:
        addon = {}
    if not addon:
        return [mb.text("Add-on not found.")], "selecting_addons"

    desc = addon.get("description", "")[:300]
    return [
        mb.text(f"**{addon.get('name','')}** — ${addon.get('price_usd',0):.0f}\n\n{desc}"),
        mb.quick_replies(
            "Add this to your booking?",
            [
                {"title": f"Add this ✅", "payload": f"ADDON_ADD_{addon_id}"},
                {"title": "No thanks ❌", "payload": "ADDON_SHOW_ALL"},
            ],
        ),
    ], "selecting_addons"


async def _show_cart(psid: str, lang: str) -> tuple[list[dict], str]:
    mb = MessengerResponse(psid)
    r = get_redis()
    raw_cart = await r.get(f"user:{psid}:addon_cart")
    cart: list = json.loads(raw_cart) if raw_cart else []

    if not cart:
        return [mb.quick_replies(
            "Your cart is empty. Add something special?",
            [{"title": "Browse add-ons 🛍️", "payload": "ADDON_SHOW_ALL"},
             {"title": "Skip add-ons →",     "payload": "ADDON_SKIP"}],
        )], "selecting_addons"

    subtotal = sum(a["price"] for a in cart)
    items = [{"title": a["name"], "subtitle": f"${a['price']:.0f}", "payload": f"ADDON_REMOVE_{a['addon_id']}", "button_label": "Remove"} for a in cart[:4]]
    cart_list = mb.list_template(f"🛍 Cart — ${subtotal:.0f} total", items)
    return mb.send_sequence([cart_list]) + [
        mb.quick_replies("Ready to pay?", [
            {"title": "Proceed to payment →", "payload": "ADDON_PROCEED"},
            {"title": "Browse more 🛍️",       "payload": "ADDON_SHOW_ALL"},
        ])
    ], "selecting_addons"


async def _proceed_to_review(psid: str, lang: str) -> tuple[list[dict], str]:
    """Move to reviewing_booking state and show payment preview."""
    await set_user_state(psid, "reviewing_booking")
    from hf_space.routers.payment import handle_booking_review_start
    return await handle_booking_review_start(psid, lang)


# ── Internal helpers ───────────────────────────────────────────────────────────

async def _fetch_recommended_addons(hotel_id: str, trip_purpose: str) -> list[dict]:
    sb = get_supabase()
    # Map trip purpose to relevant tags
    tag_map = {
        "honeymoon": ["romantic", "spa", "dining"],
        "family":    ["family", "kids", "transfer"],
        "business":  ["business", "transport", "laundry"],
        "leisure":   ["popular", "spa", "activities"],
    }
    tags = tag_map.get(trip_purpose, ["popular"])

    try:
        res = await sb.table("addons") \
            .select("*") \
            .eq("hotel_id", hotel_id) \
            .eq("active", True) \
            .limit(10).execute()
        all_addons = res.data or []
    except Exception:
        return []

    # Score addons by tag overlap
    scored = []
    for a in all_addons:
        a_tags = a.get("tags") or []
        score = len(set(tags) & set(a_tags))
        scored.append((score, a))

    scored.sort(key=lambda x: -x[0])
    return [a for _, a in scored[:6]]


def _build_opener_text(trip_purpose: str) -> str:
    openers = {
        "honeymoon": "Since it's a special occasion 💍, I've picked these for you:",
        "family":    "Perfect for families — here are some popular extras 👨‍👩‍👧:",
        "business":  "Make your business trip smoother with these add-ons 💼:",
        "study":     "Make the most of your trip with these extras 🎓:",
        "medical":   "For a comfortable stay, here are some helpful extras:",
    }
    return openers.get(trip_purpose, "Make your stay even better with these popular extras 🌟:")


def _build_addon_cards(mb: MessengerResponse, addons: list[dict], currency: str) -> dict:
    """Build a generic template carousel for add-ons."""
    elements = []
    for a in addons[:10]:
        price_display = f"${a.get('price_usd', 0):.0f}"
        elements.append({
            "title": f"{a.get('name', 'Add-on')} — {price_display}"[:80],
            "subtitle": a.get("description", "")[:80],
            "image_url": a.get("image_url", "https://via.placeholder.com/400x200"),
            "buttons": [
                {"type": "postback", "title": "Add to Booking ✅",  "payload": f"ADDON_ADD_{a['id']}"},
                {"type": "postback", "title": "Tell me more 💬",    "payload": f"ADDON_INFO_{a['id']}"},
            ],
        })
    return {
        "recipient": {"id": mb.psid},
        "message": {
            "attachment": {
                "type": "template",
                "payload": {"template_type": "generic", "elements": elements},
            }
        },
    }


# ── API endpoints ──────────────────────────────────────────────────────────────

class CartAddRequest(BaseModel):
    psid:     str
    addon_id: str


class CartRemoveRequest(BaseModel):
    psid:     str
    addon_id: str


@router.get("/recommend")
async def recommend_addons(hotel_id: str, trip_purpose: str = "leisure", psid: str = "") -> dict:
    """GET /api/addons/recommend"""
    addons = await _fetch_recommended_addons(hotel_id, trip_purpose)
    return {"addons": addons}


@router.post("/cart/add")
async def cart_add(req: CartAddRequest) -> dict:
    r = get_redis()
    raw_cart = await r.get(f"user:{req.psid}:addon_cart")
    cart: list = json.loads(raw_cart) if raw_cart else []
    sb = get_supabase()
    try:
        res = await sb.table("addons").select("id,name,price_usd").eq("id", req.addon_id).single().execute()
        addon = res.data or {}
    except Exception:
        return {"success": False, "error": "Addon not found"}
    if not any(a["addon_id"] == req.addon_id for a in cart):
        cart.append({"addon_id": req.addon_id, "name": addon.get("name",""), "price": addon.get("price_usd",0), "currency":"USD"})
        await r.set(f"user:{req.psid}:addon_cart", json.dumps(cart), ex=172800)
    return {"success": True, "cart_count": len(cart)}


@router.post("/cart/remove")
async def cart_remove(req: CartRemoveRequest) -> dict:
    r = get_redis()
    raw_cart = await r.get(f"user:{req.psid}:addon_cart")
    cart: list = json.loads(raw_cart) if raw_cart else []
    cart = [a for a in cart if a["addon_id"] != req.addon_id]
    await r.set(f"user:{req.psid}:addon_cart", json.dumps(cart), ex=172800)
    return {"success": True, "cart_count": len(cart)}


@router.get("/cart/{psid}")
async def get_cart(psid: str) -> dict:
    r = get_redis()
    raw_cart = await r.get(f"user:{psid}:addon_cart")
    cart: list = json.loads(raw_cart) if raw_cart else []
    subtotal = sum(a.get("price", 0) for a in cart)
    return {"cart": cart, "subtotal_usd": subtotal, "count": len(cart)}
