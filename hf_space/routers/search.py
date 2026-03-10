"""
hf_space/routers/search.py
---------------------------
Module 2 — Hotel Search & Discovery with full button-first UX.

Conversation flow (Section 6, Task Prompt 2):
  Step 1 — Destination (quick-reply cities or free text)
  Step 2 — Dates (quick replies: Tonight/Tomorrow/This weekend…)
  Step 3 — Guests (quick replies: 1–4+)
  Step 4 — Optional filters (stars, meal plan, amenities — or Skip)
  Step 5 — Results as hotel-card carousel (top 3 first, "Show more →")

Backend:
  POST /api/search/hotels
  GET  /api/search/suggest?query=&lang=
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import structlog
from fastapi import APIRouter, Query
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile, set_user_profile
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse, validate_messages
from services.search_service.elasticsearch_client import search_hotels as es_search
from services.search_service.ranker import rank_hotels

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Popular destinations quick reply list ─────────────────────────────────────
_POPULAR_CITIES = [
    {"title": "🗼 Paris",    "payload": "CITY_Paris"},
    {"title": "🗾 Tokyo",   "payload": "CITY_Tokyo"},
    {"title": "🏙️ Dubai",   "payload": "CITY_Dubai"},
    {"title": "🌴 Bali",    "payload": "CITY_Bali"},
    {"title": "🎭 London",  "payload": "CITY_London"},
    {"title": "🗽 New York","payload": "CITY_New_York"},
    {"title": "✏️ Type city…", "payload": "CITY_MANUAL"},
]

_DATE_OPTIONS = [
    {"title": "Tonight 🌙",     "payload": "CHECKIN_tonight"},
    {"title": "Tomorrow ☀️",    "payload": "CHECKIN_tomorrow"},
    {"title": "This weekend 🎉", "payload": "CHECKIN_weekend"},
    {"title": "Next week 📅",   "payload": "CHECKIN_next_week"},
    {"title": "Next month 🗓",  "payload": "CHECKIN_next_month"},
    {"title": "📅 Pick date",   "payload": "CHECKIN_picker"},
]

_GUEST_OPTIONS = [
    {"title": "1 Guest 🧍",   "payload": "GUESTS_1"},
    {"title": "2 Guests 👫",  "payload": "GUESTS_2"},
    {"title": "3 Guests 👨‍👩‍👦", "payload": "GUESTS_3"},
    {"title": "4 Guests 👨‍👩‍👧‍👦","payload": "GUESTS_4"},
    {"title": "4+ Guests 🏢", "payload": "GUESTS_4plus"},
]

_FILTER_OPTIONS = [
    {"title": "⭐ 3 Stars",          "payload": "FILTER_stars_3"},
    {"title": "⭐⭐ 4 Stars",        "payload": "FILTER_stars_4"},
    {"title": "⭐⭐⭐ 5 Stars",      "payload": "FILTER_stars_5"},
    {"title": "🍳 Breakfast incl.",  "payload": "FILTER_breakfast"},
    {"title": "🏊 Pool",             "payload": "FILTER_pool"},
    {"title": "💼 Business",         "payload": "FILTER_business"},
    {"title": "Skip filters →",      "payload": "FILTER_skip"},
]

_SORT_OPTIONS = [
    {"title": "Show 4 more →",    "payload": "SEARCH_SHOW_MORE"},
    {"title": "Change filters 🔧","payload": "SEARCH_CHANGE_FILTERS"},
    {"title": "New search 🔄",    "payload": "SEARCH_NEW"},
    {"title": "Sort by price ↕",  "payload": "SEARCH_SORT_PRICE"},
]


# ── Request/response models ───────────────────────────────────────────────────

class HotelSearchRequest(BaseModel):
    psid:       str
    city:       str
    check_in:   str
    check_out:  str
    num_guests: int = 1
    filters:    dict = {}


# ── Conversation handlers (called by app.py router) ───────────────────────────

async def handle_search_start(
    psid: str, text: str, lang: str
) -> tuple[list[dict], str]:
    """
    Entry point after greeting.
    Send destination step: city quick replies.
    """
    mb = MessengerResponse(psid)
    await set_user_state(psid, "searching")

    # Clear any stale search session
    r = get_redis()
    await r.delete(f"user:{psid}:search_draft")

    msgs = mb.send_sequence([
        mb.text("Where would you like to stay? 🌍"),
        mb.quick_replies(
            "Choose a popular destination or type any city:",
            _POPULAR_CITIES,
        ),
    ])
    return msgs, "searching"


async def handle_search_input(
    psid: str, text: str, lang: str
) -> tuple[list[dict], str]:
    """
    Stateful handler — walks the user through search steps 1–5.
    Draft stored in Redis: user:{psid}:search_draft
    """
    r = get_redis()
    raw = await r.get(f"user:{psid}:search_draft")
    draft: dict = json.loads(raw) if raw else {}
    mb = MessengerResponse(psid)

    # ── Step 1: resolve city ──────────────────────────────────────────────────
    if "city" not in draft:
        city = _resolve_city(text)
        if not city:
            return (
                [mb.quick_replies(
                    "I didn't find that city. Try one of these or type a city name:",
                    _POPULAR_CITIES,
                )],
                "searching",
            )
        draft["city"] = city
        await r.set(f"user:{psid}:search_draft", json.dumps(draft), ex=3600)

        msgs = mb.send_sequence([
            mb.text(f"Great, searching in **{city}**! 🌟"),
            mb.text("When are you checking in? 📅"),
            mb.quick_replies("Pick a check-in date:", _DATE_OPTIONS),
        ])
        return msgs, "searching"

    # ── Step 2: check-in date ─────────────────────────────────────────────────
    if "check_in" not in draft:
        check_in = _resolve_date(text)
        if not check_in:
            return (
                [mb.quick_replies("Please pick a check-in date:", _DATE_OPTIONS)],
                "searching",
            )
        draft["check_in"] = check_in
        await r.set(f"user:{psid}:search_draft", json.dumps(draft), ex=3600)

        # Ask check-out
        checkout_options = _checkout_options(check_in)
        msgs = mb.send_sequence([
            mb.text(f"Check-in: **{check_in}** ✅"),
            mb.quick_replies("And when are you checking out? 🏁", checkout_options),
        ])
        return msgs, "searching"

    # ── Step 3: check-out date ────────────────────────────────────────────────
    if "check_out" not in draft:
        check_out = _resolve_checkout_date(text, draft["check_in"])
        if not check_out:
            co_opts = _checkout_options(draft["check_in"])
            return (
                [mb.quick_replies("Please pick a check-out date:", co_opts)],
                "searching",
            )
        draft["check_out"] = check_out
        await r.set(f"user:{psid}:search_draft", json.dumps(draft), ex=3600)

        msgs = mb.send_sequence([
            mb.text(f"Check-out: **{check_out}** ✅"),
            mb.quick_replies("How many guests? 👥", _GUEST_OPTIONS),
        ])
        return msgs, "searching"

    # ── Step 4: guests ────────────────────────────────────────────────────────
    if "num_guests" not in draft:
        guests = _resolve_guests(text)
        if guests == -1:
            # 4+ — ask for free text number
            return (
                [mb.text("How many guests? Please type the number (e.g., 6):")],
                "searching",
            )
        if guests is None:
            # Try to parse as free text integer
            try:
                guests = int(text.strip())
            except ValueError:
                return (
                    [mb.quick_replies("Please select the number of guests:", _GUEST_OPTIONS)],
                    "searching",
                )
        draft["num_guests"] = guests
        await r.set(f"user:{psid}:search_draft", json.dumps(draft), ex=3600)

        msgs = mb.send_sequence([
            mb.text(f"{guests} guest(s) ✅"),
            mb.quick_replies("Any preferences? (optional)", _FILTER_OPTIONS),
        ])
        return msgs, "searching"

    # ── Step 5: filters → run search ─────────────────────────────────────────
    if "filters" not in draft:
        filters = _resolve_filter(text)
        draft["filters"] = filters
        await r.set(f"user:{psid}:search_draft", json.dumps(draft), ex=3600)

    # Run search
    return await _run_search(psid, draft, lang)


async def handle_hotel_text_input(
    psid: str, text: str, lang: str
) -> tuple[list[dict], str]:
    """Handle free text while viewing hotel results (e.g., pagination commands)."""
    mb = MessengerResponse(psid)
    tl = text.lower()
    if any(k in tl for k in ("more", "next", "show")):
        return await _show_more_hotels(psid, lang)
    if any(k in tl for k in ("filter", "change", "sort")):
        r = get_redis()
        raw = await r.get(f"user:{psid}:search_draft")
        draft = json.loads(raw) if raw else {}
        if "filters" in draft:
            del draft["filters"]
            await r.set(f"user:{psid}:search_draft", json.dumps(draft), ex=3600)
        return (
            [mb.quick_replies("Update your preferences:", _FILTER_OPTIONS)],
            "searching",
        )
    # New search
    return await handle_search_start(psid, text, lang)


async def handle_hotel_details(
    psid: str, hotel_id: str, lang: str
) -> tuple[list[dict], str]:
    """Return brief details for a specific hotel."""
    mb = MessengerResponse(psid)
    try:
        sb = get_supabase()
        result = await sb.table("hotels").select("*").eq("id", hotel_id).single().execute()
        h = result.data
        if not h:
            return [mb.text("Hotel not found. Please try another.")], "viewing_hotels"

        text_parts = [
            f"🏨 **{h.get('name')}** {'⭐' * int(h.get('star_rating', 0))}",
            f"📍 {h.get('address', '')}",
            f"📝 {h.get('description', '')[:300]}",
            f"🏊 Amenities: {', '.join((h.get('amenities') or [])[:6])}",
        ]
        detail_text = "\n".join(text_parts)
        msgs = mb.send_sequence([
            mb.text(detail_text),
            mb.quick_replies(
                "What would you like to do?",
                [
                    {"title": "✅ Select Hotel",  "payload": f"HOTEL_SELECT_{hotel_id}"},
                    {"title": "❓ Ask a Question","payload": f"HOTEL_FAQ_{hotel_id}"},
                    {"title": "← Back to list",  "payload": "SEARCH_SHOW_MORE"},
                ],
            ),
        ])
        return msgs, "viewing_hotels"
    except Exception as exc:
        log.error("hotel_details error: %s", exc)
        return [mb.text("Could not load hotel details. Please try again.")], "viewing_hotels"


# ── API endpoint ──────────────────────────────────────────────────────────────

@router.post("/hotels")
async def search_hotels_endpoint(req: HotelSearchRequest) -> dict:
    """
    POST /api/search/hotels
    Full hotel search with ES + ranking + currency conversion.
    """
    profile = await get_user_profile(req.psid)
    currency = (profile or {}).get("currency", "USD")

    # Call Elasticsearch
    try:
        es_result = es_search(
            city=req.city,
            check_in=req.check_in,
            check_out=req.check_out,
            num_guests=req.num_guests,
            filters=req.filters,
        )
        hits = es_result.get("hits", [])
    except Exception as exc:
        log.error("ES search failed: %s", exc)
        hits = []

    # Rank
    hotels = rank_hotels(hits, psid=req.psid)

    # Currency conversion
    rates = await _get_fx_rates(currency)
    for h in hotels:
        usd_price = h.get("price_from_usd", 0)
        h["price_from"] = round(usd_price * rates.get(currency, 1))
        h["currency"] = currency
        h["price_usd"] = usd_price

    # Cache results
    r = get_redis()
    cache_key = f"search:{req.city}:{req.check_in}:{req.check_out}"
    await r.set(cache_key, json.dumps(hotels[:10]), ex=300)

    return {"hotels": hotels[:10], "total": len(hotels)}


@router.get("/suggest")
async def suggest_cities(
    query: str = Query(..., min_length=1),
    lang: str = Query("en"),
) -> dict:
    """
    GET /api/search/suggest?query=tok&lang=en
    City autocomplete from pre-loaded city list in Redis.
    """
    r = get_redis()
    all_cities_raw = await r.get("cities:all")
    if not all_cities_raw:
        return {"suggestions": []}

    all_cities: list[str] = json.loads(all_cities_raw)
    q = query.lower()
    matches = [c for c in all_cities if q in c.lower()][:8]
    return {"suggestions": matches}


# ── Internal helpers ───────────────────────────────────────────────────────────

async def _run_search(
    psid: str, draft: dict, lang: str
) -> tuple[list[dict], str]:
    """Execute the hotel search and return Messenger hotel cards."""
    mb = MessengerResponse(psid)
    city       = draft["city"]
    check_in   = draft["check_in"]
    check_out  = draft["check_out"]
    num_guests = draft.get("num_guests", 1)
    filters    = draft.get("filters", {})

    profile = await get_user_profile(psid)
    currency = (profile or {}).get("currency", "USD")

    thinking_msgs = [mb.text(f"Searching hotels in {city}… 🔍")]

    try:
        es_result = es_search(
            city=city,
            check_in=check_in,
            check_out=check_out,
            num_guests=num_guests,
            filters=filters,
        )
        hits = es_result.get("hits", [])
    except Exception as exc:
        log.error("Search failed: %s", exc)
        hits = []

    if not hits:
        msgs = thinking_msgs + [
            mb.quick_replies(
                f"No hotels found in {city} for those dates. Try adjusting your search:",
                [
                    {"title": "📅 Change dates",   "payload": "SEARCH_CHANGE_DATES"},
                    {"title": "🏙️ Change city",    "payload": "SEARCH_CHANGE_CITY"},
                    {"title": "🔄 New search",      "payload": "SEARCH_NEW"},
                ],
            )
        ]
        return msgs, "searching"

    hotels = rank_hotels(hits, psid=psid)
    rates  = await _get_fx_rates(currency)

    for h in hotels:
        usd_price = h.get("price_from_usd", 0)
        h["price_from"] = round(usd_price * rates.get(currency, 1))
        h["currency"]   = currency
        h["price_usd"]  = usd_price
        h.setdefault("thumbnail_url", "https://via.placeholder.com/400x200")
        h.setdefault("top_feature", "Great location")

    # Store full results for pagination
    r = get_redis()
    await r.set(f"user:{psid}:search_results", json.dumps(hotels[:10]), ex=600)
    await r.set(f"user:{psid}:search_offset", "0", ex=600)
    await set_user_state(psid, "viewing_hotels")

    top3 = hotels[:3]
    card_msg = mb.hotel_cards(top3)

    msgs = thinking_msgs + mb.send_sequence([card_msg]) + [
        mb.quick_replies(
            f"Found {len(hotels)} hotels in {city} 🎉",
            _sort_options_with_more(len(hotels)),
        )
    ]
    return msgs, "viewing_hotels"


async def _show_more_hotels(psid: str, lang: str) -> tuple[list[dict], str]:
    r = get_redis()
    raw     = await r.get(f"user:{psid}:search_results")
    raw_off = await r.get(f"user:{psid}:search_offset")
    mb = MessengerResponse(psid)

    if not raw:
        return [mb.text("No search results found. Please start a new search.")], "new"

    hotels = json.loads(raw)
    offset = int(raw_off or 0) + 3
    batch = hotels[offset: offset + 3]

    if not batch:
        return (
            [mb.quick_replies("No more results. Start a new search?", [
                {"title": "🔄 New search", "payload": "SEARCH_NEW"},
            ])],
            "viewing_hotels",
        )

    await r.set(f"user:{psid}:search_offset", str(offset), ex=600)
    card_msg = mb.hotel_cards(batch)
    remaining = len(hotels) - offset - len(batch)
    opts = _sort_options_with_more(remaining)
    return mb.send_sequence([card_msg]) + [mb.quick_replies("More options:", opts)], "viewing_hotels"


def _sort_options_with_more(remaining: int) -> list[dict]:
    opts = list(_SORT_OPTIONS)
    if remaining <= 0:
        opts = [o for o in opts if o["payload"] != "SEARCH_SHOW_MORE"]
    else:
        for o in opts:
            if o["payload"] == "SEARCH_SHOW_MORE":
                o["title"] = f"Show {min(remaining,3)} more →"
    return opts[:5]


def _resolve_city(text: str) -> str | None:
    """Extract city from text or postback payload."""
    if text.startswith("CITY_"):
        city_raw = text[5:].replace("_", " ")
        return None if city_raw == "MANUAL" else city_raw
    t = text.strip()
    return t if len(t) >= 2 else None


def _resolve_date(text: str) -> str | None:
    """Resolve date quick-reply payload or free text to YYYY-MM-DD."""
    today = date.today()
    payloads = {
        "CHECKIN_tonight":    str(today),
        "CHECKIN_tomorrow":   str(today + timedelta(days=1)),
        "CHECKIN_weekend":    str(_next_weekday(today, 5)),
        "CHECKIN_next_week":  str(today + timedelta(weeks=1)),
        "CHECKIN_next_month": str(today + timedelta(days=30)),
        "CHECKIN_picker":     None,  # handled by webview
    }
    if text in payloads:
        return payloads[text]
    # Try ISO parse
    try:
        from datetime import datetime
        parsed = datetime.strptime(text.strip(), "%Y-%m-%d").date()
        if parsed >= today:
            return str(parsed)
    except ValueError:
        pass
    return None


def _resolve_checkout_date(text: str, check_in_str: str) -> str | None:
    """Resolve checkout quick replies (e.g. CHECKOUT_1night) or free text."""
    from datetime import datetime
    try:
        check_in = datetime.strptime(check_in_str, "%Y-%m-%d").date()
    except ValueError:
        return None

    night_map = {
        "CHECKOUT_1night": check_in + timedelta(days=1),
        "CHECKOUT_2nights": check_in + timedelta(days=2),
        "CHECKOUT_3nights": check_in + timedelta(days=3),
        "CHECKOUT_5nights": check_in + timedelta(days=5),
        "CHECKOUT_7nights": check_in + timedelta(days=7),
    }
    if text in night_map:
        return str(night_map[text])
    # ISO date parse
    try:
        parsed = datetime.strptime(text.strip(), "%Y-%m-%d").date()
        if parsed > check_in:
            return str(parsed)
    except ValueError:
        pass
    return None


def _checkout_options(check_in_str: str) -> list[dict]:
    """Build dynamic checkout quick replies relative to check-in."""
    from datetime import datetime
    try:
        ci = datetime.strptime(check_in_str, "%Y-%m-%d").date()
    except ValueError:
        ci = date.today()

    return [
        {"title": f"1 night ({ci + timedelta(1)})",  "payload": "CHECKOUT_1night"},
        {"title": f"2 nights ({ci + timedelta(2)})", "payload": "CHECKOUT_2nights"},
        {"title": f"3 nights ({ci + timedelta(3)})", "payload": "CHECKOUT_3nights"},
        {"title": f"5 nights ({ci + timedelta(5)})", "payload": "CHECKOUT_5nights"},
        {"title": f"7 nights ({ci + timedelta(7)})", "payload": "CHECKOUT_7nights"},
        {"title": "📅 Pick date",                    "payload": "CHECKOUT_picker"},
    ]


def _resolve_guests(text: str) -> int | None:
    """Parse guest quick reply or return None for free text."""
    mapping = {
        "GUESTS_1": 1, "GUESTS_2": 2, "GUESTS_3": 3, "GUESTS_4": 4,
    }
    if text in mapping:
        return mapping[text]
    if text == "GUESTS_4plus":
        return -1  # signal to ask free text
    return None


def _resolve_filter(text: str) -> dict:
    """Build filters dict from quick-reply payload."""
    filters: dict = {}
    if text == "FILTER_skip":
        return filters
    if text.startswith("FILTER_stars_"):
        filters["stars"] = int(text[-1])
    elif text == "FILTER_breakfast":
        filters["amenities"] = ["breakfast"]
    elif text == "FILTER_pool":
        filters["amenities"] = ["pool"]
    elif text == "FILTER_business":
        filters["amenities"] = ["business_center"]
    return filters


def _next_weekday(d: date, weekday: int) -> date:
    """Return the next occurrence of weekday (0=Mon, 5=Sat, 6=Sun)."""
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)


async def _get_fx_rates(currency: str) -> dict:
    """
    Fetch FX rate for currency from Upstash Redis (cached 1h) or Open Exchange Rates.
    Returns {currency: rate_vs_usd}.
    """
    if currency == "USD":
        return {"USD": 1.0}

    r = get_redis()
    cached = await r.get(f"fx:{currency}")
    if cached:
        return {currency: float(cached)}

    import os
    import httpx as _httpx
    app_id = os.environ.get("OPEN_EXCHANGE_RATES_APP_ID", "")
    if not app_id:
        return {currency: 1.0}

    try:
        async with _httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                "https://openexchangerates.org/api/latest.json",
                params={"app_id": app_id, "symbols": currency, "base": "USD"},
            )
            data = resp.json()
            rate = data["rates"].get(currency, 1.0)
            await r.set(f"fx:{currency}", str(rate), ex=3600)
            return {currency: rate}
    except Exception as exc:
        log.warning("FX fetch failed for %s: %s", currency, exc)
        return {currency: 1.0}
