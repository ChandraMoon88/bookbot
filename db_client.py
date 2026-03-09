"""
db_client.py
------------
All Supabase REST API calls for BookBot.
Uses SUPABASE_URL + SUPABASE_KEY (service_role) — works on free tier, no IPv4 needed.

Tables used:
  users        — register / lookup guest
  hotels       — hotel info, amenities, photos
  inventory    — room types, availability, rate_plans per date
  bookings     — create / read / cancel bookings
  sessions     — conversation state persistence
  engagements  — ratings, feedback
  v_available_rooms   — view: available rooms per hotel per date
  v_booking_summary   — view: booking list for a user
"""

from __future__ import annotations
import os
import json
import urllib.request
import urllib.error
import urllib.parse
import logging
from datetime import datetime, date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")


def _headers() -> dict:
    return {
        "apikey":          SUPABASE_KEY,
        "Authorization":   f"Bearer {SUPABASE_KEY}",
        "Content-Type":    "application/json",
        "Prefer":          "return=representation",
    }


def _get(endpoint: str, params: dict | list | None = None) -> list | dict:
    """GET from Supabase REST API.
    params can be a dict or a list of (key, value) tuples to allow
    duplicate query-string keys (needed for range queries like date=gte.X&date=lt.Y).
    """
    url = SUPABASE_URL + endpoint
    if params:
        if isinstance(params, dict):
            url += "?" + urllib.parse.urlencode(params)
        else:
            url += "?" + urllib.parse.urlencode(params)  # list of tuples
    req = urllib.request.Request(url, headers=_headers())
    try:
        r = urllib.request.urlopen(req, timeout=15)
        return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        logger.error("GET %s → %s %s", endpoint, e.code, e.read().decode()[:200])
        return []
    except Exception as e:
        logger.error("GET %s error: %s", endpoint, e)
        return []


def _post(endpoint: str, body: dict) -> dict | None:
    url = SUPABASE_URL + endpoint
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        r = urllib.request.urlopen(req, timeout=15)
        result = json.loads(r.read().decode())
        return result[0] if isinstance(result, list) and result else result
    except urllib.error.HTTPError as e:
        logger.error("POST %s → %s %s", endpoint, e.code, e.read().decode()[:200])
        return None
    except Exception as e:
        logger.error("POST %s error: %s", endpoint, e)
        return None


def _patch(endpoint: str, body: dict, match: dict) -> dict | None:
    params = urllib.parse.urlencode({k: f"eq.{v}" for k, v in match.items()})
    url = SUPABASE_URL + endpoint + "?" + params
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=_headers(), method="PATCH")
    try:
        r = urllib.request.urlopen(req, timeout=15)
        result = json.loads(r.read().decode())
        return result[0] if isinstance(result, list) and result else result
    except urllib.error.HTTPError as e:
        logger.error("PATCH %s → %s %s", endpoint, e.code, e.read().decode()[:200])
        return None
    except Exception as e:
        logger.error("PATCH %s error: %s", endpoint, e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HOTEL SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def search_hotels(city: str, check_in: str, check_out: str,
                  num_adults: int = 1, num_children: int = 0) -> list[dict]:
    """
    Search active hotels in a city that have available rooms for the dates.
    Queries the inventory table directly (no view dependency).
    Returns list of hotel dicts with available room types embedded.
    """
    # Step 1: find active hotels in the city
    rows = _get("/rest/v1/hotels", {
        "city":      f"ilike.{city}",
        "is_active": "eq.true",
        "select":    "id,name,description,city,country,star_rating,amenities,"
                     "currency,check_in_time,check_out_time,thumbnail_url,photos",
    })
    if not rows:
        return []

    results = []
    for hotel in rows:
        hotel_id = hotel["id"]
        nights   = _nights(check_in, check_out)

        # Query inventory with a date range using list-of-tuples for duplicate keys
        avail_rows = _get("/rest/v1/inventory", [
            ("hotel_id",        f"eq.{hotel_id}"),
            ("date",            f"gte.{check_in}"),
            ("date",            f"lt.{check_out}"),
            ("available_count", "gt.0"),
            ("is_blackout",     "eq.false"),
            ("select",          "room_type_code,room_type_name,date,"
                                "available_count,max_adults,max_children,rate_plans"),
        ])
        if not avail_rows:
            continue

        # Build expected date set for the stay
        all_dates: set[str] = set()
        d = datetime.strptime(check_in, "%Y-%m-%d").date()
        end_d = datetime.strptime(check_out, "%Y-%m-%d").date()
        while d < end_d:
            all_dates.add(str(d))
            d += timedelta(days=1)

        # Count how many nights each room type appears as available
        from collections import defaultdict
        date_per_room: dict[str, set] = defaultdict(set)
        room_info: dict[str, dict] = {}
        for row in avail_rows:
            dt  = row["date"][:10]
            rtc = row["room_type_code"]
            date_per_room[rtc].add(dt)
            if rtc not in room_info:
                room_info[rtc] = row

        available_rooms = []
        for rtc, info in room_info.items():
            # Room must be available every night of the stay
            if not all_dates.issubset(date_per_room[rtc]):
                continue
            rate_plans = info.get("rate_plans") or {}
            if isinstance(rate_plans, str):
                try:
                    rate_plans = json.loads(rate_plans)
                except Exception:
                    rate_plans = {}
            price_per_night = _cheapest_rate(rate_plans)
            available_rooms.append({
                "room_type_code":  rtc,
                "room_type_name":  info.get("room_type_name", rtc),
                "max_adults":      info.get("max_adults", 2),
                "max_children":    info.get("max_children", 0),
                "price_per_night": price_per_night,
                "total_price":     price_per_night * nights if price_per_night else None,
                "rate_plans":      rate_plans,
            })

        if not available_rooms:
            continue

        # Prefer rooms that fit the guest count; fall back to showing all
        fitting = [
            r for r in available_rooms
            if r["max_adults"] >= num_adults and r["max_children"] >= num_children
        ]
        if not fitting:
            fitting = available_rooms

        results.append({**hotel, "available_rooms": fitting, "nights": nights})

    return results


def _nights(check_in: str, check_out: str) -> int:
    try:
        d1 = datetime.strptime(check_in, "%Y-%m-%d").date()
        d2 = datetime.strptime(check_out, "%Y-%m-%d").date()
        return max((d2 - d1).days, 1)
    except Exception:
        return 1


def _cheapest_rate(rate_plans: dict) -> float | None:
    """Return the minimum price from rate_plans JSON."""
    if not rate_plans:
        return None
    prices = []
    # rate_plans may be {plan_code: {price_per_night: X}} or [{price: X}] etc.
    if isinstance(rate_plans, dict):
        for v in rate_plans.values():
            if isinstance(v, dict):
                for key in ("price_per_night", "price", "rate", "amount"):
                    if key in v and v[key]:
                        try:
                            prices.append(float(v[key]))
                        except Exception:
                            pass
    elif isinstance(rate_plans, list):
        for v in rate_plans:
            if isinstance(v, dict):
                for key in ("price_per_night", "price", "rate", "amount"):
                    if key in v and v[key]:
                        try:
                            prices.append(float(v[key]))
                        except Exception:
                            pass
    return min(prices) if prices else None


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC HOTEL SEARCH  (HuggingFace sentence-transformers, CPU-safe)
# ─────────────────────────────────────────────────────────────────────────────

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_model


def semantic_hotel_search(query: str, hotels: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-rank a list of hotels using cosine similarity between the user's query
    and a text representation of each hotel (name + description + amenities).
    Falls back to the original list if embeddings not available.
    """
    if not hotels:
        return hotels
    try:
        model = _get_embed_model()
        import numpy as np

        # Build hotel text corpus
        def _hotel_text(h: dict) -> str:
            amenities = h.get("amenities") or []
            if isinstance(amenities, str):
                try:
                    amenities = json.loads(amenities)
                except Exception:
                    amenities = [amenities]
            return (
                f"{h.get('name', '')} in {h.get('city', '')}. "
                f"{h.get('description', '')}. "
                f"Amenities: {', '.join(str(a) for a in amenities)}. "
                f"{h.get('star_rating', '')} stars."
            )

        corpus   = [_hotel_text(h) for h in hotels]
        q_emb    = model.encode([query], normalize_embeddings=True)
        c_embs   = model.encode(corpus,  normalize_embeddings=True)
        scores   = (c_embs @ q_emb.T).flatten()
        ranked   = sorted(zip(scores, hotels), key=lambda x: x[0], reverse=True)
        return [h for _, h in ranked[:top_k]]
    except Exception as e:
        logger.warning("Semantic search failed (%s), using original order.", e)
        return hotels[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# USERS
# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_user(messenger_id: str, first_name: str = "",
                       last_name: str = "", lang: str = "en") -> dict | None:
    """
    Look up user by Messenger platform ID stored in the phone field
    as 'messenger:{messenger_id}'. This avoids unreliable JSONB filtering.
    Creates a new user record if not found.
    """
    phone_key = f"messenger:{messenger_id}"
    rows = _get("/rest/v1/users", {
        "phone":  f"eq.{phone_key}",
        "select": "id,first_name,last_name,preferred_language,loyalty_tier,"
                  "loyalty_points_available",
    })
    if rows:
        return rows[0]

    # Create new user
    return _post("/rest/v1/users", {
        "first_name":         first_name or "Guest",
        "last_name":          last_name  or "",
        "preferred_language": lang,
        "phone":              phone_key,
        "oauth_accounts":     [{"provider": "messenger", "provider_user_id": messenger_id}],
        "is_active":          True,
        "loyalty_tier":       "bronze",
        "loyalty_points_available": 0,
        "loyalty_points_total":     0,
    })


def get_user_bookings(user_id: str) -> list[dict]:
    """Fetch user's recent bookings. Tries the view first, then falls back
    to a direct bookings + hotels query if the view does not exist."""
    rows = _get("/rest/v1/v_booking_summary", {
        "user_id": f"eq.{user_id}",
        "order":   "created_at.desc",
        "limit":   "5",
        "select":  "booking_reference,hotel_name,city,country,room_type_code,"
                   "check_in,check_out,num_adults,total_amount,currency,"
                   "status,payment_status,primary_guest_name",
    })
    if rows:  # view exists and returned data
        return rows

    # Fallback: query bookings table directly and enrich with hotel details
    bookings = _get("/rest/v1/bookings", {
        "user_id": f"eq.{user_id}",
        "order":   "created_at.desc",
        "limit":   "5",
        "select":  "booking_reference,hotel_id,room_type_code,check_in,check_out,"
                   "num_adults,num_children,total_amount,currency,status,"
                   "payment_status,primary_guest_name",
    })
    if not bookings:
        return []

    # Fetch hotel names in bulk
    hotel_ids = list({b["hotel_id"] for b in bookings if b.get("hotel_id")})
    hotel_map: dict[str, dict] = {}
    for hid in hotel_ids:
        h = _get("/rest/v1/hotels", {"id": f"eq.{hid}", "select": "id,name,city,country"})
        if h:
            hotel_map[hid] = h[0]

    for b in bookings:
        hd = hotel_map.get(b.get("hotel_id", ""), {})
        b["hotel_name"] = hd.get("name", "Hotel")
        b["city"]       = hd.get("city", "")
        b["country"]    = hd.get("country", "")
    return bookings


# ─────────────────────────────────────────────────────────────────────────────
# BOOKINGS
# ─────────────────────────────────────────────────────────────────────────────

def create_booking(user_id: str, hotel_id: str, room_type_code: str,
                   check_in: str, check_out: str, num_adults: int,
                   num_children: int, primary_guest_name: str,
                   primary_guest_email: str, primary_guest_phone: str,
                   total_amount: float, currency: str,
                   special_requests: str = "", rate_plan: str = "room_only",
                   meal_plan: str = "room_only") -> dict | None:
    import random, string
    ref = "BB" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
    # NOTE: 'nights' is a GENERATED ALWAYS AS column in the DB schema —
    # do NOT pass it explicitly or PostgreSQL will raise an error.
    return _post("/rest/v1/bookings", {
        "booking_reference":    ref,
        "user_id":              user_id,
        "hotel_id":             hotel_id,
        "room_type_code":       room_type_code,
        "rate_plan":            rate_plan,
        "meal_plan":            meal_plan,
        "check_in":             check_in,
        "check_out":            check_out,
        "num_adults":           num_adults,
        "num_children":         num_children,
        "num_rooms":            1,
        "primary_guest_name":   primary_guest_name,
        "primary_guest_email":  primary_guest_email,
        "primary_guest_phone":  primary_guest_phone or "",
        "total_amount":         total_amount,
        "base_amount":          total_amount,
        "tax_amount":           0,
        "currency":             currency,
        "special_requests":     special_requests,
        "status":               "confirmed",
        "payment_status":       "pending",
    })


def cancel_booking(booking_reference: str, reason: str = "guest_request") -> bool:
    result = _patch(
        "/rest/v1/bookings",
        {"status": "cancelled", "cancellation_reason": reason,
         "cancelled_at": datetime.utcnow().isoformat()},
        {"booking_reference": booking_reference}
    )
    return result is not None


def get_booking_by_ref(ref: str) -> dict | None:
    """Fetch booking details by reference. Tries view first, falls back to direct query."""
    rows = _get("/rest/v1/v_booking_summary", {
        "booking_reference": f"eq.{ref}",
        "select": "*",
    })
    if rows:
        return rows[0]
    # Fallback — direct query on bookings table
    rows = _get("/rest/v1/bookings", {
        "booking_reference": f"eq.{ref}",
        "select": "booking_reference,hotel_id,room_type_code,check_in,check_out,"
                  "num_adults,num_children,total_amount,currency,status,"
                  "payment_status,primary_guest_name,primary_guest_email,"
                  "cancellation_reason",
    })
    if not rows:
        return None
    b = rows[0]
    # Enrich with hotel name
    if b.get("hotel_id"):
        h = _get("/rest/v1/hotels", {"id": f"eq.{b['hotel_id']}",
                                     "select": "id,name,city,country"})
        if h:
            b["hotel_name"] = h[0].get("name", "Hotel")
            b["city"]       = h[0].get("city", "")
            b["country"]    = h[0].get("country", "")
    return b


# ─────────────────────────────────────────────────────────────────────────────
# SESSIONS (conversation persistence)
# ─────────────────────────────────────────────────────────────────────────────

def get_hotel_details(hotel_id: str) -> dict | None:
    """Fetch full hotel record by ID."""
    rows = _get("/rest/v1/hotels", {
        "id":     f"eq.{hotel_id}",
        "select": "id,name,description,city,country,star_rating,amenities,"
                  "currency,check_in_time,check_out_time,policy,photos,thumbnail_url",
    })
    return rows[0] if rows else None


def upsert_session(session_key: str, user_id: str | None,
                   lang: str, turn_count: int, last_intent: str,
                   messages_snapshot: list) -> None:
    """Store/update conversation session for analytics & handoff."""
    existing = _get("/rest/v1/sessions", {
        "session_key": f"eq.{session_key}",
        "select": "id",
    })
    body = {
        "session_key":       session_key,
        "user_id":           user_id,
        "channel":           "messenger",
        "detected_language": lang,
        "total_turns":       turn_count,
        "last_intent":       last_intent,
        "messages":          messages_snapshot[-20:],  # keep last 20 turns
    }
    if existing:
        _patch("/rest/v1/sessions", body, {"session_key": session_key})
    else:
        _post("/rest/v1/sessions", body)
