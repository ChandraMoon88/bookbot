"""
db_client.py
------------
Synchronous PostgreSQL client for BookBot processor.
Uses psycopg2 with DATABASE_URL — no Supabase SDK needed.

Tables used (see hotel_ai_supabase_schema.sql):
  hotels, inventory, bookings, users

All functions are SYNCHRONOUS — they run from the booking state
machine in processor.py which is a regular (non-async) context.
A new connection is opened and closed per call; Supabase's session-mode
pooler (port 5432) handles true connection pooling.
"""

import json
import logging
import os
import random
import re
import string
from datetime import datetime, date
from typing import Optional

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# ── Sentence-transformers (optional — graceful degradation) ───────────────────
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _sem_model: Optional[SentenceTransformer] = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
    _SEM_AVAILABLE = True
except Exception:
    _sem_model = None
    _SEM_AVAILABLE = False


# ── Database connection ────────────────────────────────────────────────────────

def _get_conn() -> psycopg2.extensions.connection:
    """Open a new psycopg2 connection from DATABASE_URL."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    # Supabase requires SSL — add sslmode if not already present
    if "sslmode" not in url and "supabase.com" in url:
        sep = "&" if "?" in url else "?"
        url = url + sep + "sslmode=require"
    conn = psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)
    conn.autocommit = False
    return conn


def _gen_booking_ref() -> str:
    return "BB" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


# ── Hotel search ──────────────────────────────────────────────────────────────

def search_hotels(
    city: str,
    checkin: str,
    checkout: str,
    num_adults: int = 1,
    num_children: int = 0,
) -> list:
    """
    Return available hotels in *city* for the given date range/guest count.
    Each hotel dict includes an 'available_rooms' key with per-room pricing.
    """
    try:
        ci = datetime.strptime(checkin, "%Y-%m-%d").date()
        co = datetime.strptime(checkout, "%Y-%m-%d").date()
    except ValueError:
        return []

    nights = (co - ci).days
    if nights <= 0:
        return []

    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                # 1. Active hotels in city
                cur.execute(
                    """
                    SELECT id, name, description, city, country,
                           star_rating, amenities, currency,
                           check_in_time, check_out_time, thumbnail_url
                    FROM   hotels
                    WHERE  is_active = TRUE
                      AND  LOWER(city) = LOWER(%s)
                    LIMIT  20
                    """,
                    (city,),
                )
                hotels = [dict(r) for r in cur.fetchall()]
                if not hotels:
                    return []

                hotel_ids = [str(h["id"]) for h in hotels]
                ph = ",".join(["%s"] * len(hotel_ids))

                # 2. Aggregate inventory for the date range
                cur.execute(
                    f"""
                    SELECT
                        hotel_id::text,
                        room_type_code,
                        room_type_name,
                        MIN(available_count)                     AS min_avail,
                        MAX(max_adults)                          AS max_adults,
                        MAX(max_children)                        AS max_children,
                        (array_agg(rate_plans ORDER BY date))[1] AS rate_plans
                    FROM   inventory
                    WHERE  hotel_id::text IN ({ph})
                      AND  date >= %s
                      AND  date <  %s
                      AND  available_count > 0
                      AND  is_blackout = FALSE
                    GROUP  BY hotel_id, room_type_code, room_type_name
                    HAVING COUNT(*) >= %s
                    """,
                    hotel_ids + [ci, co, nights],
                )
                rooms_rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        logger.error("search_hotels DB error: %s", e, exc_info=True)
        return []

    # Group rooms by hotel
    rooms_by_hotel: dict[str, list] = {}
    for row in rooms_rows:
        row = dict(row)
        hid = str(row["hotel_id"])

        rate_plans = row.get("rate_plans") or {}
        if isinstance(rate_plans, str):
            try:
                rate_plans = json.loads(rate_plans)
            except Exception:
                rate_plans = {}

        # Cheapest per-night price across all meal plans
        prices = []
        for plan_data in rate_plans.values():
            if isinstance(plan_data, dict):
                p = plan_data.get("price_per_night")
                if p:
                    try:
                        prices.append(float(p))
                    except (TypeError, ValueError):
                        pass

        rooms_by_hotel.setdefault(hid, []).append({
            "room_type_code":  row["room_type_code"],
            "room_type_name":  row["room_type_name"],
            "available_count": int(row.get("min_avail") or 0),
            "max_adults":      int(row.get("max_adults") or 2),
            "max_children":    int(row.get("max_children") or 0),
            "price_per_night": min(prices) if prices else None,
            "rate_plans":      rate_plans,
        })

    # Attach rooms to hotels; filter by capacity
    results = []
    for h in hotels:
        hid   = str(h["id"])
        rooms = rooms_by_hotel.get(hid, [])
        eligible = [
            r for r in rooms
            if r["max_adults"] >= num_adults
            and (num_children == 0 or r["max_children"] >= num_children)
            and r["available_count"] > 0
        ]
        if eligible:
            if isinstance(h.get("amenities"), str):
                try:
                    h["amenities"] = json.loads(h["amenities"])
                except Exception:
                    h["amenities"] = []
            h["available_rooms"] = eligible
            results.append(h)

    return results


def semantic_hotel_search(query: str, hotels: list, top_k: int = 5) -> list:
    """Rerank hotel list by semantic similarity. Falls back to original order."""
    if not _SEM_AVAILABLE or not hotels or _sem_model is None:
        return hotels[:top_k]
    try:
        texts = [
            f"{h.get('name', '')} {h.get('description', '')} {h.get('city', '')} "
            + " ".join(str(a) for a in (h.get("amenities") or [])[:5])
            for h in hotels
        ]
        q_emb  = _sem_model.encode(query, convert_to_tensor=True)
        t_emb  = _sem_model.encode(texts,  convert_to_tensor=True)
        scores = st_util.cos_sim(q_emb, t_emb)[0].tolist()
        ranked = sorted(zip(scores, hotels), key=lambda x: x[0], reverse=True)
        return [h for _, h in ranked[:top_k]]
    except Exception as e:
        logger.error("semantic_hotel_search error: %s", e)
        return hotels[:top_k]


# ── User management ───────────────────────────────────────────────────────────

def get_or_create_user(
    messenger_id: str, first_name: str = ""
) -> Optional[dict]:
    """
    Get or create a user record.
    Uses phone = "messenger:{messenger_id}" as the unique identifier.
    """
    phone_key = f"messenger:{messenger_id}"
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM users WHERE phone = %s LIMIT 1",
                    (phone_key,),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)

                cur.execute(
                    """
                    INSERT INTO users (first_name, phone)
                    VALUES (%s, %s)
                    RETURNING *
                    """,
                    (first_name or "Guest", phone_key),
                )
                new_row = cur.fetchone()
                conn.commit()
                return dict(new_row) if new_row else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("get_or_create_user error: %s", e)
        return None


# ── Booking operations ────────────────────────────────────────────────────────

def get_user_bookings(user_id: str) -> list:
    """Return the 10 most-recent bookings for a user (joined with hotel name)."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT b.*, h.name AS hotel_name
                    FROM   bookings b
                    LEFT   JOIN hotels h ON b.hotel_id = h.id
                    WHERE  b.user_id = %s::uuid
                    ORDER  BY b.created_at DESC
                    LIMIT  10
                    """,
                    (user_id,),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception as e:
        logger.error("get_user_bookings error: %s", e)
        return []


def create_booking(
    user_id: str,
    hotel_id: str,
    room_type_code: str,
    check_in: str,
    check_out: str,
    num_adults: int = 1,
    num_children: int = 0,
    primary_guest_name: str = "",
    primary_guest_email: str = "",
    primary_guest_phone: str = "",
    total_amount: float = 0.0,
    currency: str = "INR",
    special_requests: str = "",
    rate_plan: str = "room_only",
    meal_plan: str = "room_only",
    num_rooms: int = 1,
) -> Optional[dict]:
    """Insert a new booking row. Returns the inserted dict (with booking_reference)."""
    booking_ref = _gen_booking_ref()
    # Tax calculation (18% GST inclusive)
    tax_rate    = 0.18
    base_amount = round(float(total_amount) / (1 + tax_rate), 2)
    tax_amount  = round(float(total_amount) - base_amount, 2)

    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                # Ensure unique reference
                for _ in range(5):
                    cur.execute(
                        "SELECT 1 FROM bookings WHERE booking_reference = %s",
                        (booking_ref,),
                    )
                    if not cur.fetchone():
                        break
                    booking_ref = _gen_booking_ref()

                uid = user_id  if user_id  else None
                hid = hotel_id if hotel_id else None

                cur.execute(
                    """
                    INSERT INTO bookings (
                        booking_reference, user_id, hotel_id, room_type_code,
                        rate_plan, meal_plan, check_in, check_out,
                        num_adults, num_children, num_rooms,
                        primary_guest_name, primary_guest_email, primary_guest_phone,
                        total_amount, base_amount, tax_amount, currency,
                        special_requests, status, payment_status
                    ) VALUES (
                        %s, %s::uuid, %s::uuid, %s,
                        %s, %s, %s::date, %s::date,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, 'confirmed', 'pending'
                    ) RETURNING *
                    """,
                    (
                        booking_ref, uid, hid, room_type_code,
                        rate_plan, meal_plan, check_in, check_out,
                        num_adults, num_children, num_rooms,
                        primary_guest_name, primary_guest_email, primary_guest_phone,
                        total_amount, base_amount, tax_amount, currency,
                        special_requests,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
                return dict(row) if row else {"booking_reference": booking_ref}
        finally:
            conn.close()
    except Exception as e:
        logger.error("create_booking error: %s", e, exc_info=True)
        return None


def cancel_booking(booking_id: str) -> bool:
    """Cancel a booking by ID. Returns True on success."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE bookings
                    SET    status       = 'cancelled',
                           cancelled_at = NOW(),
                           updated_at   = NOW()
                    WHERE  id = %s::uuid
                    """,
                    (booking_id,),
                )
                conn.commit()
                return cur.rowcount > 0
        finally:
            conn.close()
    except Exception as e:
        logger.error("cancel_booking error: %s", e)
        return False


def get_booking_by_ref(booking_ref: str) -> Optional[dict]:
    """Fetch a booking by its booking_reference (case-insensitive)."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT b.*, h.name AS hotel_name
                    FROM   bookings b
                    LEFT   JOIN hotels h ON b.hotel_id = h.id
                    WHERE  UPPER(b.booking_reference) = UPPER(%s)
                    LIMIT  1
                    """,
                    (booking_ref.upper(),),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("get_booking_by_ref error: %s", e)
        return None
