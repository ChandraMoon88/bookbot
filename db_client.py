"""
db_client.py
------------
Synchronous PostgreSQL client for BookBot processor.
Uses psycopg2 with DATABASE_URL — no Supabase SDK needed.

Tables used (see hotel_ai_supabase_schema.sql):
  hotel_partners, room_types, room_availability, room_rates,
  bookings, guests, cancellations, users

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
                           check_in_time, check_out_time, thumbnail_url,
                           contact_email
                    FROM   hotel_partners
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

                # 2. Aggregate availability across room_types + room_availability + room_rates
                cur.execute(
                    f"""
                    SELECT
                        rt.hotel_id::text,
                        rt.id::text                                    AS room_type_id,
                        rt.code                                        AS room_type_code,
                        rt.name                                        AS room_type_name,
                        rt.max_adults,
                        rt.max_children,
                        MIN(ra.available_count)                        AS min_avail,
                        jsonb_object_agg(
                            COALESCE(rr.rate_type, 'room_only'),
                            jsonb_build_object('price_per_night', rr.base_price)
                        )                                              AS rate_plans
                    FROM   room_types rt
                    JOIN   room_availability ra ON ra.room_type_id = rt.id
                    LEFT   JOIN room_rates rr   ON rr.room_type_id = rt.id
                                               AND rr.valid_from   <= %s
                                               AND rr.valid_to     >= %s
                    WHERE  rt.hotel_id::text IN ({ph})
                      AND  ra.date >= %s
                      AND  ra.date <  %s
                      AND  ra.available_count > 0
                      AND  ra.is_blackout = FALSE
                    GROUP  BY rt.hotel_id, rt.id, rt.code, rt.name,
                              rt.max_adults, rt.max_children
                    HAVING COUNT(DISTINCT ra.date) >= %s
                    """,
                    [ci, co] + hotel_ids + [ci, co, nights],
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
            "room_type_id":    row["room_type_id"],
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
    """Return the 10 most-recent bookings for a user (joined with hotel/room/guest)."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        b.*,
                        h.name                                         AS hotel_name,
                        h.contact_email                                AS hotel_email,
                        rt.name                                        AS room_name,
                        rt.code                                        AS room_type_code,
                        b.num_adults                                   AS adults,
                        b.num_children                                 AS children,
                        b.meal_plan                                    AS meal_plan_code,
                        g.first_name || ' ' || COALESCE(g.last_name, '') AS guest_name,
                        g.email                                        AS guest_email,
                        g.phone                                        AS guest_phone
                    FROM   bookings b
                    LEFT   JOIN hotel_partners h ON b.hotel_id    = h.id
                    LEFT   JOIN room_types rt    ON b.room_type_id = rt.id
                    LEFT   JOIN guests g         ON g.booking_id  = b.id
                                               AND g.is_primary   = TRUE
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
    room_type_id: str = "",
) -> Optional[dict]:
    """Insert a booking row + primary guest row. Returns the inserted dict."""
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

                # Resolve room_type_id UUID from code if not supplied directly
                rtid: Optional[str] = room_type_id if room_type_id else None
                if not rtid and hid and room_type_code:
                    cur.execute(
                        """
                        SELECT id FROM room_types
                        WHERE  hotel_id = %s::uuid AND code = %s
                        LIMIT  1
                        """,
                        (hid, room_type_code),
                    )
                    rt_row = cur.fetchone()
                    rtid = str(rt_row["id"]) if rt_row else None

                cur.execute(
                    """
                    INSERT INTO bookings (
                        booking_reference, user_id, hotel_id, room_type_id,
                        rate_plan, meal_plan, check_in, check_out,
                        num_adults, num_children, num_rooms,
                        total_amount, base_amount, tax_amount, currency,
                        special_requests, status, payment_status
                    ) VALUES (
                        %s, %s::uuid, %s::uuid, %s::uuid,
                        %s, %s, %s::date, %s::date,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, 'confirmed', 'pending'
                    ) RETURNING *
                    """,
                    (
                        booking_ref, uid, hid, rtid,
                        rate_plan, meal_plan, check_in, check_out,
                        num_adults, num_children, num_rooms,
                        total_amount, base_amount, tax_amount, currency,
                        special_requests,
                    ),
                )
                row = cur.fetchone()
                if not row:
                    conn.rollback()
                    return None
                booking_id = row["id"]

                # Insert primary guest into guests table
                if primary_guest_name:
                    name_parts = primary_guest_name.strip().split(" ", 1)
                    first = name_parts[0] or "Guest"
                    last  = name_parts[1] if len(name_parts) > 1 else "-"
                    cur.execute(
                        """
                        INSERT INTO guests (
                            booking_id, first_name, last_name,
                            email, phone, is_primary
                        ) VALUES (%s, %s, %s, %s, %s, TRUE)
                        """,
                        (booking_id, first, last,
                         primary_guest_email or None,
                         primary_guest_phone or None),
                    )

                conn.commit()
                result = dict(row)
                # Expose guest fields for downstream use
                result["guest_name"]  = primary_guest_name
                result["guest_email"] = primary_guest_email
                result["guest_phone"] = primary_guest_phone
                return result
        finally:
            conn.close()
    except Exception as e:
        logger.error("create_booking error: %s", e, exc_info=True)
        return None


def cancel_booking(booking_ref: str, cancelled_by: str = "guest") -> bool:
    """Cancel a booking by booking_reference. Returns True on success."""
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
                    WHERE  UPPER(booking_reference) = UPPER(%s)
                      AND  status NOT IN ('cancelled', 'no_show')
                    RETURNING id, booking_reference
                    """,
                    (booking_ref,),
                )
                row = cur.fetchone()
                if not row:
                    conn.rollback()
                    return False
                # Record in cancellations table
                cur.execute(
                    """
                    INSERT INTO cancellations (
                        booking_id, initiated_by, reason
                    ) VALUES (%s, %s, 'Guest requested via chat')
                    """,
                    (row["id"], cancelled_by),
                )
                conn.commit()
                return True
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
                    SELECT
                        b.*,
                        h.name                                         AS hotel_name,
                        h.contact_email                                AS hotel_email,
                        rt.name                                        AS room_name,
                        rt.code                                        AS room_type_code,
                        b.num_adults                                   AS adults,
                        b.num_children                                 AS children,
                        b.meal_plan                                    AS meal_plan_code,
                        g.first_name || ' ' || COALESCE(g.last_name, '') AS guest_name,
                        g.email                                        AS guest_email,
                        g.phone                                        AS guest_phone
                    FROM   bookings b
                    LEFT   JOIN hotel_partners h ON b.hotel_id    = h.id
                    LEFT   JOIN room_types rt    ON b.room_type_id = rt.id
                    LEFT   JOIN guests g         ON g.booking_id  = b.id
                                               AND g.is_primary   = TRUE
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


# ── Voucher validation ────────────────────────────────────────────────────────

def validate_voucher(code: str, user_id: str = "") -> dict:
    """
    Validate a voucher code against the vouchers + voucher_redemptions tables.
    Returns {"valid": True/False, "discount_pct": float, "message": str}.
    Falls back to a small built-in set when DB is unavailable.
    """
    _FALLBACK = {"WELCOME20": 0.20, "SAVE10": 0.10, "DEAL15": 0.15}
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, discount_type, discount_value,
                           max_uses, valid_from, valid_to, is_active
                    FROM   vouchers
                    WHERE  UPPER(code) = UPPER(%s)
                    LIMIT  1
                    """,
                    (code,),
                )
                v = cur.fetchone()
                if not v:
                    return {"valid": False, "discount_pct": 0,
                            "message": f"Voucher '{code}' not found or has expired."}
                v = dict(v)
                if not v["is_active"]:
                    return {"valid": False, "discount_pct": 0,
                            "message": f"Voucher '{code}' is no longer active."}
                today = date.today()
                if v["valid_from"] and today < v["valid_from"]:
                    return {"valid": False, "discount_pct": 0,
                            "message": f"Voucher '{code}' is not valid yet."}
                if v["valid_to"] and today > v["valid_to"]:
                    return {"valid": False, "discount_pct": 0,
                            "message": f"Voucher '{code}' has expired."}
                # Check usage count
                cur.execute(
                    "SELECT COUNT(*) AS cnt FROM voucher_redemptions WHERE voucher_id = %s",
                    (v["id"],),
                )
                usage = (cur.fetchone() or {}).get("cnt", 0)
                if v["max_uses"] and usage >= v["max_uses"]:
                    return {"valid": False, "discount_pct": 0,
                            "message": f"Voucher '{code}' has already been fully redeemed."}
                # Check per-user single use
                if user_id:
                    cur.execute(
                        """
                        SELECT 1 FROM voucher_redemptions vr
                        JOIN   bookings b ON b.id = vr.booking_id
                        WHERE  vr.voucher_id = %s AND b.user_id = %s::uuid
                        LIMIT  1
                        """,
                        (v["id"], user_id),
                    )
                    if cur.fetchone():
                        return {"valid": False, "discount_pct": 0,
                                "message": f"Voucher '{code}' has already been used on your account."}

                pct = float(v["discount_value"]) if v["discount_type"] == "percentage" else 0.0
                return {"valid": True, "voucher_id": str(v["id"]),
                        "discount_pct": pct, "discount_value": float(v["discount_value"]),
                        "discount_type": v["discount_type"],
                        "message": f"Voucher '{code}' applied!"}
        finally:
            conn.close()
    except Exception as e:
        logger.error("validate_voucher DB error: %s", e)
        # Graceful fallback to hardcoded set
        code_u = code.upper()
        if code_u in _FALLBACK:
            return {"valid": True, "voucher_id": None,
                    "discount_pct": _FALLBACK[code_u],
                    "discount_value": _FALLBACK[code_u],
                    "discount_type": "percentage",
                    "message": f"Voucher '{code_u}' applied!"}
        return {"valid": False, "discount_pct": 0,
                "message": f"Voucher '{code}' not found or has expired."}


def redeem_voucher(voucher_id: str, booking_id: str, discount_amount: float) -> bool:
    """Record a voucher_redemptions row. Called after booking is created."""
    if not voucher_id:
        return False
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO voucher_redemptions (voucher_id, booking_id, discount_amount)
                    VALUES (%s::uuid, %s::uuid, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (voucher_id, booking_id, discount_amount),
                )
                conn.commit()
                return True
        finally:
            conn.close()
    except Exception as e:
        logger.error("redeem_voucher error: %s", e)
        return False


# ── Payment recording ─────────────────────────────────────────────────────────

def record_payment(
    booking_id: str,
    amount: float,
    currency: str,
    payment_method: str,
    status: str = "pending",
    gateway_ref: str = "",
) -> Optional[dict]:
    """Insert a row into the payments table. Returns the inserted row dict."""
    # Normalize payment_method string to schema's gateway enum
    _gw_map = {
        "stripe": "stripe", "card": "stripe", "credit card": "stripe",
        "paypal": "paypal",
        "upi": "upi",
        "pay at hotel": "hotel", "cash": "hotel", "hotel": "hotel",
        "net banking": "netbanking", "netbanking": "netbanking",
        "crypto": "crypto", "bitcoin": "crypto",
        "bizum": "bizum", "split": "split", "corporate": "corporate",
    }
    gateway = _gw_map.get((payment_method or "").lower(), "stripe")
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO payments (
                        booking_id, amount, currency,
                        gateway, payment_method_type, transaction_id, status
                    ) VALUES (
                        %s::uuid, %s, %s, %s, %s, %s, %s
                    ) RETURNING *
                    """,
                    (booking_id, amount, currency,
                     gateway, payment_method or None,
                     gateway_ref or None, status),
                )
                row = cur.fetchone()
                conn.commit()
                return dict(row) if row else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("record_payment error: %s", e)
        return None


# ── Loyalty ───────────────────────────────────────────────────────────────────

def get_or_create_loyalty_account(user_id: str) -> Optional[dict]:
    """Fetch or create a loyalty_accounts row for the user."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM loyalty_accounts WHERE user_id = %s::uuid LIMIT 1",
                    (user_id,),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
                cur.execute(
                    """
                    INSERT INTO loyalty_accounts (user_id, tier)
                    VALUES (%s::uuid, 'bronze')
                    ON CONFLICT (user_id) DO UPDATE SET updated_at = NOW()
                    RETURNING *
                    """,
                    (user_id,),
                )
                new_row = cur.fetchone()
                conn.commit()
                return dict(new_row) if new_row else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("get_or_create_loyalty_account error: %s", e)
        return None


def add_loyalty_points(
    user_id: str,
    points: int,
    transaction_type: str = "earn",
    description: str = "",
    booking_id: str = "",
) -> Optional[dict]:
    """
    Add (or deduct if negative) loyalty points.
    Inserts a loyalty_transactions row and updates loyalty_accounts.
    Returns updated loyalty_accounts row.
    """
    if points == 0:
        return None
    # Map caller's transaction_type to schema's action_type enum
    _type_map = {
        "earn": "booking", "booking": "booking",
        "redeem": "redemption", "redemption": "redemption",
        "referral": "referral", "review": "review",
        "checkin": "checkin", "expiry": "expiry",
        "adjustment": "adjustment",
    }
    action_type = _type_map.get(transaction_type, "adjustment")
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                # Ensure account exists
                cur.execute(
                    """
                    INSERT INTO loyalty_accounts (user_id, tier)
                    VALUES (%s::uuid, 'bronze')
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    (user_id,),
                )
                # Fetch account id and current total_points
                cur.execute(
                    "SELECT id, total_points, available_points FROM loyalty_accounts WHERE user_id = %s::uuid",
                    (user_id,),
                )
                acct_row = cur.fetchone()
                if not acct_row:
                    return None
                acct_id = acct_row["id"]
                new_total = max(0, (acct_row["total_points"] or 0) + points)
                # Update balances
                cur.execute(
                    """
                    UPDATE loyalty_accounts
                    SET    total_points     = %s,
                           available_points = GREATEST(0, available_points + %s),
                           updated_at       = NOW()
                    WHERE  user_id = %s::uuid
                    RETURNING *
                    """,
                    (new_total, points, user_id),
                )
                acct = cur.fetchone()
                # Log transaction
                points_earned = points if points > 0 else 0
                points_spent  = abs(points) if points < 0 else 0
                cur.execute(
                    """
                    INSERT INTO loyalty_transactions (
                        user_id, loyalty_account_id, booking_id,
                        action_type, points_earned, points_spent,
                        running_balance, description
                    ) VALUES (
                        %s::uuid, %s, %s,
                        %s, %s, %s,
                        %s, %s
                    )
                    """,
                    (user_id, acct_id,
                     booking_id if booking_id else None,
                     action_type, points_earned, points_spent,
                     new_total, description or None),
                )
                conn.commit()
                return dict(acct) if acct else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("add_loyalty_points error: %s", e)
        return None


# ── Handoff / support ticket ──────────────────────────────────────────────────

def create_support_ticket(
    user_id: str,
    booking_ref: str = "",
    subject: str = "",
    message: str = "",
    priority: str = "normal",
) -> Optional[dict]:
    """Create a support_ticket row. Returns the inserted row dict."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                # Resolve booking_id from ref if provided
                booking_id = None
                if booking_ref:
                    cur.execute(
                        "SELECT id FROM bookings WHERE UPPER(booking_reference) = UPPER(%s) LIMIT 1",
                        (booking_ref,),
                    )
                    b = cur.fetchone()
                    if b:
                        booking_id = b["id"]

                # Generate unique ticket_ref  (e.g. TKT-20260310-4821)
                date_part = datetime.utcnow().strftime("%Y%m%d")
                ticket_ref = f"TKT-{date_part}-{random.randint(1000, 9999)}"

                cur.execute(
                    """
                    INSERT INTO support_tickets (
                        ticket_ref, user_id, booking_id,
                        subject, description, priority, status
                    ) VALUES (
                        %s, %s::uuid, %s,
                        %s, %s, %s, 'open'
                    ) RETURNING *
                    """,
                    (ticket_ref, user_id or None, booking_id,
                     subject or "Chat support request",
                     message or None, priority),
                )
                ticket = cur.fetchone()
                if ticket and message:
                    cur.execute(
                        """
                        INSERT INTO support_ticket_messages (ticket_id, sender_type, message)
                        VALUES (%s, 'user', %s)
                        """,
                        (ticket["id"], message),
                    )
                conn.commit()
                return dict(ticket) if ticket else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("create_support_ticket error: %s", e)
        return None


def create_handoff_request(
    user_id: str,
    booking_ref: str = "",
    reason: str = "",
    channel: str = "messenger",
) -> Optional[dict]:
    """Insert a handoff_requests row. Returns the inserted dict."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                # handoff_requests has no booking_id or channel columns;
                # store booking_ref and channel in slots_filled JSONB
                slots: dict = {}
                if booking_ref:
                    slots["booking_ref"] = booking_ref
                if channel:
                    slots["channel"] = channel
                cur.execute(
                    """
                    INSERT INTO handoff_requests (
                        user_id, reason, booking_intent, slots_filled, status
                    ) VALUES (
                        %s::uuid, %s, %s, %s::jsonb, 'pending'
                    ) RETURNING *
                    """,
                    (user_id or None, reason or None,
                     booking_ref or None,
                     json.dumps(slots)),
                )
                row = cur.fetchone()
                conn.commit()
                return dict(row) if row else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("create_handoff_request error: %s", e)
        return None


def get_last_booking(user_id: str) -> Optional[dict]:
    """Return the most recent non-cancelled booking for a user (SELECT only)."""
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT b.id, b.booking_reference, b.check_in, b.check_out,
                           b.status, b.total_amount, b.currency,
                           b.num_adults, b.num_children,
                           h.name  AS hotel_name, h.city AS hotel_city, h.id AS hotel_id,
                           rt.name AS room_name,  rt.code AS room_type_code, rt.id AS room_type_id
                    FROM   bookings b
                    LEFT JOIN hotel_partners h  ON b.hotel_id     = h.id
                    LEFT JOIN room_types     rt ON b.room_type_id = rt.id
                    WHERE  b.user_id = %s::uuid
                      AND  b.status NOT IN ('cancelled', 'no_show', 'pending')
                    ORDER  BY b.created_at DESC
                    LIMIT  1
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            conn.close()
    except Exception as e:
        logger.error("get_last_booking error: %s", e)
        return None
