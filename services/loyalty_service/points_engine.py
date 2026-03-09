"""
services/loyalty_service/points_engine.py
-------------------------------------------
Awards and redeems loyalty points.

Earning rule  : 10 points per USD spent
Redemption    : 100 points = $1 discount
Expiry        : Points expire after 365 days (handled by scheduled task)
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta

import urllib.request

log = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

POINTS_PER_USD    = 10
POINTS_PER_DOLLAR = 100   # redemption rate


def _headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def _rpc(fn: str, args: dict) -> dict:
    url  = f"{SUPABASE_URL}/rest/v1/rpc/{fn}"
    data = json.dumps(args).encode()
    req  = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get(path: str) -> list:
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _patch(path: str, payload: dict) -> list:
    url  = f"{SUPABASE_URL}/rest/v1/{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=_headers(), method="PATCH")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _insert(table: str, payload: dict) -> dict:
    url  = f"{SUPABASE_URL}/rest/v1/{table}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result[0] if isinstance(result, list) else result


def award(guest_id: str, amount_usd: float, booking_id: str) -> dict:
    """Awards points for a completed booking."""
    points_earned = int(amount_usd * POINTS_PER_USD)
    if points_earned <= 0:
        return {"awarded": 0}

    # Upsert loyalty_accounts
    rows = _get(f"loyalty_accounts?guest_id=eq.{guest_id}")
    expires = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()

    if rows:
        acc    = rows[0]
        new_pts = acc["points_balance"] + points_earned
        _patch(f"loyalty_accounts?guest_id=eq.{guest_id}", {"points_balance": new_pts})
    else:
        _insert("loyalty_accounts", {
            "guest_id":       guest_id,
            "points_balance": points_earned,
            "tier":           "bronze",
        })

    # Record transaction
    _insert("loyalty_transactions", {
        "guest_id":    guest_id,
        "booking_id":  booking_id,
        "points":      points_earned,
        "type":        "earn",
        "expires_at":  expires,
        "created_at":  datetime.now(timezone.utc).isoformat(),
    })

    log.info("Awarded %d points to guest %s for booking %s", points_earned, guest_id, booking_id)
    return {"awarded": points_earned}


def redeem(guest_id: str, points_to_redeem: int, booking_id: str) -> dict:
    """
    Redeems points for a discount.  Returns {discount_usd, remaining_balance}.
    Raises ValueError if insufficient balance.
    """
    rows = _get(f"loyalty_accounts?guest_id=eq.{guest_id}")
    if not rows or rows[0]["points_balance"] < points_to_redeem:
        raise ValueError("Insufficient loyalty points")

    acc     = rows[0]
    new_pts = acc["points_balance"] - points_to_redeem
    discount_usd = round(points_to_redeem / POINTS_PER_DOLLAR, 2)

    _patch(f"loyalty_accounts?guest_id=eq.{guest_id}", {"points_balance": new_pts})

    _insert("loyalty_transactions", {
        "guest_id":    guest_id,
        "booking_id":  booking_id,
        "points":      -points_to_redeem,
        "type":        "redeem",
        "created_at":  datetime.now(timezone.utc).isoformat(),
    })

    return {"discount_usd": discount_usd, "remaining_balance": new_pts}


def get_balance(guest_id: str) -> dict:
    rows = _get(f"loyalty_accounts?guest_id=eq.{guest_id}")
    if not rows:
        return {"points_balance": 0, "tier": "bronze"}
    acc = rows[0]
    return {
        "points_balance": acc.get("points_balance", 0),
        "tier":           acc.get("tier", "bronze"),
        "discount_usd":   round(acc.get("points_balance", 0) / POINTS_PER_DOLLAR, 2),
    }
