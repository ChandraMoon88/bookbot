"""
services/loyalty_service/points_engine.py
-------------------------------------------
Awards and redeems loyalty points.

Earning rule  : 10 points per USD spent
Redemption    : 100 points = $1 discount
Expiry        : Points expire after 365 days (handled by scheduled task)
"""

import os
import logging
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

POINTS_PER_USD    = 10
POINTS_PER_DOLLAR = 100   # redemption rate


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def award(guest_id: str, amount_usd: float, booking_id: str) -> dict:
    """Awards points for a completed booking."""
    points_earned = int(amount_usd * POINTS_PER_USD)
    if points_earned <= 0:
        return {"awarded": 0}

    expires = datetime.now(timezone.utc) + timedelta(days=365)
    with _get_conn() as conn:
        with conn.cursor() as cur:
            # Upsert loyalty_accounts
            cur.execute(
                "INSERT INTO loyalty_accounts (guest_id, points_balance, tier) "
                "VALUES (%s, %s, 'bronze') "
                "ON CONFLICT (guest_id) DO UPDATE "
                "SET points_balance = loyalty_accounts.points_balance + EXCLUDED.points_balance",
                (guest_id, points_earned),
            )
            # Record transaction
            cur.execute(
                "INSERT INTO loyalty_transactions "
                "(guest_id, booking_id, points, type, expires_at, created_at) "
                "VALUES (%s,%s,%s,'earn',%s,%s)",
                (guest_id, booking_id, points_earned, expires, datetime.now(timezone.utc)),
            )
            conn.commit()

    log.info("Awarded %d points to guest %s for booking %s", points_earned, guest_id, booking_id)
    return {"awarded": points_earned}


def redeem(guest_id: str, points_to_redeem: int, booking_id: str) -> dict:
    """
    Redeems points for a discount.  Returns {discount_usd, remaining_balance}.
    Raises ValueError if insufficient balance.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT points_balance FROM loyalty_accounts WHERE guest_id=%s FOR UPDATE",
                (guest_id,),
            )
            row = cur.fetchone()
            if not row or row["points_balance"] < points_to_redeem:
                raise ValueError("Insufficient loyalty points")
            new_pts = row["points_balance"] - points_to_redeem
            discount_usd = round(points_to_redeem / POINTS_PER_DOLLAR, 2)
            cur.execute(
                "UPDATE loyalty_accounts SET points_balance=%s WHERE guest_id=%s",
                (new_pts, guest_id),
            )
            cur.execute(
                "INSERT INTO loyalty_transactions "
                "(guest_id, booking_id, points, type, created_at) VALUES (%s,%s,%s,'redeem',%s)",
                (guest_id, booking_id, -points_to_redeem, datetime.now(timezone.utc)),
            )
            conn.commit()
    return {"discount_usd": discount_usd, "remaining_balance": new_pts}


def get_balance(guest_id: str) -> dict:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT points_balance, tier FROM loyalty_accounts WHERE guest_id=%s",
                (guest_id,),
            )
            row = cur.fetchone()
    if not row:
        return {"points_balance": 0, "tier": "bronze"}
    acc = dict(row)
    return {
        "points_balance": acc.get("points_balance", 0),
        "tier":           acc.get("tier", "bronze"),
        "discount_usd":   round(acc.get("points_balance", 0) / POINTS_PER_DOLLAR, 2),
    }
