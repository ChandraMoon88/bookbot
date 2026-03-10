"""
services/corporate_service/rate_lookup.py
--------------------------------------------
Retrieves negotiated corporate rates from Supabase.
Falls back to the standard best-available rate if no contract exists.
"""

import os
import logging
from datetime import datetime

log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def get_corporate_rate(
    corporate_account_id: str,
    hotel_id:             str,
    room_type_id:         str,
    check_in:             str,
) -> dict:
    """
    Returns:
        rate_per_night   float
        currency         str
        contract_id      str | None
        is_negotiated    bool
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM corporate_rates "
                "WHERE corporate_account_id=%s AND hotel_id=%s AND room_type_id=%s "
                "AND valid_from<=%s AND valid_to>=%s AND is_active=TRUE LIMIT 1",
                (corporate_account_id, hotel_id, room_type_id, check_in, check_in),
            )
            row = cur.fetchone()

    if row:
        r = dict(row)
        return {
            "rate_per_night": float(r["rate_per_night"]),
            "currency":       r.get("currency", "USD"),
            "contract_id":    r["id"],
            "is_negotiated":  True,
            "discount_label": f"Corporate rate – contract {str(r['id'])[:8]}",
        }

    # Fallback: fetch standard room rate
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT rate_per_night, currency FROM room_rates "
                "WHERE room_type_id=%s AND rate_type='standard' LIMIT 1",
                (room_type_id,),
            )
            row2 = cur.fetchone()

    std_rate = float(dict(row2)["rate_per_night"]) if row2 else 100.0
    return {
        "rate_per_night": std_rate,
        "currency":       "USD",
        "contract_id":    None,
        "is_negotiated":  False,
        "discount_label": "Best available rate",
    }
