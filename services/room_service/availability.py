"""
services/room_service/availability.py
----------------------------------------
Room availability checker against Supabase.
Returns minimum available count across all nights of a stay.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Any

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def check_availability(
    room_type_id: str,
    check_in:     str,
    check_out:    str,
    num_guests:   int = 1,
) -> dict:
    """
    Return availability info for a given room type across the stay.

    available_count = MIN of daily counts (bottleneck-night approach).
    """
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT date, available_count FROM room_availability "
                    "WHERE room_type_id = %s AND date >= %s",
                    (room_type_id, check_in),
                )
                rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        logger.error("availability query error: %s", e)
        rows = []

    # Build required date set
    required: set[str] = set()
    d = date.fromisoformat(check_in)
    end = date.fromisoformat(check_out)
    while d < end:
        required.add(str(d))
        d += timedelta(days=1)

    available_map = {r["date"][:10]: r["available_count"] for r in rows}

    # All dates must be present and available
    for dt in required:
        if dt not in available_map or available_map[dt] <= 0:
            return {"available": False, "available_count": 0}

    min_count = min(available_map[dt] for dt in required)

    return {
        "available":       min_count > 0,
        "available_count": min_count,
    }
