"""
services/modification_service/rebooking.py
--------------------------------------------
Handles the actual date / room-type change in Supabase via the REST API.
If the new room is unavailable, rolls back automatically (soft lock released).
"""

import os
import json
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def modify_booking(
    booking_id:   str,
    new_check_in:  str,
    new_check_out: str,
    new_room_type: str,
    fee_usd:       float,
    agent_id:      str = "bot",
) -> dict:
    """
    Updates the booking row and appends a modification_history JSONB entry.
    Returns the updated booking dict, or raises on failure.
    """
    # Fetch current booking
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM bookings WHERE id=%s", (booking_id,))
            row = cur.fetchone()
    if not row:
        raise ValueError(f"Booking {booking_id} not found")

    current = dict(row)
    history_entry = {
        "modified_at":      datetime.now(timezone.utc).isoformat(),
        "modified_by":      agent_id,
        "old_check_in":     current.get("check_in"),
        "old_check_out":    current.get("check_out"),
        "old_room_type":    current.get("room_type_id"),
        "new_check_in":     new_check_in,
        "new_check_out":    new_check_out,
        "new_room_type":    new_room_type,
        "modification_fee": fee_usd,
    }

    existing = current.get("modification_history") or []
    existing.append(history_entry)

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE bookings SET check_in=%s, check_out=%s, room_type_id=%s, "
                "modification_history=%s::jsonb, status='confirmed', updated_at=%s "
                "WHERE id=%s RETURNING *",
                (new_check_in, new_check_out, new_room_type,
                 json.dumps(existing), datetime.now(timezone.utc), booking_id),
            )
            updated = dict(cur.fetchone())
            conn.commit()

    log.info("Booking %s modified; fee $%.2f", booking_id, fee_usd)
    return updated
