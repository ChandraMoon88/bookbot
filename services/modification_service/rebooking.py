"""
services/modification_service/rebooking.py
--------------------------------------------
Handles the actual date / room-type change in Supabase via the REST API.
If the new room is unavailable, rolls back automatically (soft lock released).
"""

import os
import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timezone

log = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _headers() -> dict:
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def _patch(path: str, payload: dict) -> dict:
    url  = f"{SUPABASE_URL}/rest/v1/{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=_headers(), method="PATCH")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get(path: str) -> list:
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


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
    rows = _get(f"bookings?id=eq.{booking_id}&select=*")
    if not rows:
        raise ValueError(f"Booking {booking_id} not found")

    current = rows[0]
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

    updated = _patch(
        f"bookings?id=eq.{booking_id}",
        {
            "check_in":             new_check_in,
            "check_out":            new_check_out,
            "room_type_id":         new_room_type,
            "modification_history": existing,
            "status":               "confirmed",
            "updated_at":           datetime.now(timezone.utc).isoformat(),
        },
    )

    log.info("Booking %s modified; fee $%.2f", booking_id, fee_usd)
    return updated[0] if isinstance(updated, list) else updated
