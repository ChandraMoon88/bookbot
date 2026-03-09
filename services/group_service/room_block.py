"""
services/group_service/room_block.py
--------------------------------------
Manages group / rooming-list room blocks.
A room block reserves N rooms of a type from date_from to date_to,
with a pickup deadline and optional deposit schedule.
"""

import os
import json
import logging
from datetime import datetime, timezone
import urllib.request

log = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def _post(table: str, payload: dict) -> dict:
    url  = f"{SUPABASE_URL}/rest/v1/{table}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result[0] if isinstance(result, list) else result


def _get(path: str) -> list:
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def create_block(
    hotel_id:       str,
    room_type_id:   str,
    rooms_reserved: int,
    date_from:      str,
    date_to:        str,
    group_name:     str,
    pickup_deadline: str,
    rate_per_night:  float,
    organiser_email: str,
) -> dict:
    block = _post("room_blocks", {
        "hotel_id":        hotel_id,
        "room_type_id":    room_type_id,
        "rooms_reserved":  rooms_reserved,
        "rooms_picked_up": 0,
        "date_from":       date_from,
        "date_to":         date_to,
        "group_name":      group_name,
        "pickup_deadline": pickup_deadline,
        "rate_per_night":  rate_per_night,
        "organiser_email": organiser_email,
        "status":          "active",
        "created_at":      datetime.now(timezone.utc).isoformat(),
    })
    log.info("Room block created: %s (%d rooms)", group_name, rooms_reserved)
    return block


def pickup_room(block_id: str, booking_id: str) -> dict:
    """Marks one room from the block as picked up."""
    blocks = _get(f"room_blocks?id=eq.{block_id}")
    if not blocks:
        raise ValueError(f"Block {block_id} not found")
    block = blocks[0]
    if block["rooms_picked_up"] >= block["rooms_reserved"]:
        raise ValueError("All rooms already picked up")

    url  = f"{SUPABASE_URL}/rest/v1/room_blocks?id=eq.{block_id}"
    data = json.dumps({
        "rooms_picked_up": block["rooms_picked_up"] + 1,
        "updated_at":       datetime.now(timezone.utc).isoformat(),
    }).encode()
    req = urllib.request.Request(url, data=data, headers=_headers(), method="PATCH")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result[0] if isinstance(result, list) else result


def get_block(block_id: str) -> dict:
    rows = _get(f"room_blocks?id=eq.{block_id}")
    if not rows:
        raise ValueError(f"Block {block_id} not found")
    return rows[0]
