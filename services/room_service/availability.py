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

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_KEY", ""))


def _get(endpoint: str, params: dict) -> list:
    import urllib.request, urllib.parse, json
    url = SUPABASE_URL + endpoint + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    })
    try:
        r = urllib.request.urlopen(req, timeout=10)
        return json.loads(r.read().decode())
    except Exception as e:
        logger.error("availability _get error: %s", e)
        return []


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
    rows = _get("/rest/v1/room_availability", {
        "room_type_id": f"eq.{room_type_id}",
        "date":         f"gte.{check_in}",
        "select":       "date,available_count",
    })

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
