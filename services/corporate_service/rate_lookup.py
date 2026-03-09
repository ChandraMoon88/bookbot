"""
services/corporate_service/rate_lookup.py
--------------------------------------------
Retrieves negotiated corporate rates from Supabase.
Falls back to the standard best-available rate if no contract exists.
"""

import os
import json
import logging
import urllib.request
from datetime import datetime

log = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


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
    url = (
        f"{SUPABASE_URL}/rest/v1/corporate_rates"
        f"?corporate_account_id=eq.{corporate_account_id}"
        f"&hotel_id=eq.{hotel_id}"
        f"&room_type_id=eq.{room_type_id}"
        f"&valid_from=lte.{check_in}"
        f"&valid_to=gte.{check_in}"
        f"&is_active=eq.true"
        f"&select=*&limit=1"
    )
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req) as resp:
        rows = json.loads(resp.read())

    if rows:
        r = rows[0]
        return {
            "rate_per_night": float(r["rate_per_night"]),
            "currency":       r.get("currency", "USD"),
            "contract_id":    r["id"],
            "is_negotiated":  True,
            "discount_label": f"Corporate rate – contract {r['id'][:8]}",
        }

    # Fallback: fetch standard room rate
    url2 = (
        f"{SUPABASE_URL}/rest/v1/room_rates"
        f"?room_type_id=eq.{room_type_id}"
        f"&rate_type=eq.standard"
        f"&select=rate_per_night,currency&limit=1"
    )
    req2 = urllib.request.Request(url2, headers=_headers())
    with urllib.request.urlopen(req2) as resp:
        rows2 = json.loads(resp.read())

    std_rate = float(rows2[0]["rate_per_night"]) if rows2 else 100.0
    return {
        "rate_per_night": std_rate,
        "currency":       "USD",
        "contract_id":    None,
        "is_negotiated":  False,
        "discount_label": "Best available rate",
    }
