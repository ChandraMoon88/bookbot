"""
services/force_majeure_service/relocation.py
----------------------------------------------
Finds alternative hotels when a force-majeure event affects the original hotel.
Calls the search service to find available hotels in nearby cities.
"""

import os
import json
import logging
import urllib.request

log = logging.getLogger(__name__)

SEARCH_SERVICE_URL = os.environ.get("SEARCH_SERVICE_URL", "http://localhost:8002")


def find_alternatives(
    original_city: str,
    check_in:      str,
    check_out:     str,
    guests:        int = 1,
    budget:        float = 500.0,
    stars:         int  = 3,
    radius_km:     int  = 50,
) -> list[dict]:
    """
    Calls the search service for nearby hotels and returns relocation options.
    """
    payload = json.dumps({
        "city":       original_city,
        "check_in":   check_in,
        "check_out":  check_out,
        "guests":     guests,
        "max_price":  budget,
        "stars":      stars,
        "radius_km":  radius_km,
        "limit":      5,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{SEARCH_SERVICE_URL}/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("hotels", [])
    except Exception as exc:
        log.error("Search service unavailable for relocation: %s", exc)
        return []


def build_relocation_message(
    alternatives: list[dict],
    original_ref: str,
) -> str:
    """Returns a human-readable relocation offer message."""
    if not alternatives:
        return (
            f"We're sorry, no suitable alternatives are available right now. "
            f"Your booking {original_ref} has been marked for full refund."
        )
    lines = [
        f"Due to a force-majeure event, we're offering you a free move for booking {original_ref}.\n"
        "Available alternatives:\n"
    ]
    for i, h in enumerate(alternatives[:3], 1):
        name  = h.get("name", "Hotel")
        price = h.get("price_per_night", "—")
        stars = "★" * int(h.get("stars", 3))
        lines.append(f"{i}. {name} {stars} — ${price}/night")
    lines.append("\nReply with the number to confirm or type REFUND for a full refund.")
    return "\n".join(lines)
