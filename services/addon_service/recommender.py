"""
services/addon_service/recommender.py
---------------------------------------
Rule-based + collaborative-filter add-on recommender.
Trip purpose is inferred from the booking context supplied by the caller.
"""

from typing import Optional

ADDONS_CATALOG = {
    "spa_package":       {"name": "Spa & Wellness Package",      "price": 89,  "currency": "USD"},
    "champagne_arrival": {"name": "Champagne on Arrival",        "price": 35,  "currency": "USD"},
    "couples_dinner":    {"name": "Romantic Couples Dinner",     "price": 79,  "currency": "USD"},
    "airport_transfer":  {"name": "Airport Transfer",            "price": 45,  "currency": "USD"},
    "late_checkout":     {"name": "Late Check-out (2 PM)",       "price": 30,  "currency": "USD"},
    "early_checkin":     {"name": "Early Check-in (10 AM)",      "price": 30,  "currency": "USD"},
    "breakfast":         {"name": "Daily Breakfast Included",    "price": 25,  "currency": "USD"},
    "kids_club":         {"name": "Kids Club Access",            "price": 20,  "currency": "USD"},
    "crib_rental":       {"name": "Baby Crib / Co-sleeper",      "price": 15,  "currency": "USD"},
    "city_tour":         {"name": "Half-Day City Tour",          "price": 55,  "currency": "USD"},
    "meeting_room":      {"name": "Meeting Room (half day)",     "price": 120, "currency": "USD"},
    "business_lounge":   {"name": "Business Lounge Access",      "price": 40,  "currency": "USD"},
    "anniversary_decor": {"name": "Room Anniversary Decoration", "price": 60,  "currency": "USD"},
    "pet_amenities":     {"name": "Pet Welcome Kit",             "price": 18,  "currency": "USD"},
}

PURPOSE_MAP = {
    "honeymoon":   ["champagne_arrival", "couples_dinner", "spa_package", "anniversary_decor"],
    "anniversary": ["champagne_arrival", "anniversary_decor", "couples_dinner", "spa_package"],
    "family":      ["crib_rental", "kids_club", "breakfast", "airport_transfer"],
    "business":    ["late_checkout", "early_checkin", "meeting_room", "business_lounge", "airport_transfer"],
    "leisure":     ["breakfast", "city_tour", "spa_package", "airport_transfer"],
    "solo":        ["breakfast", "city_tour", "airport_transfer"],
    "pet":         ["pet_amenities", "breakfast", "airport_transfer"],
}


def recommend(
    purpose:     Optional[str] = None,
    guests:      int = 1,
    nights:      int = 1,
    hotel_stars: int = 3,
) -> list[dict]:
    """
    Returns a ranked list of add-on dicts to present to the guest.
    Each dict: {id, name, price, currency}
    """
    purpose = (purpose or "leisure").lower()
    addon_ids = PURPOSE_MAP.get(purpose, PURPOSE_MAP["leisure"])

    # For single-night stays always surface late/early check options
    if nights == 1 and "late_checkout" not in addon_ids:
        addon_ids = ["late_checkout"] + addon_ids

    # 4-5 star → upgrade spa priority
    if hotel_stars >= 4 and "spa_package" not in addon_ids[:2]:
        addon_ids = ["spa_package"] + [a for a in addon_ids if a != "spa_package"]

    results = []
    for aid in addon_ids[:5]:
        if aid in ADDONS_CATALOG:
            item = {"id": aid}
            item.update(ADDONS_CATALOG[aid])
            results.append(item)

    return results
