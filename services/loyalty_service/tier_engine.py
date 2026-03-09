"""
services/loyalty_service/tier_engine.py
-----------------------------------------
Upgrades / downgrades loyalty tier based on lifetime points.

Tiers:
  Bronze  :     0 – 4 999 points
  Silver  :  5 000 – 19 999 points
  Gold    : 20 000 – 49 999 points
  Platinum: 50 000+ points
"""

TIER_THRESHOLDS = [
    ("platinum", 50_000),
    ("gold",     20_000),
    ("silver",    5_000),
    ("bronze",        0),
]

TIER_PERKS = {
    "bronze":   {"late_checkout": False, "lounge_access": False, "discount_pct": 0},
    "silver":   {"late_checkout": False, "lounge_access": False, "discount_pct": 5},
    "gold":     {"late_checkout": True,  "lounge_access": False, "discount_pct": 10},
    "platinum": {"late_checkout": True,  "lounge_access": True,  "discount_pct": 15},
}


def evaluate(lifetime_points: int) -> str:
    """Returns the tier name for the given lifetime points total."""
    for tier_name, threshold in TIER_THRESHOLDS:
        if lifetime_points >= threshold:
            return tier_name
    return "bronze"


def get_perks(tier: str) -> dict:
    return TIER_PERKS.get(tier, TIER_PERKS["bronze"])


def next_tier_info(lifetime_points: int) -> dict:
    """Returns how many points away the guest is from the next tier."""
    current = evaluate(lifetime_points)
    tiers   = [t for t, _ in TIER_THRESHOLDS]
    idx     = tiers.index(current)

    if idx == 0:
        return {"next_tier": None, "points_needed": 0, "current_tier": current}

    next_tier_name, next_threshold = TIER_THRESHOLDS[idx - 1]
    return {
        "current_tier": current,
        "next_tier":    next_tier_name,
        "points_needed": next_threshold - lifetime_points,
    }
