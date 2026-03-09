"""
services/loyalty_service/gamification.py
------------------------------------------
Leaderboard and challenge badges using Redis sorted sets.

Leaderboard key  : loyalty:leaderboard
Challenge keys   : loyalty:challenge:{name}:{guest_id}
"""

import os
import redis
import logging
from typing import Optional

log = logging.getLogger(__name__)

_r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=True)

LB_KEY = "loyalty:leaderboard"

BADGES = {
    "first_booking":  {"label": "First Booking",  "points_bonus": 200},
    "five_stays":     {"label": "5 Stays",         "points_bonus": 500},
    "ten_stays":      {"label": "10 Stays",         "points_bonus": 1000},
    "globe_trotter":  {"label": "Globe Trotter",   "points_bonus": 1500},   # 3+ cities
    "loyalty_fan":    {"label": "Loyalty Fan",     "points_bonus": 300},    # app review submitted
    "speed_booker":   {"label": "Speed Booker",    "points_bonus": 100},    # books within 5 min
}


def update_leaderboard(guest_id: str, total_points: int) -> None:
    """Adds/updates the guest's score in the global leaderboard."""
    _r.zadd(LB_KEY, {guest_id: total_points})


def top_guests(n: int = 10) -> list[dict]:
    """Returns the top-N guests with their scores."""
    entries = _r.zrevrange(LB_KEY, 0, n - 1, withscores=True)
    return [{"guest_id": gid, "points": int(score)} for gid, score in entries]


def get_rank(guest_id: str) -> Optional[int]:
    """Returns 1-based rank of the guest (None if not in leaderboard)."""
    rank = _r.zrevrank(LB_KEY, guest_id)
    return None if rank is None else rank + 1


def award_badge(guest_id: str, badge_name: str) -> dict:
    """
    Marks a badge as earned for the guest.
    Returns {badge, points_bonus, already_earned}.
    """
    key = f"loyalty:badge:{badge_name}:{guest_id}"
    already = _r.exists(key)
    if already:
        return {"badge": badge_name, "points_bonus": 0, "already_earned": True}

    _r.set(key, "1")
    bonus = BADGES.get(badge_name, {}).get("points_bonus", 0)
    log.info("Badge '%s' awarded to guest %s (+%d pts)", badge_name, guest_id, bonus)
    return {"badge": badge_name, "points_bonus": bonus, "already_earned": False}


def list_badges(guest_id: str) -> list[str]:
    """Returns list of badge names earned by the guest."""
    earned = []
    for badge_name in BADGES:
        key = f"loyalty:badge:{badge_name}:{guest_id}"
        if _r.exists(key):
            earned.append(badge_name)
    return earned
