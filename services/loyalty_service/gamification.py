"""
services/loyalty_service/gamification.py
------------------------------------------
Leaderboard and challenge badges using Redis sorted sets.

Redis is optional — if no URL is configured, all functions degrade
gracefully (leaderboard returns empty, badges are PostgreSQL-backed).

Redis keys:
  loyalty:leaderboard          — ZSET  guest_id → points
  loyalty:badge:{name}:{guest} — STRING "1"
"""

import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

LB_KEY = "loyalty:leaderboard"

BADGES = {
    "first_booking":  {"label": "First Booking",  "points_bonus": 200},
    "five_stays":     {"label": "5 Stays",         "points_bonus": 500},
    "ten_stays":      {"label": "10 Stays",         "points_bonus": 1000},
    "globe_trotter":  {"label": "Globe Trotter",   "points_bonus": 1500},   # 3+ cities
    "loyalty_fan":    {"label": "Loyalty Fan",     "points_bonus": 300},    # app review submitted
    "speed_booker":   {"label": "Speed Booker",    "points_bonus": 100},    # books within 5 min
}


def _get_redis():
    """
    Lazily return a Redis client using UPSTASH_REDIS_URL / REDIS_URL.
    Returns None if no Redis URL is configured so callers can degrade gracefully.
    """
    url = (
        os.environ.get("UPSTASH_REDIS_URL")
        or os.environ.get("REDIS_URL")
        or ""
    )
    if not url:
        return None
    try:
        import redis as _redis
        return _redis.from_url(url, decode_responses=True, socket_connect_timeout=3)
    except Exception as exc:
        log.warning("Redis unavailable: %s", exc)
        return None


def update_leaderboard(guest_id: str, total_points: int) -> None:
    """Adds/updates the guest's score in the global leaderboard."""
    r = _get_redis()
    if r:
        try:
            r.zadd(LB_KEY, {guest_id: total_points})
        except Exception as exc:
            log.warning("leaderboard_update_failed: %s", exc)


def top_guests(n: int = 10) -> list[dict]:
    """Returns the top-N guests with their scores."""
    r = _get_redis()
    if not r:
        return []
    try:
        entries = r.zrevrange(LB_KEY, 0, n - 1, withscores=True)
        return [{"guest_id": gid, "points": int(score)} for gid, score in entries]
    except Exception as exc:
        log.warning("leaderboard_top_failed: %s", exc)
        return []


def get_rank(guest_id: str) -> Optional[int]:
    """Returns 1-based rank of the guest (None if not in leaderboard)."""
    r = _get_redis()
    if not r:
        return None
    try:
        rank = r.zrevrank(LB_KEY, guest_id)
        return None if rank is None else rank + 1
    except Exception:
        return None


def award_badge(guest_id: str, badge_name: str) -> dict:
    """
    Marks a badge as earned for the guest.
    Returns {badge, points_bonus, already_earned}.
    """
    r = _get_redis()
    if not r:
        # No Redis — treat as not yet earned (allows re-award on next session)
        bonus = BADGES.get(badge_name, {}).get("points_bonus", 0)
        return {"badge": badge_name, "points_bonus": bonus, "already_earned": False}

    key = f"loyalty:badge:{badge_name}:{guest_id}"
    try:
        already = r.exists(key)
        if already:
            return {"badge": badge_name, "points_bonus": 0, "already_earned": True}
        r.set(key, "1")
    except Exception as exc:
        log.warning("badge_award_failed: %s", exc)
        already = False

    bonus = BADGES.get(badge_name, {}).get("points_bonus", 0)
    log.info("Badge '%s' awarded to guest %s (+%d pts)", badge_name, guest_id, bonus)
    return {"badge": badge_name, "points_bonus": bonus, "already_earned": False}


def list_badges(guest_id: str) -> list[str]:
    """Returns list of badge names earned by the guest."""
    r = _get_redis()
    if not r:
        return []
    earned = []
    for badge_name in BADGES:
        try:
            key = f"loyalty:badge:{badge_name}:{guest_id}"
            if r.exists(key):
                earned.append(badge_name)
        except Exception:
            pass
    return earned

