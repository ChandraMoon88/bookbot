"""
services/room_service/soft_lock.py
-------------------------------------
Redis soft lock for rooms during booking session.

Uses an atomic Lua script so all dates are locked or none (all-or-nothing).
Lock key: soft_lock:{room_type_id}:{date}  EX {lock_mins*60}
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta, datetime

logger   = logging.getLogger(__name__)
REDIS_URL = os.getenv("REDIS_URL", "")

_redis = None


def _get_redis():
    global _redis
    if _redis is None and REDIS_URL:
        import redis as rlib
        _redis = rlib.from_url(REDIS_URL, decode_responses=True)
    return _redis


# Lua script: lock every date or abort if any is already locked by someone else
_LOCK_SCRIPT = """
local session = ARGV[1]
local ttl     = tonumber(ARGV[2])
-- Check all keys first
for i = 1, #KEYS do
    local cur = redis.call('GET', KEYS[i])
    if cur and cur ~= session then
        return 0
    end
end
-- Set all
for i = 1, #KEYS do
    redis.call('SET', KEYS[i], session, 'EX', ttl)
end
return 1
"""


def acquire_lock(
    room_type_id:  str,
    check_in:      str,
    check_out:     str,
    session_id:    str,
    lock_minutes:  int = 15,
) -> dict:
    """
    Atomically lock all nights of a stay for a session.

    Returns: { locked: bool, expires_at: str | None }
    """
    r = _get_redis()
    if not r:
        # No Redis — allow booking without soft lock
        return {"locked": True, "expires_at": None}

    # Build date list
    keys: list[str] = []
    d = date.fromisoformat(check_in)
    end = date.fromisoformat(check_out)
    while d < end:
        keys.append(f"soft_lock:{room_type_id}:{d}")
        d += timedelta(days=1)

    ttl  = lock_minutes * 60
    lock_fn = r.register_script(_LOCK_SCRIPT)
    try:
        result = lock_fn(keys=keys, args=[session_id, ttl])
        locked = bool(result)
        expires_at = (
            datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            if locked else None
        )
        return {"locked": locked, "expires_at": expires_at}
    except Exception as e:
        logger.error("soft_lock acquire error: %s", e)
        return {"locked": False, "expires_at": None}


def release_lock(room_type_id: str, check_in: str, check_out: str, session_id: str) -> None:
    """Release all date locks for this session (used on cancellation/restart)."""
    r = _get_redis()
    if not r:
        return
    d = date.fromisoformat(check_in)
    end = date.fromisoformat(check_out)
    keys = []
    while d < end:
        keys.append(f"soft_lock:{room_type_id}:{d}")
        d += timedelta(days=1)
    try:
        pipe = r.pipeline()
        for key in keys:
            val = r.get(key)
            if val == session_id:
                pipe.delete(key)
        pipe.execute()
    except Exception as e:
        logger.error("soft_lock release error: %s", e)


def refresh_lock(room_type_id: str, check_in: str, check_out: str,
                 session_id: str, lock_minutes: int = 15) -> bool:
    """Extend TTL on existing locks (called every 5 min during payment)."""
    r = _get_redis()
    if not r:
        return True
    ttl = lock_minutes * 60
    d   = date.fromisoformat(check_in)
    end = date.fromisoformat(check_out)
    try:
        pipe = r.pipeline()
        while d < end:
            key = f"soft_lock:{room_type_id}:{d}"
            pipe.expire(key, ttl)
            d += timedelta(days=1)
        pipe.execute()
        return True
    except Exception as e:
        logger.error("soft_lock refresh error: %s", e)
        return False
