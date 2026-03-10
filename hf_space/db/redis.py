"""
hf_space/db/redis.py
----------------------
Production Redis client for the HuggingFace Space.

Priority order:
  1. Upstash HTTP REST (UPSTASH_REDIS_URL + UPSTASH_REDIS_TOKEN) — preferred
     HF Spaces blocks outbound TCP on non-standard ports; HTTP always works.
  2. Standard async Redis via REDIS_URL (rediss://...upstash.io TLS) — fallback

Do NOT use a plain redis://localhost URL in production.
Set UPSTASH_REDIS_URL + UPSTASH_REDIS_TOKEN in HF Space secrets.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from upstash_redis.asyncio import Redis as UpstashRedis
    _USE_UPSTASH = True
except ImportError:
    _USE_UPSTASH = False

_client: Optional[object] = None


async def init_redis() -> object:
    """
    Initialise Redis client and verify connectivity.
    Called once from app lifespan.
    """
    global _client

    upstash_url   = os.environ.get("UPSTASH_REDIS_URL", "")
    upstash_token = os.environ.get("UPSTASH_REDIS_TOKEN", "")

    if _USE_UPSTASH and upstash_url and upstash_token:
        # ── Upstash HTTP REST (preferred on HF Spaces) ────────────────────────
        _client = UpstashRedis(url=upstash_url, token=upstash_token)
        result = await _client.ping()
        if result is not True and result != "PONG":
            raise ConnectionError(f"Upstash Redis PING failed: {result!r}")
        logger.info("✅ Upstash HTTP Redis connected")
    else:
        # ── Standard async Redis via REDIS_URL (Upstash TLS or other) ─────────
        redis_url = os.environ.get("REDIS_URL", "")
        if not redis_url:
            raise RuntimeError(
                "Redis not configured — set UPSTASH_REDIS_URL + UPSTASH_REDIS_TOKEN "
                "in HF Space secrets (preferred), or set REDIS_URL=rediss://..."
            )
        import ssl
        import redis.asyncio as aioredis
        kwargs: dict = {"encoding": "utf-8", "decode_responses": True}
        if redis_url.startswith("rediss://") or ".upstash.io" in redis_url:
            # Upstash TLS via TCP — normalise scheme and disable cert check
            redis_url = redis_url.replace("redis://", "rediss://", 1)
            kwargs["ssl_cert_reqs"] = ssl.CERT_NONE
        _client = aioredis.from_url(redis_url, **kwargs)
        await _client.ping()
        logger.info("✅ Redis TLS connected (REDIS_URL)")

    return _client


def get_redis() -> object:
    """Return the initialised Redis client singleton."""
    if _client is None:
        raise RuntimeError(
            "Redis has not been initialised — call init_redis() during startup"
        )
    return _client


# ── Convenience helpers (state machine — Section 7) ───────────────────────────

VALID_STATES = {
    "new", "greeting", "searching", "viewing_hotels", "selecting_room",
    "choosing_rate", "filling_guest_form", "selecting_addons",
    "reviewing_booking", "paying", "booking_confirmed", "modifying",
    "cancelling", "faq_browsing", "handoff_active", "post_stay_review",
    "loyalty_browsing",
}


async def get_user_state(psid: str) -> str:
    """Return user's current conversation state (default: 'new')."""
    r = get_redis()
    state = await r.get(f"user:{psid}:state")
    return state or "new"


async def set_user_state(psid: str, state: str) -> None:
    """Transition to new state — validates against VALID_STATES."""
    if state not in VALID_STATES:
        logger.warning("Invalid state transition attempted: %s for psid=%s", state, psid)
        return
    r = get_redis()
    await r.set(f"user:{psid}:state", state, ex=172800)  # 48-hour TTL


async def get_user_profile(psid: str) -> dict | None:
    """Retrieve user profile from Redis (30-day TTL)."""
    import json
    r = get_redis()
    raw = await r.get(f"user:{psid}:profile")
    return json.loads(raw) if raw else None


async def set_user_profile(psid: str, profile: dict) -> None:
    """Persist user profile to Redis with 30-day TTL."""
    import json
    r = get_redis()
    await r.set(f"user:{psid}:profile", json.dumps(profile), ex=2592000)


async def get_booking_draft(psid: str) -> dict | None:
    """Retrieve in-progress booking draft."""
    import json
    r = get_redis()
    raw = await r.get(f"user:{psid}:booking_draft")
    return json.loads(raw) if raw else None


async def set_booking_draft(psid: str, draft: dict) -> None:
    """Save booking draft with 48-hour TTL."""
    import json
    r = get_redis()
    await r.set(f"user:{psid}:booking_draft", json.dumps(draft), ex=172800)
