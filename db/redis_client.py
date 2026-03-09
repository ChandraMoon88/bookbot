"""
db/redis_client.py
------------------
Redis connection clients.
Provides both async (for FastAPI) and sync (for Celery/Rasa) clients.
"""

import os
import ssl as _ssl
import redis
import redis.asyncio as aioredis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

_REDIS_SSL_KWARGS: dict = {}

def _resolve_url(url: str) -> tuple[str, dict]:
    """Return (resolved_url, ssl_kwargs) for the given Redis URL."""
    kwargs: dict = {}
    if url.startswith("rediss://") or ".upstash.io" in url:
        kwargs["ssl_cert_reqs"] = _ssl.CERT_NONE
        url = url.replace("redis://", "rediss://", 1)
    return url, kwargs


def _sync_client() -> redis.Redis:
    url, kwargs = _resolve_url(REDIS_URL)
    return redis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
        **kwargs,
    )


def _async_client() -> aioredis.Redis:
    url, kwargs = _resolve_url(REDIS_URL)
    return aioredis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
        **kwargs,
    )


# Lazily initialised singletons
_sync: redis.Redis | None   = None
_async: aioredis.Redis | None = None


def sync_redis() -> redis.Redis:
    global _sync
    if _sync is None:
        _sync = _sync_client()
    return _sync


def async_redis() -> aioredis.Redis:
    global _async
    if _async is None:
        _async = _async_client()
    return _async


# ── Convenience wrappers used across the project ────────────────────────────

class RedisKeys:
    """Centralised Redis key patterns — never hardcode keys anywhere else."""

    @staticmethod
    def session_lang(session_id: str)      -> str: return f"session:{session_id}:lang"
    @staticmethod
    def session_user(session_id: str)      -> str: return f"session:{session_id}:user_id"
    @staticmethod
    def session_booking(session_id: str)   -> str: return f"session:{session_id}:booking_state"
    @staticmethod
    def soft_lock(room_type_id, date: str) -> str: return f"soft_lock:{room_type_id}:{date}"
    @staticmethod
    def pay_attempts(user_id: str)         -> str: return f"user:{user_id}:payment_attempts:1h"
    @staticmethod
    def login_attempts(user_id: str)       -> str: return f"user:{user_id}:login_attempts"
    @staticmethod
    def tts_cache(lang: str, text_hash: str) -> str: return f"tts_cache:{lang}:{text_hash}"
    @staticmethod
    def leaderboard(yyyymm: str)           -> str: return f"leaderboard:monthly:{yyyymm}"
    @staticmethod
    def booking_months(user_id: str)       -> str: return f"user:{user_id}:booking_months"
    @staticmethod
    def rasa_tracker(sender_id: str)       -> str: return f"rasa_tracker:{sender_id}"
    @staticmethod
    def token_blacklist(token_hash: str)   -> str: return f"blacklist:{token_hash}"
    @staticmethod
    def force_majeure(country_code: str)   -> str: return f"force_majeure:{country_code}"
    @staticmethod
    def bb_lang(sender_id: str)            -> str: return f"bblang:{sender_id}"
