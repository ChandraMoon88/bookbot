import os
import redis

_redis_client = None

def get_redis():
    """Lazy Redis connection — only connects on first use, not at import time."""
    global _redis_client
    if _redis_client is None:
        url = os.getenv("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL environment variable is not set")
        _redis_client = redis.from_url(url, decode_responses=True)
    return _redis_client


def save_session(messenger_id, data):
    get_redis().setex(
        f"session:{messenger_id}",
        1800,  # expires in 30 minutes
        str(data)
    )


def get_session(messenger_id):
    return get_redis().get(f"session:{messenger_id}")