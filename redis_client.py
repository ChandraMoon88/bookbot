import os
import redis_client as redis_lib

_client = None

def get_redis():
    """Lazy Redis connection — only connects on first use, not at import time."""
    global _client
    if _client is None:
        url = os.getenv("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL environment variable is not set")
        _client = redis_lib.from_url(url, decode_responses=True)
    return _client


def save_session(messenger_id, data):
    get_redis().setex(f"session:{messenger_id}", 1800, str(data))


def get_session(messenger_id):
    return get_redis().get(f"session:{messenger_id}")