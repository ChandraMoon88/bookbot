import os
import ssl as _ssl
import redis as redis_lib

_client = None

def get_redis():
    """Lazy Redis connection — only connects on first use, not at import time.
    Only passes SSL kwargs when the URL uses the rediss:// scheme.
    redis:// (plain) URLs must not receive SSL kwargs — causes TypeError.
    """
    global _client
    if _client is None:
        url = os.getenv("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL environment variable is not set")
        kwargs = {"decode_responses": True}
        if url.startswith("rediss://") or ".upstash.io" in url:
            kwargs["ssl_cert_reqs"] = _ssl.CERT_NONE
            # Normalise scheme so the redis library uses TLS
            url = url.replace("redis://", "rediss://", 1)
        _client = redis_lib.from_url(url, **kwargs)
    return _client


def save_session(messenger_id, data):
    get_redis().setex(f"session:{messenger_id}", 1800, str(data))


def get_session(messenger_id):
    return get_redis().get(f"session:{messenger_id}")