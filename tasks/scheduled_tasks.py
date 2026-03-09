"""
tasks/scheduled_tasks.py
--------------------------
Celery beat-scheduled maintenance tasks.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
import urllib.request

from .celery_app import celery_app

log = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
REDIS_URL    = os.environ.get("REDIS_URL", "redis://localhost:6379/0")


def _sb_headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


@celery_app.task(name="tasks.scheduled_tasks.trigger_review_requests")
def trigger_review_requests():
    """
    Finds bookings that checked out yesterday and enqueues review requests.
    """
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"{SUPABASE_URL}/rest/v1/bookings"
        f"?check_out=eq.{yesterday}&status=eq.confirmed&select=id"
    )
    req = urllib.request.Request(url, headers=_sb_headers())
    with urllib.request.urlopen(req) as resp:
        bookings = json.loads(resp.read())

    queued = 0
    for b in bookings:
        celery_app.send_task(
            "review_service.send_review_request",
            kwargs={"booking_id": b["id"]},
            countdown=3600,  # 1 h after checkout
        )
        queued += 1
    log.info("Queued %d review request tasks for %s", queued, yesterday)
    return {"queued": queued, "date": yesterday}


@celery_app.task(name="tasks.scheduled_tasks.expire_loyalty_points")
def expire_loyalty_points():
    """
    Marks loyalty_transactions expired where expires_at < now.
    (Points deduction handled by a Supabase function / trigger in production.)
    """
    now = datetime.now(timezone.utc).isoformat()
    url = (
        f"{SUPABASE_URL}/rest/v1/loyalty_transactions"
        f"?expires_at=lt.{now}&expired=eq.false"
    )
    headers = {**_sb_headers(), "Prefer": "return=representation"}
    data    = json.dumps({"expired": True}).encode()
    req     = urllib.request.Request(url, data=data, headers=headers, method="PATCH")
    with urllib.request.urlopen(req) as resp:
        updated = json.loads(resp.read())
    log.info("Expired %d loyalty point records", len(updated))
    return {"expired": len(updated)}


@celery_app.task(name="tasks.scheduled_tasks.release_stale_soft_locks")
def release_stale_soft_locks():
    """
    Redis soft locks expire via TTL automatically, but this task cleans up
    any orphaned keys whose session no longer exists.
    Iterates soft_lock:* keys older than LOCK_TTL without matching session.
    """
    import redis as redis_lib
    r = redis_lib.from_url(REDIS_URL, decode_responses=True)
    pattern  = "soft_lock:*"
    cursor   = 0
    released = 0

    while True:
        cursor, keys = r.scan(cursor, match=pattern, count=200)
        for key in keys:
            ttl = r.ttl(key)
            if ttl == -1:   # no TTL set → stale
                r.delete(key)
                released += 1
        if cursor == 0:
            break

    log.info("Released %d stale soft locks", released)
    return {"released": released}
