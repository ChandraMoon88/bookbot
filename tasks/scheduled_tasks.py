"""
tasks/scheduled_tasks.py
--------------------------
Celery beat-scheduled maintenance tasks.
"""

import os
import logging
from datetime import datetime, timezone, timedelta

from .celery_app import celery_app

log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
REDIS_URL    = os.environ.get("REDIS_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


@celery_app.task(name="tasks.scheduled_tasks.trigger_review_requests")
def trigger_review_requests():
    """
    Finds bookings that checked out yesterday and enqueues review requests.
    """
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM bookings WHERE check_out = %s AND status = 'confirmed'",
                (yesterday,),
            )
            bookings = [dict(r) for r in cur.fetchall()]

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
    now = datetime.now(timezone.utc)
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE loyalty_transactions SET expired = TRUE "
                "WHERE expires_at < %s AND expired = FALSE",
                (now,),
            )
            updated_count = cur.rowcount
            conn.commit()
    log.info("Expired %d loyalty point records", updated_count)
    return {"expired": updated_count}


@celery_app.task(name="tasks.scheduled_tasks.release_stale_soft_locks")
def release_stale_soft_locks():
    """
    Redis soft locks expire via TTL automatically, but this task cleans up
    any orphaned keys whose session no longer exists.
    Iterates soft_lock:* keys older than LOCK_TTL without matching session.
    """
    import redis as redis_lib
    if not REDIS_URL:
        log.warning("REDIS_URL not set; skipping stale lock cleanup")
        return {"released": 0}
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
