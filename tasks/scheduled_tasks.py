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


@celery_app.task(name="tasks.scheduled_tasks.send_checkin_reminders")
def send_checkin_reminders():
    """
    K5 — Send a proactive check-in reminder 24 hours before the guest's check-in date.
    Pushes a Messenger message via the /send_proactive endpoint on the main.py server.
    """
    import requests
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    MAIN_URL = os.environ.get("MAIN_SERVICE_URL", "")
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT b.booking_reference,
                       b.check_in,
                       b.user_id,
                       h.name  AS hotel_name,
                       rt.name AS room_name,
                       u.messenger_psid AS psid
                FROM   bookings b
                LEFT JOIN hotel_partners  h  ON b.hotel_id     = h.id
                LEFT JOIN room_types      rt ON b.room_type_id = rt.id
                LEFT JOIN users           u  ON b.user_id      = u.id
                WHERE  b.check_in = %s
                  AND  b.status   = 'confirmed'
                  AND  u.messenger_psid IS NOT NULL
                """,
                (tomorrow,),
            )
            bookings = [dict(r) for r in cur.fetchall()]

    sent = 0
    for b in bookings:
        psid = b.get("psid")
        if not psid or not MAIN_URL:
            continue
        msg = (
            f"Heads up! Your check-in at {b.get('hotel_name', 'your hotel')} is TOMORROW.\n"
            f"{tomorrow} | {b.get('room_name', '')}\n\n"
            f"Booking: {b.get('booking_reference')}\n\n"
            "Quick services before you arrive:"
        )
        buttons = [
            {"title": "Add Airport Transfer", "payload": "AIRPORT_TRANSFER"},
            {"title": "Book a Spa Treatment", "payload": "SPA_BOOKING"},
            {"title": "Request Early Check-in","payload": "EARLY_CHECKIN"},
            {"title": "Special Occasion Setup","payload": "SPECIAL_OCCASION"},
            {"title": "View Full Booking",    "payload": f"BOOKING_DETAIL_{b.get('booking_reference')}"},
            {"title": "Cancel Booking",       "payload": "CANCEL_BOOKING"},
        ]
        try:
            requests.post(
                f"{MAIN_URL}/send_proactive",
                json={"psid": psid, "text": msg, "buttons": buttons},
                timeout=10,
            )
            sent += 1
        except Exception as exc:
            log.warning("Failed to send check-in reminder to %s: %s", psid, exc)

    log.info("Sent %d check-in reminders for %s", sent, tomorrow)
    return {"sent": sent, "date": tomorrow}


@celery_app.task(name="tasks.scheduled_tasks.recover_abandoned_bookings")
def recover_abandoned_bookings():
    """
    K4 — Re-engage users who started a booking (reached payment step) but did not complete it.
    Checks Redis for 'abandoned' sessions older than 2 hours and pushes a reminder.
    """
    import json
    import redis as redis_lib
    import requests

    MAIN_URL  = os.environ.get("MAIN_SERVICE_URL", "")
    if not REDIS_URL or not MAIN_URL:
        log.warning("REDIS_URL or MAIN_SERVICE_URL not set; skipping abandoned booking recovery")
        return {"recovered": 0}

    r   = redis_lib.from_url(REDIS_URL, decode_responses=True)
    now = datetime.now(timezone.utc)
    TWO_HOURS = timedelta(hours=2)
    recovered = 0

    cursor = 0
    while True:
        cursor, keys = r.scan(cursor, match="state:*", count=500)
        for key in keys:
            try:
                raw = r.get(key)
                if not raw:
                    continue
                session = json.loads(raw)
                # Only target sessions that reached the payment step
                if session.get("step") not in ("payment", "pay_method"):
                    continue
                last_seen_str = session.get("last_seen")
                if not last_seen_str:
                    continue
                last_seen = datetime.fromisoformat(last_seen_str)
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)
                if (now - last_seen) < TWO_HOURS:
                    continue  # not stale yet

                psid  = key.split(":", 1)[-1]
                hotel = session.get("hotel_name", "your selected hotel")
                city  = session.get("city", "")
                checkin  = session.get("checkin", "")
                checkout = session.get("checkout", "")
                guests   = session.get("num_adults", 2)
                name     = (session.get("guest_name") or "there").split()[0]
                price_s  = session.get("price_str", "")

                date_line = f"{checkin} – {checkout} | " if checkin else ""
                msg = (
                    f"Hey {name}!\n\n"
                    f"You were so close to booking {hotel}{' in ' + city if city else ''}!\n"
                    f"Your search is still saved:\n\n"
                    f"{hotel}\n"
                    f"{date_line}{guests} guests{' | ' + price_s if price_s else ''}\n\n"
                    "Prices may change — complete your booking now to lock in this rate!"
                )
                buttons = [
                    {"title": "Complete My Booking", "payload": "RESUME_BOOKING"},
                    {"title": "Start a New Search",  "payload": "ACTION_BOOK"},
                    {"title": "I Need Help",          "payload": "AGENT_HANDOFF"},
                ]
                try:
                    requests.post(
                        f"{MAIN_URL}/send_proactive",
                        json={"psid": psid, "text": msg, "buttons": buttons},
                        timeout=10,
                    )
                    recovered += 1
                except Exception as exc:
                    log.warning("Failed to send recovery message to %s: %s", psid, exc)
            except Exception as exc:
                log.warning("Error processing key %s: %s", key, exc)
        if cursor == 0:
            break

    log.info("Sent %d abandoned-booking recovery messages", recovered)
    return {"recovered": recovered}
