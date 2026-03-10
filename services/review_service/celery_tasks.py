"""
services/review_service/celery_tasks.py
-----------------------------------------
Celery tasks for sending post-stay review requests.
Triggered 24 h after checkout via a Celery beat schedule.
"""

from celery import shared_task
from datetime import datetime, timezone
import os
import json
import logging
import urllib.request

log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
NOTIFY_URL   = os.environ.get("NOTIFICATION_SERVICE_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


@shared_task(name="review_service.send_review_request", bind=True, max_retries=3)
def send_review_request(self, booking_id: str):
    """
    Fetches booking info and sends review request email + Messenger message.
    """
    try:
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT b.*, row_to_json(g.*) AS guests "
                    "FROM bookings b LEFT JOIN guests g ON g.booking_id=b.id "
                    "WHERE b.id=%s",
                    (booking_id,),
                )
                row = cur.fetchone()

        if not row:
            log.warning("Review task: booking %s not found", booking_id)
            return

        booking = dict(row)
        guest   = booking.pop("guests") or {}
        ctx     = {
            "guest_name":  guest.get("first_name", "Guest"),
            "hotel_name":  booking.get("hotel_id"),
            "booking_ref": booking.get("ref"),
            "review_url":  f"https://bookhotel.ai/review/{booking.get('ref')}",
        }

        if guest.get("email"):
            payload = json.dumps({
                "channel":        "email",
                "to_email":       guest["email"],
                "to_name":        guest.get("first_name"),
                "email_subject":  "How was your stay? Leave a review",
                "email_template": "post_stay_review_request.html",
                "template_ctx":   ctx,
            }).encode()
            notify_req = urllib.request.Request(
                f"{NOTIFY_URL}/notify",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(notify_req)

        log.info("Review request sent for booking %s", booking_id)

    except Exception as exc:
        log.error("Review task failed: %s", exc)
        raise self.retry(exc=exc, countdown=300)
