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

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
NOTIFY_URL   = os.environ.get("NOTIFICATION_SERVICE_URL", "http://localhost:8007")


def _headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


@shared_task(name="review_service.send_review_request", bind=True, max_retries=3)
def send_review_request(self, booking_id: str):
    """
    Fetches booking info and sends review request email + Messenger message.
    """
    try:
        url  = f"{SUPABASE_URL}/rest/v1/bookings?id=eq.{booking_id}&select=*,guests(*)"
        req  = urllib.request.Request(url, headers=_headers())
        with urllib.request.urlopen(req) as resp:
            rows = json.loads(resp.read())

        if not rows:
            log.warning("Review task: booking %s not found", booking_id)
            return

        booking = rows[0]
        guest   = (booking.get("guests") or [{}])[0]
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
