"""
tasks/celery_app.py
---------------------
Celery application factory.
Broker  : Redis (REDIS_URL env var)
Backend : Redis (for result storage)
"""

import os
from celery import Celery
from celery.schedules import crontab

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "bookhotel",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "tasks.scheduled_tasks",
        "services.review_service.celery_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    result_expires=3600,
    # Beat schedule
    beat_schedule={
        "send-review-requests-daily": {
            "task":     "tasks.scheduled_tasks.trigger_review_requests",
            "schedule": crontab(hour=10, minute=0),   # 10:00 UTC daily
        },
        "expire-loyalty-points-monthly": {
            "task":     "tasks.scheduled_tasks.expire_loyalty_points",
            "schedule": crontab(day_of_month=1, hour=0, minute=0),
        },
        "release-stale-soft-locks": {
            "task":     "tasks.scheduled_tasks.release_stale_soft_locks",
            "schedule": crontab(minute="*/15"),   # every 15 min
        },
        "send-checkin-reminders-daily": {
            "task":     "tasks.scheduled_tasks.send_checkin_reminders",
            "schedule": crontab(hour=9, minute=0),    # 09:00 UTC daily (K5)
        },
        "recover-abandoned-bookings": {
            "task":     "tasks.scheduled_tasks.recover_abandoned_bookings",
            "schedule": crontab(minute="*/30"),        # every 30 min (K4)
        },
    },
)
