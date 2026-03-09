"""
services/group_service/deposit_scheduler.py
---------------------------------------------
Schedules deposit reminders for group bookings via Celery.
30% deposit at block creation → 70% balance due 30 days before event.
"""

import logging
from datetime import datetime, timezone, timedelta

log = logging.getLogger(__name__)


def schedule_deposits(
    block_id:       str,
    total_amount:   float,
    date_from:      str,
    organiser_email: str,
) -> dict:
    """
    Schedules Celery tasks to send deposit reminder emails.
    Returns the schedule as a dict for logging/testing.
    """
    try:
        from tasks.celery_app import celery_app
    except ImportError:
        log.warning("Celery not available; deposit schedule logged only")
        celery_app = None

    deposit_30 = round(total_amount * 0.30, 2)
    balance_70 = round(total_amount * 0.70, 2)
    event_date = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    balance_due = event_date - timedelta(days=30)
    now         = datetime.now(timezone.utc)

    schedule = {
        "deposit_30_usd":         deposit_30,
        "deposit_30_due":         "immediately",
        "balance_70_usd":         balance_70,
        "balance_70_due":         balance_due.strftime("%Y-%m-%d"),
    }

    if celery_app:
        eta_balance = balance_due if balance_due > now else now + timedelta(minutes=1)
        celery_app.send_task(
            "group_service.send_balance_reminder",
            kwargs={
                "block_id":       block_id,
                "amount":         balance_70,
                "organiser_email": organiser_email,
            },
            eta=eta_balance,
        )

    log.info("Deposit schedule for block %s: %s", block_id, schedule)
    return schedule
