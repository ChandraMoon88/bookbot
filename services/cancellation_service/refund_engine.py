"""
services/cancellation_service/refund_engine.py
------------------------------------------------
Calculates refund amount based on cancellation policy.

Policy (from SKILL.md):
  >72 h before check-in  → Full refund
  24–72 h                → 80% refund
  <24 h                  → No refund
  Already checked in     → No refund
  Force majeure          → Full refund
"""

from datetime import datetime, timezone
from typing import Optional


def calculate(
    check_in_str:  str,
    amount_paid:   float,
    reason:        Optional[str] = None,
) -> dict:
    """
    Returns:
        refund_usd       float
        refund_pct       int  (0 | 80 | 100)
        policy_label     str
        hours_until_checkin float
    """
    now      = datetime.now(timezone.utc)
    check_in = datetime.strptime(check_in_str, "%Y-%m-%d").replace(
                   hour=15, minute=0, tzinfo=timezone.utc
               )
    hours_left = (check_in - now).total_seconds() / 3600

    # Force majeure → always full refund
    if reason and "force_majeure" in reason.lower():
        pct   = 100
        label = "Force Majeure – Full Refund"
    elif hours_left <= 0:
        pct   = 0
        label = "No refund – already checked in or past check-in"
    elif hours_left < 24:
        pct   = 0
        label = "No refund – < 24 h before check-in"
    elif hours_left < 72:
        pct   = 80
        label = "Partial refund (80%) – 24-72 h before check-in"
    else:
        pct   = 100
        label = "Full refund – > 72 h before check-in"

    refund_usd = round(amount_paid * pct / 100, 2)

    return {
        "refund_usd":          refund_usd,
        "refund_pct":          pct,
        "policy_label":        label,
        "hours_until_checkin": round(hours_left, 1),
    }
