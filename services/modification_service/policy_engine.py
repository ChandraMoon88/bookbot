"""
services/modification_service/policy_engine.py
------------------------------------------------
Determines whether a modification is allowed and what fee applies.

Rules (from SKILL.md):
  - Free change if >72 h before check-in
  - 1-night fee   if 24-72 h before check-in
  - 2-night fee   if <24 h before check-in
  - No change     if check-in has already passed
"""

from datetime import datetime, timezone
from typing import Optional


def evaluate(
    check_in_str:  str,
    amount_per_night: float,
    reason:        Optional[str] = None,
) -> dict:
    """
    Args:
        check_in_str:       ISO date string  'YYYY-MM-DD'
        amount_per_night:   One-night room rate in USD
        reason:             Optional reason (force_majeure → fee_waived)

    Returns:
        {allowed, fee_usd, fee_nights, policy_label, hours_until_checkin}
    """
    now       = datetime.now(timezone.utc)
    check_in  = datetime.strptime(check_in_str, "%Y-%m-%d").replace(
                    hour=15, minute=0, tzinfo=timezone.utc
                )
    hours_left = (check_in - now).total_seconds() / 3600

    # Force majeure waiver
    if reason and "force_majeure" in reason.lower():
        return {
            "allowed":             True,
            "fee_usd":             0.0,
            "fee_nights":          0,
            "policy_label":        "Force Majeure – Waived",
            "hours_until_checkin": round(hours_left, 1),
        }

    if hours_left <= 0:
        return {
            "allowed":             False,
            "fee_usd":             0.0,
            "fee_nights":          0,
            "policy_label":        "Check-in already passed",
            "hours_until_checkin": round(hours_left, 1),
        }
    elif hours_left < 24:
        fee_nights = 2
        label      = "Late change fee (< 24 h)"
    elif hours_left < 72:
        fee_nights = 1
        label      = "Short-notice change fee (24-72 h)"
    else:
        fee_nights = 0
        label      = "Free modification (> 72 h)"

    return {
        "allowed":             True,
        "fee_usd":             round(fee_nights * amount_per_night, 2),
        "fee_nights":          fee_nights,
        "policy_label":        label,
        "hours_until_checkin": round(hours_left, 1),
    }
