"""
services/payment_service/fraud_check.py
-----------------------------------------
Lightweight fraud-risk scoring before charging.

Risk bands
----------
0-30  → ALLOW
31-70 → REQUIRE_3DS
71+   → BLOCK
"""

import os
from datetime import datetime, timezone
from typing import Optional

import redis

_r = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), decode_responses=True)


def _velocity_count(user_id: str) -> int:
    """How many payment attempts this user made in the last hour."""
    key = f"pay_attempts:{user_id}"
    pipe = _r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 3600)
    count, _ = pipe.execute()
    return count


def score(
    user_id:        str,
    amount_usd:     float,
    card_country:   Optional[str] = None,
    user_country:   Optional[str] = None,
    account_age_days: int = 365,
) -> dict:
    """
    Returns:
        risk_score    int  0-100
        decision      str  "ALLOW" | "REQUIRE_3DS" | "BLOCK"
        reasons       list[str]
    """
    risk = 0
    reasons = []

    # Rule 1: high-value transaction
    if amount_usd > 2000:
        risk += 20
        reasons.append("high_value_transaction")
    elif amount_usd > 1000:
        risk += 10

    # Rule 2: new account (<7 days) + any purchase
    if account_age_days < 7:
        risk += 25
        reasons.append("new_account")
    elif account_age_days < 30:
        risk += 10

    # Rule 3: card country mismatch
    if card_country and user_country and card_country.upper() != user_country.upper():
        risk += 20
        reasons.append("card_country_mismatch")

    # Rule 4: velocity – more than 5 attempts in past hour
    attempts = _velocity_count(user_id)
    if attempts > 5:
        risk += 30
        reasons.append("high_velocity")
    elif attempts > 3:
        risk += 10

    risk = min(risk, 100)

    if risk <= 30:
        decision = "ALLOW"
    elif risk <= 70:
        decision = "REQUIRE_3DS"
    else:
        decision = "BLOCK"

    return {"risk_score": risk, "decision": decision, "reasons": reasons}
