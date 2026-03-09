"""
services/cancellation_service/stripe_refund.py
------------------------------------------------
Issues a refund through the Stripe API.
"""

import os
import logging
from typing import Optional

import stripe

log = logging.getLogger(__name__)
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")


def issue_refund(
    payment_intent_id: str,
    amount_usd:        Optional[float] = None,
    reason:            Optional[str]   = "requested_by_customer",
) -> dict:
    """
    Issues a full or partial Stripe refund.

    Args:
        payment_intent_id: Stripe PI id (pi_xxx)
        amount_usd:        Pass None for full refund; float for partial.
        reason:            Stripe reason code.

    Returns dict with {refund_id, status, amount_usd}
    """
    kwargs: dict = {
        "payment_intent": payment_intent_id,
        "reason":         reason,
    }
    if amount_usd is not None:
        kwargs["amount"] = int(round(amount_usd * 100))

    try:
        refund = stripe.Refund.create(**kwargs)
        return {
            "refund_id": refund.id,
            "status":    refund.status,
            "amount_usd": round((refund.amount or 0) / 100, 2),
        }
    except stripe.error.StripeError as e:
        log.error("Stripe refund error: %s", e)
        raise
