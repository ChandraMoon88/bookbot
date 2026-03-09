"""
services/payment_service/stripe_adapter.py
--------------------------------------------
Thin wrapper around the Stripe Python SDK.

Card data NEVER passes through this service.
Stripe.js / Payment Element on the front-end tokenises the card;
we only handle the PaymentIntent lifecycle here.
"""

import os
import hashlib
import hmac
import logging
from typing import Optional

import stripe

log = logging.getLogger(__name__)

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


def create_payment_intent(
    amount_usd:    float,
    booking_id:    str,
    customer_email: str,
    currency:       str = "usd",
    require_3ds:    bool = False,
) -> dict:
    """
    Creates a Stripe PaymentIntent.

    Args:
        amount_usd:      Float in dollars – will be converted to cents.
        booking_id:      Used as idempotency key (booking-<id>).
        customer_email:  For receipt.
        currency:        Default USD.
        require_3ds:     If True forces payment_method_options.card.request_three_d_secure = "any".

    Returns dict with {client_secret, payment_intent_id, status}.
    """
    amount_cents = int(round(amount_usd * 100))

    kwargs: dict = {
        "amount":   amount_cents,
        "currency": currency.lower(),
        "metadata": {"booking_id": booking_id},
        "receipt_email": customer_email,
        "description": f"Hotel booking {booking_id}",
    }

    if require_3ds:
        kwargs["payment_method_options"] = {
            "card": {"request_three_d_secure": "any"}
        }

    try:
        intent = stripe.PaymentIntent.create(
            **kwargs,
            idempotency_key=f"booking-{booking_id}",
        )
        return {
            "client_secret":       intent.client_secret,
            "payment_intent_id":   intent.id,
            "status":              intent.status,
            "amount_usd":          amount_usd,
            "currency":            currency,
        }
    except stripe.error.StripeError as e:
        log.error("Stripe error creating intent: %s", e)
        raise


def capture_payment_intent(payment_intent_id: str) -> dict:
    intent = stripe.PaymentIntent.capture(payment_intent_id)
    return {"payment_intent_id": intent.id, "status": intent.status}


def verify_webhook(payload_body: bytes, sig_header: str) -> Optional[dict]:
    """
    Verifies the Stripe webhook signature (HMAC-SHA256) and returns the event dict.
    Returns None if signature is invalid.
    """
    try:
        event = stripe.Webhook.construct_event(
            payload_body, sig_header, WEBHOOK_SECRET
        )
        return dict(event)
    except (stripe.error.SignatureVerificationError, ValueError) as e:
        log.warning("Invalid Stripe webhook signature: %s", e)
        return None
