"""
services/payment_service/main.py
----------------------------------
FastAPI payment microservice.
"""

import os
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

from .fraud_check import score as fraud_score
from .stripe_adapter import create_payment_intent, capture_payment_intent, verify_webhook

log = logging.getLogger(__name__)
app = FastAPI(title="Payment Service")


class PayRequest(BaseModel):
    booking_id:       str
    user_id:          str
    amount_usd:       float
    customer_email:   str
    currency:         Optional[str] = "usd"
    card_country:     Optional[str] = None
    user_country:     Optional[str] = None
    account_age_days: Optional[int] = 365


@app.post("/pay")
def pay(req: PayRequest):
    # 1. Fraud check
    fraud = fraud_score(
        user_id=req.user_id,
        amount_usd=req.amount_usd,
        card_country=req.card_country,
        user_country=req.user_country,
        account_age_days=req.account_age_days,
    )
    if fraud["decision"] == "BLOCK":
        raise HTTPException(status_code=402, detail={"error": "payment_blocked", "fraud": fraud})

    require_3ds = fraud["decision"] == "REQUIRE_3DS"

    # 2. Create Stripe PaymentIntent
    intent = create_payment_intent(
        amount_usd=req.amount_usd,
        booking_id=req.booking_id,
        customer_email=req.customer_email,
        currency=req.currency,
        require_3ds=require_3ds,
    )

    return {
        "client_secret":     intent["client_secret"],
        "payment_intent_id": intent["payment_intent_id"],
        "status":            intent["status"],
        "fraud":             fraud,
    }


@app.post("/capture/{payment_intent_id}")
def capture(payment_intent_id: str):
    result = capture_payment_intent(payment_intent_id)
    return result


@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    event = verify_webhook(payload, sig_header)
    if event is None:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature")

    etype = event.get("type", "")
    log.info("Stripe webhook: %s", etype)

    if etype == "payment_intent.succeeded":
        data = event["data"]["object"]
        booking_id = data.get("metadata", {}).get("booking_id")
        log.info("Payment succeeded for booking %s", booking_id)
        # TODO: emit Kafka booking.payment.succeeded event

    return {"received": True}


@app.get("/health")
def health():
    return {"status": "ok", "service": "payment_service"}
