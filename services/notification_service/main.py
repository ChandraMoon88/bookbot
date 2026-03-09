"""
services/notification_service/main.py
---------------------------------------
FastAPI unified notification microservice.
Routes to email / WhatsApp / push depending on the channel field.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

from .email_sender    import send as send_email
from .whatsapp_sender import send_template as send_wa_template, send_text as send_wa_text
from .push_sender     import send as send_push

app = FastAPI(title="Notification Service")


class NotifyRequest(BaseModel):
    channel:       str           # "email" | "whatsapp" | "push" | "all"
    booking_ref:   Optional[str] = None

    # Email
    to_email:      Optional[str] = None
    to_name:       Optional[str] = None
    email_subject: Optional[str] = None
    email_template: Optional[str] = "booking_confirmation.html"
    template_ctx:  Optional[dict] = None

    # WhatsApp
    wa_phone:      Optional[str] = None
    wa_template:   Optional[str] = "booking_confirmation"
    wa_language:   Optional[str] = "en_US"
    wa_components: Optional[list] = None

    # Push
    push_subscription: Optional[dict] = None
    push_title:        Optional[str]  = None
    push_body:         Optional[str]  = None
    push_url:          Optional[str]  = None


@app.post("/notify")
def notify(req: NotifyRequest):
    results = {}

    ctx = req.template_ctx or {}
    if req.booking_ref:
        ctx.setdefault("booking_ref", req.booking_ref)

    # Email
    if req.channel in ("email", "all") and req.to_email:
        results["email"] = send_email(
            to_email=req.to_email,
            subject=req.email_subject or f"Booking Confirmation – {req.booking_ref}",
            template=req.email_template,
            context=ctx,
            to_name=req.to_name,
        )

    # WhatsApp
    if req.channel in ("whatsapp", "all") and req.wa_phone:
        results["whatsapp"] = send_wa_template(
            to=req.wa_phone,
            template_name=req.wa_template,
            language=req.wa_language,
            components=req.wa_components,
        )

    # Push
    if req.channel in ("push", "all") and req.push_subscription:
        results["push"] = send_push(
            subscription_info=req.push_subscription,
            title=req.push_title or "Booking Update",
            body=req.push_body  or ctx.get("message", ""),
            url=req.push_url,
        )

    return {"results": results}


@app.get("/health")
def health():
    return {"status": "ok", "service": "notification_service"}
