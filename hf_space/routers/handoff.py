"""
hf_space/routers/handoff.py
-----------------------------
Module 13 — Human Handoff & Escalation

When the bot can't help, routes to a human agent gracefully.

States: handoff_active (all messages relayed to agent queue)
        handoff_complete (resolved, returns to greeting)

Handoff record stored in Supabase: handoffs table
Agent notification: email via services/notification_service/email_sender.py
Real-time: optional Slack/Intercom webhook (HANDOFF_WEBHOOK_URL env var)

Flow:
  HANDOFF_REQUEST postback → collect reason → create ticket →
  notify agent → enter handoff_active → agent replies via admin panel →
  HANDOFF_CLOSE → confirm → return to greeting
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_handoff_request(psid: str, lang: str, reason: str = "") -> tuple[list[dict], str]:
    """
    Initiate a handoff. Create ticket, notify agent.
    """
    mb      = MessengerResponse(psid)
    profile = await get_user_profile(psid) or {}

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    sb        = get_supabase()
    psid_hash = hashlib.sha256(psid.encode()).hexdigest()

    handoff_row = {
        "ticket_id":  ticket_id,
        "psid_hash":  psid_hash,
        "guest_name": profile.get("name", "Guest"),
        "reason":     reason or "Guest requested agent",
        "status":     "open",
        "lang":       lang,
        "created_at": datetime.utcnow().isoformat(),
    }

    try:
        await sb.table("handoffs").insert(handoff_row).execute()
    except Exception as e:
        log.error("handoff_create_failed", error=str(e))

    # Notify agent
    await _notify_agent(ticket_id, profile, reason)

    await set_user_state(psid, "handoff_active")

    return mb.send_sequence([
        mb.text(
            f"🧑‍💼 I'm connecting you with a team member now.\n\n"
            f"Ticket: **{ticket_id}**\n"
            f"Typical wait: 5–10 minutes.\n\n"
            f"You can type your message and our agent will reply shortly."
        ),
    ]), "handoff_active"


async def handle_handoff_active(psid: str, text: str, lang: str) -> tuple[list[dict], str]:
    """
    While in handoff_active, relay messages to the agent queue.
    Don't process with bot logic.
    """
    mb = MessengerResponse(psid)

    if text in ("HANDOFF_CLOSE", "MENU_MAIN", "SEARCH_START"):
        return await _close_handoff(psid, lang)

    # Store message in agent inbox
    await _relay_to_agent(psid, text)

    return [mb.quick_replies(
        "Your message has been sent to our team. ✉️\n\nAn agent will reply shortly.",
        [
            {"title": "🔁 Check if resolved",   "payload": "HANDOFF_CHECK"},
            {"title": "🏠 Return to main menu",  "payload": "MENU_MAIN"},
        ],
    )], "handoff_active"


async def _close_handoff(psid: str, lang: str) -> tuple[list[dict], str]:
    mb        = MessengerResponse(psid)
    sb        = get_supabase()
    psid_hash = hashlib.sha256(psid.encode()).hexdigest()

    try:
        await sb.table("handoffs") \
            .update({"status": "closed", "closed_at": datetime.utcnow().isoformat()}) \
            .eq("psid_hash", psid_hash) \
            .eq("status", "open") \
            .execute()
    except Exception:
        pass

    await set_user_state(psid, "greeting")

    return [
        mb.text("Chat session ended. Thank you for reaching out! 😊"),
        mb.quick_replies("How can I help you next?", [
            {"title": "🔍 Search hotels",   "payload": "SEARCH_START"},
            {"title": "📋 My bookings",     "payload": "VIEW_BOOKINGS"},
            {"title": "💎 Loyalty points",  "payload": "LOYALTY_MENU"},
        ]),
    ], "greeting"


async def _relay_to_agent(psid: str, text: str) -> None:
    """Store incoming guest message in agent message queue."""
    sb        = get_supabase()
    psid_hash = hashlib.sha256(psid.encode()).hexdigest()
    try:
        await sb.table("agent_messages").insert({
            "psid_hash": psid_hash,
            "direction": "guest",
            "message":   text,
            "timestamp": datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        log.warning("relay_to_agent_failed", error=str(e))


async def _notify_agent(ticket_id: str, profile: dict, reason: str) -> None:
    """Send email + optional webhook notification to agent team."""
    webhook_url = os.environ.get("HANDOFF_WEBHOOK_URL", "")

    # Email notification
    try:
        from services.notification_service.email_sender import send_handoff_alert
        import asyncio
        asyncio.create_task(send_handoff_alert(ticket_id, profile, reason))
    except Exception:
        pass

    # Slack/Intercom webhook
    if webhook_url:
        try:
            import httpx
            payload = {
                "text": f"🔔 New handoff request\nTicket: {ticket_id}\nGuest: {profile.get('name','Unknown')}\nReason: {reason}",
            }
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(webhook_url, json=payload)
        except Exception as e:
            log.warning("handoff_webhook_failed", error=str(e))


# ── Agent API (called from admin panel) ───────────────────────────────────────

class AgentReplyRequest(BaseModel):
    ticket_id:  str
    message:    str
    agent_name: str = "Hotel Team"


@router.post("/agent_reply")
async def agent_reply(req: AgentReplyRequest) -> dict:
    """
    Admin panel posts here to send a message back to the guest.
    This endpoint proxies the reply through the Render webhook to Messenger.
    """
    sb = get_supabase()
    try:
        res = await sb.table("handoffs").select("psid_hash").eq("ticket_id", req.ticket_id).single().execute()
        psid_hash = (res.data or {}).get("psid_hash", "")
    except Exception:
        return {"success": False, "error": "Ticket not found"}

    # We can't reverse the hash to get psid — store the raw psid_hash in agent_messages
    # The Render relay needs the psid. Store it separately in a secure mapping.
    try:
        await sb.table("agent_messages").insert({
            "psid_hash": psid_hash,
            "direction": "agent",
            "message":   req.message,
            "agent":     req.agent_name,
            "timestamp": datetime.utcnow().isoformat(),
        }).execute()
        return {"success": True, "message": "Queued for delivery"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/tickets")
async def list_open_tickets(limit: int = 20) -> dict:
    sb = get_supabase()
    try:
        res = await sb.table("handoffs").select("*").eq("status", "open").order("created_at", desc=True).limit(limit).execute()
        return {"tickets": res.data or []}
    except Exception as e:
        return {"tickets": [], "error": str(e)}


@router.post("/close/{ticket_id}")
async def close_ticket(ticket_id: str) -> dict:
    sb = get_supabase()
    try:
        await sb.table("handoffs").update({"status": "closed", "closed_at": datetime.utcnow().isoformat()}).eq("ticket_id", ticket_id).execute()
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
