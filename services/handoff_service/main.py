"""
services/handoff_service/main.py
----------------------------------
Transfers conversation to a live human agent via Chatwoot.
Creates a Chatwoot conversation and notifies the agent inbox.
"""

import os
import json
import logging
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

log = logging.getLogger(__name__)
app = FastAPI(title="Handoff Service")

CHATWOOT_BASE    = os.environ.get("CHATWOOT_BASE_URL", "https://app.chatwoot.com")
CHATWOOT_TOKEN   = os.environ.get("CHATWOOT_API_TOKEN", "")
CHATWOOT_INBOX   = os.environ.get("CHATWOOT_INBOX_ID", "1")
CHATWOOT_ACCOUNT = os.environ.get("CHATWOOT_ACCOUNT_ID", "1")


def _cw_headers() -> dict:
    return {
        "api_access_token": CHATWOOT_TOKEN,
        "Content-Type":     "application/json",
    }


def _post_cw(path: str, payload: dict) -> dict:
    url  = f"{CHATWOOT_BASE}/api/v1/accounts/{CHATWOOT_ACCOUNT}{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=_cw_headers(), method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


class HandoffRequest(BaseModel):
    sender_id:     str
    sender_name:   Optional[str] = "Guest"
    issue:         Optional[str] = "Agent requested by user"
    language:      Optional[str] = "en"
    booking_ref:   Optional[str] = None
    transcript:    Optional[list] = None


@app.post("/handoff")
def handoff(req: HandoffRequest):
    """
    Creates a Chatwoot contact + conversation and returns the conversation URL.
    """
    # 1. Upsert Chatwoot contact
    try:
        contact_resp = _post_cw(
            "/contacts",
            {
                "name":       req.sender_name,
                "identifier": req.sender_id,
                "additional_attributes": {
                    "messenger_psid": req.sender_id,
                    "booking_ref":    req.booking_ref,
                    "language":       req.language,
                },
            },
        )
        contact_id = contact_resp["id"]
    except Exception as exc:
        log.error("Failed to create Chatwoot contact: %s", exc)
        raise HTTPException(status_code=502, detail="Could not create agent handoff")

    # 2. Create conversation
    try:
        convo_resp = _post_cw(
            "/conversations",
            {
                "contact_id":  contact_id,
                "inbox_id":    int(CHATWOOT_INBOX),
                "additional_attributes": {"issue": req.issue},
            },
        )
        convo_id  = convo_resp["id"]
        convo_url = f"{CHATWOOT_BASE}/app/accounts/{CHATWOOT_ACCOUNT}/conversations/{convo_id}"
    except Exception as exc:
        log.error("Failed to create Chatwoot conversation: %s", exc)
        raise HTTPException(status_code=502, detail="Could not create agent handoff")

    # 3. Post transcript as first message if provided
    if req.transcript:
        text = "\n".join(
            f"[{m.get('sender','?')}] {m.get('text','')}" for m in req.transcript[-20:]
        )
        try:
            _post_cw(
                f"/conversations/{convo_id}/messages",
                {
                    "content": f"**Transcript (last 20 messages)**\n\n{text}",
                    "message_type": "outgoing",
                    "private": True,
                },
            )
        except Exception as exc:
            log.warning("Could not post transcript: %s", exc)

    return {
        "conversation_id":  convo_id,
        "conversation_url": convo_url,
        "contact_id":       contact_id,
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "handoff_service"}
