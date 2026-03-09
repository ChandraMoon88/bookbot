"""
main.py — Render webhook proxy for BookHotel Bot
-------------------------------------------------
KEY FIX: Facebook requires a 200 OK response within 20 seconds.
If Render takes longer (waiting for HF Spaces), Facebook retries
the webhook — causing duplicate "Sorry" replies.

Solution: return 200 to Facebook IMMEDIATELY, then process in
a background task. Facebook never retries, no duplicate replies.
"""

from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
import httpx
import os
import json
import base64
import time
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

VERIFY_TOKEN      = os.getenv("VERIFY_TOKEN", "mybot123")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
HF_PROCESSOR_URL  = os.getenv("HF_PROCESSOR_URL", "").rstrip("/")

print(f"TOKEN LOADED: {PAGE_ACCESS_TOKEN[:20] if PAGE_ACCESS_TOKEN else 'NOT FOUND!'}", flush=True)
print(f"HF URL: {HF_PROCESSOR_URL}", flush=True)

# ── Message deduplication (prevents duplicate replies from Facebook retries) ──
# Facebook retries a webhook up to 5–7 times if it doesn't get 200 within 20 s.
# When Render has a cold-start the 200 arrives late, so Facebook retries.
# We track seen message IDs for 5 minutes to silently drop duplicates.
_processed_mids: dict[str, float] = {}
_MID_TTL = 300  # seconds


def _is_duplicate(mid: str) -> bool:
    """Return True if this mid was already processed recently."""
    now = time.time()
    # Prune expired entries
    for k in [k for k, t in list(_processed_mids.items()) if now - t > _MID_TTL]:
        _processed_mids.pop(k, None)
    if mid in _processed_mids:
        return True
    _processed_mids[mid] = now
    return False

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "bot": "bookhotel-webhook"}


# ─────────────────────────────────────────────────────────────────────────────
# WEBHOOK VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/webhook")
async def verify_webhook(
    hub_mode:         str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge:    str = Query(None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("Webhook verified ✅", flush=True)
        return PlainTextResponse(content=hub_challenge, status_code=200)
    print(f"Webhook verification failed ❌ token={hub_verify_token}", flush=True)
    return PlainTextResponse(content="Forbidden", status_code=403)


# ─────────────────────────────────────────────────────────────────────────────
# WEBHOOK RECEIVER
# Returns 200 to Facebook INSTANTLY — processes message in background.
# This prevents Facebook from retrying (which caused duplicate replies).
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/webhook")
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    print(f"Webhook received: {json.dumps(data)[:200]}", flush=True)

    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for event in entry.get("messaging", []):
                sender_id = event["sender"]["id"]

                # ── Get started button ────────────────────────────────────────
                if "postback" in event:
                    pb_mid = event["postback"].get("mid", "")
                    if pb_mid and _is_duplicate(pb_mid):
                        print(f"Duplicate postback mid={pb_mid}, skipping.", flush=True)
                        continue
                    if event["postback"].get("payload") == "GET_STARTED":
                        background_tasks.add_task(
                            call_processor_and_reply, sender_id, "hello", "text"
                        )

                # ── Text or voice message ─────────────────────────────────────
                elif "message" in event:
                    msg = event["message"]
                    mid = msg.get("mid", "")
                    if mid and _is_duplicate(mid):
                        print(f"Duplicate mid={mid}, skipping.", flush=True)
                        continue

                    attachments = msg.get("attachments", [])
                    audio_att   = next(
                        (a for a in attachments if a.get("type") == "audio"), None
                    )

                    if audio_att:
                        audio_url = audio_att["payload"]["url"]
                        background_tasks.add_task(handle_voice, sender_id, audio_url)
                    else:
                        user_message = msg.get("text", "")
                        if user_message:
                            print(f"Text from {sender_id}: {user_message}", flush=True)
                            background_tasks.add_task(
                                call_processor_and_reply, sender_id, user_message, "text"
                            )

    # ── Return 200 immediately BEFORE any processing ─────────────────────────
    # Facebook marks the webhook as failed and retries if this takes >20s.
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND TASKS
# ─────────────────────────────────────────────────────────────────────────────

async def handle_voice(sender_id: str, audio_url: str):
    """Download voice audio then forward to HF Spaces."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                audio_url,
                headers={"Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"},
                follow_redirects=True,
            )
        audio_b64 = base64.b64encode(resp.content).decode()
        await call_processor_and_reply(sender_id, None, "voice", audio_b64)
    except Exception as e:
        print(f"handle_voice error: {e}", flush=True)
        await send_text(sender_id, "Sorry, I couldn't process your voice message.")


async def call_processor_and_reply(
    sender_id: str,
    message:   str,
    msg_type:  str,
    audio_b64: str = None,
):
    """Check HF Spaces is ready, call /process, send reply to Messenger."""
    try:
        # ── Check if HF Spaces models are ready ───────────────────────────────
        # Use a 30-second timeout — HF Spaces free tier can be slow to respond.
        # Wrap in its own try/except so a cold-starting / sleeping Space sends a
        # friendly "waking up" message instead of the generic "Sorry" error.
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                h = await client.get(f"{HF_PROCESSOR_URL}/health")
            if h.status_code != 200:
                raise RuntimeError(f"Health check HTTP {h.status_code}")
            try:
                health = h.json()
            except Exception:
                # HF Spaces returns an HTML loading page during cold-start
                health = {}
        except Exception as hc_err:
            print(f"HF Spaces unreachable (cold-start?): {hc_err}", flush=True)
            await send_text(
                sender_id,
                "I'm waking up ⏳ — please send your message again in 1–2 minutes!",
            )
            return

        if not health.get("models_ready", False):
            print("HF Spaces still warming up.", flush=True)
            await send_text(
                sender_id,
                "I'm warming up ⏳ — please send your message again in 1–2 minutes!",
            )
            return

        # ── Call /process ─────────────────────────────────────────────────────
        payload = {
            "sender_id": sender_id,
            "type":      msg_type,
            "message":   message,
            "audio_b64": audio_b64,
        }
        print(f"Calling HF Spaces /process for {sender_id}…", flush=True)
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp   = await client.post(f"{HF_PROCESSOR_URL}/process", json=payload)
            result = resp.json()

        print(f"HF response: {str(result)[:150]}", flush=True)

        # ── Send text reply ───────────────────────────────────────────────────
        text = result.get("text", "Sorry, something went wrong.")
        await send_text(sender_id, text)

        # ── Send audio reply if available ─────────────────────────────────────
        if result.get("audio_b64"):
            await send_audio(sender_id, base64.b64decode(result["audio_b64"]))

    except Exception as e:
        print(f"call_processor_and_reply error: {e}", flush=True)
        await send_text(sender_id, "Sorry, I'm having trouble right now.")


# ─────────────────────────────────────────────────────────────────────────────
# FACEBOOK SEND HELPERS
# ─────────────────────────────────────────────────────────────────────────────

async def send_text(recipient_id: str, message_text: str):
    url     = "https://graph.facebook.com/v17.0/me/messages"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {PAGE_ACCESS_TOKEN}",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url, headers=headers,
                json={"recipient": {"id": recipient_id}, "message": {"text": message_text}},
            )
        if resp.status_code == 200:
            print(f"✅ Text sent to {recipient_id}", flush=True)
        else:
            print(f"send_text ERROR: {resp.text}", flush=True)
    except Exception as e:
        print(f"send_text error: {e}", flush=True)


async def send_audio(recipient_id: str, audio_bytes: bytes):
    url     = "https://graph.facebook.com/v17.0/me/messages"
    headers = {"Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"}
    data    = {
        "recipient": json.dumps({"id": recipient_id}),
        "message":   json.dumps({
            "attachment": {"type": "audio", "payload": {"is_reusable": True}}
        }),
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url, headers=headers,
                data=data, files={"filedata": ("voice.mp3", audio_bytes, "audio/mpeg")},
            )
        if resp.status_code == 200:
            print(f"✅ Audio sent to {recipient_id}", flush=True)
        else:
            print(f"send_audio ERROR: {resp.text}", flush=True)
    except Exception as e:
        print(f"send_audio error: {e}", flush=True)