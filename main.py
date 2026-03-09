"""
main.py — Render webhook proxy for BookHotel Bot
-------------------------------------------------
KEY FIX: Facebook requires a 200 OK response within 20 seconds.
If Render takes longer (waiting for HF Spaces), Facebook retries
the webhook — causing duplicate "Sorry" replies.

Solution: return 200 to Facebook IMMEDIATELY, then process in
a background task. Facebook never retries, no duplicate replies.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse
import httpx
import os
import json
import base64
import time
from dotenv import load_dotenv
load_dotenv()


async def _setup_messenger_profile():
    """
    Register the Get Started button and greeting text with Facebook.
    Called once at Render startup.
    """
    if not PAGE_ACCESS_TOKEN:
        print("WARNING: PAGE_ACCESS_TOKEN missing - skipping Messenger profile setup.", flush=True)
        return
    url     = "https://graph.facebook.com/v17.0/me/messenger_profile"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"}
    payload = {
        "get_started": {"payload": "GET_STARTED"},
        "greeting": [
            {
                "locale":  "default",
                "text":    "Hi {{user_first_name}}! Welcome to BookBot."
                           " Tap Get Started to choose your language and begin.",
            }
        ],
        "persistent_menu": [
            {
                "locale": "default",
                "composer_input_disabled": False,
                "call_to_actions": [
                    {"type": "postback", "title": "Book a Hotel",    "payload": "ACTION_BOOK"},
                    {"type": "postback", "title": "Help",            "payload": "ACTION_HELP"},
                    {"type": "postback", "title": "Change Language", "payload": "ACTION_CHANGE_LANG"},
                    {"type": "postback", "title": "Start Over",      "payload": "RESTART"},
                ],
            }
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code == 200:
            print("Messenger profile registered (Get Started + persistent menu).", flush=True)
        else:
            print(f"Messenger profile setup failed: {resp.text}", flush=True)
    except Exception as e:
        print(f"Messenger profile setup error: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _setup_messenger_profile()
    yield


app = FastAPI(lifespan=lifespan)

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

                # -- Get started / restart postbacks ---------------------------
                if "postback" in event:
                    pb = event["postback"]
                    pb_mid = pb.get("mid", "")
                    if pb_mid and _is_duplicate(pb_mid):
                        print(f"Duplicate postback mid={pb_mid}, skipping.", flush=True)
                        continue
                    payload_val = pb.get("payload", "")
                    if payload_val == "GET_STARTED":
                        background_tasks.add_task(
                            call_processor_and_reply, sender_id, "GET_STARTED", "text"
                        )
                    elif payload_val == "RESTART":
                        background_tasks.add_task(
                            call_processor_and_reply, sender_id, "RESTART", "text"
                        )
                    elif payload_val in ("ACTION_BOOK", "ACTION_HELP", "ACTION_CHANGE_LANG"):
                        background_tasks.add_task(
                            call_processor_and_reply, sender_id, payload_val, "text"
                        )
                    elif payload_val.startswith("LANG_"):
                        # Language selection button tapped
                        background_tasks.add_task(
                            call_processor_and_reply, sender_id, payload_val[5:], "text"
                        )

                # -- Text or voice message, including quick reply taps ----------
                elif "message" in event:
                    msg = event["message"]
                    mid = msg.get("mid", "")
                    if mid and _is_duplicate(mid):
                        print(f"Duplicate mid={mid}, skipping.", flush=True)
                        continue

                    # Quick reply buttons send message.quick_reply.payload
                    qr_payload = (msg.get("quick_reply") or {}).get("payload", "")

                    attachments = msg.get("attachments", [])
                    audio_att   = next(
                        (a for a in attachments if a.get("type") == "audio"), None
                    )

                    if audio_att:
                        audio_url = audio_att["payload"]["url"]
                        background_tasks.add_task(handle_voice, sender_id, audio_url)
                    elif qr_payload:
                        # Route quick-reply button taps by their payload
                        print(f"Quick reply from {sender_id}: payload={qr_payload}", flush=True)
                        if qr_payload in ("RESTART", "ACTION_BOOK", "ACTION_HELP", "ACTION_CHANGE_LANG"):
                            background_tasks.add_task(
                                call_processor_and_reply, sender_id, qr_payload, "text"
                            )
                        elif qr_payload.startswith("LANG_"):
                            # Language button — strip prefix, send the code as text
                            background_tasks.add_task(
                                call_processor_and_reply, sender_id, qr_payload[5:], "text"
                            )
                        else:
                            # Unknown payload — fall back to the button's display title
                            user_message = msg.get("text", qr_payload)
                            background_tasks.add_task(
                                call_processor_and_reply, sender_id, user_message, "text"
                            )
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
        await send_text(sender_id, "Sorry, I could not process your voice message.")


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
                "I am waking up. Please send your message again in 1-2 minutes.",
            )
            return

        if not health.get("models_ready", False):
            print("HF Spaces still warming up.", flush=True)
            await send_text(
                sender_id,
                "I am warming up. Please send your message again in 1-2 minutes.",
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

        text    = result.get("text", "Sorry, something went wrong.")
        buttons = result.get("buttons", [])

        # -- Send audio FIRST (if any) -----------------------------------------
        # Quick replies are only shown on the LAST message in Messenger.
        # Sending audio after the buttons message would hide the buttons.
        # By sending audio first, the text+buttons message is always last.
        if result.get("audio_b64"):
            await send_audio(sender_id, base64.b64decode(result["audio_b64"]))

        # -- Send text + quick-reply buttons in ONE message (always last) ------
        if buttons:
            await send_quick_replies(sender_id, text, buttons)
        else:
            await send_text(sender_id, text)

    except Exception as e:
        print(f"call_processor_and_reply error: {e}", flush=True)
        await send_text(sender_id, "Sorry, I am having trouble right now. Please try again.")


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


async def send_quick_replies(recipient_id: str, prompt_text: str, buttons: list):
    """Send a message with Messenger quick-reply buttons (max 13)."""
    url     = "https://graph.facebook.com/v17.0/me/messages"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {PAGE_ACCESS_TOKEN}",
    }
    payload = {
        "recipient": {"id": recipient_id},
        "message": {
            "text":          prompt_text,
            "quick_replies": buttons[:13],
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            print(f"send_quick_replies ERROR: {resp.text}", flush=True)
    except Exception as e:
        print(f"send_quick_replies error: {e}", flush=True)


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