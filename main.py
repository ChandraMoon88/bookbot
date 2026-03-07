from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
import httpx
import os
import json
import base64
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "mybot123")
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN")
HF_PROCESSOR_URL = os.getenv("HF_PROCESSOR_URL")

print(f"TOKEN LOADED: {PAGE_ACCESS_TOKEN[:20] if PAGE_ACCESS_TOKEN else 'NOT FOUND!'}")
print(f"HF URL: {HF_PROCESSOR_URL}")

@app.get("/health")
async def health():
    return {"status": "ok", "bot": "bookhotel-webhook"}

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(content=hub_challenge, status_code=200)
    return PlainTextResponse(content="Forbidden", status_code=403)

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()

    if data.get("object") == "page":
        for entry in data["entry"]:
            for event in entry.get("messaging", []):
                sender_id = event["sender"]["id"]

                if "postback" in event:
                    payload = event["postback"].get("payload", "")
                    if payload == "GET_STARTED":
                        await call_processor_and_reply(
                            sender_id, "hello", "text"
                        )

                elif "message" in event:
                    msg = event["message"]
                    attachments = msg.get("attachments", [])
                    audio_attachment = next(
                        (a for a in attachments if a.get("type") == "audio"), None
                    )

                    if audio_attachment:
                        audio_url = audio_attachment["payload"]["url"]
                        # Download audio
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            response = await client.get(
                                audio_url,
                                headers={"Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"},
                                follow_redirects=True
                            )
                            audio_bytes = response.content
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        await call_processor_and_reply(
                            sender_id, None, "voice", audio_b64
                        )
                    else:
                        user_message = msg.get("text", "")
                        if user_message:
                            await call_processor_and_reply(
                                sender_id, user_message, "text"
                            )

    return {"status": "ok"}


async def call_processor_and_reply(
    sender_id: str,
    message: str,
    msg_type: str,
    audio_b64: str = None
):
    try:
        # Call HF Space for AI processing
        payload = {
            "sender_id": sender_id,
            "type": msg_type,
            "message": message,
            "audio_b64": audio_b64
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{HF_PROCESSOR_URL}/process",
                json=payload
            )
            result = response.json()

        # Send text reply to Facebook
        text = result.get("text", "Sorry, something went wrong.")
        await send_text(sender_id, text)

        # Send audio reply if available
        audio_data = result.get("audio_b64")
        if audio_data:
            audio_bytes = base64.b64decode(audio_data)
            await send_audio(sender_id, audio_bytes)

    except Exception as e:
        print(f"Error calling processor: {e}")
        await send_text(sender_id, "Sorry, I'm having trouble right now.")


async def send_text(recipient_id: str, message_text: str):
    url = "https://graph.facebook.com/v17.0/me/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"
    }
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                print("Text sent successfully ✅")
            else:
                print(f"ERROR: {response.json()}")
    except Exception as e:
        print(f"send_text error: {e}")


async def send_audio(recipient_id: str, audio_bytes: bytes):
    url = "https://graph.facebook.com/v17.0/me/messages"
    headers = {"Authorization": f"Bearer {PAGE_ACCESS_TOKEN}"}
    data = {
        "recipient": json.dumps({"id": recipient_id}),
        "message": json.dumps({
            "attachment": {
                "type": "audio",
                "payload": {"is_reusable": True}
            }
        })
    }
    files = {"filedata": ("voice.mp3", audio_bytes, "audio/mpeg")}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url, headers=headers, data=data, files=files
            )
            if response.status_code == 200:
                print("Audio sent successfully ✅")
            else:
                print(f"Audio ERROR: {response.json()}")
    except Exception as e:
        print(f"send_audio error: {e}")