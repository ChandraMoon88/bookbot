from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
import httpx
import os
import json
from autotranslator import (
    detect_language, get_user_language, set_user_language,
    translate_to, translate_to_english, text_to_speech_bytes,
    speech_to_text, download_audio
)

app = FastAPI()

VERIFY_TOKEN = "mybot123"
PAGE_ACCESS_TOKEN = "EAAWVRMLpTZAUBQyFZAULnQT6IYHPuAEZBLKTYTrNwfUrPLSTavpnAHVhDlP6weUQpuwi6jXUwZAVCI9O1ZBF8axTI2wIKcHwyaiThXl74qyZCovEGOKuyiCcD6BnSe6ZAstYlkZAgXzc20B0kmO4D9fKY7FQEfuGxWS9HgNQ7oasz1QZB7C9ibtWzkT2THNpcS4iR728Ih0oTlQZDZD"

WELCOME_MESSAGE = (
    "Welcome to BookBot!\n\n"
    "I'm your hotel booking assistant. I can help you with:\n"
    "- Search & book hotel rooms\n"
    "- Check availability\n"
    "- Manage your reservations\n\n"
    "Type 'book' to start a new booking or 'help' to see all options."
)

GREETING_KEYWORDS = {
    "hi", "hello", "hey", "hlo", "hii", "howdy", "sup", "yo",
    "greetings", "morning", "afternoon", "evening", "night",
    "good morning", "good afternoon", "good evening", "good night",
    "good day", "how are you", "what's up", "whats up",
    "welcome", "start", "begin", "help"
}


# Webhook Verification
@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        print("Webhook Verified!")
        return PlainTextResponse(content=hub_challenge, status_code=200)
    return PlainTextResponse(content="Forbidden", status_code=403)


# Receive Messages
@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()

    if data.get("object") == "page":
        for entry in data["entry"]:
            for event in entry.get("messaging", []):
                sender_id = event["sender"]["id"]

                # Handle "Get Started" button postback
                if "postback" in event:
                    payload = event["postback"].get("payload", "")
                    print(f"Postback from {sender_id}: {payload}")
                    if payload == "GET_STARTED":
                        await reply_with_translation(sender_id, WELCOME_MESSAGE)

                elif "message" in event:
                    msg = event["message"]
                    attachments = msg.get("attachments", [])

                    # Voice input
                    audio_attachment = next(
                        (a for a in attachments if a.get("type") == "audio"), None
                    )

                    if audio_attachment:
                        audio_url = audio_attachment["payload"]["url"]
                        print(f"Voice message from {sender_id}: {audio_url}")

                        lang = get_user_language(sender_id) or "en"
                        audio_bytes = await download_audio(audio_url, PAGE_ACCESS_TOKEN)
                        user_message = speech_to_text(audio_bytes, lang)

                        if not user_message:
                            await send_text(
                                sender_id,
                                "Sorry, I couldn't understand that voice message. Please try typing."
                            )
                            continue

                        print(f"Transcribed voice [{sender_id}]: {user_message}")

                        # Always re-detect language from transcribed text
                        detected = detect_language(user_message)
                        set_user_language(sender_id, detected)
                        print(f"Language detected for {sender_id}: {detected}")

                    else:
                        # Text input
                        user_message = msg.get("text", "")
                        if not user_message:
                            continue

                        print(f"Message from {sender_id}: {user_message}")

                        # Always re-detect language so users can switch languages freely
                        detected = detect_language(user_message)
                        set_user_language(sender_id, detected)
                        print(f"Language detected for {sender_id}: {detected}")

                    # Translate user message to English for intent detection
                    english_message = translate_to_english(user_message)
                    print(f"English intent [{sender_id}]: {english_message}")

                    lowered = english_message.strip().lower()
                    is_greeting = any(kw in lowered for kw in GREETING_KEYWORDS)

                    # Build bot response (always in English internally)
                    if is_greeting:
                        bot_response = WELCOME_MESSAGE
                    else:
                        bot_response = f"You said: {user_message}"

                    # Send translated text + voice response
                    await reply_with_translation(sender_id, bot_response)

    return {"status": "ok"}


async def reply_with_translation(sender_id: str, english_text: str):
    """Translate to user's language then send both text and voice."""
    lang = get_user_language(sender_id) or "en"

    translated = translate_to(english_text, lang)
    print(f"Replying to {sender_id} in [{lang}]: {translated[:60]}...")

    # Send text
    await send_text(sender_id, translated)

    # Send voice
    audio_bytes = text_to_speech_bytes(translated, lang)
    if audio_bytes:
        await send_audio(sender_id, audio_bytes)


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
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        result = response.json()
        if response.status_code == 200:
            print("Text sent successfully")
        else:
            print(f"ERROR sending text [HTTP {response.status_code}]:", result)


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
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data, files=files)
        result = response.json()
        if response.status_code == 200:
            print("Audio sent successfully")
        else:
            print(f"ERROR sending audio [HTTP {response.status_code}]:", result)


# Subscribe page to webhook events
@app.get("/setup")
async def setup():
    url = "https://graph.facebook.com/v17.0/me/subscribed_apps"
    params = {
        "subscribed_fields": "messages,messaging_postbacks",
        "access_token": PAGE_ACCESS_TOKEN
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, params=params)
        result = response.json()
        print("Setup result:", result)
        return result


# Health check
@app.get("/")
def root():
    return {"status": "Bot is running!"}
