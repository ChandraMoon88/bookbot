from fastapi import FastAPI, Request
from autotranslator import (
    detect_language, get_user_language, set_user_language,
    translate_to, translate_to_english,
    text_to_speech_bytes, speech_to_text
)
import base64
import os

app = FastAPI()

GREETING_KEYWORDS = {
    "hi", "hello", "hey", "hlo", "hii", "howdy", "sup", "yo",
    "greetings", "morning", "afternoon", "evening", "night",
    "good morning", "good afternoon", "good evening", "good night",
    "good day", "how are you", "what's up", "whats up",
    "welcome", "start", "begin", "help"
}

WELCOME_MESSAGE = (
    "Welcome to BookBot!\n\n"
    "I'm your hotel booking assistant. I can help you with:\n"
    "- Search & book hotel rooms\n"
    "- Check availability\n"
    "- Manage your reservations\n\n"
    "Type or Voice 'book' to start a new booking."
)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI processor"}

@app.post("/process")
async def process_message(request: Request):
    try:
        data = await request.json()
        sender_id = data.get("sender_id")
        message_type = data.get("type", "text")

        if message_type == "voice":
            audio_b64 = data.get("audio_b64")
            audio_bytes = base64.b64decode(audio_b64)
            # Pass stored language as hint — helps Whisper with low-resource
            # languages (Telugu, Tamil, etc.) it may otherwise mis-identify
            stored_lang = get_user_language(sender_id)
            user_message, detected = speech_to_text(audio_bytes, lang_hint=stored_lang)
            if not user_message:
                return {
                    "text": "Sorry, could not understand voice.",
                    "audio_b64": None,
                    "lang": "en"
                }
            set_user_language(sender_id, detected)
        else:
            user_message = data.get("message", "")
            detected = detect_language(user_message)
            set_user_language(sender_id, detected)

        # Translate to English for intent
        english_message = translate_to_english(user_message, detected)
        lowered = english_message.strip().lower()
        is_greeting = any(kw in lowered for kw in GREETING_KEYWORDS)

        # Generate response
        bot_response = WELCOME_MESSAGE if is_greeting else f"You said: {user_message}"

        # Translate to user language
        lang = get_user_language(sender_id) or "en"
        translated = translate_to(bot_response, lang)

        # Generate audio
        audio_bytes = text_to_speech_bytes(translated, lang)
        audio_b64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None

        return {
            "text": translated,
            "audio_b64": audio_b64,
            "lang": lang
        }

    except Exception as e:
        print(f"Processing error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "text": "Sorry, something went wrong.",
            "audio_b64": None,
            "lang": "en"
        }