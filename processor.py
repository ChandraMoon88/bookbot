"""
processor.py
------------
FastAPI AI processor for BookHotel Bot.

Key change vs previous version:
  @app.on_event("startup") pre-warms ALL models (NLLB-200, SpeechT5, MMS-TTS,
  Whisper) so the container is fully ready before the first request arrives.
  Without this, the first request would hang for ~8–10 minutes while models
  download/load — which is what was causing the silent startup log.
"""

from contextlib import asynccontextmanager
import asyncio
import base64
import os
import logging

from fastapi import FastAPI, Request
from autotranslator import (
    detect_language,
    get_user_language,
    set_user_language,
    translate_to,
    translate_to_english,
    text_to_speech_bytes,
    speech_to_text,
    # Import the private loader functions so we can pre-warm them at startup
    _get_whisper,
    _get_nllb,
    _get_speecht5,
    _get_mms,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STARTUP PRE-WARM
# Loads every model into memory before the first request arrives.
# Prevents the 8–10 minute silent hang on cold start.
# ─────────────────────────────────────────────────────────────────────────────

# MMS languages to pre-warm — must match MMS_PRELOAD in download_model.py
# Uses ISO 639-3 codes (MMS model suffix), not ISO 639-1
_MMS_PREWARM = ["eng", "hin", "tel", "tam", "fra", "spa", "ara"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm all models on startup so cold requests are instant."""
    logger.info("===== Pre-warming models on startup =====")

    loop = asyncio.get_event_loop()

    def _warm_all():
        print("[startup] Loading Whisper…", flush=True)
        _get_whisper()
        print("[startup] ✅ Whisper ready.", flush=True)

        print("[startup] Loading NLLB-200…", flush=True)
        _get_nllb()
        print("[startup] ✅ NLLB-200 ready.", flush=True)

        print("[startup] Loading SpeechT5…", flush=True)
        _get_speecht5()
        print("[startup] ✅ SpeechT5 ready.", flush=True)

        for code in _MMS_PREWARM:
            print(f"[startup] Loading MMS-TTS [{code}]…", flush=True)
            try:
                _get_mms(code)
                print(f"[startup] ✅ MMS-TTS [{code}] ready.", flush=True)
            except Exception as e:
                print(f"[startup] ⚠️  MMS-TTS [{code}] skipped: {e}", flush=True)

        print("[startup] ===== All models loaded — ready for requests =====", flush=True)

    # Run blocking model loads in a thread so we don't block the event loop
    await loop.run_in_executor(None, _warm_all)

    yield  # App runs here

    # (shutdown hook — nothing needed)


app = FastAPI(lifespan=lifespan)

# ─────────────────────────────────────────────────────────────────────────────
# INTENT / GREETING DETECTION
# ─────────────────────────────────────────────────────────────────────────────

GREETING_KEYWORDS = {
    "hi", "hello", "hey", "hlo", "hii", "howdy", "sup", "yo",
    "greetings", "morning", "afternoon", "evening", "night",
    "good morning", "good afternoon", "good evening", "good night",
    "good day", "how are you", "what's up", "whats up",
    "welcome", "start", "begin", "help",
}

WELCOME_MESSAGE = (
    "Welcome to BookBot!\n\n"
    "I'm your hotel booking assistant. I can help you with:\n"
    "- Search & book hotel rooms\n"
    "- Check availability\n"
    "- Manage your reservations\n\n"
    "Type or Voice 'book' to start a new booking."
)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "AI processor"}


@app.post("/process")
async def process_message(request: Request):
    try:
        data          = await request.json()
        sender_id     = data.get("sender_id")
        message_type  = data.get("type", "text")

        # ── Voice input ───────────────────────────────────────────────────────
        if message_type == "voice":
            audio_b64   = data.get("audio_b64")
            audio_bytes = base64.b64decode(audio_b64)
            stored_lang = get_user_language(sender_id)

            user_message, detected = speech_to_text(audio_bytes, lang_hint=stored_lang)
            if not user_message:
                return {
                    "text":      "Sorry, could not understand voice.",
                    "audio_b64": None,
                    "lang":      stored_lang or "en",
                }
            set_user_language(sender_id, detected)

        # ── Text input ────────────────────────────────────────────────────────
        else:
            user_message = data.get("message", "")
            stored_lang  = get_user_language(sender_id)
            detected     = detect_language(user_message)

            # langdetect is unreliable on very short words (≤6 chars).
            # Trust the already-confirmed stored language instead.
            if stored_lang and len(user_message.strip()) <= 6:
                detected = stored_lang
            else:
                set_user_language(sender_id, detected)

        # ── Translate to English for intent matching ──────────────────────────
        english_message = translate_to_english(user_message, detected)
        lowered         = english_message.strip().lower()
        is_greeting     = any(kw in lowered for kw in GREETING_KEYWORDS)

        # ── Generate response ─────────────────────────────────────────────────
        bot_response = WELCOME_MESSAGE if is_greeting else f"You said: {user_message}"

        # ── Translate back to user language ───────────────────────────────────
        lang       = get_user_language(sender_id) or "en"
        translated = translate_to(bot_response, lang)

        # ── TTS ───────────────────────────────────────────────────────────────
        audio_bytes = text_to_speech_bytes(translated, lang)
        audio_b64   = base64.b64encode(audio_bytes).decode() if audio_bytes else None

        return {
            "text":      translated,
            "audio_b64": audio_b64,
            "lang":      lang,
        }

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        return {
            "text":      "Sorry, something went wrong.",
            "audio_b64": None,
            "lang":      "en",
        }