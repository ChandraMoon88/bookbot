"""
processor.py
------------
FastAPI AI processor for BookHotel Bot.

Key fix vs previous version:
  The old lifespan hook blocked uvicorn startup until ALL models were loaded
  (~10 min on free CPU). HF Spaces has a startup timeout and killed the process
  before it ever became healthy — causing the permanent "Starting" state.

  New approach:
    1. uvicorn starts INSTANTLY, HF Spaces sees it as healthy right away
    2. Models load in a background thread after the server is already live
    3. Requests arriving before models finish get a friendly "warming up" reply
    4. /health shows per-model status so you can watch progress in real time
"""

from contextlib import asynccontextmanager
import base64
import logging
import threading

from fastapi import FastAPI, Request
from autotranslator import (
    detect_language,
    get_user_language,
    set_user_language,
    translate_to,
    translate_to_english,
    text_to_speech_bytes,
    speech_to_text,
    _get_whisper,
    _get_nllb,
    _get_speecht5,
    _get_mms,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL READINESS STATE
# ─────────────────────────────────────────────────────────────────────────────

_MMS_PREWARM   = ["eng", "hin", "tel", "tam", "fra", "spa", "ara"]
_models_ready  = False
_models_status: dict = {}


def _load_models_background():
    """
    Loads all models after the server is already running.
    Any request arriving before this finishes gets WARMING_UP_MESSAGE.
    """
    global _models_ready

    def _load(name: str, fn):
        print(f"[warmup] Loading {name}…", flush=True)
        try:
            fn()
            _models_status[name] = "ready"
            print(f"[warmup] ✅ {name} ready.", flush=True)
        except Exception as e:
            _models_status[name] = f"error: {e}"
            print(f"[warmup] ❌ {name} failed: {e}", flush=True)

    _load("Whisper",  _get_whisper)
    _load("NLLB-200", _get_nllb)
    _load("SpeechT5", _get_speecht5)
    for code in _MMS_PREWARM:
        _load(f"MMS-{code}", lambda c=code: _get_mms(c))

    _models_ready = True
    print("[warmup] ===== All models loaded — fully ready for requests =====", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — server starts first, models load in background
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Server starting — kicking off background model warmup…", flush=True)
    threading.Thread(target=_load_models_background, daemon=True).start()
    print("[startup] ✅ Server LIVE. Models loading in background.", flush=True)
    yield


app = FastAPI(lifespan=lifespan)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
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

WARMING_UP_MESSAGE = (
    "I'm just starting up ⏳ — please send your message again in 1–2 minutes!"
)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "models_ready":  _models_ready,
        "models_status": _models_status,
        "service":       "AI processor",
    }


@app.post("/process")
async def process_message(request: Request):
    # Return friendly message while models are still loading
    if not _models_ready:
        return {
            "text":      WARMING_UP_MESSAGE,
            "audio_b64": None,
            "lang":      "en",
        }

    try:
        data         = await request.json()
        sender_id    = data.get("sender_id")
        message_type = data.get("type", "text")

        # ── Voice ─────────────────────────────────────────────────────────────
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

        # ── Text ──────────────────────────────────────────────────────────────
        else:
            user_message = data.get("message", "")
            stored_lang  = get_user_language(sender_id)
            detected     = detect_language(user_message)
            if stored_lang and len(user_message.strip()) <= 6:
                detected = stored_lang
            else:
                set_user_language(sender_id, detected)

        # ── Intent ────────────────────────────────────────────────────────────
        english_message = translate_to_english(user_message, detected)
        is_greeting     = any(kw in english_message.strip().lower() for kw in GREETING_KEYWORDS)
        bot_response    = WELCOME_MESSAGE if is_greeting else f"You said: {user_message}"

        # ── Translate + TTS ───────────────────────────────────────────────────
        lang        = get_user_language(sender_id) or "en"
        translated  = translate_to(bot_response, lang)
        audio_bytes = text_to_speech_bytes(translated, lang)
        audio_b64   = base64.b64encode(audio_bytes).decode() if audio_bytes else None

        return {"text": translated, "audio_b64": audio_b64, "lang": lang}

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        return {"text": "Sorry, something went wrong.", "audio_b64": None, "lang": "en"}