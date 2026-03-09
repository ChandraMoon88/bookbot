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

WARMING_UP_MESSAGE = (
    "I'm just starting up ⏳ — please send your message again in 1–2 minutes!"
)

# ── Per-user session state ────────────────────────────────────────────────────
# Tracks language selection progress. Resets on server restart (acceptable for
# free-tier HF Spaces — user just re-selects language on cold start).
_user_states: dict[str, dict] = {}


def _get_state(sender_id: str) -> dict:
    if sender_id not in _user_states:
        _user_states[sender_id] = {"lang_confirmed": False, "awaiting_lang": True}
    return _user_states[sender_id]


# ── Language selection menu ───────────────────────────────────────────────────
LANGUAGE_MENU = {
    "1": "en", "2": "hi", "3": "te", "4": "ta", "5": "kn",
    "6": "ml", "7": "bn", "8": "fr", "9": "es", "10": "ar",
}
LANGUAGE_LABELS = {
    "en": "English",             "hi": "Hindi (हिंदी)",
    "te": "Telugu (తెలుగు)",     "ta": "Tamil (தமிழ்)",
    "kn": "Kannada (ಕನ್ನಡ)",   "ml": "Malayalam (മലയാളം)",
    "bn": "Bengali (বাংলা)",    "fr": "French (Français)",
    "es": "Spanish (Español)",  "ar": "Arabic (العربية)",
}
_LANG_BY_NAME = {
    "english": "en", "hindi": "hi",    "telugu": "te",  "tamil": "ta",
    "kannada": "kn", "malayalam": "ml", "bengali": "bn",
    "french":  "fr", "spanish": "es",  "arabic": "ar",
}
LANGUAGE_SELECTION_MSG = (
    "Welcome to BookBot! 🏨\n\n"
    "Please choose your language:\n\n"
    "1. English\n"
    "2. Hindi (हिंदी)\n"
    "3. Telugu (తెలుగు)\n"
    "4. Tamil (தமிழ்)\n"
    "5. Kannada (ಕನ್ನಡ)\n"
    "6. Malayalam (മലയാളം)\n"
    "7. Bengali (বাংলা)\n"
    "8. French (Français)\n"
    "9. Spanish (Español)\n"
    "10. Arabic (العربية)\n\n"
    "Reply with a number — e.g. '1' for English, '2' for Hindi 😊"
)

# Phrases (checked against English translation) that trigger language change
_CHANGE_LANG_TRIGGERS = {
    "change language", "change lang", "switch language",
    "select language", "choose language", "language change",
    "different language", "other language",
}


def _parse_lang_selection(text: str) -> str | None:
    """Return ISO language code if text is a valid language selection, else None."""
    t = text.strip().lower()
    if t in LANGUAGE_MENU:
        return LANGUAGE_MENU[t]
    for name, code in _LANG_BY_NAME.items():
        if name in t:
            return code
    return None


def _human_response(en_text: str) -> str | None:
    """
    Returns a natural, human-like English response for conversational inputs.
    Returns None if no intent matched (caller handles as unknown input).
    All text is English — translated to the user's language by the caller.
    """
    t = en_text.strip().lower()

    # ── Greetings ─────────────────────────────────────────────────────────────
    _greet_kw = {"hi", "hello", "hey", "hlo", "hii", "howdy", "yo", "sup", "greetings"}
    if any(t == kw or t.startswith(kw + " ") or t.startswith(kw + "!") for kw in _greet_kw):
        return (
            "Hello there! 👋 So great to hear from you!\n\n"
            "I'm BookBot, your hotel booking assistant. "
            "How can I help you today?\n\n"
            "Type 'book' to start a new hotel booking! 🏨"
        )

    # ── How are you ────────────────────────────────────────────────────────────
    if any(kw in t for kw in ("how are you", "how r u", "how are u", "hows it going", "how do you do")):
        return (
            "I'm doing absolutely wonderful, thanks for asking! 😊\n\n"
            "I'm all set and ready to help you find the perfect hotel. "
            "What can I do for you today?"
        )

    # ── What's up ─────────────────────────────────────────────────────────────
    if any(kw in t for kw in ("what's up", "whats up")):
        return "Not much, just here and ready to find you a great hotel! 😄 What can I do for you?"

    # ── Time-of-day greetings ─────────────────────────────────────────────────
    if "good morning" in t:
        return "Good morning! ☀️ Hope your day is off to a fantastic start! I'm here to help with your hotel booking whenever you're ready. 😊"
    if "good afternoon" in t:
        return "Good afternoon! 😊 Hope you're having a lovely day! How can I assist you with your hotel booking?"
    if "good evening" in t:
        return "Good evening! 🌙 Hope you had a wonderful day! I'm here to help you find a great hotel. 😊"
    if "good night" in t:
        return "Good night! 🌙 Sweet dreams! Come back anytime you need a hotel booking — I'm always here! 😊"

    # ── Thanks ────────────────────────────────────────────────────────────────
    if any(kw in t for kw in ("thank you", "thanks", "thank u", "thx", "thankyou")):
        return "You're most welcome! 😊 It's my absolute pleasure to help. Is there anything else I can do for you?"

    # ── Goodbye ───────────────────────────────────────────────────────────────
    if any(kw in t for kw in ("bye", "goodbye", "see you", "take care", "good bye", "cya")):
        return "Goodbye! 👋 It was lovely talking with you! Have a wonderful day and come back anytime. Take care! 😊"

    # ── Help / capabilities ───────────────────────────────────────────────────
    if any(kw in t for kw in ("what can you do", "what do you do", "help", "capabilities", "what are you", "who are you")):
        return (
            "I'm BookBot — your personal hotel booking assistant! 🏨\n\n"
            "Here's what I can do for you:\n"
            "✅ Search hotels by city & dates\n"
            "✅ Check room availability\n"
            "✅ Make & manage your bookings\n"
            "✅ Understand voice & text messages\n"
            "✅ Talk to you in your own language\n\n"
            "Just type 'book' to get started! 😊"
        )

    # ── Welcome / start / begin ────────────────────────────────────────────────
    if any(kw in t for kw in ("welcome", "start", "begin")):
        return (
            "Welcome! Great to have you here! 🎉\n\n"
            "I'm BookBot, your hotel booking assistant.\n"
            "Type 'book' to start a new booking, or ask me anything! 🏨"
        )

    # ── Book / reserve ────────────────────────────────────────────────────────
    if any(kw in t for kw in ("book", "booking", "reserve", "reservation", "hotel", "room")):
        return (
            "Awesome! Let's find you the perfect hotel! 🏨\n\n"
            "To get started, I'll need a few details:\n\n"
            "📍 Which city are you looking in?\n"
            "📅 What's your check-in date?\n"
            "📅 What's your check-out date?\n"
            "👥 How many guests?\n\n"
            "Let's start — which city would you like to stay in? 😊"
        )

    return None

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
    if not _models_ready:
        return {"text": WARMING_UP_MESSAGE, "audio_b64": None, "lang": "en"}

    try:
        data         = await request.json()
        sender_id    = data.get("sender_id")
        message_type = data.get("type", "text")
        state        = _get_state(sender_id)
        lang         = get_user_language(sender_id) or "en"
        stt_lang: str | None = None

        # ── Step 1: Get the message text (transcribe voice if needed) ──────────
        if message_type == "voice":
            audio_b64_in = data.get("audio_b64")
            raw_bytes    = base64.b64decode(audio_b64_in)
            # Pass the user's confirmed language as a hint so Whisper biases
            # toward the right script (critical for Indian languages).
            user_message, stt_lang = speech_to_text(
                raw_bytes, lang_hint=lang if state["lang_confirmed"] else None
            )
            if not user_message:
                return {
                    "text":      "Sorry, I couldn't understand that. Could you please try again? 😊",
                    "audio_b64": None,
                    "lang":      lang,
                }
        else:
            user_message = data.get("message", "")

        # ── Step 2: Language selection flow ────────────────────────────────────
        # Every new user (and any user who requested a language change) goes here.
        if state["awaiting_lang"]:
            lang_choice = _parse_lang_selection(user_message)

            if lang_choice is None:
                # Cannot parse — show the menu
                audio_out = text_to_speech_bytes(LANGUAGE_SELECTION_MSG, "en")
                a64 = base64.b64encode(audio_out).decode() if audio_out else None
                return {"text": LANGUAGE_SELECTION_MSG, "audio_b64": a64, "lang": "en"}

            # User chose a language — confirm and store
            set_user_language(sender_id, lang_choice)
            lang = lang_choice
            state["lang_confirmed"] = True
            state["awaiting_lang"]  = False

            label   = LANGUAGE_LABELS.get(lang_choice, "English")
            body_en = (
                f"Perfect! I'll respond in {label} from now on. 😊\n\n"
                "Welcome to BookBot! 🏨\n"
                "I'm your hotel booking assistant.\n\n"
                "Here's what I can do:\n"
                "✅ Search & book hotel rooms\n"
                "✅ Check room availability\n"
                "✅ Manage your reservations\n\n"
                "Type 'book' to start your first booking!"
            )
            body_text = translate_to(body_en, lang_choice) if lang_choice != "en" else body_en
            full_text = body_text + "\n\n_(💬 Type 'change language' anytime to switch)_"
            audio_out = text_to_speech_bytes(body_text, lang_choice)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": full_text, "audio_b64": a64, "lang": lang_choice}

        # ── Step 3: Normal conversation (language already confirmed) ───────────
        # Always respond in the user's chosen language regardless of input language.
        input_lang    = stt_lang or detect_language(user_message)
        english_input = translate_to_english(user_message, input_lang)
        en_lower      = english_input.strip().lower()

        # Check if user wants to change language
        if (
            any(trigger in en_lower for trigger in _CHANGE_LANG_TRIGGERS)
            or "change language" in user_message.lower()
        ):
            state["lang_confirmed"] = False
            state["awaiting_lang"]  = True
            audio_out = text_to_speech_bytes(LANGUAGE_SELECTION_MSG, "en")
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": LANGUAGE_SELECTION_MSG, "audio_b64": a64, "lang": "en"}

        # Get a natural human-like response (in English), then translate
        bot_en = _human_response(en_lower) or (
            "I'm not sure I understood that. Here's what I can help with:\n\n"
            "• Type 'book' — to start a hotel booking 🏨\n"
            "• Type 'help' — to see all my features\n"
            "• Type 'change language' — to switch your language"
        )
        response_text = translate_to(bot_en, lang) if lang != "en" else bot_en

        # Hint appended to text only (not read aloud by TTS)
        full_text = response_text + "\n\n_(💬 Type 'change language' anytime to switch)_"
        audio_out = text_to_speech_bytes(response_text, lang)
        a64 = base64.b64encode(audio_out).decode() if audio_out else None
        return {"text": full_text, "audio_b64": a64, "lang": lang}

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        return {"text": "Sorry, something went wrong. Please try again! 😊", "audio_b64": None, "lang": "en"}