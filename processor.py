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
import os
import threading

try:
    import redis as _redis_lib
except ImportError:
    _redis_lib = None

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
    "I'm just starting up. Please send your message again in 1-2 minutes."
)

# ── Per-user session state ────────────────────────────────────────────────────
# In-memory store; bootstrapped from Redis on first access so language choice
# survives HF Spaces restarts (Redis is optional — falls back to in-memory).
_user_states: dict[str, dict] = {}
_redis_conn = None


def _get_redis_conn():
    """Return a Redis connection if REDIS_URL is set, otherwise None.
    Upstash Redis requires SSL — use ssl_cert_reqs=None to allow self-signed certs.
    """
    global _redis_conn
    if _redis_conn is None and _redis_lib is not None:
        url = os.getenv("REDIS_URL")
        if url:
            try:
                # ssl_cert_reqs=None is required for Upstash (and most managed Redis)
                _redis_conn = _redis_lib.from_url(
                    url,
                    decode_responses=True,
                    ssl_cert_reqs=None,
                )
                _redis_conn.ping()
                print("[redis] Connected — language preferences will persist across restarts.",
                      flush=True)
            except Exception as e:
                print(f"[redis] Unavailable ({e}) — using in-memory state.", flush=True)
                _redis_conn = None
    return _redis_conn


def _redis_get_lang(sender_id: str) -> str | None:
    r = _get_redis_conn()
    if r:
        try:
            return r.get(f"bblang:{sender_id}")
        except Exception:
            pass
    return None


def _redis_set_lang(sender_id: str, lang: str) -> None:
    r = _get_redis_conn()
    if r:
        try:
            r.setex(f"bblang:{sender_id}", 86400 * 90, lang)  # 90-day TTL
        except Exception:
            pass


def _get_state(sender_id: str) -> dict:
    if sender_id not in _user_states:
        # Try to restore language from Redis (survives HF Spaces restarts)
        persisted = _redis_get_lang(sender_id)
        if persisted:
            set_user_language(sender_id, persisted)
            _user_states[sender_id] = {
                "lang_confirmed": True,
                "awaiting_lang":  False,
                "lang_page":      1,
            }
        else:
            _user_states[sender_id] = {
                "lang_confirmed": False,
                "awaiting_lang":  True,
                "lang_page":      1,
            }
    return _user_states[sender_id]


# ── Language catalogue (globally popular, ordered by speaker count) ───────────
# Page 1 — top 20 by global speakers
_LANG_PAGE1 = [
    ("1",  "en",  "English"),
    ("2",  "zh",  "Chinese"),
    ("3",  "hi",  "Hindi"),
    ("4",  "es",  "Spanish"),
    ("5",  "fr",  "French"),
    ("6",  "ar",  "Arabic"),
    ("7",  "bn",  "Bengali"),
    ("8",  "pt",  "Portuguese"),
    ("9",  "ru",  "Russian"),
    ("10", "ur",  "Urdu"),
    ("11", "id",  "Indonesian"),
    ("12", "de",  "German"),
    ("13", "ja",  "Japanese"),
    ("14", "te",  "Telugu"),
    ("15", "ta",  "Tamil"),
    ("16", "mr",  "Marathi"),
    ("17", "tr",  "Turkish"),
    ("18", "ko",  "Korean"),
    ("19", "it",  "Italian"),
    ("20", "ml",  "Malayalam"),
]

# Page 2 — next 20 by global speakers
_LANG_PAGE2 = [
    ("21", "kn",  "Kannada"),
    ("22", "gu",  "Gujarati"),
    ("23", "pa",  "Punjabi"),
    ("24", "pl",  "Polish"),
    ("25", "uk",  "Ukrainian"),
    ("26", "nl",  "Dutch"),
    ("27", "th",  "Thai"),
    ("28", "vi",  "Vietnamese"),
    ("29", "fa",  "Persian"),
    ("30", "sw",  "Swahili"),
    ("31", "ms",  "Malay"),
    ("32", "fil", "Filipino"),
    ("33", "ro",  "Romanian"),
    ("34", "el",  "Greek"),
    ("35", "cs",  "Czech"),
    ("36", "hu",  "Hungarian"),
    ("37", "he",  "Hebrew"),
    ("38", "sv",  "Swedish"),
    ("39", "fi",  "Finnish"),
    ("40", "or",  "Odia"),
]

# Combined lookup: number → code
LANGUAGE_MENU: dict[str, str] = {
    num: code for num, code, _ in (_LANG_PAGE1 + _LANG_PAGE2)
}

# Combined lookup: code → display label
LANGUAGE_LABELS: dict[str, str] = {
    code: label for _, code, label in (_LANG_PAGE1 + _LANG_PAGE2)
}

# All known language names → code (for free-text input)
# Covers all NLLB-200 supported languages + common alternate names
_LANG_BY_NAME: dict[str, str] = {
    # Major world languages
    "english": "en",       "chinese": "zh",       "mandarin": "zh",
    "cantonese": "zh",     "hindi": "hi",          "spanish": "es",
    "castilian": "es",     "french": "fr",         "arabic": "ar",
    "bengali": "bn",       "bangla": "bn",         "portuguese": "pt",
    "russian": "ru",       "urdu": "ur",           "indonesian": "id",
    "bahasa": "id",        "german": "de",         "japanese": "ja",
    "telugu": "te",        "tamil": "ta",          "marathi": "mr",
    "turkish": "tr",       "korean": "ko",         "italian": "it",
    "malayalam": "ml",     "kannada": "kn",        "gujarati": "gu",
    "punjabi": "pa",       "polish": "pl",         "ukrainian": "uk",
    "dutch": "nl",         "flemish": "nl",        "thai": "th",
    "vietnamese": "vi",    "persian": "fa",        "farsi": "fa",
    "swahili": "sw",       "kiswahili": "sw",      "malay": "ms",
    "filipino": "fil",     "tagalog": "fil",       "romanian": "ro",
    "greek": "el",         "czech": "cs",          "hungarian": "hu",
    "hebrew": "he",        "swedish": "sv",        "finnish": "fi",
    "odia": "or",          "oriya": "or",
    # Additional European
    "norwegian": "no",     "danish": "da",         "slovak": "sk",
    "bulgarian": "bg",     "croatian": "hr",       "serbian": "sr",
    "slovenian": "sl",     "lithuanian": "lt",     "latvian": "lv",
    "estonian": "et",      "catalan": "ca",        "galician": "gl",
    "basque": "eu",        "welsh": "cy",          "irish": "ga",
    "icelandic": "is",     "maltese": "mt",
    # Middle East / Central Asia
    "azerbaijani": "az",   "kazakh": "kk",         "uzbek": "uz",
    "kyrgyz": "ky",        "georgian": "ka",       "armenian": "hy",
    # South / SE Asia
    "sinhala": "si",       "sinhalese": "si",      "nepali": "ne",
    "burmese": "my",       "myanmar": "my",        "khmer": "km",
    "cambodian": "km",     "lao": "lo",            "mongolian": "mn",
    "tibetan": "bo",
    # East Asia
    "taiwanese": "zh",
    # Africa
    "amharic": "am",       "yoruba": "yo",         "igbo": "ig",
    "zulu": "zu",          "xhosa": "xh",          "somali": "so",
    "afrikaans": "af",
}


def _strip_for_tts(text: str) -> str:
    """Remove markdown and formatting symbols before passing text to TTS."""
    import re
    # Remove markdown italic/bold wrappers like _(text)_
    text = re.sub(r'_\(([^)]+)\)_', r'\1', text)
    # Remove bullet characters
    text = re.sub(r'^[•*-] ', '', text, flags=re.MULTILINE)
    # Remove checkmark symbols
    text = text.replace('\u2705', '').replace('\u274c', '')
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _build_lang_menu_text(page: int) -> str:
    items = _LANG_PAGE1 if page == 1 else _LANG_PAGE2
    lines = "\n".join(f"{num}. {label}" for num, _, label in items)
    if page == 1:
        footer = "\nM - More languages (21-40)\nOr type any language name, e.g. Portuguese"
    else:
        footer = "\nB - Back to previous page (1-20)\nOr type any language name, e.g. Swahili"
    return (
        "Welcome to BookBot!\n\n"
        "Please choose your language:\n\n"
        + lines + footer +
        "\n\nReply with a number or language name."
    )


def _build_lang_buttons(page: int) -> list:
    """Build Messenger quick-reply button list for the language menu."""
    items = _LANG_PAGE1 if page == 1 else _LANG_PAGE2
    buttons = [
        {"content_type": "text", "title": label, "payload": f"LANG_{code}"}
        for _, code, label in items
    ]
    if page == 1:
        buttons.append({"content_type": "text", "title": "More (21-40)", "payload": "LANG_PAGE_2"})
    else:
        buttons.append({"content_type": "text", "title": "Back (1-20)", "payload": "LANG_PAGE_1"})
    # Facebook allows max 13 quick replies
    return buttons[:13]


LANGUAGE_SELECTION_MSG = _build_lang_menu_text(1)

# Phrases (checked against English translation) that trigger language change
_CHANGE_LANG_TRIGGERS = {
    "change language", "change lang", "switch language",
    "select language", "choose language", "language change",
    "different language", "other language",
}


def _parse_lang_selection(text: str, current_page: int = 1) -> str | tuple | None:
    """
    Returns:
      - ISO language code string  → confirmed selection
      - ("page", 2)               → user wants more languages
      - ("page", 1)               → user wants to go back
      - None                      → cannot parse
    """
    t = text.strip().lower()

    # Navigation
    if t in ("m", "more", "more languages", "next", "next page"):
        return ("page", 2)
    if t in ("b", "back", "previous", "back page", "previous page"):
        return ("page", 1)

    # Number selection
    if t in LANGUAGE_MENU:
        return LANGUAGE_MENU[t]

    # Free-text language name — check our comprehensive name map
    for name, code in _LANG_BY_NAME.items():
        if name in t:
            return code

    # Last resort: auto-detect the language the user is TYPING IN
    # e.g. user types "Telugu" in Telugu script → detect as "te"
    try:
        detected = detect_language(text)
        if detected and detected != "en":
            return ("autodetect", detected)
    except Exception:
        pass

    return None


def _human_response(en_text: str) -> str | None:
    """
    Returns a natural, human-like English response for conversational inputs.
    Returns None if no intent matched (caller handles as unknown input).
    All text is English — translated to the user's language by the caller.
    No emojis — clean plain text for correct TTS pronunciation.
    """
    t = en_text.strip().lower()

    # ── Greetings ─────────────────────────────────────────────────────────────
    _greet_kw = {"hi", "hello", "hey", "hlo", "hii", "howdy", "yo", "sup", "greetings"}
    if any(t == kw or t.startswith(kw + " ") or t.startswith(kw + "!") for kw in _greet_kw):
        return (
            "Hello! Great to hear from you.\n\n"
            "I am BookBot, your hotel booking assistant. "
            "How can I help you today?\n\n"
            "Tap Book a Hotel to get started."
        )

    # ── How are you ────────────────────────────────────────────────────────────
    if any(kw in t for kw in ("how are you", "how r u", "how are u", "hows it going", "how do you do")):
        return (
            "I am doing well, thank you for asking.\n\n"
            "I am ready to help you find the perfect hotel. "
            "What can I do for you today?"
        )

    # ── What's up ─────────────────────────────────────────────────────────────
    if any(kw in t for kw in ("what's up", "whats up")):
        return "Just here and ready to find you a great hotel. What can I do for you?"

    # ── Time-of-day greetings ─────────────────────────────────────────────────
    if "good morning" in t:
        return "Good morning! Hope your day is off to a great start. I am here to help with your hotel booking whenever you are ready."
    if "good afternoon" in t:
        return "Good afternoon! Hope you are having a lovely day. How can I assist you with your hotel booking?"
    if "good evening" in t:
        return "Good evening! Hope you had a wonderful day. I am here to help you find a great hotel."
    if "good night" in t:
        return "Good night! Come back anytime you need a hotel booking. I am always here."

    # ── Thanks ────────────────────────────────────────────────────────────────
    if any(kw in t for kw in ("thank you", "thanks", "thank u", "thx", "thankyou")):
        return "You are most welcome. It is my pleasure to help. Is there anything else I can do for you?"

    # ── Goodbye ───────────────────────────────────────────────────────────────
    if any(kw in t for kw in ("bye", "goodbye", "see you", "take care", "good bye", "cya")):
        return "Goodbye! It was lovely talking with you. Have a wonderful day and come back anytime."

    # ── Help / capabilities ───────────────────────────────────────────────────
    if any(kw in t for kw in ("what can you do", "what do you do", "help", "capabilities", "what are you", "who are you")):
        return (
            "I am BookBot, your personal hotel booking assistant.\n\n"
            "Here is what I can do for you:\n"
            "- Search hotels by city and dates\n"
            "- Check room availability\n"
            "- Make and manage your bookings\n"
            "- Understand voice and text messages\n"
            "- Talk to you in your own language\n\n"
            "Tap Book a Hotel to get started."
        )

    # ── Welcome / start / begin / get started ─────────────────────────────────
    if any(kw in t for kw in ("welcome", "start", "begin", "get started", "get_started")):
        return (
            "Welcome! Great to have you here.\n\n"
            "I am BookBot, your hotel booking assistant.\n"
            "Tap Book a Hotel to start a new booking, or tap Help to see what I can do."
        )

    # ── Book / reserve ────────────────────────────────────────────────────────
    if any(kw in t for kw in ("book", "booking", "reserve", "reservation", "hotel", "room")):
        return (
            "Let us find you the perfect hotel.\n\n"
            "To get started, I will need a few details:\n"
            "- Which city are you looking in?\n"
            "- What is your check-in date?\n"
            "- What is your check-out date?\n"
            "- How many guests?\n\n"
            "Which city would you like to stay in?"
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
        return {"text": WARMING_UP_MESSAGE, "buttons": [], "audio_b64": None, "lang": "en"}

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
            user_message, stt_lang = speech_to_text(
                raw_bytes, lang_hint=lang if state["lang_confirmed"] else None
            )
            if not user_message:
                return {
                    "text":      "Sorry, I could not understand that audio. Please try again.",
                    "buttons":   [],
                    "audio_b64": None,
                    "lang":      lang,
                }
        else:
            user_message = data.get("message", "")

        # ── Restart / refresh postback — reset user state completely ───────────
        if user_message.strip().upper() in ("RESTART", "REFRESH", "START_OVER"):
            _user_states.pop(sender_id, None)
            set_user_language(sender_id, "en")
            _redis_set_lang(sender_id, "en")
            state = _get_state(sender_id)
            # Force language selection flow
            state["awaiting_lang"] = True
            state["lang_confirmed"] = False

        # ── Step 2: Language selection flow ────────────────────────────────────
        if state["awaiting_lang"]:
            result = _parse_lang_selection(user_message, state["lang_page"])

            # Navigation: user wants page 2 or back to page 1
            if isinstance(result, tuple) and result[0] == "page":
                state["lang_page"] = result[1]
                menu_text = _build_lang_menu_text(result[1])
                buttons   = _build_lang_buttons(result[1])
                tts_text  = _strip_for_tts(menu_text)
                audio_out = text_to_speech_bytes(tts_text, "en")
                a64 = base64.b64encode(audio_out).decode() if audio_out else None
                return {"text": menu_text, "buttons": buttons, "audio_b64": a64, "lang": "en"}

            # Auto-detected language from what the user typed
            if isinstance(result, tuple) and result[0] == "autodetect":
                detected_code  = result[1]
                detected_label = LANGUAGE_LABELS.get(detected_code, detected_code.upper())
                confirm_en = (
                    f"It looks like you are writing in {detected_label}. "
                    f"Would you like me to respond in {detected_label}?\n\n"
                    "Tap Yes to confirm, or choose a different language below."
                )
                state["pending_autodetect"] = detected_code
                confirm_buttons = [
                    {"content_type": "text", "title": "Yes",              "payload": f"LANG_{detected_code}"},
                    {"content_type": "text", "title": "Choose different", "payload": "LANG_PAGE_1"},
                ]
                audio_out = text_to_speech_bytes(confirm_en, "en")
                a64 = base64.b64encode(audio_out).decode() if audio_out else None
                return {"text": confirm_en, "buttons": confirm_buttons, "audio_b64": a64, "lang": "en"}

            # User confirming auto-detected language
            if result is None and state.get("pending_autodetect"):
                t_lower = user_message.strip().lower()
                if any(w in t_lower for w in ("yes", "ok", "sure", "correct", "right", "yeah")):
                    result = state["pending_autodetect"]
                else:
                    state.pop("pending_autodetect", None)
                    menu_text = _build_lang_menu_text(state["lang_page"])
                    buttons   = _build_lang_buttons(state["lang_page"])
                    tts_text  = _strip_for_tts(menu_text)
                    audio_out = text_to_speech_bytes(tts_text, "en")
                    a64 = base64.b64encode(audio_out).decode() if audio_out else None
                    return {"text": menu_text, "buttons": buttons, "audio_b64": a64, "lang": "en"}

            if result is None:
                # Cannot parse — re-show the current page
                menu_text = (
                    _build_lang_menu_text(state["lang_page"]) +
                    "\n\nYou can also type the language name, e.g. Norwegian or Swahili."
                )
                buttons   = _build_lang_buttons(state["lang_page"])
                tts_text  = _strip_for_tts(_build_lang_menu_text(state["lang_page"]))
                audio_out = text_to_speech_bytes(tts_text, "en")
                a64 = base64.b64encode(audio_out).decode() if audio_out else None
                return {"text": menu_text, "buttons": buttons, "audio_b64": a64, "lang": "en"}

            lang_choice = result  # confirmed ISO code

            # User chose a language — confirm, persist, and store
            set_user_language(sender_id, lang_choice)
            _redis_set_lang(sender_id, lang_choice)
            state.pop("pending_autodetect", None)
            lang = lang_choice
            state["lang_confirmed"] = True
            state["awaiting_lang"]  = False

            label   = LANGUAGE_LABELS.get(lang_choice, "English")
            body_en = (
                f"Language set to {label}.\n\n"
                "Welcome to BookBot!\n"
                "I am your hotel booking assistant.\n\n"
                "What I can do for you:\n"
                "- Search and book hotel rooms\n"
                "- Check room availability\n"
                "- Manage your reservations\n\n"
                "Tap Book a Hotel to start your first booking."
            )
            body_text = translate_to(body_en, lang_choice) if lang_choice != "en" else body_en
            full_text = body_text + "\n\nType 'change language' anytime to switch."
            tts_text  = _strip_for_tts(body_text)
            audio_out = text_to_speech_bytes(tts_text, lang_choice)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            main_buttons = [
                {"content_type": "text", "title": "Book a Hotel",    "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "Help",            "payload": "ACTION_HELP"},
                {"content_type": "text", "title": "Change Language", "payload": "ACTION_CHANGE_LANG"},
            ]
            return {"text": full_text, "buttons": main_buttons, "audio_b64": a64, "lang": lang_choice}

        # ── Step 3: Normal conversation (language already confirmed) ───────────
        input_lang    = stt_lang or detect_language(user_message)
        english_input = translate_to_english(user_message, input_lang)
        en_lower      = english_input.strip().lower()

        # Handle quick-reply button payloads
        raw_upper = user_message.strip().upper()
        if raw_upper == "ACTION_CHANGE_LANG" or any(trigger in en_lower for trigger in _CHANGE_LANG_TRIGGERS) or "change language" in user_message.lower():
            state["lang_confirmed"] = False
            state["awaiting_lang"]  = True
            state["lang_page"]      = 1
            menu_text = _build_lang_menu_text(1)
            buttons   = _build_lang_buttons(1)
            tts_text  = _strip_for_tts(menu_text)
            audio_out = text_to_speech_bytes(tts_text, "en")
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": menu_text, "buttons": buttons, "audio_b64": a64, "lang": "en"}

        if raw_upper == "ACTION_BOOK":
            en_lower = "book"
        elif raw_upper == "ACTION_HELP":
            en_lower = "help"

        # Get a natural human-like response (in English), then translate
        bot_en = _human_response(en_lower) or (
            "I am not sure I understood that. Here is what I can help with:\n\n"
            "- Tap Book a Hotel to start a hotel booking\n"
            "- Tap Help to see all my features\n"
            "- Tap Change Language to switch your language"
        )
        response_text = translate_to(bot_en, lang) if lang != "en" else bot_en
        full_text     = response_text + "\n\nType 'change language' anytime to switch."
        tts_text      = _strip_for_tts(response_text)
        audio_out     = text_to_speech_bytes(tts_text, lang)
        a64 = base64.b64encode(audio_out).decode() if audio_out else None
        main_buttons = [
            {"content_type": "text", "title": "Book a Hotel",    "payload": "ACTION_BOOK"},
            {"content_type": "text", "title": "Help",            "payload": "ACTION_HELP"},
            {"content_type": "text", "title": "Change Language", "payload": "ACTION_CHANGE_LANG"},
        ]
        return {"text": full_text, "buttons": main_buttons, "audio_b64": a64, "lang": lang}

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        return {"text": "Sorry, something went wrong. Please try again.", "buttons": [], "audio_b64": None, "lang": "en"}