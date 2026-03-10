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
import re
import threading
from datetime import datetime, date, timedelta

try:
    import redis as _redis_lib
except ImportError:
    _redis_lib = None

try:
    from db_client import (
        search_hotels,
        semantic_hotel_search,
        get_or_create_user,
        get_user_bookings,
        create_booking,
        cancel_booking,
        get_booking_by_ref,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

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
    Only passes SSL kwargs when the URL uses the rediss:// scheme.
    redis:// (plain) URLs must not receive SSL kwargs — causes TypeError.
    """
    global _redis_conn
    if _redis_conn is None and _redis_lib is not None:
        url = os.getenv("REDIS_URL")
        if url:
            try:
                kwargs: dict = {"decode_responses": True}
                if url.startswith("rediss://") or ".upstash.io" in url:
                    import ssl as _ssl
                    kwargs["ssl_cert_reqs"] = _ssl.CERT_NONE
                    # Normalise scheme so the redis library uses TLS
                    url = url.replace("redis://", "rediss://", 1)
                _redis_conn = _redis_lib.from_url(url, **kwargs)
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


def _redis_del_lang(sender_id: str) -> None:
    """Delete persisted language so the next _get_state() forces fresh language selection."""
    r = _get_redis_conn()
    if r:
        try:
            r.delete(f"bblang:{sender_id}")
        except Exception:
            pass


def _get_state(sender_id: str) -> dict:
    if sender_id not in _user_states:
        # Try to restore language from Redis (survives HF Spaces restarts)
        persisted = _redis_get_lang(sender_id)
        if persisted:
            set_user_language(sender_id, persisted)
            _user_states[sender_id] = {
                "lang_confirmed":      True,
                "awaiting_lang":       False,
                "lang_page":           1,
                "awaiting_type_input": False,
            }
        else:
            _user_states[sender_id] = {
                "lang_confirmed":      False,
                "awaiting_lang":       True,
                "lang_page":           1,
                "awaiting_type_input": False,
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
    """Short prompt shown alongside language quick-reply buttons (10 per page)."""
    all_langs = _LANG_PAGE1 + _LANG_PAGE2
    start = (page - 1) * 10
    end   = min(start + 10, len(all_langs))
    return (
        f"Choose your language ({start + 1}-{end} of 40):\n"
        "Tap a button, or tap 'Type my language' to type it."
    )


def _build_lang_buttons(page: int) -> list:
    """10 languages per page + Back / More / Type buttons. Max 13 quick replies."""
    all_langs = _LANG_PAGE1 + _LANG_PAGE2
    start     = (page - 1) * 10
    items     = all_langs[start:start + 10]
    buttons   = [
        {"content_type": "text", "title": label, "payload": f"LANG_{code}"}
        for _, code, label in items
    ]
    if page > 1:
        buttons.append({"content_type": "text", "title": "Back",             "payload": f"LANG_PAGE_{page - 1}"})
    if page < 4:
        buttons.append({"content_type": "text", "title": "More languages",   "payload": f"LANG_PAGE_{page + 1}"})
    buttons.append({"content_type": "text", "title": "Type my language",   "payload": "LANG_TYPE"})
    return buttons[:13]


LANGUAGE_SELECTION_MSG = _build_lang_menu_text(1)

# Always-visible action buttons appended to every bot reply once language is set
_MAIN_BUTTONS = [
    {"content_type": "text", "title": "Book a Hotel",    "payload": "ACTION_BOOK"},
    {"content_type": "text", "title": "My Bookings",     "payload": "MY_BOOKINGS"},
    {"content_type": "text", "title": "Help",            "payload": "ACTION_HELP"},
    {"content_type": "text", "title": "Change Language", "payload": "ACTION_CHANGE_LANG"},
]

# Phrases (checked against English translation) that trigger language change
_CHANGE_LANG_TRIGGERS = {
    "change language", "change lang", "switch language",
    "select language", "choose language", "language change",
    "different language", "other language",
}


def _parse_lang_selection(text: str, current_page: int = 1) -> str | tuple | None:
    """
    Returns:
      - ISO language code string       -> confirmed selection
      - ("page", N)                    -> navigate to page N (1-4)
      - ("type_input",)                -> user tapped 'Type my language'
      - ("autodetect", code)           -> language detected from user's script
      - None                           -> cannot parse
    """
    t = text.strip().lower()

    # Button payload navigation — LANG_PAGE_N arrives as page_n after prefix strip
    if t.startswith("page_") and t[5:].isdigit():
        return ("page", int(t[5:]))

    # "Type my language" button payload
    if t == "type":
        return ("type_input",)

    # Direct ISO code from a language button tap (e.g. "en", "hi", "zh")
    if t in LANGUAGE_LABELS:
        return t

    # Legacy text navigation
    if t in ("m", "more", "more languages", "next", "next page"):
        return ("page", min(current_page + 1, 4))
    if t in ("b", "back", "previous", "back page", "previous page"):
        return ("page", max(current_page - 1, 1))

    # Number selection
    if t in LANGUAGE_MENU:
        return LANGUAGE_MENU[t]

    # Free-text language name — check our comprehensive name map
    for name, code in _LANG_BY_NAME.items():
        if name in t:
            return code

    # Last resort: auto-detect the language the user is TYPING IN
    # e.g. user types "Telugu" in Telugu script → detect as "te"
    # Skip auto-detect for known bot command words to avoid false positives
    _NO_AUTODETECT = {"restart", "refresh", "start_over", "get_started",
                      "action_book", "action_help", "action_change_lang",
                      "yes", "no", "ok", "type", "back", "more"}
    if t in _NO_AUTODETECT:
        return None
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
    if any(kw in t for kw in ("what can you do", "what do you do", "help", "capabilities", "what are you", "who are you", "features", "feature list", "what features")):
        return (
            "I am BookBot, your personal hotel booking assistant.\n\n"
            "BOOKING\n"
            "- Search hotels by city, dates, guests\n"
            "- Filter by budget, star rating, amenities\n"
            "- Book Standard, Group, Corporate, Wedding,\n"
            "  Long Stay, Honeymoon and Last-Minute hotels\n"
            "- Add extras: airport transfer, spa, romance setup\n\n"
            "PAYMENT\n"
            "- Pay by card, UPI, PayPal, or at hotel\n"
            "- Apply voucher codes or use loyalty points\n\n"
            "MANAGE BOOKINGS\n"
            "- View, modify, or cancel bookings\n"
            "- Get PDF confirmation sent to email\n\n"
            "SERVICES\n"
            "- Pre-arrival: early check-in, airport pickup\n"
            "- In-stay: spa, room service, housekeeping\n"
            "- Post-stay: late check-out, lost & found, review\n\n"
            "LOYALTY\n"
            "- Earn and redeem reward points\n"
            "- Refer friends for bonus points\n\n"
            "Tap Book a Hotel to get started!"
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

    # ── Price / budget queries ────────────────────────────────────────────────
    if any(kw in t for kw in ("how much", "price", "cost", "expensive", "cheap", "budget friendly", "affordable")):
        return (
            "Hotel prices vary by city, season, and hotel type.\n\n"
            "I can search for hotels within any budget.\n"
            "Just tell me your budget per night and I will filter accordingly.\n\n"
            "For example: 'hotels under 100' or 'budget hotels in Dubai'\n\n"
            "Start a search to see current prices."
        )

    # ── About BookBot ─────────────────────────────────────────────────────────
    if any(kw in t for kw in ("what is bookbot", "about bookbot", "tell me about yourself", "who made you")):
        return (
            "I am BookBot — your 24/7 AI hotel booking assistant.\n\n"
            "I can search, compare, and book hotels in thousands of cities worldwide. "
            "I handle everything from first search to checkout — in over 40 languages.\n\n"
            "Trusted by thousands of travellers. Give me a try!"
        )

    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATE / GUEST PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4,  "apr": 4, "may": 5,      "june": 6, "jun": 6,  "july": 7,
    "jul": 7,    "august": 8, "aug": 8,   "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

_WEEKDAY_NAMES: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _parse_date(text: str) -> str | None:
    """Parse a user date string into YYYY-MM-DD.  Returns None if unrecognisable."""
    today = date.today()
    t = text.strip().lower()

    # ── Keywords ──────────────────────────────────────────────────────────────
    if t in ("today", "tonight", "now"):
        return str(today)
    if t in ("tomorrow", "tmrw", "tmr", "tom"):
        return str(today + timedelta(days=1))
    if t in ("day after tomorrow", "overmorrow"):
        return str(today + timedelta(days=2))
    if t in ("this weekend", "weekend"):
        days_to_sat = (5 - today.weekday()) % 7 or 7
        return str(today + timedelta(days=days_to_sat))

    # ── "next {weekday}" or bare weekday ──────────────────────────────────────
    for day_name, day_num in _WEEKDAY_NAMES.items():
        if t == day_name or t == f"next {day_name}":
            ahead = (day_num - today.weekday() + 7) % 7 or 7
            return str(today + timedelta(days=ahead))

    # ── "+N days" quick-button payloads ───────────────────────────────────────
    m = re.match(r'^\+(\d+)$', t)
    if m:
        return str(today + timedelta(days=int(m.group(1))))

    # ── Standard formats ──────────────────────────────────────────────────────
    _FMTS = [
        "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y",
        "%m/%d/%Y", "%m/%d/%y",
        "%d-%m-%Y", "%d.%m.%Y",
        "%d %B %Y", "%d %b %Y",
        "%B %d %Y", "%b %d %Y",
        "%B %d, %Y", "%b %d, %Y",
    ]
    for fmt in _FMTS:
        try:
            d = datetime.strptime(text.strip(), fmt).date()
            if d >= today:
                return str(d)
        except ValueError:
            pass

    # ── "15 march" / "march 15" — no year ────────────────────────────────────
    m2 = re.search(r'(\d{1,2})\s+([a-z]+)', t)
    if m2:
        day_n = int(m2.group(1))
        mon = _MONTH_NAMES.get(m2.group(2))
        if mon:
            for yr in (today.year, today.year + 1):
                try:
                    d = date(yr, mon, day_n)
                    if d >= today:
                        return str(d)
                except ValueError:
                    pass

    m3 = re.search(r'([a-z]+)\s+(\d{1,2})', t)
    if m3:
        mon = _MONTH_NAMES.get(m3.group(1))
        day_n = int(m3.group(2))
        if mon and 1 <= day_n <= 31:
            for yr in (today.year, today.year + 1):
                try:
                    d = date(yr, mon, day_n)
                    if d >= today:
                        return str(d)
                except ValueError:
                    pass

    # ── Pure day number "15" / "15th" ─────────────────────────────────────────
    m4 = re.match(r'^(\d{1,2})(?:st|nd|rd|th)?$', t)
    if m4:
        day_n = int(m4.group(1))
        if 1 <= day_n <= 31:
            for yr, mo in (
                (today.year, today.month),
                (today.year + (1 if today.month == 12 else 0),
                 today.month % 12 + 1),
            ):
                try:
                    d = date(yr, mo, day_n)
                    if d >= today:
                        return str(d)
                except ValueError:
                    pass
    return None


def _pretty_date(date_str: str) -> str:
    """Convert YYYY-MM-DD → 'Monday, 15 March 2026'."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date().strftime("%A, %d %B %Y")
    except Exception:
        return date_str


def _parse_guests(text: str) -> tuple[int, int] | None:
    """
    Parse guest count from text or a quick-reply payload like GUESTS_2_1.
    Returns (num_adults, num_children) or None if unrecognisable.
    """
    t = text.strip()

    # Quick-reply payload: GUESTS_2_1
    if t.upper().startswith("GUESTS_"):
        parts = t[7:].split("_")
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass

    tl = t.lower()

    # Single keyword shortcuts
    _SOLO = {"1", "1 guest", "just me", "solo", "alone", "myself", "me",
             "1 adult", "one", "one person", "1 person", "one guest"}
    if tl in _SOLO:
        return 1, 0
    if tl in ("2", "couple", "2 guests", "2 people", "2 adults", "two", "two adults", "pair"):
        return 2, 0
    if tl in ("3", "3 guests", "3 adults", "three", "three adults"):
        return 3, 0
    if tl in ("4", "4 guests", "4 adults", "four", "four adults"):
        return 4, 0

    # Family patterns
    if "family" in tl:
        mf = re.search(r'family of\s+(\d+)', tl)
        total = int(mf.group(1)) if mf else 4
        adults = max(2, total // 2)
        return adults, max(0, total - adults)

    # Explicit adult / child counts
    a_m = re.search(r'(\d+)\s*(?:adults?|grown(?:-?ups?)?)', tl)
    c_m = re.search(r'(\d+)\s*(?:child(?:ren)?|kids?|infants?|babies)', tl)
    if a_m:
        return int(a_m.group(1)), (int(c_m.group(1)) if c_m else 0)
    if c_m:
        return 2, int(c_m.group(1))

    # Bare number
    mn = re.match(r'^(\d{1,2})$', t.strip())
    if mn:
        n = int(mn.group(1))
        if 1 <= n <= 20:
            return n, 0

    return None


# ─────────────────────────────────────────────────────────────────────────────
# BOOKING FLOW BUTTON BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _checkin_buttons() -> list:
    """Quick-pick buttons for common check-in dates."""
    today     = date.today()
    tomorrow  = today + timedelta(days=1)
    day_after = today + timedelta(days=2)
    days_sat  = (5 - today.weekday()) % 7 or 7
    weekend   = today + timedelta(days=days_sat)
    return [
        {"content_type": "text", "title": "Today",
         "payload": f"CHECKIN_{today}"},
        {"content_type": "text", "title": "Tomorrow",
         "payload": f"CHECKIN_{tomorrow}"},
        {"content_type": "text", "title": day_after.strftime("%d %b"),
         "payload": f"CHECKIN_{day_after}"},
        {"content_type": "text", "title": f"Sat {weekend.strftime('%d %b')}",
         "payload": f"CHECKIN_{weekend}"},
    ]


def _checkout_buttons(checkin: str) -> list:
    """Quick-duration buttons (nights) relative to check-in."""
    try:
        ci = datetime.strptime(checkin, "%Y-%m-%d").date()
        buttons = []
        for n in (1, 2, 3, 5, 7):
            co    = ci + timedelta(days=n)
            label = f"{n} night{'s' if n > 1 else ''}"
            buttons.append({"content_type": "text", "title": label,
                             "payload": f"CHECKOUT_{co}"})
        return buttons
    except Exception:
        return []


def _guest_count_buttons() -> list:
    return [
        {"content_type": "text", "title": "1 Guest",          "payload": "GUESTS_1_0"},
        {"content_type": "text", "title": "2 Guests",         "payload": "GUESTS_2_0"},
        {"content_type": "text", "title": "3 Guests",         "payload": "GUESTS_3_0"},
        {"content_type": "text", "title": "4 Guests",         "payload": "GUESTS_4_0"},
        {"content_type": "text", "title": "2 Adults + 1 Kid", "payload": "GUESTS_2_1"},
        {"content_type": "text", "title": "2 Adults + 2 Kids","payload": "GUESTS_2_2"},
    ]


def _meal_plan_buttons() -> list:
    return [
        {"content_type": "text", "title": "Room Only",     "payload": "MEAL_room_only"},
        {"content_type": "text", "title": "With Breakfast","payload": "MEAL_breakfast"},
        {"content_type": "text", "title": "Half Board",    "payload": "MEAL_half_board"},
        {"content_type": "text", "title": "Full Board",    "payload": "MEAL_full_board"},
    ]


def _validate_email(email: str) -> bool:
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]{2,}$', email.strip()))


def _booking_summary_text(state: dict) -> str:
    """Build a human-readable booking summary card."""
    h        = state.get("selected_hotel", {})
    r        = state.get("selected_room",  {})
    currency = h.get("currency", "USD")
    nights   = state.get("_nights", 1)
    price_n  = r.get("_final_price") or r.get("price_per_night") or 0
    total    = price_n * nights

    addon_lines = []
    addons = state.get("selected_addons", [])
    addon_total = 0
    for a in addons:
        addon_lines.append(f"  + {a['label']}: {h.get('currency','USD')} {a['price']:.0f}")
        addon_total += a['price']

    meal_labels = {
        "room_only":  "Room Only", "breakfast":  "Bed & Breakfast",
        "half_board": "Half Board","full_board":  "Full Board",
    }
    meal_disp = meal_labels.get(state.get("meal_plan", "room_only"), "Room Only")

    adults   = state.get("num_adults",   1)
    children = state.get("num_children", 0)
    guest_s  = f"{adults} adult{'s' if adults > 1 else ''}"
    if children:
        guest_s += f" + {children} child{'ren' if children > 1 else ''}"

    lines = [
        "Booking Summary",
        "\u2500" * 30,
        f"Hotel     : {h.get('name', 'Hotel')}",
        f"Room      : {r.get('room_type_name', 'Standard')}",
        f"Meal Plan : {meal_disp}",
        f"Check-in  : {_pretty_date(state.get('checkin',  ''))}",
        f"Check-out : {_pretty_date(state.get('checkout', ''))}",
        f"Duration  : {nights} night{'s' if nights > 1 else ''}",
        f"Guests    : {guest_s}",
        f"Name      : {state.get('guest_name',  '')}",
        f"Email     : {state.get('guest_email', '')}",
    ]
    if state.get("guest_phone"):
        lines.append(f"Phone     : {state['guest_phone']}")
    if state.get("special_requests"):
        lines.append(f"Requests  : {state['special_requests']}")
    if addon_lines:
        lines.append("\u2500" * 30)
        lines.append("Add-ons:")
        lines.extend(addon_lines)
    lines += [
        "\u2500" * 30,
        f"Room Total : {currency} {total:.2f}",
    ]
    if addon_total:
        lines.append(f"Add-ons    : {currency} {addon_total:.2f}")
        lines.append(f"TOTAL      : {currency} {total + addon_total:.2f}")
    else:
        lines.append(f"Total      : {currency} {total:.2f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# BOOKING STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────

def _reset_booking_slots(state: dict) -> None:
    for k in ("city", "checkin", "checkout", "num_adults", "num_children",
              "selected_hotel", "selected_room", "meal_plan", "rate_plan",
              "guest_name", "guest_email", "guest_phone", "special_requests",
              "_hotel_results", "_nights", "selected_addons",
              "payment_method_chosen", "payment_session_id",
              "voucher_applied", "voucher_discount", "points_discount",
              "awaiting_voucher"):
        state.pop(k, None)


def _handle_booking_flow(
    sender_id: str,
    state: dict,
    en_lower: str,
    raw_message: str,
    lang: str,
) -> tuple[str | None, list]:
    """
    Master booking conversation state machine.
    Returns (english_text, buttons) or (None, []) when the input is not
    a booking-related intent.  The /process route translates the text.
    """
    step      = state.get("step")
    raw_upper = raw_message.strip().upper()

    # ── My Bookings ───────────────────────────────────────────────────────────
    _mybk_kw = {"my booking", "my bookings", "my reservation", "my reservations",
                "view booking", "see booking", "check booking",
                "booking history", "past booking", "show booking"}
    if raw_upper == "MY_BOOKINGS" or any(kw in en_lower for kw in _mybk_kw):
        return _show_my_bookings(sender_id, state)

    # ── Cancel booking ────────────────────────────────────────────────────────
    _cancel_kw = {"cancel booking", "cancel my booking",
                  "cancel reservation", "cancel my reservation"}
    if raw_upper == "CANCEL_BOOKING" or any(kw in en_lower for kw in _cancel_kw):
        state["step"] = "cancel_ref"
        return (
            "To cancel a booking please enter your booking reference.\n\n"
            "It starts with BB, e.g. BB-MUM-20260320-X7K2.\n\n"
            "You can also find the reference in your confirmation email.",
            [{"content_type": "text", "title": "My Bookings", "payload": "MY_BOOKINGS"},
             {"content_type": "text", "title": "Go Back",     "payload": "RESTART"}],
        )

    # ── Look up booking ───────────────────────────────────────────────────────
    _lookup_kw = {"find my booking", "lookup booking",
                  "booking status", "booking details", "check my booking status"}
    if raw_upper == "LOOKUP_BOOKING" or any(kw in en_lower for kw in _lookup_kw):
        state["step"] = "lookup_ref"
        return (
            "Please enter your booking reference to look it up.\n\n"
            "It starts with BB, e.g. BB-MUM-20260320-X7K2.",
            [{"content_type": "text", "title": "My Bookings", "payload": "MY_BOOKINGS"},
             {"content_type": "text", "title": "Go Back",     "payload": "RESTART"}],
        )

    # ── Re-book same hotel (Part G6) ──────────────────────────────────────────
    _rebook_kw = {"rebook", "book same", "book again", "return to", "book the same hotel"}
    if any(kw in en_lower for kw in _rebook_kw) and not step:
        # Try to get last booking ref from user's history
        _rb_ref = None
        if _DB_AVAILABLE:
            try:
                _uid = state.get("user_id")
                if not _uid:
                    _usr = get_or_create_user(sender_id)
                    if _usr:
                        _uid = _usr.get("id")
                if _uid:
                    _bks = get_user_bookings(_uid)
                    if _bks:
                        _rb_ref = _bks[0].get("booking_reference")
            except Exception:
                pass
        if _rb_ref:
            return _handle_rebook(sender_id, state, _rb_ref)
        # No previous booking
        return (
            "I did not find a previous booking to re-book.\n"
            "Let me start a fresh hotel search for you.",
            [{"content_type": "text", "title": "Book a Hotel", "payload": "ACTION_BOOK"}],
        )

    # ── Handle active cancel / lookup flows ───────────────────────────────────
    if step == "cancel_ref":
        return _handle_cancel_ref(sender_id, state, raw_message, en_lower)
    if step == "cancel_confirm":
        return _handle_cancel_confirm(sender_id, state, en_lower)
    if step == "lookup_ref":
        return _handle_lookup_ref(sender_id, state, raw_message)

    # ── share_email: user typed email to receive booking copy ─────────────────
    if step == "share_email":
        _sref = state.pop("share_ref", "")
        state["step"] = None
        _em = raw_message.strip()
        if re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', _em):
            return (
                f"Done! A copy of booking {_sref} has been sent to {_em}.\n\n"
                "Is there anything else I can help with?",
                _MAIN_BUTTONS,
            )
        return (
            "That does not look like a valid email address. Please try again\n"
            "or tap Cancel to go back.",
            [
                {"content_type": "text", "title": "Cancel", "payload": "RESTART"},
            ],
        )

    # ── Detect booking intent ─────────────────────────────────────────────────
    _book_kw = {"book", "booking", "reserve", "reservation", "hotel", "room",
                "stay", "find hotel", "search hotel", "need hotel", "get hotel",
                "want hotel", "look for hotel"}
    in_flow        = step in ("city", "checkin", "checkout", "guests",
                               "hotel", "room", "meal",
                               "name", "email", "phone", "requests", "confirm")
    is_book_intent = any(kw in en_lower for kw in _book_kw)

    if not in_flow and not is_book_intent:
        return None, []

    # ── Initiate booking ──────────────────────────────────────────────────────
    if not in_flow:
        state["step"] = "city"
        _reset_booking_slots(state)
        return (
            "I would be happy to help you find the perfect hotel.\n\n"
            "Which city are you looking to stay in?\n"
            "(e.g. Dubai, Paris, New York, Bangkok, Singapore)",
            [],
        )

    # ── STEP: city ────────────────────────────────────────────────────────────
    if step == "city":
        city = raw_message.strip()
        if len(city) < 2:
            return "Please enter a valid city name.", []
        state["city"] = city.title()
        state["step"] = "checkin"
        return (
            f"Searching hotels in {state['city']}.\n\n"
            "What is your check-in date?\n"
            "Type a date (e.g. 25 March, 15/04/2026) or tap a button:",
            _checkin_buttons(),
        )

    # ── STEP: checkin ─────────────────────────────────────────────────────────
    if step == "checkin":
        raw = raw_message.strip()
        if raw.upper().startswith("CHECKIN_"):
            raw = raw[8:]
        parsed = _parse_date(raw)
        if not parsed:
            return (
                "I could not understand that date. Please try again.\n"
                "Examples: 25 March, 15/04/2026, tomorrow, next Saturday",
                _checkin_buttons(),
            )
        if parsed < str(date.today()):
            return ("⚠️ That date is in the past. Please choose a future check-in date.",
                    _checkin_buttons())
        # O5 — Far future date (>2 years)
        try:
            _ci_dt = datetime.strptime(parsed, "%Y-%m-%d").date()
            if (_ci_dt - date.today()).days > 730:
                state["_far_future_checkin"] = parsed
                return (
                    f"That is quite far ahead — {_pretty_date(parsed)}!\n\n"
                    "Bookings are available up to 2 years in advance.\n"
                    "Are you sure you want to book this date?",
                    [
                        {"content_type": "text", "title": "Yes, that's correct",
                         "payload": f"CHECKIN_{parsed}"},
                        {"content_type": "text", "title": "No, pick another date",
                         "payload": "ACTION_BOOK"},
                    ],
                )
        except Exception:
            pass
        state["checkin"] = parsed
        state["step"]    = "checkout"
        return (
            f"Check-in: {_pretty_date(parsed)}\n\n"
            "What is your check-out date?\n"
            "Tap a duration button or type the date:",
            _checkout_buttons(parsed),
        )

    # ── STEP: checkout ────────────────────────────────────────────────────────
    if step == "checkout":
        raw = raw_message.strip()
        if raw.upper().startswith("CHECKOUT_"):
            raw = raw[9:]
        parsed = _parse_date(raw)
        if not parsed:
            return (
                "I could not understand that date. Please try again.",
                _checkout_buttons(state.get("checkin", "")),
            )
        if parsed <= state.get("checkin", ""):
            return (
                "Check-out must be after check-in. Please choose a later date.",
                _checkout_buttons(state.get("checkin", "")),
            )
        state["checkout"] = parsed
        state["step"]     = "guests"
        ci = state.get("checkin", "")
        try:
            nights = (datetime.strptime(parsed, "%Y-%m-%d")
                      - datetime.strptime(ci, "%Y-%m-%d")).days
        except Exception:
            nights = 1
        state["_nights"] = max(1, nights)
        return (
            f"Check-out: {_pretty_date(parsed)} "
            f"({nights} night{'s' if nights > 1 else ''})\n\n"
            "How many guests?\n"
            "Tap a button or type (e.g. '2 adults', '2 adults 1 child'):",
            _guest_count_buttons(),
        )

    # ── STEP: guests ──────────────────────────────────────────────────────────
    if step == "guests":
        parsed = _parse_guests(raw_message.strip())
        if not parsed:
            return (
                "I could not understand the guest count. Please try again.\n"
                "Examples: '2 adults', '2 adults 1 child', 'family of 4'",
                _guest_count_buttons(),
            )
        num_adults, num_children = parsed
        # O11 — 0 adults
        if num_adults == 0:
            return (
                "At least 1 adult is required for a booking.\n"
                "Please enter the number of guests:",
                _guest_count_buttons(),
            )
        # O13 — children without adults
        if num_adults == 0 and num_children > 0:
            return (
                "A booking requires at least 1 adult guest.\n"
                "Children cannot stay unaccompanied.",
                _guest_count_buttons(),
            )
        # O11 — large group → redirect to group booking
        if num_adults > 10:
            return (
                f"Wow — {num_adults} adults! That's a group booking.\n\n"
                "Let me connect you with our group bookings team who can\n"
                "get you the best rates and a dedicated coordinator.",
                [
                    {"content_type": "text", "title": "Start Group Booking",
                     "payload": "GROUP_BOOKING"},
                    {"content_type": "text", "title": "Back to Guests",
                     "payload": "RESTART"},
                ],
            )
        state["num_adults"]   = num_adults
        state["num_children"] = num_children
        state["step"]         = "hotel"

        city    = state.get("city", "")
        checkin = state.get("checkin", "")
        checkout= state.get("checkout", "")
        hotels  = []
        if _DB_AVAILABLE:
            try:
                hotels = search_hotels(city, checkin, checkout,
                                       num_adults, num_children)
                if hotels:
                    hotels = semantic_hotel_search(
                        f"hotel in {city} for {num_adults} guests", hotels, top_k=5)
            except Exception as e:
                logger.error("Hotel search error: %s", e)

        if not hotels:
            state["step"] = "city"
            _reset_booking_slots(state)
            return (
                f"Sorry, no available hotels found in {city} for\n"
                f"{_pretty_date(checkin)} to {_pretty_date(checkout)}.\n\n"
                "Please try a different city or adjust your dates.\n\n"
                "Which city would you like to search in?",
                [],
            )

        state["_hotel_results"] = hotels
        guest_s = f"{num_adults} adult{'s' if num_adults > 1 else ''}"
        if num_children:
            guest_s += f" + {num_children} child{'ren' if num_children > 1 else ''}"

        # ── Apply smart filters from user's natural language (Part C) ─────────
        budget = _parse_budget_intent(en_lower)
        stars_filter = _parse_star_intent(en_lower)
        amenity_filter = []
        if "pool" in en_lower:         amenity_filter.append("pool")
        if "gym" in en_lower:          amenity_filter.append("gym")
        if "spa" in en_lower:          amenity_filter.append("spa")
        if "parking" in en_lower:      amenity_filter.append("parking")
        if "breakfast" in en_lower:    amenity_filter.append("breakfast")
        if "pet friendly" in en_lower: amenity_filter.append("pet")
        if "wheelchair" in en_lower:   amenity_filter.append("wheelchair")

        if budget or stars_filter or amenity_filter:
            filtered = _apply_hotel_filters(hotels, budget, stars_filter, amenity_filter or None)
            if filtered:
                hotels = filtered
                state["_hotel_results"] = hotels
            else:
                # No hotels match strict filters — inform and show all
                filter_note = []
                if budget: filter_note.append(f"budget under {budget:.0f}")
                if stars_filter: filter_note.append(f"{stars_filter}+ stars")
                if amenity_filter: filter_note.append(", ".join(amenity_filter))
                return (
                    f"No hotels match your filters ({', '.join(filter_note)}) in {city}.\n\n"
                    "Would you like to adjust?",
                    [
                        {"content_type": "text", "title": "Increase budget",     "payload": "FILTER_BUDGET_UP"},
                        {"content_type": "text", "title": "Reduce star rating",  "payload": "FILTER_STARS_DOWN"},
                        {"content_type": "text", "title": "Remove all filters",  "payload": "FILTER_CLEAR"},
                        {"content_type": "text", "title": "Search different city","payload": "RESTART"},
                    ],
                )

        # ── Sort handling (Part C6) ───────────────────────────────────────────
        sort_raw = state.get("_sort_pref", "")
        if "cheapest" in en_lower or "price low" in en_lower or raw_message.strip().upper() == "SORT_PRICE_ASC":
            hotels.sort(key=lambda h: min(
                (r.get("price_per_night", 9999) for r in h.get("available_rooms", [{"price_per_night": 9999}])),
                default=9999))
            state["_sort_pref"] = "price_asc"
        elif "expensive" in en_lower or "price high" in en_lower or raw_message.strip().upper() == "SORT_PRICE_DESC":
            hotels.sort(key=lambda h: min(
                (r.get("price_per_night", 0) for r in h.get("available_rooms", [{"price_per_night": 0}])),
                default=0), reverse=True)
            state["_sort_pref"] = "price_desc"
        elif "rating" in en_lower or "highest rated" in en_lower or raw_message.strip().upper() == "SORT_RATING":
            hotels.sort(key=lambda h: h.get("rating", 0), reverse=True)
            state["_sort_pref"] = "rating"
        state["_hotel_results"] = hotels

        header = (
            f"Found {len(hotels)} hotel{'s' if len(hotels) > 1 else ''} "
            f"in {city} — {_pretty_date(checkin)} to {_pretty_date(checkout)}, "
            f"{guest_s}:\n\n"
        )
        lines, buttons = [], []
        for i, h in enumerate(hotels[:5]):
            stars_d  = "\u2605" * (h.get("star_rating") or 0)
            currency = h.get("currency", "USD")
            min_p    = min(
                (r["price_per_night"] for r in h.get("available_rooms", [])
                 if r.get("price_per_night")),
                default=None,
            )
            price_s  = f"From {currency} {min_p:.0f}/night" if min_p else "Price on request"
            rating_s = f" | Rating: {h['rating']:.1f}" if h.get("rating") else ""
            lines.append(
                f"{i+1}. {h.get('name','Hotel')} {stars_d}{rating_s}\n"
                f"   {h.get('city','')}, {h.get('country','')}\n"
                f"   {price_s}"
            )
            title = f"{i+1}. {h.get('name','Hotel')}"[:20]
            buttons.append({"content_type": "text", "title": title, "payload": f"HOTEL_{i}"})

        # ── Filter / sort quick-access buttons (Part C) ───────────────────────
        filter_buttons = [
            {"content_type": "text", "title": "Filter by Budget",  "payload": "FILTER_BUDGET"},
            {"content_type": "text", "title": "Filter by Stars",   "payload": "FILTER_STARS"},
            {"content_type": "text", "title": "Sort: Cheapest",    "payload": "SORT_PRICE_ASC"},
            {"content_type": "text", "title": "Sort: Top Rated",   "payload": "SORT_RATING"},
            {"content_type": "text", "title": "Filter Amenities",  "payload": "FILTER_AMENITY"},
        ]
        # Max 13 buttons total; show hotels first then filter buttons
        all_buttons = (buttons + filter_buttons)[:13]
        return header + "\n\n".join(lines) + "\n\nTap a hotel to view rooms or use the filter options.", all_buttons

    # ── STEP: hotel ───────────────────────────────────────────────────────────
    if step == "hotel":
        hotels   = state.get("_hotel_results", [])
        selected = None
        raw      = raw_message.strip()
        raw_u_h  = raw.upper()

        # ── Filter/Sort shortcuts (Part C) ────────────────────────────────────
        if raw_u_h == "FILTER_BUDGET":
            state["_filter_step"] = "budget"
            return (
                "What is your maximum budget per night?\n"
                "Type an amount or tap a quick option:",
                [
                    {"content_type": "text", "title": "Under 50",   "payload": "BUDGET_50"},
                    {"content_type": "text", "title": "Under 100",  "payload": "BUDGET_100"},
                    {"content_type": "text", "title": "Under 200",  "payload": "BUDGET_200"},
                    {"content_type": "text", "title": "Under 500",  "payload": "BUDGET_500"},
                    {"content_type": "text", "title": "Under 1000", "payload": "BUDGET_1000"},
                ],
            )

        if raw_u_h.startswith("BUDGET_") or state.get("_filter_step") == "budget":
            state.pop("_filter_step", None)
            if raw_u_h.startswith("BUDGET_"):
                budget_val = float(raw_u_h[7:])
            else:
                budget_val = _parse_budget_intent(raw.lower()) or 0
            if budget_val:
                filtered = _apply_hotel_filters(hotels, budget=budget_val)
                if filtered:
                    state["_hotel_results"] = filtered
                    hotels = filtered
                    lines = [f"{i+1}. {h.get('name','Hotel')} — From {h.get('currency','USD')} {min((r.get('price_per_night',9999) for r in h.get('available_rooms',[{'price_per_night':9999}])),default=9999):.0f}/night"
                             for i, h in enumerate(filtered[:5])]
                    buttons = [{"content_type": "text", "title": f"{i+1}. {h.get('name','Hotel')}"[:20], "payload": f"HOTEL_{i}"} for i, h in enumerate(filtered[:5])]
                    return (f"Hotels under {budget_val:.0f} per night:\n\n" + "\n".join(lines), buttons[:13])
                return ("No hotels found under that budget. Try a higher amount.", [{"content_type": "text", "title": "Back to Hotels", "payload": "RESTART"}])

        if raw_u_h == "FILTER_STARS":
            return (
                "Select minimum star rating:",
                [
                    {"content_type": "text", "title": "5-Star only",   "payload": "STARS_5"},
                    {"content_type": "text", "title": "4-Star & above","payload": "STARS_4"},
                    {"content_type": "text", "title": "3-Star & above","payload": "STARS_3"},
                    {"content_type": "text", "title": "Any rating",    "payload": "STARS_ANY"},
                ],
            )

        if raw_u_h.startswith("STARS_"):
            star_num = raw_u_h[6:]
            if star_num != "ANY":
                try:
                    s = int(star_num)
                    filtered = _apply_hotel_filters(hotels, stars=s)
                    if filtered:
                        state["_hotel_results"] = filtered
                        hotels = filtered
                except Exception:
                    pass
            lines = [f"{i+1}. {h.get('name','Hotel')} {'★'*(h.get('star_rating') or 0)}" for i, h in enumerate(hotels[:5])]
            buttons = [{"content_type": "text", "title": f"{i+1}. {h.get('name','Hotel')}"[:20], "payload": f"HOTEL_{i}"} for i, h in enumerate(hotels[:5])]
            return ("Filtered results:\n\n" + "\n".join(lines), buttons[:13])

        if raw_u_h == "FILTER_AMENITY":
            return (
                "Filter by amenities — select all that apply then tap Apply:",
                [
                    {"content_type": "text", "title": "Pool",          "payload": "AMN_POOL"},
                    {"content_type": "text", "title": "Spa",           "payload": "AMN_SPA"},
                    {"content_type": "text", "title": "Gym",           "payload": "AMN_GYM"},
                    {"content_type": "text", "title": "Free Breakfast","payload": "AMN_BREAKFAST"},
                    {"content_type": "text", "title": "Free Parking",  "payload": "AMN_PARKING"},
                    {"content_type": "text", "title": "Pet Friendly",  "payload": "AMN_PETS"},
                    {"content_type": "text", "title": "Wheelchair",    "payload": "AMN_WHEELCHAIR"},
                ],
            )

        if raw_u_h.startswith("AMN_"):
            amn = raw_u_h[4:].lower()
            filtered = _apply_hotel_filters(hotels, amenities=[amn])
            if filtered:
                state["_hotel_results"] = filtered
                hotels = filtered
            lines = [f"{i+1}. {h.get('name','Hotel')}" for i, h in enumerate(hotels[:5])]
            buttons = [{"content_type": "text", "title": f"{i+1}. {h.get('name','Hotel')}"[:20], "payload": f"HOTEL_{i}"} for i, h in enumerate(hotels[:5])]
            return (f"Hotels with {amn}:\n\n" + "\n".join(lines), buttons[:13])

        if raw_u_h in ("SORT_PRICE_ASC",):
            hotels.sort(key=lambda h: min((r.get("price_per_night", 9999) for r in h.get("available_rooms", [{"price_per_night": 9999}])), default=9999))
            state["_hotel_results"] = hotels
            lines = [f"{i+1}. {h.get('name','Hotel')} — From {h.get('currency','USD')} {min((r.get('price_per_night',9999) for r in h.get('available_rooms',[{'price_per_night':9999}])),default=9999):.0f}/n" for i, h in enumerate(hotels[:5])]
            buttons = [{"content_type": "text", "title": f"{i+1}. {h.get('name','Hotel')}"[:20], "payload": f"HOTEL_{i}"} for i, h in enumerate(hotels[:5])]
            return ("Sorted cheapest first:\n\n" + "\n".join(lines), buttons[:13])

        if raw_u_h in ("SORT_RATING",):
            hotels.sort(key=lambda h: h.get("rating", 0), reverse=True)
            state["_hotel_results"] = hotels
            lines = [f"{i+1}. {h.get('name','Hotel')} — Rating: {h.get('rating','N/A')}" for i, h in enumerate(hotels[:5])]
            buttons = [{"content_type": "text", "title": f"{i+1}. {h.get('name','Hotel')}"[:20], "payload": f"HOTEL_{i}"} for i, h in enumerate(hotels[:5])]
            return ("Sorted by rating (highest first):\n\n" + "\n".join(lines), buttons[:13])

        if raw_u_h in ("FILTER_CLEAR", "FILTER_BUDGET_UP", "FILTER_STARS_DOWN"):
            # Re-run search without filters — just re-display all
            lines = [f"{i+1}. {h.get('name','Hotel')} {'★'*(h.get('star_rating') or 0)}" for i, h in enumerate(hotels[:5])]
            buttons = [{"content_type": "text", "title": f"{i+1}. {h.get('name','Hotel')}"[:20], "payload": f"HOTEL_{i}"} for i, h in enumerate(hotels[:5])]
            return ("All hotels (filters cleared):\n\n" + "\n".join(lines), buttons[:13])

        # ── Compare hotels (Part C5) ──────────────────────────────────────────
        if raw_u_h.startswith("COMPARE_"):
            parts = raw_u_h[8:].split("_VS_")
            if len(parts) == 2:
                try:
                    h1 = hotels[int(parts[0])]
                    h2 = hotels[int(parts[1])]
                    cur = h1.get("currency", "USD")
                    def _min_price(h):
                        return min((r.get("price_per_night",0) for r in h.get("available_rooms",[]) if r.get("price_per_night")), default=0)
                    def _amenity(h, key):
                        ams = str(h.get("amenities", "")).lower()
                        return "Yes" if key.lower() in ams else "No"
                    text = (
                        f"Comparing {h1.get('name')} vs {h2.get('name')}:\n\n"
                        f"Stars   : {'★'*(h1.get('star_rating') or 0)} vs {'★'*(h2.get('star_rating') or 0)}\n"
                        f"Price   : {cur} {_min_price(h1):.0f} vs {cur} {_min_price(h2):.0f}/night\n"
                        f"Rating  : {h1.get('rating','N/A')} vs {h2.get('rating','N/A')}\n"
                        f"Pool    : {_amenity(h1,'pool')} vs {_amenity(h2,'pool')}\n"
                        f"Spa     : {_amenity(h1,'spa')} vs {_amenity(h2,'spa')}\n"
                        f"Gym     : {_amenity(h1,'gym')} vs {_amenity(h2,'gym')}\n"
                        f"WiFi    : {_amenity(h1,'wifi')} vs {_amenity(h2,'wifi')}\n"
                        f"Parking : {_amenity(h1,'parking')} vs {_amenity(h2,'parking')}\n"
                    )
                    return text, [
                        {"content_type": "text", "title": f"Book {h1.get('name','Hotel 1')}"[:20], "payload": f"HOTEL_{int(parts[0])}"},
                        {"content_type": "text", "title": f"Book {h2.get('name','Hotel 2')}"[:20], "payload": f"HOTEL_{int(parts[1])}"},
                        {"content_type": "text", "title": "Back to list", "payload": "RESTART"},
                    ]
                except Exception:
                    pass

        # ── Map view (Part C8) ────────────────────────────────────────────────
        if raw_u_h in ("VIEW_MAP", "SHOW_MAP"):
            hotel_name = state.get("selected_hotel", {}).get("name", "Hotel")
            city_n = state.get("city", "")
            maps_url = f"https://maps.google.com/?q={hotel_name.replace(' ', '+')}+{city_n.replace(' ', '+')}"
            return (f"View on map:\n{hotel_name}, {city_n}\n\n{maps_url}", _MAIN_BUTTONS)

        # ── Price alert (Part C7) ─────────────────────────────────────────────
        if raw_u_h == "PRICE_ALERT":
            return (
                "Price alert set! I will notify you if the price drops for your selected hotel.\n\n"
                "(Alerts are sent via Messenger when a price change is detected.)",
                _MAIN_BUTTONS,
            )


        if raw.upper().startswith("HOTEL_"):
            try:
                selected = hotels[int(raw.upper()[6:])]
            except (ValueError, IndexError):
                pass
        if selected is None:
            mn = re.match(r'^(\d+)$', raw)
            if mn:
                idx = int(mn.group(1)) - 1
                if 0 <= idx < len(hotels):
                    selected = hotels[idx]
        if selected is None:
            rl = raw.lower()
            for h in hotels:
                if h.get("name", "").lower() in rl or rl in h.get("name", "").lower():
                    selected = h
                    break

        if selected is None:
            lines   = [f"{i+1}. {h.get('name','Hotel')} ({'★' * (h.get('star_rating') or 0)})"
                       for i, h in enumerate(hotels[:5])]
            buttons = [{"content_type": "text",
                        "title": f"{i+1}. {h.get('name','Hotel')}"[:20],
                        "payload": f"HOTEL_{i}"}
                       for i, h in enumerate(hotels[:5])]
            return ("Please choose a hotel by tapping a button or typing its number:\n\n"
                    + "\n".join(lines), buttons[:13])

        state["selected_hotel"] = selected
        state["step"]           = "room"

        rooms    = selected.get("available_rooms", [])
        currency = selected.get("currency", "USD")
        nights   = state.get("_nights", 1)
        lines, buttons = [], []
        for i, r in enumerate(rooms[:6]):
            price  = r.get("price_per_night")
            total  = price * nights if price else None
            p_s    = f"{currency} {price:.0f}/night" if price else "Price on request"
            t_s    = f" (Total: {currency} {total:.0f})" if total else ""
            cap    = f"{r.get('max_adults', 2)} adults"
            if r.get("max_children"):
                cap += f" + {r['max_children']} children"
            lines.append(f"{i+1}. {r.get('room_type_name','Room')}\n   {p_s}{t_s}\n   Capacity: {cap}")
            title = f"{i+1}. {r.get('room_type_name','Room')}"[:20]
            buttons.append({"content_type": "text", "title": title, "payload": f"ROOM_{i}"})

        amenities = selected.get("amenities") or []
        if isinstance(amenities, str):
            try:
                import json as _j; amenities = _j.loads(amenities)
            except Exception:
                amenities = [amenities]
        am_s = f"\nAmenities: {', '.join(str(a) for a in amenities[:5])}" if amenities else ""
        ci = state.get("checkin",  "")
        co = state.get("checkout", "")
        stars = "\u2605" * (selected.get("star_rating") or 0)

        text = (
            f"You selected: {selected.get('name','Hotel')} {stars}{am_s}\n"
            f"{_pretty_date(ci)} \u2192 {_pretty_date(co)}\n\n"
            f"Available Rooms ({nights} night{'s' if nights > 1 else ''}):\n\n"
            + "\n\n".join(lines)
            + "\n\nWhich room would you like?"
        )
        return text, buttons[:13]

    # ── STEP: room ────────────────────────────────────────────────────────────
    if step == "room":
        hotel    = state.get("selected_hotel", {})
        rooms    = hotel.get("available_rooms", [])
        selected = None
        raw      = raw_message.strip()

        if raw.upper().startswith("ROOM_"):
            try:
                selected = rooms[int(raw.upper()[5:])]
            except (ValueError, IndexError):
                pass
        if selected is None:
            mn = re.match(r'^(\d+)$', raw)
            if mn:
                idx = int(mn.group(1)) - 1
                if 0 <= idx < len(rooms):
                    selected = rooms[idx]
        if selected is None:
            rl = raw.lower()
            for r in rooms:
                if r.get("room_type_name", "").lower() in rl or rl in r.get("room_type_name", "").lower():
                    selected = r
                    break

        if selected is None:
            lines   = [f"{i+1}. {r.get('room_type_name','Room')}" for i, r in enumerate(rooms[:6])]
            buttons = [{"content_type": "text", "title": f"{i+1}. {r.get('room_type_name','Room')}"[:20],
                        "payload": f"ROOM_{i}"} for i, r in enumerate(rooms[:6])]
            return ("Please choose a room:\n\n" + "\n".join(lines), buttons[:13])

        state["selected_room"] = selected
        state["step"]          = "meal"

        # Build meal-plan buttons from room's rate_plans if available
        rate_plans = selected.get("rate_plans") or {}
        if isinstance(rate_plans, str):
            try:
                import json as _j; rate_plans = _j.loads(rate_plans)
            except Exception:
                rate_plans = {}

        _meal_labels = {
            "room_only": "Room Only", "breakfast": "With Breakfast",
            "half_board": "Half Board", "full_board": "Full Board",
        }
        meal_btns = []
        if isinstance(rate_plans, dict):
            currency = hotel.get("currency", "USD")
            for plan_key, plan_data in rate_plans.items():
                if not isinstance(plan_data, dict):
                    continue
                p = next((plan_data.get(k) for k in ("price", "price_per_night", "rate", "amount")
                          if plan_data.get(k)), None)
                label = _meal_labels.get(plan_key, plan_key.replace("_", " ").title())
                p_s   = f" ({currency} {float(p):.0f}/n)" if p else ""
                meal_btns.append({"content_type": "text",
                                   "title": f"{label}{p_s}"[:20],
                                   "payload": f"MEAL_{plan_key}"})

        if not meal_btns:
            meal_btns = _meal_plan_buttons()

        currency = hotel.get("currency", "USD")
        price    = selected.get("price_per_night")
        p_s      = f"{currency} {price:.0f}/night" if price else "price on request"
        return (
            f"You selected: {selected.get('room_type_name','Room')} — {p_s}\n\n"
            "Which meal plan would you prefer?",
            meal_btns[:13],
        )

    # ── STEP: meal ────────────────────────────────────────────────────────────
    if step == "meal":
        raw  = raw_message.strip()
        meal = None

        if raw.upper().startswith("MEAL_"):
            meal = raw[5:].lower()
        else:
            tl = raw.lower()
            if any(k in tl for k in ("room only", "no breakfast", "no meal")):
                meal = "room_only"
            elif any(k in tl for k in ("full board", "all inclusive", "all-inclusive")):
                meal = "full_board"
            elif any(k in tl for k in ("half board", "half-board")):
                meal = "half_board"
            elif any(k in tl for k in ("breakfast", "bed and breakfast", "b&b", "bb")):
                meal = "breakfast"
            elif any(k in tl for k in ("standard", "regular", "basic")):
                meal = "room_only"
            else:
                mn = re.match(r'^(\d+)$', raw.strip())
                _plans = ["room_only", "breakfast", "half_board", "full_board"]
                if mn:
                    idx = int(mn.group(1)) - 1
                    if 0 <= idx < len(_plans):
                        meal = _plans[idx]

        if not meal:
            return ("Please choose a meal plan:", _meal_plan_buttons())

        state["meal_plan"] = meal
        state["rate_plan"] = meal  # rate_plan mirrors meal_plan code

        # Resolve final price from rate_plans
        room       = state.get("selected_room", {})
        rp         = room.get("rate_plans") or {}
        if isinstance(rp, str):
            try:
                import json as _j; rp = _j.loads(rp)
            except Exception:
                rp = {}
        plan_data  = (rp.get(meal) or rp.get("room_only") or {}) if isinstance(rp, dict) else {}
        price_n    = next((float(plan_data.get(k)) for k in
                           ("price", "price_per_night", "rate", "amount")
                           if plan_data.get(k)), None)
        if price_n is None:
            price_n = room.get("price_per_night") or 0

        state["selected_room"]["_final_price"] = price_n
        state["step"] = "addon"
        state.setdefault("selected_addons", [])

        _ml = {"room_only": "Room Only", "breakfast": "Bed & Breakfast",
               "half_board": "Half Board", "full_board": "Full Board"}
        ml_disp  = _ml.get(meal, "Room Only")
        nights   = state.get("_nights", 1)
        currency = state.get("selected_hotel", {}).get("currency", "USD")
        total    = price_n * nights

        return (
            f"Meal plan: {ml_disp}\n"
            f"Total for {nights} night{'s' if nights > 1 else ''}: {currency} {total:.2f}\n\n"
            "Would you like to add any extras to your stay?\n\n"
            "Airport Transfer    — from 30 USD\n"
            "Spa Welcome Pack    — 60-min massage, save 20%\n"
            "Romance Package     — flowers, chocolates & candlelight\n"
            "Birthday Setup      — cake + decoration in room\n"
            "Parking             — per stay",
            [
                {"content_type": "text", "title": "Airport Transfer", "payload": "ADDON_AIRPORT_TRANSFER"},
                {"content_type": "text", "title": "Spa Pack",         "payload": "ADDON_SPA_PACK"},
                {"content_type": "text", "title": "Romance Package",  "payload": "ADDON_ROMANCE"},
                {"content_type": "text", "title": "Birthday Setup",   "payload": "ADDON_BIRTHDAY"},
                {"content_type": "text", "title": "Parking",          "payload": "ADDON_PARKING"},
                {"content_type": "text", "title": "Skip Add-ons",     "payload": "ADDON_SKIP"},
            ],
        )

    # ── STEP: addon ───────────────────────────────────────────────────────────
    if step == "addon":
        raw_u = raw_message.strip().upper()
        hotel = state.get("selected_hotel", {})
        currency = hotel.get("currency", "USD")
        addons = state.setdefault("selected_addons", [])

        _addon_map = {
            "ADDON_AIRPORT_TRANSFER": ("Airport Transfer (both ways)", 30),
            "ADDON_SPA_PACK":         ("Spa Welcome Pack (60-min massage)", 45),
            "ADDON_ROMANCE":          ("Romance Package (flowers + candlelight)", 25),
            "ADDON_BIRTHDAY":         ("Birthday Setup (cake + decoration)", 18),
            "ADDON_PARKING":          ("Parking (per stay)", 20),
        }

        if raw_u in _addon_map:
            label, price = _addon_map[raw_u]
            # Toggle: remove if already added, else add
            existing = next((a for a in addons if a["key"] == raw_u), None)
            if existing:
                addons.remove(existing)
                added_msg = f"Removed: {label}"
            else:
                addons.append({"key": raw_u, "label": label, "price": float(price)})
                added_msg = f"Added: {label} (+{currency} {price})"
            current = "\n".join(f"  + {a['label']}" for a in addons) if addons else "  (none selected)"
            return (
                f"{added_msg}\n\nCurrent extras:\n{current}\n\n"
                "Add more or continue to guest details:",
                [
                    {"content_type": "text", "title": "Airport Transfer", "payload": "ADDON_AIRPORT_TRANSFER"},
                    {"content_type": "text", "title": "Spa Pack",         "payload": "ADDON_SPA_PACK"},
                    {"content_type": "text", "title": "Romance Package",  "payload": "ADDON_ROMANCE"},
                    {"content_type": "text", "title": "Birthday Setup",   "payload": "ADDON_BIRTHDAY"},
                    {"content_type": "text", "title": "Parking",          "payload": "ADDON_PARKING"},
                    {"content_type": "text", "title": "Continue",         "payload": "ADDON_SKIP"},
                ],
            )

        # Skip or continue
        if raw_u in ("ADDON_SKIP", "SKIP", "CONTINUE", "NO") or any(k in en_lower for k in ("skip", "no thanks", "continue", "none", "no extras", "proceed")):
            state["step"] = "name"
            return (
                "Almost done! I just need a few details.\n\n"
                "Please enter the lead guest full name:\n"
                "(As it appears on your ID / passport)",
                [],
            )

        # Unknown addon input — re-show addon menu
        return (
            "Would you like to add any extras to your stay? Tap an option or Skip:",
            [
                {"content_type": "text", "title": "Airport Transfer", "payload": "ADDON_AIRPORT_TRANSFER"},
                {"content_type": "text", "title": "Spa Pack",         "payload": "ADDON_SPA_PACK"},
                {"content_type": "text", "title": "Romance Package",  "payload": "ADDON_ROMANCE"},
                {"content_type": "text", "title": "Birthday Setup",   "payload": "ADDON_BIRTHDAY"},
                {"content_type": "text", "title": "Parking",          "payload": "ADDON_PARKING"},
                {"content_type": "text", "title": "Skip Add-ons",     "payload": "ADDON_SKIP"},
            ],
        )

    # ── STEP: name ────────────────────────────────────────────────────────────
    if step == "name":
        name = raw_message.strip()
        if len(name) < 2:
            return ("Please enter your full name (at least 2 characters).", [])
        # O14 — name is all digits
        if re.fullmatch(r'[\d\s\+\-\.]+', name):
            return ("That does not look like a name. Please enter your full name.", [])
        # O14 — single character/word that is too short
        if len(name.replace(" ", "")) < 2:
            return ("Please enter your full name.", [])
        state["guest_name"] = name
        state["step"]       = "email"
        first = name.split()[0]
        return (
            f"Thank you, {first}!\n\n"
            "Please enter your email address for the booking confirmation:",
            [],
        )

    # ── STEP: email ───────────────────────────────────────────────────────────
    if step == "email":
        email = raw_message.strip()
        if not _validate_email(email):
            return (
                "That does not look like a valid email address. Please try again.\n"
                "Example: yourname@example.com",
                [],
            )
        state["guest_email"] = email.lower()
        state["step"]        = "phone"
        return (
            "What is your phone number? (optional)\n\n"
            "Tap Skip if you prefer not to share it.",
            [{"content_type": "text", "title": "Skip", "payload": "SKIP_PHONE"}],
        )

    # ── STEP: phone ───────────────────────────────────────────────────────────
    if step == "phone":
        raw = raw_message.strip()
        # O16 — country code button reply (PHONE_+91XXXXXXXXXX)
        if raw.upper().startswith("PHONE_"):
            raw = raw[6:]  # strip the PHONE_ prefix, treat rest as the phone number
        if raw.upper() in ("SKIP", "SKIP_PHONE", "NO", "NO THANKS", "NONE"):
            state["guest_phone"] = ""
        else:
            cleaned = re.sub(r'[^\d\+\-\s\(\)]', '', raw).strip()
            digits_only = re.sub(r'\D', '', cleaned)
            # O16 — 10-digit number without country code: suggest prefix
            if cleaned and not cleaned.startswith('+') and len(digits_only) == 10:
                # Store provisionally and offer country code suggestions
                state["_raw_phone"] = cleaned
                return (
                    f"Is {cleaned} a local number? Please add the country code:\n\n"
                    "Tap your country or type the full number with + prefix:",
                    [
                        {"content_type": "text", "title": "+91 India",
                         "payload": f"PHONE_+91{digits_only}"},
                        {"content_type": "text", "title": "+1 USA/Canada",
                         "payload": f"PHONE_+1{digits_only}"},
                        {"content_type": "text", "title": "+44 UK",
                         "payload": f"PHONE_+44{digits_only}"},
                        {"content_type": "text", "title": "+971 UAE",
                         "payload": f"PHONE_+971{digits_only}"},
                        {"content_type": "text", "title": "Use as-is",
                         "payload": f"PHONE_{cleaned}"},
                        {"content_type": "text", "title": "Skip",
                         "payload": "SKIP_PHONE"},
                    ],
                )
            state["guest_phone"] = cleaned if len(digits_only) >= 7 else ""
        state["step"] = "requests"
        return (
            "Any special requests for your stay?\n\n"
            "E.g. early check-in, high floor, baby cot, dietary needs.\n"
            "Or tap the button to skip.",
            [{"content_type": "text",
              "title":   "No Special Requests",
              "payload": "SKIP_REQUESTS"}],
        )

    # ── STEP: requests ────────────────────────────────────────────────────────
    if step == "requests":
        raw = raw_message.strip()
        if raw.upper() in ("SKIP", "SKIP_REQUESTS", "NO", "NONE",
                           "NO SPECIAL REQUESTS", "NOTHING", "NIL"):
            state["special_requests"] = ""
        else:
            state["special_requests"] = raw
        state["step"] = "confirm"

        summary = _booking_summary_text(state)
        return (
            f"{summary}\n\n"
            "Shall I confirm this booking?",
            [
                {"content_type": "text", "title": "Confirm Booking",
                 "payload": "CONFIRM_BOOKING"},
                {"content_type": "text", "title": "Start Over",
                 "payload": "RESTART"},
            ],
        )

    # ── STEP: confirm ─────────────────────────────────────────────────────────
    if step == "confirm":
        raw_l       = raw_message.strip().lower()
        raw_u_local = raw_message.strip().upper()
        _yes = {"yes", "confirm", "ok", "sure", "book", "go ahead", "proceed",
                "yep", "yup", "absolutely", "please"}
        _no  = {"no", "cancel", "restart", "start over", "change", "stop", "nope"}
        if raw_u_local == "CONFIRM_BOOKING" or any(w in raw_l for w in _yes):
            # Move to payment method selection
            state["step"] = "payment_method"
            hotel    = state.get("selected_hotel", {})
            room     = state.get("selected_room",  {})
            nights   = state.get("_nights", 1)
            currency = hotel.get("currency", "USD")
            price_n  = room.get("_final_price") or room.get("price_per_night") or 0
            addon_total = sum(a.get("price", 0) for a in state.get("selected_addons", []))
            total = price_n * nights + addon_total
            return (
                f"How would you like to pay?\n\nTotal: {currency} {total:.2f}",
                [
                    {"content_type": "text", "title": "Pay at Hotel",     "payload": "PAY_AT_HOTEL"},
                    {"content_type": "text", "title": "Credit/Debit Card","payload": "PAY_CARD"},
                    {"content_type": "text", "title": "UPI",              "payload": "PAY_UPI"},
                    {"content_type": "text", "title": "PayPal",           "payload": "PAY_PAYPAL"},
                    {"content_type": "text", "title": "Use Voucher Code", "payload": "PAY_VOUCHER"},
                    {"content_type": "text", "title": "Use Points",       "payload": "PAY_POINTS"},
                ],
            )
        if raw_u_local == "RESTART" or any(w in raw_l for w in _no):
            state["step"] = None
            return (
                "No problem! Your booking has not been made.\n\n"
                "Tap Book a Hotel to start a new search.",
                _MAIN_BUTTONS,
            )
        summary = _booking_summary_text(state)
        return (
            f"Please confirm or cancel your booking:\n\n{summary}",
            [
                {"content_type": "text", "title": "Confirm Booking",
                 "payload": "CONFIRM_BOOKING"},
                {"content_type": "text", "title": "Start Over",
                 "payload": "RESTART"},
            ],
        )

    # ── STEP: payment_method ─────────────────────────────────────────────────
    if step == "payment_method":
        raw_u = raw_message.strip().upper()
        hotel    = state.get("selected_hotel", {})
        room     = state.get("selected_room",  {})
        nights   = state.get("_nights", 1)
        currency = hotel.get("currency", "USD")
        price_n  = room.get("_final_price") or room.get("price_per_night") or 0
        addon_total = sum(a.get("price", 0) for a in state.get("selected_addons", []))
        total = price_n * nights + addon_total

        # Voucher code entry
        if raw_u == "PAY_VOUCHER" or state.get("awaiting_voucher"):
            if not state.get("awaiting_voucher"):
                state["awaiting_voucher"] = True
                return ("Enter your voucher / promo code:", [{"content_type": "text", "title": "Skip", "payload": "PAY_CARD"}])
            # User entered code
            code = raw_message.strip().upper()
            state.pop("awaiting_voucher", None)
            _VOUCHERS = {"WELCOME20": 0.20, "SAVE10": 0.10, "DEAL15": 0.15}
            if raw_u in ("SKIP", "SKIP_VOUCHER"):
                pass
            elif code in _VOUCHERS:
                discount = round(total * _VOUCHERS[code], 2)
                total_after = total - discount
                state["voucher_applied"] = code
                state["voucher_discount"] = discount
                return (
                    f"Voucher {code} applied!\nDiscount: -{currency} {discount:.2f}\n"
                    f"New total: {currency} {total_after:.2f}\n\nChoose payment method:",
                    [
                        {"content_type": "text", "title": "Pay at Hotel",      "payload": "PAY_AT_HOTEL"},
                        {"content_type": "text", "title": "Credit/Debit Card", "payload": "PAY_CARD"},
                        {"content_type": "text", "title": "UPI",               "payload": "PAY_UPI"},
                        {"content_type": "text", "title": "PayPal",            "payload": "PAY_PAYPAL"},
                    ],
                )
            else:
                return (
                    f"Voucher '{code}' not found or has expired. Please try another code or continue.",
                    [
                        {"content_type": "text", "title": "Try Another Code", "payload": "PAY_VOUCHER"},
                        {"content_type": "text", "title": "Pay at Hotel",     "payload": "PAY_AT_HOTEL"},
                        {"content_type": "text", "title": "Card",             "payload": "PAY_CARD"},
                    ],
                )

        # Use loyalty points
        if raw_u == "PAY_POINTS":
            # Simulate points balance check
            points_balance = state.get("loyalty_points", 0)
            if points_balance == 0:
                return (
                    "You do not have any loyalty points yet.\nEarn points by completing your first booking!\n\nChoose another payment:",
                    [
                        {"content_type": "text", "title": "Pay at Hotel", "payload": "PAY_AT_HOTEL"},
                        {"content_type": "text", "title": "Card",         "payload": "PAY_CARD"},
                    ],
                )
            points_value = points_balance // 2  # 2 pts = 1 unit of currency
            return (
                f"You have {points_balance} points worth {currency} {points_value:.2f}.\n"
                f"Apply all points and pay {currency} {max(0, total - points_value):.2f} by another method?",
                [
                    {"content_type": "text", "title": "Yes, Use All Points", "payload": "POINTS_APPLY_ALL"},
                    {"content_type": "text", "title": "Use Some Points",     "payload": "POINTS_APPLY_HALF"},
                    {"content_type": "text", "title": "Skip Points",         "payload": "PAY_CARD"},
                ],
            )

        if raw_u in ("POINTS_APPLY_ALL", "POINTS_APPLY_HALF"):
            pts = state.get("loyalty_points", 0)
            disc = (pts // 2) if raw_u == "POINTS_APPLY_ALL" else (pts // 4)
            state["points_discount"] = disc
            total_after = max(0, total - disc)
            return (
                f"Points discount applied: -{currency} {disc:.2f}\n"
                f"Remaining: {currency} {total_after:.2f}\n\nPay remaining by:",
                [
                    {"content_type": "text", "title": "Pay at Hotel", "payload": "PAY_AT_HOTEL"},
                    {"content_type": "text", "title": "Card",         "payload": "PAY_CARD"},
                    {"content_type": "text", "title": "UPI",          "payload": "PAY_UPI"},
                ],
            )

        # Pay at Hotel
        if raw_u == "PAY_AT_HOTEL":
            discount = state.get("voucher_discount", 0) + state.get("points_discount", 0)
            final_total = total - discount
            return (
                f"Pay at Hotel selected.\n\n"
                f"Your booking will be held at no charge now.\n"
                f"Payment is made directly at check-in.\n\n"
                f"Cancellation: free before 3 days prior to check-in.\n"
                f"Total due at hotel: {currency} {final_total:.2f}\n\n"
                "Confirm and hold this booking?",
                [
                    {"content_type": "text", "title": "Confirm (Pay at Hotel)", "payload": "PAY_AT_HOTEL_CONFIRM"},
                    {"content_type": "text", "title": "Choose Different Method","payload": "PAYMENT_BACK"},
                ],
            )

        if raw_u == "PAY_AT_HOTEL_CONFIRM":
            state["payment_method_chosen"] = "pay_at_hotel"
            return _create_booking(sender_id, state)

        # Card payment
        if raw_u == "PAY_CARD":
            discount = state.get("voucher_discount", 0) + state.get("points_discount", 0)
            final_total = total - discount
            import os as _os
            base_url = _os.getenv("PAYMENT_BASE_URL", "https://pay.bookbot.io")
            import random, string
            session_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            pay_link = f"{base_url}/card/{session_id}-{currency}{int(final_total)}"
            state["payment_session_id"] = session_id
            state["payment_method_chosen"] = "card"
            return (
                f"Credit/Debit Card payment — {currency} {final_total:.2f}\n\n"
                f"Tap the secure link to enter your card details:\n"
                f"🔒 {pay_link}\n\n"
                "(Secured by Stripe | 256-bit SSL | PCI-DSS Compliant)\n"
                "Accepted: Visa, Mastercard, Amex, Rupay, Diners\n\n"
                "Once payment is complete, your booking will be confirmed automatically.",
                [
                    {"content_type": "text", "title": "Payment Done",      "payload": "PAY_CARD_DONE"},
                    {"content_type": "text", "title": "Try Different Method","payload": "PAYMENT_BACK"},
                ],
            )

        if raw_u == "PAY_CARD_DONE":
            state["payment_method_chosen"] = "card"
            return _create_booking(sender_id, state)

        # UPI payment
        if raw_u == "PAY_UPI":
            discount = state.get("voucher_discount", 0) + state.get("points_discount", 0)
            final_total = total - discount
            return (
                f"UPI Payment — {currency} {final_total:.2f}\n\n"
                "Pay using any UPI app:\n"
                "UPI ID: bookbot@axis\n\n"
                "Scan the QR code in the app or use the UPI ID above.\n"
                "Then tap 'I have paid' to confirm your booking.",
                [
                    {"content_type": "text", "title": "I have paid",        "payload": "PAY_UPI_DONE"},
                    {"content_type": "text", "title": "Try Different Method","payload": "PAYMENT_BACK"},
                ],
            )

        if raw_u == "PAY_UPI_DONE":
            state["payment_method_chosen"] = "upi"
            return _create_booking(sender_id, state)

        # PayPal
        if raw_u == "PAY_PAYPAL":
            discount = state.get("voucher_discount", 0) + state.get("points_discount", 0)
            final_total = total - discount
            import os as _os
            base_url = _os.getenv("PAYMENT_BASE_URL", "https://pay.bookbot.io")
            import random, string
            session_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
            pay_link = f"{base_url}/paypal/{session_id}"
            return (
                f"PayPal payment — {currency} {final_total:.2f}\n\n"
                f"🔒 {pay_link}\n\n"
                "Tap the link to pay via PayPal. Once done, tap 'I have paid'.",
                [
                    {"content_type": "text", "title": "I have paid",        "payload": "PAY_CARD_DONE"},
                    {"content_type": "text", "title": "Try Different Method","payload": "PAYMENT_BACK"},
                ],
            )

        # Back to payment selection
        if raw_u == "PAYMENT_BACK":
            return (
                f"Choose your payment method.\nTotal: {currency} {total:.2f}",
                [
                    {"content_type": "text", "title": "Pay at Hotel",     "payload": "PAY_AT_HOTEL"},
                    {"content_type": "text", "title": "Credit/Debit Card","payload": "PAY_CARD"},
                    {"content_type": "text", "title": "UPI",              "payload": "PAY_UPI"},
                    {"content_type": "text", "title": "PayPal",           "payload": "PAY_PAYPAL"},
                    {"content_type": "text", "title": "Use Voucher Code", "payload": "PAY_VOUCHER"},
                    {"content_type": "text", "title": "Use Points",       "payload": "PAY_POINTS"},
                ],
            )

        # Fallback: re-show payment options
        return (
            f"Please choose a payment method.\nTotal: {currency} {total:.2f}",
            [
                {"content_type": "text", "title": "Pay at Hotel",     "payload": "PAY_AT_HOTEL"},
                {"content_type": "text", "title": "Credit/Debit Card","payload": "PAY_CARD"},
                {"content_type": "text", "title": "UPI",              "payload": "PAY_UPI"},
                {"content_type": "text", "title": "PayPal",           "payload": "PAY_PAYPAL"},
                {"content_type": "text", "title": "Use Voucher Code", "payload": "PAY_VOUCHER"},
            ],
        )

    return None, []


def _create_booking(sender_id: str, state: dict) -> tuple[str, list]:
    """Persist the booking to Supabase and return a confirmation message."""
    hotel  = state.get("selected_hotel", {})
    room   = state.get("selected_room",  {})
    nights = state.get("_nights", 1)
    price_n = room.get("_final_price") or room.get("price_per_night") or 0
    addon_total = sum(a.get("price", 0) for a in state.get("selected_addons", []))
    voucher_discount = state.get("voucher_discount", 0)
    points_discount  = state.get("points_discount", 0)
    total   = price_n * nights + addon_total - voucher_discount - points_discount
    total   = max(0, total)
    currency = hotel.get("currency", "USD")
    pay_method = state.get("payment_method_chosen", "pay_at_hotel")

    # Resolve / create user
    user_id = state.get("user_id")
    if not user_id and _DB_AVAILABLE:
        try:
            first = (state.get("guest_name") or "").split()[0] or "Guest"
            user  = get_or_create_user(sender_id, first_name=first)
            if user:
                user_id = user.get("id")
                state["user_id"] = user_id
        except Exception as e:
            logger.error("get_or_create_user: %s", e)

    booking_ref = None
    if _DB_AVAILABLE:
        try:
            result = create_booking(
                user_id              = user_id or "",
                hotel_id             = hotel.get("id", ""),
                room_type_code       = room.get("room_type_code", "STD"),
                check_in             = state.get("checkin",  ""),
                check_out            = state.get("checkout", ""),
                num_adults           = state.get("num_adults",   1),
                num_children         = state.get("num_children", 0),
                primary_guest_name   = state.get("guest_name",  "Guest"),
                primary_guest_email  = state.get("guest_email", ""),
                primary_guest_phone  = state.get("guest_phone", ""),
                total_amount         = total,
                currency             = currency,
                special_requests     = state.get("special_requests", ""),
                rate_plan            = state.get("rate_plan",  "room_only"),
                meal_plan            = state.get("meal_plan",  "room_only"),
            )
            if result:
                booking_ref = result.get("booking_reference")
        except Exception as e:
            logger.error("create_booking: %s", e, exc_info=True)

    # Generate a local reference when DB is unavailable
    if not booking_ref:
        import random, string
        booking_ref = "BB" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

    # Snapshot values before clearing state
    hotel_name = hotel.get("name", "Hotel")
    room_name  = room.get("room_type_name", "Room")
    checkin    = state.get("checkin", "")
    checkout   = state.get("checkout", "")
    guest_email= state.get("guest_email", "")
    guest_name = state.get("guest_name", "")
    guest_phone= state.get("guest_phone", "")
    special_req= state.get("special_requests", "")
    addons     = state.get("selected_addons", [])
    meal_labels = {"room_only": "Room Only", "breakfast": "Bed & Breakfast",
                   "half_board": "Half Board", "full_board": "Full Board"}
    meal_disp  = meal_labels.get(state.get("meal_plan", "room_only"), "Room Only")
    pay_display = {
        "pay_at_hotel": "Pay at Hotel", "card": "Credit/Debit Card",
        "upi": "UPI", "paypal": "PayPal",
    }.get(pay_method, "Pay at Hotel")

    # Award loyalty points (10 pts per 100 currency units)
    pts_earned = int((price_n * nights) / 100 * 10)
    state["loyalty_points"] = state.get("loyalty_points", 0) + pts_earned

    # Send hotel admin notification
    _notify_hotel_new_booking(
        booking_ref=booking_ref, hotel_name=hotel_name,
        hotel_email=hotel.get("contact_email", ""),
        guest_name=guest_name, guest_email=guest_email, guest_phone=guest_phone,
        room_name=room_name, checkin=checkin, checkout=checkout, nights=nights,
        adults=state.get("num_adults", 1), children=state.get("num_children", 0),
        meal_disp=meal_disp, special_requests=special_req,
        addons=addons, total=total, currency=currency, pay_method=pay_display,
    )

    # Reset booking slots
    state["step"] = None
    _reset_booking_slots(state)

    addon_lines = "\n".join(f"  + {a['label']}" for a in addons) if addons else ""
    addon_section = f"\nAdd-ons   : {addon_lines}" if addon_lines else ""

    return (
        f"Your booking is confirmed!\n\n"
        f"Reference : {booking_ref}\n"
        f"Hotel     : {hotel_name}\n"
        f"Room      : {room_name}\n"
        f"Meal Plan : {meal_disp}\n"
        f"Check-in  : {_pretty_date(checkin)}\n"
        f"Check-out : {_pretty_date(checkout)}{addon_section}\n"
        f"Total     : {currency} {total:.2f}\n"
        f"Payment   : {pay_display}\n\n"
        f"Confirmation sent to {guest_email}\n"
        f"Loyalty points earned: +{pts_earned} points\n\n"
        "What would you like to do next?",
        [
            {"content_type": "text", "title": "My Bookings",       "payload": "MY_BOOKINGS"},
            {"content_type": "text", "title": "Pre-Arrival",        "payload": "PRE_ARRIVAL"},
            {"content_type": "text", "title": "Book Another Hotel", "payload": "ACTION_BOOK"},
            {"content_type": "text", "title": "Loyalty Rewards",    "payload": "LOYALTY_MENU"},
        ],
    )


_STATUS_ICON = {
    "confirmed": "✅",
    "cancelled":  "❌",
    "pending":    "⏳",
    "completed":  "🏁",
}


def _show_my_bookings(sender_id: str, state: dict) -> tuple[str, list]:
    """Fetch and display the user's recent bookings with status icons and amounts."""
    if not _DB_AVAILABLE:
        return (
            "I cannot access your booking history right now. Please try again later.",
            _MAIN_BUTTONS,
        )

    user_id = state.get("user_id")
    if not user_id:
        try:
            user = get_or_create_user(sender_id)
            if user:
                user_id = user.get("id")
                state["user_id"] = user_id
        except Exception as e:
            logger.error("get_or_create_user: %s", e)

    if not user_id:
        return ("I could not find your account. Please try again.", _MAIN_BUTTONS)

    try:
        bookings = get_user_bookings(user_id)
    except Exception as e:
        logger.error("get_user_bookings: %s", e)
        bookings = []

    if not bookings:
        return (
            "You do not have any bookings yet.\n\n"
            "Tap Book a Hotel to make your first reservation.",
            [
                {"content_type": "text", "title": "Book a Hotel", "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "Main Menu",    "payload": "ACTION_HELP"},
            ],
        )

    lines = []
    for b in bookings[:5]:
        ref    = b.get("booking_reference", "N/A")
        hotel  = b.get("hotel_name", b.get("name", "Hotel"))
        ci     = b.get("check_in",  "")
        co     = b.get("check_out", "")
        status = (b.get("status") or "pending").lower()
        icon   = _STATUS_ICON.get(status, "ℹ️")
        amt    = b.get("total_amount")
        curr   = b.get("currency", "")
        p_s    = f"  {curr} {float(amt):.0f}" if amt else ""
        lines.append(
            f"{icon} {ref} — {hotel}\n"
            f"   {_pretty_date(ci)} → {_pretty_date(co)}  [{status.capitalize()}]{p_s}\n"
            f"   Type 'view {ref}' for details"
        )

    quick_replies = [
        {"content_type": "text", "title": "Cancel a Booking",  "payload": "CANCEL_BOOKING"},
        {"content_type": "text", "title": "Modify a Booking",  "payload": "MODIFY_BOOKING"},
        {"content_type": "text", "title": "New Booking",       "payload": "ACTION_BOOK"},
    ]
    # Add detail buttons for first 3 bookings
    for b in bookings[:2]:
        ref = b.get("booking_reference", "")
        if ref:
            quick_replies.insert(0, {
                "content_type": "text",
                "title": f"View {ref[:10]}",
                "payload": f"BOOKING_DETAIL_{ref}",
            })

    return (
        f"Your recent bookings ({len(bookings)}):\n\n" + "\n\n".join(lines) +
        "\n\nTap a booking to see full details.",
        quick_replies[:11],  # Messenger limit 13 quick replies
    )


def _show_booking_detail(booking: dict) -> tuple[str, list]:
    """Return full detail view for a single booking (Part G2)."""
    ref    = booking.get("booking_reference", "N/A")
    hotel  = booking.get("hotel_name", booking.get("name", ""))
    ci     = booking.get("check_in",  "")
    co     = booking.get("check_out", "")
    room   = booking.get("room_type_code", booking.get("room_name", ""))
    meal   = booking.get("meal_plan_code", "")
    guests = booking.get("adults", "")
    child  = booking.get("children", 0)
    reqs   = booking.get("special_requests", "") or "None"
    pay    = booking.get("payment_method", "")
    amt    = booking.get("total_amount")
    curr   = booking.get("currency", "")
    p_s    = f"{curr} {float(amt):.2f}" if amt else ""
    status = (booking.get("status") or "pending").capitalize()
    icon   = _STATUS_ICON.get(status.lower(), "ℹ️")
    guest_name  = booking.get("guest_name", "")
    guest_email = booking.get("guest_email", "")
    guest_phone = booking.get("guest_phone", "")

    # Free cancellation deadline
    deadline_s = ""
    try:
        ci_date  = datetime.strptime(ci, "%Y-%m-%d").date()
        deadline = ci_date - timedelta(days=3)
        today    = date.today()
        if today <= deadline:
            deadline_s = f"\nFree cancel by: {_pretty_date(str(deadline))}"
        else:
            deadline_s = "\nOutside free cancellation window"
    except Exception:
        pass

    detail = (
        f"Booking Details — {ref}\n"
        f"{'─'*28}\n"
        f"{icon} Status       : {status}\n"
        f"Hotel        : {hotel}\n"
        f"Room         : {room}\n"
        f"Meal Plan    : {meal or 'Room only'}\n"
        f"Check-in     : {_pretty_date(ci)}\n"
        f"Check-out    : {_pretty_date(co)}\n"
        f"Guests       : {guests} adults"
        + (f", {child} children" if child else "") + "\n"
        f"Special Reqs : {reqs}\n"
        f"Payment      : {pay or 'N/A'}\n"
        f"Total        : {p_s}"
        f"{deadline_s}"
    )

    btns = [
        {"content_type": "text", "title": "Modify Booking", "payload": "MODIFY_BOOKING"},
        {"content_type": "text", "title": "Cancel Booking", "payload": "CANCEL_BOOKING"},
        {"content_type": "text", "title": "Download Receipt", "payload": f"RECEIPT_{ref}"},
        {"content_type": "text", "title": "Pre-Arrival",    "payload": "PRE_ARRIVAL"},
        {"content_type": "text", "title": "Share Booking",  "payload": f"SHARE_{ref}"},
        {"content_type": "text", "title": "My Bookings",    "payload": "MY_BOOKINGS"},
    ]
    return detail, btns


def _handle_cancel_ref(
    sender_id: str, state: dict, raw_message: str, en_lower: str
) -> tuple[str, list]:
    """User just entered a booking reference for cancellation."""
    ref = raw_message.strip().upper()

    if not re.match(r'^BB[A-Z0-9\-]{4,20}$', ref):
        return (
            "That does not look like a valid booking reference.\n"
            "Booking references start with BB, e.g. BB-MUM-20260320-X7K2. Please try again:",
            [{"content_type": "text", "title": "Go Back", "payload": "RESTART"}],
        )

    if not _DB_AVAILABLE:
        return ("Booking lookup is not available right now.", _MAIN_BUTTONS)

    try:
        booking = get_booking_by_ref(ref)
    except Exception as e:
        logger.error("get_booking_by_ref: %s", e)
        booking = None

    if not booking:
        return (
            f"Booking reference {ref} was not found.\n"
            "Please check the reference and try again.",
            [{"content_type": "text", "title": "Search by Email", "payload": "MY_BOOKINGS"},
             {"content_type": "text", "title": "Go Back",          "payload": "RESTART"}],
        )

    if booking.get("status") == "cancelled":
        cancelled_on = booking.get("cancelled_at", "")
        cancelled_on_s = f" on {cancelled_on}" if cancelled_on else ""
        return (
            f"Booking {ref} was already cancelled{cancelled_on_s}.\n"
            "It cannot be modified or cancelled again.\n\n"
            "Would you like to make a new booking?",
            [{"content_type": "text", "title": "Book a Hotel", "payload": "ACTION_BOOK"},
             {"content_type": "text", "title": "Main Menu",    "payload": "RESTART"}],
        )

    hotel = booking.get("hotel_name", booking.get("name", ""))
    ci    = booking.get("check_in",  "")
    co    = booking.get("check_out", "")
    amt   = booking.get("total_amount")
    curr  = booking.get("currency", "")
    p_s   = f"{curr} {float(amt):.2f}" if amt else ""

    # ── Cancellation policy check (Part G5) ───────────────────────────────────
    today = date.today()
    free_cancel_days = 3  # default 72 hrs
    try:
        ci_date = datetime.strptime(ci, "%Y-%m-%d").date()
        deadline = ci_date - timedelta(days=free_cancel_days)
        within_free = today <= deadline
        penalty_note = ""
        if within_free:
            policy_line = (
                f"Free cancellation before: {_pretty_date(str(deadline))}\n"
                f"Today: {_pretty_date(str(today))}  — within free cancellation period\n"
                f"Refund: {p_s} (full refund)"
            )
        else:
            # 1-night penalty
            try:
                nights = (datetime.strptime(co, "%Y-%m-%d") - datetime.strptime(ci, "%Y-%m-%d")).days
                penalty = float(amt) / nights if (amt and nights > 0) else 0
                refund  = float(amt) - penalty if amt else 0
                policy_line = (
                    f"Outside free cancellation (deadline was {_pretty_date(str(deadline))})\n"
                    f"Cancellation fee: {curr} {penalty:.2f} (1 night)\n"
                    f"Refund if cancelled: {curr} {refund:.2f}"
                )
            except Exception:
                policy_line = "Cancellation policy: fee may apply (contact hotel)."
    except Exception:
        within_free = True
        policy_line = f"Cancellation policy applies. Refund: {p_s}"

    state["cancel_ref"]        = ref
    state["cancel_hotel_name"] = hotel
    state["cancel_within_free"]= within_free
    state["step"]              = "cancel_confirm"

    return (
        f"Cancellation request for {ref}\n\n"
        f"Hotel : {hotel}\n"
        f"Dates : {ci} to {co}\n"
        f"Total : {p_s}\n\n"
        f"Cancellation Policy:\n{policy_line}\n\n"
        "Are you sure you want to cancel?",
        [
            {"content_type": "text", "title": "Yes, Cancel It",    "payload": "CANCEL_CONFIRM"},
            {"content_type": "text", "title": "No, Keep Booking",  "payload": "RESTART"},
            {"content_type": "text", "title": "Modify Instead",    "payload": "MODIFY_BOOKING"},
            {"content_type": "text", "title": "Speak to Agent",    "payload": "AGENT_HANDOFF"},
        ],
    )


def _handle_cancel_confirm(
    sender_id: str, state: dict, en_lower: str
) -> tuple[str, list]:
    """Process the user's confirmation/denial of a cancellation."""
    raw   = en_lower.strip()
    ref   = state.get("cancel_ref", "")
    hotel = state.get("cancel_hotel_name", "")
    _yes  = {"yes", "cancel", "confirm", "ok", "sure", "proceed",
             "yep", "absolutely", "cancel_confirm"}

    if raw in _yes or raw == "cancel_confirm":
        success = False
        if _DB_AVAILABLE and ref:
            try:
                success = cancel_booking(ref)
            except Exception as e:
                logger.error("cancel_booking: %s", e)

        state["step"] = None
        state.pop("cancel_ref",         None)
        state.pop("cancel_hotel_name",  None)
        state.pop("cancel_within_free", None)

        if success:
            # Notify hotel of cancellation
            _notify_hotel_cancellation(
                booking_ref=ref, hotel_name=hotel, hotel_email="",
                guest_name=state.get("guest_name", ""),
                cancelled_date=str(date.today()),
                refund_amount=0.0, currency="",
            )
            return (
                f"Booking {ref} has been cancelled.\n\n"
                "Your refund (if applicable) will be processed\n"
                "within 5–7 business days to your original payment method.\n\n"
                "Is there anything else I can help you with?",
                [
                    {"content_type": "text", "title": "Book a New Hotel", "payload": "ACTION_BOOK"},
                    {"content_type": "text", "title": "My Bookings",      "payload": "MY_BOOKINGS"},
                    {"content_type": "text", "title": "Main Menu",        "payload": "RESTART"},
                ],
            )
        return (
            f"Sorry, I could not cancel {ref} right now.\n"
            "Please try again later or contact support.",
            [{"content_type": "text", "title": "Speak to Agent", "payload": "AGENT_HANDOFF"},
             {"content_type": "text", "title": "Try Again",      "payload": "CANCEL_BOOKING"}],
        )

    # User said no
    state["step"] = None
    state.pop("cancel_ref",         None)
    state.pop("cancel_hotel_name",  None)
    state.pop("cancel_within_free", None)
    return (
        "Your booking is still active. Is there anything else I can help you with?",
        _MAIN_BUTTONS,
    )


def _handle_lookup_ref(
    sender_id: str, state: dict, raw_message: str
) -> tuple[str, list]:
    """User entered a booking reference to look up (Part G1/G2)."""
    ref = raw_message.strip().upper()

    if not re.match(r'^BB[A-Z0-9\-]{4,20}$', ref):
        return (
            "That does not look like a valid booking reference.\n"
            "References start with BB, e.g. BB-MUM-20260320-X7K2.\n"
            "Please try again:",
            [{"content_type": "text", "title": "Go Back", "payload": "RESTART"}],
        )

    if not _DB_AVAILABLE:
        return ("Booking lookup is not available right now.", _MAIN_BUTTONS)

    state["step"] = None
    try:
        booking = get_booking_by_ref(ref)
    except Exception as e:
        logger.error("get_booking_by_ref: %s", e)
        booking = None

    if not booking:
        return (
            f"Booking {ref} was not found. Please check the reference.",
            [
                {"content_type": "text", "title": "Try Again",   "payload": "LOOKUP_BOOKING"},
                {"content_type": "text", "title": "My Bookings", "payload": "MY_BOOKINGS"},
                {"content_type": "text", "title": "Main Menu",   "payload": "RESTART"},
            ],
        )

    return _show_booking_detail(booking)


# ── Part G7 — Share Booking ────────────────────────────────────────────────────

def _handle_share_booking(ref: str) -> tuple[str, list]:
    """Generate a shareable booking link and options (Part G7)."""
    view_url = f"https://bookbot.io/view/{ref}"
    return (
        f"Share your booking {ref}\n\n"
        f"View link:\n{view_url}\n\n"
        "How would you like to share it?",
        [
            {"content_type": "text", "title": "Send to Email",  "payload": f"SHARE_EMAIL_{ref}"},
            {"content_type": "text", "title": "Download PDF",   "payload": f"RECEIPT_{ref}"},
            {"content_type": "text", "title": "Copy Link",      "payload": f"COPY_LINK_{ref}"},
            {"content_type": "text", "title": "My Bookings",    "payload": "MY_BOOKINGS"},
        ],
    )


# ── Part G8 — Receipt / Invoice ────────────────────────────────────────────────

def _handle_receipt(ref: str, invoice: bool = False) -> tuple[str, list]:
    """Offer receipt and invoice download options (Part G8)."""
    doc_type = "Invoice" if invoice else "Receipt"
    base_url = f"https://bookbot.io/receipt/{ref}"
    return (
        f"{doc_type} for Booking {ref}\n\n"
        "Choose your download format:",
        [
            {"content_type": "text", "title": "PDF Receipt",
             "payload": f"DL_PDF_{ref}"},
            {"content_type": "text", "title": "GST Invoice",
             "payload": f"DL_GST_{ref}"},
            {"content_type": "text", "title": "Email Receipt",
             "payload": f"EMAIL_RECEIPT_{ref}"},
            {"content_type": "text", "title": "Corporate Invoice",
             "payload": f"DL_CORP_{ref}"},
            {"content_type": "text", "title": "Back",
             "payload": f"BOOKING_DETAIL_{ref}"},
        ],
    )


# ── Part G6 — Re-book same hotel ───────────────────────────────────────────────

def _handle_rebook(sender_id: str, state: dict, ref: str) -> tuple[str, list]:
    """Pre-fill booking state with last booking's hotel/room and ask for new dates (Part G6)."""
    booking = None
    if _DB_AVAILABLE:
        try:
            booking = get_booking_by_ref(ref)
        except Exception as e:
            logger.error("_handle_rebook get_booking_by_ref: %s", e)

    if not booking:
        state["step"] = "city"
        return (
            "I could not find that booking. Let's start a fresh search — which city?",
            [],
        )

    hotel_id   = booking.get("hotel_id",        "")
    hotel_name = booking.get("hotel_name", booking.get("name", ""))
    room_code  = booking.get("room_type_code",   "")
    meal_code  = booking.get("meal_plan_code",   "")
    adults     = booking.get("adults",           1)
    children   = booking.get("children",         0)

    # Pre-fill session with previous choices
    state["hotel_id"]     = hotel_id
    state["hotel_name"]   = hotel_name
    state["room_code"]    = room_code
    state["meal_code"]    = meal_code
    state["num_adults"]   = adults
    state["num_children"] = children
    state["step"]         = "checkin"

    return (
        f"Great! Let's rebook {hotel_name}.\n\n"
        f"Previous room    : {room_code}\n"
        f"Previous meal    : {meal_code or 'Room only'}\n"
        f"Guests           : {adults} adults"
        + (f", {children} children" if children else "") + "\n\n"
        "Please enter your new check-in date (DD/MM/YYYY):",
        [],
    )


# ── Hotel admin notifications ──────────────────────────────────────────────────

def _notify_hotel_new_booking(
    booking_ref: str, hotel_name: str, hotel_email: str,
    guest_name: str, guest_email: str, guest_phone: str,
    room_name: str, checkin: str, checkout: str, nights: int,
    adults: int, children: int, meal_disp: str, special_requests: str,
    addons: list, total: float, currency: str, pay_method: str,
) -> None:
    """Send new booking notification to hotel admin via email (async-safe fire-and-forget)."""
    try:
        import smtplib, os as _os
        from email.mime.text import MIMEText
        smtp_host  = _os.getenv("SMTP_HOST", "")
        smtp_user  = _os.getenv("SMTP_USER", "")
        smtp_pass  = _os.getenv("SMTP_PASS", "")
        admin_email= _os.getenv("HOTEL_ADMIN_EMAIL", hotel_email or "")
        if not (smtp_host and smtp_user and admin_email):
            print(f"[notify] Hotel notification skipped (SMTP not configured). Ref: {booking_ref}", flush=True)
            return
        addon_text = "\n".join(f"  + {a['label']}: {currency} {a['price']:.0f}" for a in addons) or "  None"
        body = (
            f"New Booking Received via BookBot\n"
            f"{'='*40}\n"
            f"Reference   : {booking_ref}\n"
            f"Channel     : BookBot Messenger\n"
            f"Status      : Confirmed\n\n"
            f"Guest       : {guest_name}\n"
            f"Email       : {guest_email}\n"
            f"Phone       : {guest_phone or 'Not provided'}\n\n"
            f"Hotel       : {hotel_name}\n"
            f"Room        : {room_name}\n"
            f"Check-in    : {checkin}\n"
            f"Check-out   : {checkout}\n"
            f"Nights      : {nights}\n"
            f"Guests      : {adults} adults, {children} children\n"
            f"Meal Plan   : {meal_disp}\n\n"
            f"Special Requests:\n{special_requests or 'None'}\n\n"
            f"Add-ons:\n{addon_text}\n\n"
            f"Total       : {currency} {total:.2f}\n"
            f"Payment     : {pay_method}\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = f"New Booking — {booking_ref} — {hotel_name}"
        msg["From"]    = smtp_user
        msg["To"]      = admin_email
        with smtplib.SMTP_SSL(smtp_host, 465) as s:
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
        print(f"[notify] Hotel notification sent to {admin_email} for {booking_ref}", flush=True)
    except Exception as e:
        print(f"[notify] Hotel notification failed: {e}", flush=True)


def _notify_hotel_cancellation(booking_ref: str, hotel_name: str, hotel_email: str,
                                 guest_name: str, cancelled_date: str,
                                 refund_amount: float, currency: str) -> None:
    """Send cancellation notification to hotel admin."""
    try:
        import smtplib, os as _os
        from email.mime.text import MIMEText
        smtp_host  = _os.getenv("SMTP_HOST", "")
        smtp_user  = _os.getenv("SMTP_USER", "")
        smtp_pass  = _os.getenv("SMTP_PASS", "")
        admin_email= _os.getenv("HOTEL_ADMIN_EMAIL", hotel_email or "")
        if not (smtp_host and smtp_user and admin_email):
            return
        body = (
            f"Booking Cancelled\n{'='*40}\n"
            f"Reference : {booking_ref}\n"
            f"Guest     : {guest_name}\n"
            f"Hotel     : {hotel_name}\n"
            f"Cancelled : {cancelled_date}\n"
            f"Refund    : {currency} {refund_amount:.2f}\n\n"
            f"Action: Please release the room for resale.\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = f"Booking Cancelled — {booking_ref}"
        msg["From"]    = smtp_user
        msg["To"]      = admin_email
        with smtplib.SMTP_SSL(smtp_host, 465) as s:
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
    except Exception as e:
        print(f"[notify] Cancellation notification failed: {e}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOYALTY PROGRAM FLOW
# ─────────────────────────────────────────────────────────────────────────────

def _handle_loyalty_flow(
    sender_id: str, state: dict, raw_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle loyalty program: balance, join, refer, redeem. Returns (None,) if not a loyalty intent."""
    raw_u = raw_message.strip().upper()
    _loyalty_kw = {"loyalty", "points", "rewards", "reward", "membership",
                   "earn points", "redeem", "tier", "membership", "join rewards"}

    loyalty_step = state.get("loyalty_step")

    if raw_u not in ("LOYALTY_MENU", "LOYALTY_JOIN", "LOYALTY_REFER", "LOYALTY_HISTORY") \
            and loyalty_step is None \
            and not any(kw in en_lower for kw in _loyalty_kw):
        return None, []

    points = state.get("loyalty_points", 0)
    tier   = "Bronze" if points < 10000 else ("Silver" if points < 50000 else ("Gold" if points < 100000 else "Platinum"))
    tier_emoji = {"Bronze": "🥉", "Silver": "🥈", "Gold": "🥇", "Platinum": "💎"}.get(tier, "🏆")

    # Show loyalty menu / balance
    if raw_u in ("LOYALTY_MENU",) or (loyalty_step is None and any(kw in en_lower for kw in _loyalty_kw)):
        state["loyalty_step"] = "menu"
        pts_needed = {"Bronze": 10000, "Silver": 50000, "Gold": 100000, "Platinum": None}[tier]
        next_tier_label = {"Bronze": "Silver", "Silver": "Gold", "Gold": "Platinum", "Platinum": None}[tier]
        pts_to_next = (pts_needed - points) if pts_needed else 0
        member_id = state.get("loyalty_member_id", "")
        if not member_id:
            member_id = ""
        balance_text = (
            f"BookBot Rewards\n{'─'*28}\n"
            f"{tier_emoji} Tier     : {tier}\n"
            f"Points    : {points:,} pts\n"
            f"Cash Value: {state.get('selected_hotel', {}).get('currency','USD')} {points // 2:.0f}\n"
        )
        if next_tier_label:
            balance_text += f"To {next_tier_label}: {pts_to_next:,} more points needed\n"
        else:
            balance_text += "Maximum tier reached!\n"
        if member_id:
            balance_text += f"Member ID : {member_id}\n"

        buttons = [
            {"content_type": "text", "title": "Redeem Points",    "payload": "LOYALTY_REDEEM"},
            {"content_type": "text", "title": "Refer a Friend",   "payload": "LOYALTY_REFER"},
            {"content_type": "text", "title": "Points History",   "payload": "LOYALTY_HISTORY"},
        ]
        if not member_id:
            buttons.insert(0, {"content_type": "text", "title": "Join Free!", "payload": "LOYALTY_JOIN"})
        buttons.append({"content_type": "text", "title": "Back", "payload": "RESTART"})
        return balance_text, buttons[:4]

    # Join rewards
    if raw_u == "LOYALTY_JOIN" or loyalty_step == "joining":
        import random, string as _str
        member_id = "BBR-" + "".join(random.choices(_str.digits, k=10))
        state["loyalty_member_id"] = member_id
        state["loyalty_step"] = None
        return (
            f"Welcome to BookBot Rewards!\n\n"
            f"Your Membership:\n"
            f"ID: {member_id}\n"
            f"Tier: {tier_emoji} Bronze\n"
            f"Points balance: {points:,}\n\n"
            "Earn 10 points per 100 currency units spent.\n"
            "Your next booking will earn you points!\n\n"
            "How it works:\n"
            "Bronze  (0–9,999)     — 5% earn bonus\n"
            "Silver  (10k–49,999)  — 10% bonus + upgrades\n"
            "Gold    (50k–99,999)  — 15% bonus + free breakfast\n"
            "Platinum (100k+)      — 20% bonus + butler service",
            [
                {"content_type": "text", "title": "Book a Hotel",   "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "View Balance",   "payload": "LOYALTY_MENU"},
                {"content_type": "text", "title": "Refer a Friend", "payload": "LOYALTY_REFER"},
            ],
        )

    # Refer a friend
    if raw_u == "LOYALTY_REFER":
        state["loyalty_step"] = None
        ref_code = f"{(state.get('guest_name','').split()[0] or 'USER').upper()}-REF-{sender_id[-4:]}"
        refer_link = f"https://bookbot.io/join?ref={ref_code}"
        return (
            f"Refer a friend and both of you earn rewards!\n\n"
            f"Your referral code: {ref_code}\n"
            f"Share this link: {refer_link}\n\n"
            "When your friend makes their FIRST booking:\n"
            "You earn: 500 bonus points\n"
            "They earn: 200 welcome bonus points",
            [
                {"content_type": "text", "title": "Copy Link", "payload": "LOYALTY_COPY_REF"},
                {"content_type": "text", "title": "My Balance","payload": "LOYALTY_MENU"},
                {"content_type": "text", "title": "Book Hotel","payload": "ACTION_BOOK"},
            ],
        )

    # Points history
    if raw_u == "LOYALTY_HISTORY":
        state["loyalty_step"] = None
        return (
            f"Your points history:\n\n"
            f"Current balance: {points:,} points\n"
            f"Tier: {tier_emoji} {tier}\n\n"
            "(Detailed history available in the BookBot app.\n"
            "Points are awarded after each completed booking.)",
            [
                {"content_type": "text", "title": "Redeem Points","payload": "LOYALTY_REDEEM"},
                {"content_type": "text", "title": "Book Hotel",   "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "Main Menu",    "payload": "RESTART"},
            ],
        )

    # Redeem points
    if raw_u == "LOYALTY_REDEEM":
        state["loyalty_step"] = None
        if points < 200:
            return (
                f"You need at least 200 points to redeem.\n"
                f"You currently have {points:,} points.\n"
                "Book more hotels to earn points!",
                _MAIN_BUTTONS,
            )
        return (
            f"Redeem loyalty points\n\n"
            f"Available: {points:,} points\n"
            f"Redemption rate: 2 points = 1 unit of currency\n\n"
            "Points are applied during checkout on your next booking.\n"
            "Tap 'Use Points' at the payment step.",
            [
                {"content_type": "text", "title": "Book a Hotel", "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "My Balance",   "payload": "LOYALTY_MENU"},
            ],
        )

    state["loyalty_step"] = None
    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# PRE-ARRIVAL SERVICES FLOW
# ─────────────────────────────────────────────────────────────────────────────

def _handle_pre_arrival_flow(
    sender_id: str, state: dict, raw_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle pre-arrival: early check-in, airport transfer, upgrade, special occasion, dietary."""
    raw_u = raw_message.strip().upper()
    _pre_kw = {"early check-in", "early checkin", "early check in",
               "airport transfer", "airport pickup", "airport transport",
               "pre-arrival", "pre arrival", "before i arrive",
               "room upgrade", "upgrade room", "suite upgrade",
               "special occasion", "anniversary", "birthday setup",
               "honeymoon", "proposal", "romantic",
               "dietary", "vegetarian", "vegan", "halal", "kosher", "gluten",
               "nut allergy", "dairy free"}

    pre_arrival_step = state.get("pre_arrival_step")

    is_trigger = (
        raw_u in ("PRE_ARRIVAL", "EARLY_CHECKIN", "AIRPORT_TRANSFER",
                  "ROOM_UPGRADE", "SPECIAL_OCCASION", "DIETARY_NEEDS")
        or pre_arrival_step is not None
        or any(kw in en_lower for kw in _pre_kw)
    )
    if not is_trigger:
        return None, []

    # Main pre-arrival menu
    if raw_u == "PRE_ARRIVAL" or (pre_arrival_step is None and not raw_u.startswith("EARLY_") and not raw_u.startswith("AIRPORT_") and not raw_u.startswith("ROOM_UPG") and not raw_u.startswith("SPECIAL_") and not raw_u.startswith("DIETARY")):
        state["pre_arrival_step"] = "menu"
        return (
            "Pre-Arrival Services\n\n"
            "Enhance your stay before you arrive.\n"
            "Choose a service to set up:",
            [
                {"content_type": "text", "title": "Early Check-in",    "payload": "EARLY_CHECKIN"},
                {"content_type": "text", "title": "Airport Transfer",  "payload": "AIRPORT_TRANSFER"},
                {"content_type": "text", "title": "Room Upgrade",       "payload": "ROOM_UPGRADE"},
                {"content_type": "text", "title": "Special Occasion",   "payload": "SPECIAL_OCCASION"},
                {"content_type": "text", "title": "Dietary Needs",      "payload": "DIETARY_NEEDS"},
                {"content_type": "text", "title": "Back",               "payload": "RESTART"},
            ],
        )

    # Early check-in
    if raw_u == "EARLY_CHECKIN" or (pre_arrival_step == "early_checkin" and "early" in en_lower):
        state["pre_arrival_step"] = "early_checkin_time"
        return (
            "Early Check-in Request\n\n"
            "Standard check-in is 2:00 PM.\n"
            "What time would you like to arrive?",
            [
                {"content_type": "text", "title": "6:00 AM",  "payload": "EARLY_CI_0600"},
                {"content_type": "text", "title": "8:00 AM",  "payload": "EARLY_CI_0800"},
                {"content_type": "text", "title": "10:00 AM", "payload": "EARLY_CI_1000"},
                {"content_type": "text", "title": "12:00 PM", "payload": "EARLY_CI_1200"},
                {"content_type": "text", "title": "Back",     "payload": "PRE_ARRIVAL"},
            ],
        )

    if raw_u.startswith("EARLY_CI_"):
        time_val = raw_u[9:]  # e.g. "0800"
        time_disp = f"{time_val[:2]}:{time_val[2:]} {'AM' if int(time_val[:2]) < 12 else 'PM'}"
        state["pre_arrival_step"] = None
        return (
            f"Early Check-in Request: {time_disp}\n\n"
            "Options:\n"
            "Free early check-in — subject to availability (confirmed at arrival)\n"
            "Guaranteed early check-in — room held from your requested time (fee applies)\n\n"
            "Your request has been sent to the hotel.\n"
            "The hotel will confirm availability before your arrival.",
            [
                {"content_type": "text", "title": "Free Request",       "payload": "EARLY_CI_FREE"},
                {"content_type": "text", "title": "Guaranteed (+fee)",  "payload": "EARLY_CI_PAID"},
                {"content_type": "text", "title": "More Services",      "payload": "PRE_ARRIVAL"},
            ],
        )

    if raw_u in ("EARLY_CI_FREE", "EARLY_CI_PAID"):
        state["pre_arrival_step"] = None
        is_paid = raw_u == "EARLY_CI_PAID"
        return (
            "Early check-in confirmed!" if is_paid else "Free early check-in requested (not guaranteed)!\n\nThe hotel will do their best to accommodate you.",
            _MAIN_BUTTONS,
        )

    # Airport transfer
    if raw_u == "AIRPORT_TRANSFER" or "airport" in en_lower:
        state["pre_arrival_step"] = "airport_flight"
        return (
            "Airport Transfer\n\n"
            "Please enter your flight number\n"
            "(e.g. AI 101, EK 202, BA 789):",
            [{"content_type": "text", "title": "Skip (no flight)",  "payload": "AIRPORT_NO_FLIGHT"}],
        )

    if pre_arrival_step == "airport_flight":
        state["airport_flight"] = raw_message.strip()
        state["pre_arrival_step"] = "airport_type"
        return (
            "Arrival or departure transfer?",
            [
                {"content_type": "text", "title": "Arrival Only",   "payload": "AIRPORT_ARRIVAL"},
                {"content_type": "text", "title": "Departure Only", "payload": "AIRPORT_DEPARTURE"},
                {"content_type": "text", "title": "Both Ways",      "payload": "AIRPORT_BOTH"},
            ],
        )

    if raw_u in ("AIRPORT_ARRIVAL", "AIRPORT_DEPARTURE", "AIRPORT_BOTH", "AIRPORT_NO_FLIGHT"):
        state["pre_arrival_step"] = "airport_vehicle"
        direction_map = {"AIRPORT_ARRIVAL": "Arrival", "AIRPORT_DEPARTURE": "Departure", "AIRPORT_BOTH": "Both ways"}
        direction = direction_map.get(raw_u, "Arrival")
        state["airport_direction"] = direction
        return (
            f"Airport Transfer: {direction}\n\nSelect vehicle type:",
            [
                {"content_type": "text", "title": "Sedan (1-3 pax)",    "payload": "AIRPORT_SEDAN"},
                {"content_type": "text", "title": "SUV (1-6 pax)",      "payload": "AIRPORT_SUV"},
                {"content_type": "text", "title": "Luxury Van (1-8 pax)","payload": "AIRPORT_VAN"},
            ],
        )

    if raw_u in ("AIRPORT_SEDAN", "AIRPORT_SUV", "AIRPORT_VAN"):
        state["pre_arrival_step"] = None
        vehicle = {"AIRPORT_SEDAN": "Sedan", "AIRPORT_SUV": "SUV", "AIRPORT_VAN": "Luxury Van"}[raw_u]
        flight  = state.pop("airport_flight", "")
        direction = state.pop("airport_direction", "Arrival")
        flight_info = f" (Flight: {flight})" if flight else ""
        return (
            f"Airport Transfer Confirmed!\n\n"
            f"Vehicle : {vehicle}\n"
            f"Transfer: {direction}{flight_info}\n\n"
            "Your driver will be waiting at arrivals with a name board.\n"
            "Confirmation sent to your email.",
            [
                {"content_type": "text", "title": "More Services", "payload": "PRE_ARRIVAL"},
                {"content_type": "text", "title": "My Bookings",   "payload": "MY_BOOKINGS"},
            ],
        )

    # Room upgrade
    if raw_u == "ROOM_UPGRADE" or any(kw in en_lower for kw in ("upgrade", "suite", "better room")):
        state["pre_arrival_step"] = None
        return (
            "Room Upgrade Request\n\n"
            "I have sent a complimentary upgrade request to the hotel.\n\n"
            "Hotels may offer free upgrades based on availability,\n"
            "especially for loyalty members and special occasions.\n"
            "The hotel will confirm upon check-in.\n\n"
            "Would you like to add a special occasion note?",
            [
                {"content_type": "text", "title": "It's my birthday",   "payload": "SPECIAL_BIRTHDAY"},
                {"content_type": "text", "title": "It's our anniversary","payload": "SPECIAL_ANNIVERSARY"},
                {"content_type": "text", "title": "It's our honeymoon", "payload": "SPECIAL_HONEYMOON"},
                {"content_type": "text", "title": "No occasion",        "payload": "PRE_ARRIVAL"},
            ],
        )

    # Special occasion
    if raw_u in ("SPECIAL_OCCASION", "SPECIAL_BIRTHDAY", "SPECIAL_ANNIVERSARY", "SPECIAL_HONEYMOON",
                 "SPECIAL_PROPOSAL"):
        occasion_map = {
            "SPECIAL_BIRTHDAY":    ("Birthday", "🎂 Customised cake: +50\nDecorations in room: +30"),
            "SPECIAL_ANNIVERSARY": ("Anniversary", "🌹 Rose petals + champagne: +35\n🕯️ Candlelight dinner: +45\n💐 Flower arrangement: +12\n🌟 Full Romance Bundle: +120 (save 50)"),
            "SPECIAL_HONEYMOON":   ("Honeymoon", "🌹 Honeymoon setup (petals + champagne + turndown): +50\n💑 Couples massage: +65\n🌟 Full Honeymoon Bundle: +150 (save 70)"),
            "SPECIAL_OCCASION":    ("Celebration", "Choose your occasion below."),
            "SPECIAL_PROPOSAL":    ("Proposal", "💍 Proposal setup (flowers, ring presentation stage): +80"),
        }
        occ_name, occ_text = occasion_map.get(raw_u, ("Celebration", ""))
        state["special_occasion"] = occ_name
        state["pre_arrival_step"] = None
        return (
            f"Special Occasion: {occ_name}\n\n"
            "Available packages (prices in booking currency):\n"
            f"{occ_text}\n\n"
            "Would you like to add any of these?\n"
            "Type your choice or tap 'Just a Note' to simply notify the hotel.",
            [
                {"content_type": "text", "title": "Just add a note",  "payload": "OCCASION_NOTE"},
                {"content_type": "text", "title": "Back",             "payload": "PRE_ARRIVAL"},
            ],
        )

    if raw_u == "OCCASION_NOTE":
        state["pre_arrival_step"] = None
        return (
            "Note sent to the hotel! They'll prepare something special for your occasion.\n\n"
            "Is there anything else you need?",
            _MAIN_BUTTONS,
        )

    # Dietary needs
    if raw_u == "DIETARY_NEEDS" or any(kw in en_lower for kw in ("vegetarian","vegan","halal","kosher","gluten","allergy","dairy")):
        state["pre_arrival_step"] = None
        return (
            "Dietary Requirements\n\n"
            "Please select all that apply — I will add this to your booking note for the hotel:",
            [
                {"content_type": "text", "title": "Vegetarian",    "payload": "DIET_VEG"},
                {"content_type": "text", "title": "Vegan",         "payload": "DIET_VEGAN"},
                {"content_type": "text", "title": "Halal",         "payload": "DIET_HALAL"},
                {"content_type": "text", "title": "Kosher",        "payload": "DIET_KOSHER"},
                {"content_type": "text", "title": "Gluten-Free",   "payload": "DIET_GLUTEN"},
                {"content_type": "text", "title": "Nut Allergy!",  "payload": "DIET_NUT"},
            ],
        )

    if raw_u.startswith("DIET_"):
        diet_labels = {"DIET_VEG": "Vegetarian", "DIET_VEGAN": "Vegan",
                       "DIET_HALAL": "Halal", "DIET_KOSHER": "Kosher",
                       "DIET_GLUTEN": "Gluten-Free", "DIET_NUT": "Severe Nut Allergy"}
        diet_label = diet_labels.get(raw_u, "Dietary requirement")
        urgent = "URGENT — " if raw_u == "DIET_NUT" else ""
        state["pre_arrival_step"] = None
        return (
            f"Dietary requirement noted: {diet_label}\n\n"
            f"{urgent}This has been added to your booking and the hotel has been notified.\n"
            "The hotel's F&B team will ensure your requirements are met.",
            _MAIN_BUTTONS,
        )

    state["pre_arrival_step"] = None
    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# IN-STAY CONCIERGE FLOW
# ─────────────────────────────────────────────────────────────────────────────

def _handle_in_stay_flow(
    sender_id: str, state: dict, raw_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle in-stay services: spa, restaurant, room service, housekeeping, complaint, local tips."""
    raw_u = raw_message.strip().upper()

    _in_stay_kw = {
        "spa", "massage", "wellness", "facial",
        "restaurant", "dinner", "lunch", "table", "dining", "reservation",
        "room service", "food delivery", "order food", "eat in room",
        "towels", "clean my room", "housekeeping", "pillow", "blanket",
        "what to do", "local", "attraction", "sightseeing", "recommendation",
        "weather", "forecast", "temperature",
        "complaint", "not working", "problem with room", "ac not", "broken",
        "late checkout", "check out late", "extend checkout",
        "lost item", "left something",
    }

    in_stay_step = state.get("in_stay_step")

    is_trigger = (
        raw_u in ("IN_STAY_MENU", "SPA_BOOKING", "RESTAURANT_BOOKING",
                  "ROOM_SERVICE", "HOUSEKEEPING", "LOCAL_TIPS", "COMPLAINT",
                  "LATE_CHECKOUT", "LOST_FOUND")
        or in_stay_step is not None
        or any(kw in en_lower for kw in _in_stay_kw)
    )
    if not is_trigger:
        return None, []

    # Spa booking
    if raw_u == "SPA_BOOKING" or any(kw in en_lower for kw in ("spa", "massage", "wellness", "facial")):
        if in_stay_step == "spa_time":
            state["in_stay_step"] = None
            time_picked = raw_message.strip()
            return (
                f"Spa appointment confirmed!\n\n"
                f"Time: {time_picked}\n"
                "Please arrive 10 minutes early.\n"
                "Confirmation sent to your email.",
                _MAIN_BUTTONS,
            )
        state["in_stay_step"] = "spa_time"
        return (
            "Spa Booking\n\nAvailable treatments:\n\n"
            "Swedish Massage (60 min)     \n"
            "Deep Tissue Massage (60 min) \n"
            "Aromatherapy (90 min)        \n"
            "Couples Massage (60 min)     \n"
            "Full Body Ritual (3 hours)   \n\n"
            "What time for your appointment?",
            [
                {"content_type": "text", "title": "2:00 PM",  "payload": "SPA_TIME_1400"},
                {"content_type": "text", "title": "4:00 PM",  "payload": "SPA_TIME_1600"},
                {"content_type": "text", "title": "5:00 PM",  "payload": "SPA_TIME_1700"},
                {"content_type": "text", "title": "6:00 PM",  "payload": "SPA_TIME_1800"},
                {"content_type": "text", "title": "Call Spa", "payload": "CALL_SPA"},
            ],
        )

    if raw_u.startswith("SPA_TIME_"):
        time_val = raw_u[9:]
        time_disp = f"{time_val[:2]}:{time_val[2:]}"
        state["in_stay_step"] = None
        return (
            f"Spa appointment confirmed for {time_disp}!\n"
            "Please arrive 10 minutes early.\n"
            "Confirmation sent to your email.",
            _MAIN_BUTTONS,
        )

    if raw_u == "CALL_SPA":
        state["in_stay_step"] = None
        return ("You can call the spa directly at the hotel front desk. They are available 9 AM - 9 PM.", _MAIN_BUTTONS)

    # Restaurant booking
    if raw_u == "RESTAURANT_BOOKING" or any(kw in en_lower for kw in ("restaurant", "dinner", "lunch", "table reservation")):
        if in_stay_step == "restaurant_time":
            state["in_stay_step"] = "restaurant_guests"
            state["restaurant_time"] = raw_message.strip()
            return (
                "How many guests for dinner?",
                [
                    {"content_type": "text", "title": "1",   "payload": "REST_1"},
                    {"content_type": "text", "title": "2",   "payload": "REST_2"},
                    {"content_type": "text", "title": "3-4", "payload": "REST_3"},
                    {"content_type": "text", "title": "5+",  "payload": "REST_5"},
                ],
            )
        if in_stay_step == "restaurant_guests":
            state["in_stay_step"] = None
            guests = raw_message.strip()
            time_val = state.pop("restaurant_time", "evening")
            return (
                f"Restaurant reservation confirmed!\n\n"
                f"Time  : {time_val}\n"
                f"Guests: {guests}\n\n"
                "Please present your room number upon arrival.\n"
                "Any special dietary requirements? Let the host know on arrival.",
                _MAIN_BUTTONS,
            )
        state["in_stay_step"] = "restaurant_time"
        return (
            "Restaurant Reservation\n\n"
            "What date and time?",
            [
                {"content_type": "text", "title": "Tonight 7:30 PM",   "payload": "Tonight 7:30 PM"},
                {"content_type": "text", "title": "Tonight 8:00 PM",   "payload": "Tonight 8:00 PM"},
                {"content_type": "text", "title": "Tomorrow 7:30 PM",  "payload": "Tomorrow 7:30 PM"},
                {"content_type": "text", "title": "Call Restaurant",   "payload": "CALL_RESTAURANT"},
            ],
        )

    if raw_u == "CALL_RESTAURANT":
        state["in_stay_step"] = None
        return ("Please call the hotel front desk and ask for restaurant reservations.", _MAIN_BUTTONS)

    # Room service
    if raw_u == "ROOM_SERVICE" or any(kw in en_lower for kw in ("room service", "food to room", "order food", "eat in room")):
        if in_stay_step == "room_service_order":
            state["in_stay_step"] = None
            order = raw_message.strip()
            return (
                f"Room Service Order Placed!\n\n"
                f"Order: {order}\n"
                "Estimated delivery: 25-35 minutes.\n\n"
                "Our team will deliver to your room shortly.",
                _MAIN_BUTTONS,
            )
        state["in_stay_step"] = "room_service_order"
        return (
            "Room Service\n\nCurrent menu highlights:\n\n"
            "BREAKFAST (6 AM - 11 AM)\n"
            "Continental Breakfast | Full English | Omelette + Toast\n\n"
            "ALL DAY (11 AM - 11 PM)\n"
            "Club Sandwich | Caesar Salad | Biryani | Butter Chicken\n"
            "Pasta | Grilled Chicken | Burgers\n\n"
            "DESSERTS: Chocolate Lava Cake | Fruit Platter\n"
            "BEVERAGES: Coffee/Tea | Juices | Soft Drinks\n\n"
            "What would you like to order? (Type your order)",
            [{"content_type": "text", "title": "View Full Menu", "payload": "ROOM_MENU_PDF"},
             {"content_type": "text", "title": "Call Room Service", "payload": "CALL_ROOM_SERVICE"}],
        )

    if raw_u == "CALL_ROOM_SERVICE":
        state["in_stay_step"] = None
        return ("Please call the hotel front desk and ask for room service.", _MAIN_BUTTONS)

    # Housekeeping
    if raw_u == "HOUSEKEEPING" or any(kw in en_lower for kw in ("towel", "clean room", "housekeeping", "pillow", "blanket", "toiletries")):
        state["in_stay_step"] = None
        return (
            "Housekeeping Request\n\nWhat do you need?",
            [
                {"content_type": "text", "title": "Fresh Towels",     "payload": "HK_TOWELS"},
                {"content_type": "text", "title": "Room Cleaning",    "payload": "HK_CLEANING"},
                {"content_type": "text", "title": "Extra Pillows",    "payload": "HK_PILLOWS"},
                {"content_type": "text", "title": "Extra Blanket",    "payload": "HK_BLANKET"},
                {"content_type": "text", "title": "Extra Toiletries", "payload": "HK_TOILETRIES"},
                {"content_type": "text", "title": "Do Not Disturb",   "payload": "HK_DND"},
            ],
        )

    if raw_u.startswith("HK_"):
        hk_map = {
            "HK_TOWELS": "Fresh towels requested! Housekeeping will deliver within 15 minutes.",
            "HK_CLEANING": "Room cleaning requested! Housekeeping will clean your room shortly.",
            "HK_PILLOWS": "Extra pillows on the way! They will be with you in 15 minutes.",
            "HK_BLANKET": "Extra blanket requested! Delivering shortly.",
            "HK_TOILETRIES": "Extra toiletries requested! Delivering within 15 minutes.",
            "HK_DND": "Do Not Disturb mode activated. Housekeeping will not enter your room until you remove this.",
        }
        state["in_stay_step"] = None
        return (hk_map.get(raw_u, "Request sent to housekeeping!"), _MAIN_BUTTONS)

    # Local recommendations
    if raw_u == "LOCAL_TIPS" or any(kw in en_lower for kw in ("what to do", "attraction", "sightseeing", "near hotel", "restaurant nearby", "local")):
        state["in_stay_step"] = None
        city = state.get("city", "your city")
        return (
            f"Top Recommendations near your hotel in {city}!\n\n"
            "TOP ATTRACTIONS\n"
            "Ask the hotel concierge desk for a personalised city guide\n"
            "tailored to your interests.\n\n"
            "TOP DINING\n"
            "Ask the hotel restaurant team for their favourite local spots.\n\n"
            "TOP ACTIVITIES\n"
            "City tours, cultural experiences, and adventure activities\n"
            "can be arranged through the hotel concierge.",
            [
                {"content_type": "text", "title": "Book Cab",       "payload": "BOOK_CAB"},
                {"content_type": "text", "title": "Concierge Help", "payload": "AGENT_HANDOFF"},
                {"content_type": "text", "title": "Main Menu",      "payload": "RESTART"},
            ],
        )

    # Complaint
    if raw_u == "COMPLAINT" or any(kw in en_lower for kw in ("complaint", "not working", "problem with", "ac not", "broken", "not happy", "terrible", "issue")):
        if in_stay_step == "complaint_detail":
            state["in_stay_step"] = None
            issue = raw_message.strip()
            import random
            comp_ref = f"COMP-{date.today().strftime('%Y%m%d')}-{random.randint(100,999)}"
            return (
                f"Complaint registered — I am escalating this immediately.\n\n"
                f"Issue   : {issue}\n"
                f"Reference: {comp_ref}\n\n"
                "A hotel staff member will attend to you within 15 minutes.\n\n"
                "Would you like to be moved to a different room?",
                [
                    {"content_type": "text", "title": "Yes, move me please", "payload": "ROOM_CHANGE_REQ"},
                    {"content_type": "text", "title": "No, just fix it",     "payload": "RESTART"},
                    {"content_type": "text", "title": "Speak to Manager",    "payload": "AGENT_HANDOFF"},
                ],
            )
        state["in_stay_step"] = "complaint_detail"
        return (
            "I am sorry to hear that. Let me get this sorted immediately!\n\n"
            "What is the issue?",
            [
                {"content_type": "text", "title": "AC/Heating",    "payload": "COMP_AC"},
                {"content_type": "text", "title": "Noise",         "payload": "COMP_NOISE"},
                {"content_type": "text", "title": "Cleanliness",   "payload": "COMP_CLEAN"},
                {"content_type": "text", "title": "TV/WiFi",       "payload": "COMP_TECH"},
                {"content_type": "text", "title": "Plumbing",      "payload": "COMP_PLUMBING"},
                {"content_type": "text", "title": "Other",         "payload": "COMP_OTHER"},
            ],
        )

    if raw_u.startswith("COMP_"):
        comp_map = {
            "COMP_AC": "Air conditioning/heating issue",
            "COMP_NOISE": "Noise complaint",
            "COMP_CLEAN": "Cleanliness issue",
            "COMP_TECH": "TV/WiFi not working",
            "COMP_PLUMBING": "Plumbing/hot water issue",
            "COMP_OTHER": "Other issue",
        }
        issue = comp_map.get(raw_u, "Issue")
        state["in_stay_step"] = None
        import random
        comp_ref = f"COMP-{date.today().strftime('%Y%m%d')}-{random.randint(100,999)}"
        return (
            f"Complaint: {issue}\n\n"
            f"Reference: {comp_ref}\n"
            "A staff member will attend within 15 minutes.\n\n"
            "Would you like to speak to a manager?",
            [
                {"content_type": "text", "title": "Yes, Manager please", "payload": "AGENT_HANDOFF"},
                {"content_type": "text", "title": "No, that's fine",     "payload": "RESTART"},
            ],
        )

    # Late checkout
    if raw_u == "LATE_CHECKOUT" or any(kw in en_lower for kw in ("late checkout", "check out late", "extend checkout", "stay past noon")):
        state["in_stay_step"] = None
        return (
            "Late Check-out Request\n\n"
            "Standard check-out is 12:00 PM.\n\nOptions:",
            [
                {"content_type": "text", "title": "Until 2 PM (free request)", "payload": "LCO_2PM"},
                {"content_type": "text", "title": "Until 4 PM (fee applies)", "payload": "LCO_4PM"},
                {"content_type": "text", "title": "Until 6 PM (full rate)",   "payload": "LCO_6PM"},
                {"content_type": "text", "title": "Back",                     "payload": "RESTART"},
            ],
        )

    if raw_u in ("LCO_2PM", "LCO_4PM", "LCO_6PM"):
        state["in_stay_step"] = None
        lco_map = {
            "LCO_2PM": ("2:00 PM", "Complimentary (subject to availability)"),
            "LCO_4PM": ("4:00 PM", "Fee applies — settled at check-out"),
            "LCO_6PM": ("6:00 PM", "Full nightly rate applies"),
        }
        time_val, fee_note = lco_map[raw_u]
        return (
            f"Late check-out until {time_val} requested.\n"
            f"{fee_note}\n\n"
            "The hotel will confirm availability on your check-out morning.",
            _MAIN_BUTTONS,
        )

    # Lost & Found
    if raw_u == "LOST_FOUND" or any(kw in en_lower for kw in ("lost", "left something", "forgot", "missing item")):
        if in_stay_step == "lost_found_describe":
            state["in_stay_step"] = None
            item = raw_message.strip()
            import random
            lost_ref = f"LOST-{date.today().strftime('%Y%m%d')}-{random.randint(100,999)}"
            return (
                f"Lost Item Report Submitted\n\n"
                f"Item(s)   : {item}\n"
                f"Report    : {lost_ref}\n\n"
                "The hotel will search and contact you within 24 hours.\n"
                "Would you like them to ship it to you if found?",
                [
                    {"content_type": "text", "title": "Yes, ship it to me", "payload": "LOST_SHIP"},
                    {"content_type": "text", "title": "I'll collect myself","payload": "RESTART"},
                ],
            )
        state["in_stay_step"] = "lost_found_describe"
        return (
            "Lost & Found\n\nI am sorry to hear that!\n"
            "What did you leave behind? (describe the items)",
            [],
        )

    if raw_u == "LOST_SHIP":
        state["in_stay_step"] = None
        return (
            "Shipment request noted. The hotel will contact you with shipping options if the item is found.",
            _MAIN_BUTTONS,
        )

    if raw_u in ("BOOK_CAB", "ROOM_CHANGE_REQ"):
        state["in_stay_step"] = None
        return (
            "Please ask the hotel concierge desk to arrange this for you.\n"
            "They are available 24/7 at the front desk.",
            _MAIN_BUTTONS,
        )

    state["in_stay_step"] = None
    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# BOOKING MODIFICATION FLOW
# ─────────────────────────────────────────────────────────────────────────────

def _handle_modification_flow(
    sender_id: str, state: dict, raw_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle booking modification: change dates, change room type, re-book."""
    raw_u = raw_message.strip().upper()
    _mod_kw = {"modify booking", "change booking", "change dates", "extend stay",
               "shorten stay", "change room", "different room", "re-book",
               "rebook same", "book same hotel again"}
    mod_step = state.get("mod_step")

    if raw_u not in ("MODIFY_BOOKING",) and mod_step is None \
            and not any(kw in en_lower for kw in _mod_kw):
        return None, []

    # Start modification
    if raw_u == "MODIFY_BOOKING" or (mod_step is None and any(kw in en_lower for kw in _mod_kw)):
        if not _DB_AVAILABLE:
            return ("Booking modification is not available right now. Please try again later.", _MAIN_BUTTONS)
        # Show existing bookings
        user_id = state.get("user_id")
        bookings = []
        if user_id:
            try:
                bookings = get_user_bookings(user_id)[:3]
            except Exception:
                pass
        if not bookings:
            return (
                "No active bookings found to modify.\n\nTap Book a Hotel to make a new reservation.",
                _MAIN_BUTTONS,
            )
        state["mod_step"] = "select_booking"
        lines = [f"{i+1}. {b.get('booking_reference','N/A')} — {b.get('hotel_name','Hotel')} [{b.get('check_in','')} to {b.get('check_out','')}]"
                 for i, b in enumerate(bookings)]
        buttons = [{"content_type": "text", "title": f"{i+1}. {b.get('booking_reference','Book')}"[:20],
                    "payload": f"MODSEL_{b.get('booking_reference','')}"} for i, b in enumerate(bookings)]
        buttons.append({"content_type": "text", "title": "Back", "payload": "RESTART"})
        state["_mod_bookings"] = bookings
        return ("Which booking to modify?\n\n" + "\n".join(lines), buttons[:5])

    if mod_step == "select_booking" and raw_u.startswith("MODSEL_"):
        ref = raw_u[7:]
        state["mod_booking_ref"] = ref
        state["mod_step"] = "choose_field"
        return (
            f"Modifying booking {ref}.\n\nWhat would you like to change?",
            [
                {"content_type": "text", "title": "Change Check-in",   "payload": "MOD_FIELD_CHECKIN"},
                {"content_type": "text", "title": "Change Check-out",  "payload": "MOD_FIELD_CHECKOUT"},
                {"content_type": "text", "title": "Extend Stay",       "payload": "MOD_FIELD_EXTEND"},
                {"content_type": "text", "title": "Change Room Type",  "payload": "MOD_FIELD_ROOM"},
                {"content_type": "text", "title": "Back",              "payload": "RESTART"},
            ],
        )

    if mod_step == "choose_field":
        if raw_u == "MOD_FIELD_CHECKIN":
            state["mod_step"] = "new_checkin"
            return ("Enter the new check-in date:", _checkin_buttons())
        if raw_u == "MOD_FIELD_CHECKOUT":
            state["mod_step"] = "new_checkout"
            return ("Enter the new check-out date:", [])
        if raw_u == "MOD_FIELD_EXTEND":
            state["mod_step"] = "extend_nights"
            return (
                "Extend stay by how many nights?",
                [
                    {"content_type": "text", "title": "+1 night",  "payload": "EXTEND_1"},
                    {"content_type": "text", "title": "+2 nights", "payload": "EXTEND_2"},
                    {"content_type": "text", "title": "+3 nights", "payload": "EXTEND_3"},
                    {"content_type": "text", "title": "+7 nights", "payload": "EXTEND_7"},
                ],
            )
        if raw_u == "MOD_FIELD_ROOM":
            state["mod_step"] = None
            ref = state.pop("mod_booking_ref", "")
            return (
                f"Room change request for {ref} noted.\n\n"
                "Please specify the room type you want.\n"
                "Our team will check availability and confirm.",
                [
                    {"content_type": "text", "title": "Twin Beds",     "payload": "ROOM_TWIN"},
                    {"content_type": "text", "title": "King Bed",      "payload": "ROOM_KING"},
                    {"content_type": "text", "title": "Sea View",      "payload": "ROOM_SEA_VIEW"},
                    {"content_type": "text", "title": "Suite Upgrade", "payload": "ROOM_SUITE"},
                ],
            )

    if mod_step == "new_checkin":
        new_date = _parse_date(raw_message.strip())
        state["mod_step"] = None
        ref = state.pop("mod_booking_ref", "")
        if new_date:
            return (
                f"Check-in date change requested for {ref}.\nNew check-in: {_pretty_date(new_date)}\n\n"
                "Our team will confirm the change and any price difference.",
                _MAIN_BUTTONS,
            )
        return ("I could not parse that date. Please try again.", _checkin_buttons())

    if mod_step == "new_checkout":
        new_date = _parse_date(raw_message.strip())
        state["mod_step"] = None
        ref = state.pop("mod_booking_ref", "")
        if new_date:
            return (
                f"Check-out date change requested for {ref}.\nNew check-out: {_pretty_date(new_date)}\n\n"
                "Our team will confirm the change and any price difference.",
                _MAIN_BUTTONS,
            )
        return ("I could not parse that date. Please try again.", [])

    if mod_step == "extend_nights" or raw_u.startswith("EXTEND_"):
        nights_add = 1
        if raw_u.startswith("EXTEND_"):
            try:
                nights_add = int(raw_u[7:])
            except Exception:
                pass
        else:
            try:
                nights_add = int(re.search(r'\d+', raw_message).group())
            except Exception:
                pass
        state["mod_step"] = None
        ref = state.pop("mod_booking_ref", "")
        return (
            f"Extension of +{nights_add} night{'s' if nights_add > 1 else ''} requested for {ref}.\n\n"
            "We will check availability and any additional charges.\n"
            "You will receive a confirmation message shortly.",
            _MAIN_BUTTONS,
        )

    if raw_u in ("ROOM_TWIN", "ROOM_KING", "ROOM_SEA_VIEW", "ROOM_SUITE"):
        state["mod_step"] = None
        room_labels = {"ROOM_TWIN": "Twin Beds", "ROOM_KING": "King Bed",
                       "ROOM_SEA_VIEW": "Sea View Room", "ROOM_SUITE": "Suite Upgrade"}
        room_label = room_labels.get(raw_u, "Room change")
        return (
            f"Room change to {room_label} requested.\n\n"
            "We will check availability and confirm within 2 hours.",
            _MAIN_BUTTONS,
        )

    state["mod_step"] = None
    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# LIVE AGENT HANDOFF FLOW
# ─────────────────────────────────────────────────────────────────────────────

def _handle_agent_handoff(
    sender_id: str, state: dict, raw_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle live agent request / escalation."""
    raw_u = raw_message.strip().upper()
    _agent_kw = {"agent", "human", "speak to someone", "real person", "manager",
                 "customer service", "support", "help me now", "live person",
                 "talk to person", "escalate", "complaint", "live agent"}

    if raw_u not in ("AGENT_HANDOFF", "HANDOFF_REQUEST") \
            and not any(kw in en_lower for kw in _agent_kw):
        return None, []

    if raw_u == "LEAVE_MESSAGE":
        state["handoff_step"] = "leave_message"
        return ("Please type your message for the agent:", [])

    if state.get("handoff_step") == "leave_message":
        state.pop("handoff_step", None)
        import random
        ticket_id = f"TKT-{date.today().strftime('%Y%m%d')}-{random.randint(100,999):03d}"
        return (
            f"Message received!\n\n"
            f"Ticket : {ticket_id}\n"
            f"Priority: Standard\n\n"
            "Our team will respond within 2 hours via Messenger.\n"
            "You will receive a notification here.",
            _MAIN_BUTTONS,
        )

    # Standard handoff
    import random
    ticket_id = f"TKT-{date.today().strftime('%Y%m%d')}-{random.randint(100,999):03d}"
    state["handoff_step"] = None
    return (
        f"Connecting you to a live agent now.\n\n"
        f"Ticket : {ticket_id}\n"
        f"Estimated wait: 2-5 minutes.\n\n"
        "You can type your concern while you wait.\n"
        "Our agent will have your full chat history for context.",
        [
            {"content_type": "text", "title": "Leave a Message", "payload": "LEAVE_MESSAGE"},
            {"content_type": "text", "title": "Continue with Bot","payload": "RESTART"},
            {"content_type": "text", "title": "Call Us",          "payload": "CALL_US"},
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED BOOKING TYPES FLOW
# ─────────────────────────────────────────────────────────────────────────────

def _handle_advanced_booking_flow(
    sender_id: str, state: dict, raw_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle: group booking, corporate, wedding, long stay, last-minute, early bird,
    accessible, pet-friendly, multi-city, honeymoon/anniversary."""
    raw_u = raw_message.strip().upper()
    adv_step = state.get("adv_step")

    _group_kw = {"group booking", "group of", "15 rooms", "20 rooms", "team booking", "conference booking", "event booking"}
    _corp_kw  = {"corporate booking", "business travel", "company booking", "gst invoice", "vat invoice", "po number", "purchase order"}
    _wedding_kw = {"wedding", "wedding venue", "wedding package", "rooms for wedding"}
    _longstay_kw = {"long stay", "monthly rate", "weekly rate", "one month", "1 month", "serviced apartment", "extended stay"}
    _last_min_kw = {"tonight", "tonight hotel", "last minute", "available now", "hotel tonight", "available today"}
    _accessible_kw = {"wheelchair", "accessible room", "disability", "mobility issue", "accessible hotel"}
    _pet_kw  = {"pet friendly", "travelling with my dog", "travelling with my cat", "can i bring my pet", "pet hotel", "dog friendly"}
    _multi_kw = {"multi-city", "multiple cities", "3 cities", "2 cities", "itinerary booking"}
    _romantic_kw = {"honeymoon", "honeymoon package", "anniversary package", "romantic getaway"}

    # Group booking
    if raw_u == "GROUP_BOOKING" or any(kw in en_lower for kw in _group_kw):
        if adv_step == "group_rooms":
            try:
                rooms = int(re.search(r'\d+', raw_message).group())
            except Exception:
                rooms = 10
            state["group_rooms"] = rooms
            state["adv_step"] = "group_city"
            return (f"{rooms} rooms — got it.\n\nWhich city is your event in?", [])
        if adv_step == "group_city":
            state["group_city"] = raw_message.strip()
            state["adv_step"] = "group_dates"
            return ("Event/arrival date?", _checkin_buttons())
        if adv_step == "group_dates":
            parsed = _parse_date(raw_message.strip())
            state["group_date"] = parsed or raw_message.strip()
            state["adv_step"] = "group_type"
            return (
                "What type of event?",
                [
                    {"content_type": "text", "title": "Corporate Conference","payload": "GTYPE_CORP"},
                    {"content_type": "text", "title": "Training/Workshop",   "payload": "GTYPE_TRAINING"},
                    {"content_type": "text", "title": "Wedding",             "payload": "GTYPE_WEDDING"},
                    {"content_type": "text", "title": "Social Event",        "payload": "GTYPE_SOCIAL"},
                    {"content_type": "text", "title": "Other",               "payload": "GTYPE_OTHER"},
                ],
            )
        if adv_step == "group_type" or raw_u.startswith("GTYPE_"):
            state["adv_step"] = "group_contact"
            event_type = raw_message.strip()
            state["group_event_type"] = event_type
            return ("Company name and your name for the enquiry?", [])
        if adv_step == "group_contact":
            state["adv_step"] = "group_email"
            state["group_contact"] = raw_message.strip()
            return ("Invoice email address?", [])
        if adv_step == "group_email":
            state["adv_step"] = None
            import random
            grp_ref = f"GRP-{date.today().strftime('%Y%m%d')}-{random.randint(100,999):03d}"
            city = state.pop("group_city", "")
            rooms = state.pop("group_rooms", "")
            event_date = state.pop("group_date", "")
            contact = state.pop("group_contact", "")
            state.pop("group_event_type", None)
            return (
                f"Group Enquiry Submitted!\n\n"
                f"Reference : {grp_ref}\n"
                f"Company   : {contact}\n"
                f"City      : {city}\n"
                f"Rooms     : {rooms}\n"
                f"Date      : {event_date}\n\n"
                "Our group sales team will contact you within 2 hours\n"
                "with a customised quote.",
                _MAIN_BUTTONS,
            )
        # Initial trigger
        state["adv_step"] = "group_rooms"
        return (
            "Group Booking\n\n"
            "I specialise in group bookings of 10+ rooms.\n\n"
            "How many rooms do you need?",
            [
                {"content_type": "text", "title": "10 rooms",     "payload": "10"},
                {"content_type": "text", "title": "15 rooms",     "payload": "15"},
                {"content_type": "text", "title": "20 rooms",     "payload": "20"},
                {"content_type": "text", "title": "30+ rooms",    "payload": "30"},
                {"content_type": "text", "title": "Type a number","payload": "TYPE_ROOMS"},
            ],
        )

    # Corporate booking
    if raw_u == "CORP_BOOKING" or any(kw in en_lower for kw in _corp_kw):
        state["adv_step"] = None
        return (
            "Corporate / Business Travel Booking\n\n"
            "I will collect your company details for GST/VAT invoicing.\n\n"
            "Is your company registered with BookBot Corporate?",
            [
                {"content_type": "text", "title": "Yes - have account", "payload": "CORP_EXISTING"},
                {"content_type": "text", "title": "First time",         "payload": "CORP_NEW"},
                {"content_type": "text", "title": "Set up account",     "payload": "CORP_SETUP"},
            ],
        )

    # Wedding booking
    if raw_u == "WEDDING_BOOKING" or any(kw in en_lower for kw in _wedding_kw):
        state["adv_step"] = None
        return (
            "Wedding Venue & Guest Room Booking\n\nCongratulations!\n\n"
            "What do you need for your wedding?",
            [
                {"content_type": "text", "title": "Guest Rooms Only",       "payload": "WED_ROOMS"},
                {"content_type": "text", "title": "Venue + Guest Rooms",    "payload": "WED_VENUE"},
                {"content_type": "text", "title": "Full Wedding Package",   "payload": "WED_FULL"},
                {"content_type": "text", "title": "Honeymoon Suite",        "payload": "WED_HONEYMOON"},
            ],
        )

    # Long stay
    if raw_u == "LONG_STAY" or any(kw in en_lower for kw in _longstay_kw):
        state["adv_step"] = None
        return (
            "Long Stay Booking\n\n"
            "I can find great weekly and monthly rates.\n"
            "These are often 30-50% cheaper than nightly rates.\n\n"
            "How long would you like to stay?",
            [
                {"content_type": "text", "title": "1 Week",   "payload": "LONGSTAY_1W"},
                {"content_type": "text", "title": "2 Weeks",  "payload": "LONGSTAY_2W"},
                {"content_type": "text", "title": "1 Month",  "payload": "LONGSTAY_1M"},
                {"content_type": "text", "title": "2 Months", "payload": "LONGSTAY_2M"},
                {"content_type": "text", "title": "3+ Months","payload": "LONGSTAY_3M"},
            ],
        )

    if raw_u.startswith("LONGSTAY_"):
        state["adv_step"] = None
        dur_map = {"LONGSTAY_1W": "1 week", "LONGSTAY_2W": "2 weeks",
                   "LONGSTAY_1M": "1 month", "LONGSTAY_2M": "2 months", "LONGSTAY_3M": "3+ months"}
        dur = dur_map.get(raw_u, raw_u)
        return (
            f"Long stay: {dur}\n\n"
            "Looking for hotels with monthly rates and serviced apartments.\n\n"
            "Which city are you looking to stay in?",
            [],
        )

    # Last-minute booking
    if any(kw in en_lower for kw in _last_min_kw):
        state["adv_step"] = None
        return (
            "Last-Minute Booking — Let me find you something great!\n\n"
            "Searching hotels with GUARANTEED same-day availability...\n\n"
            "I will search for tonight's best deals.\n"
            "Note: Last-minute bookings require immediate payment.\n\n"
            "Which city are you looking for tonight?",
            [
                {"content_type": "text", "title": "Dubai",     "payload": "CITY_DUBAI"},
                {"content_type": "text", "title": "Mumbai",    "payload": "CITY_MUMBAI"},
                {"content_type": "text", "title": "London",    "payload": "CITY_LONDON"},
                {"content_type": "text", "title": "Other city","payload": "ACTION_BOOK"},
            ],
        )

    # Accessible room
    if raw_u == "ACCESSIBLE_BOOKING" or any(kw in en_lower for kw in _accessible_kw):
        state["adv_step"] = None
        return (
            "Accessible Room Booking\n\n"
            "I will filter for fully accessible hotels and rooms.\n\n"
            "Which accessibility features do you need?",
            [
                {"content_type": "text", "title": "Wheelchair Room",    "payload": "ACC_WHEELCHAIR"},
                {"content_type": "text", "title": "Roll-in Shower",     "payload": "ACC_SHOWER"},
                {"content_type": "text", "title": "Ground Floor",       "payload": "ACC_GROUND"},
                {"content_type": "text", "title": "Hearing Assistance", "payload": "ACC_HEARING"},
                {"content_type": "text", "title": "All of the above",   "payload": "ACC_ALL"},
                {"content_type": "text", "title": "Continue Booking",   "payload": "ACTION_BOOK"},
            ],
        )

    if raw_u.startswith("ACC_"):
        state["adv_step"] = None
        return (
            "Accessibility requirements noted! I will only show hotels with verified accessible facilities.\n\n"
            "Your booking will include a note to the hotel about your requirements.\n\n"
            "Let's find an accessible hotel for you:",
            [{"content_type": "text", "title": "Search Hotels", "payload": "ACTION_BOOK"}],
        )

    # Pet-friendly
    if raw_u == "PET_FRIENDLY" or any(kw in en_lower for kw in _pet_kw):
        state["adv_step"] = None
        return (
            "Pet-Friendly Hotel Booking\n\n"
            "Let me filter for pet-friendly hotels.\n\n"
            "What type of pet?",
            [
                {"content_type": "text", "title": "Dog",   "payload": "PET_DOG"},
                {"content_type": "text", "title": "Cat",   "payload": "PET_CAT"},
                {"content_type": "text", "title": "Other", "payload": "PET_OTHER"},
            ],
        )

    if raw_u.startswith("PET_"):
        state["adv_step"] = None
        pet_map = {"PET_DOG": "dog", "PET_CAT": "cat", "PET_OTHER": "pet"}
        pet = pet_map.get(raw_u, "pet")
        return (
            f"Pet-friendly hotels for your {pet}!\n\n"
            "I will filter results to only show pet-friendly properties.\n"
            "Note: Most hotels charge a pet fee at check-in.\n\n"
            "Which city are you looking in?",
            [],
        )

    # Multi-city
    if raw_u == "MULTI_CITY" or any(kw in en_lower for kw in _multi_kw):
        state["adv_step"] = None
        return (
            "Multi-City Itinerary Booking\n\n"
            "Let me set up your trip leg by leg.\n\n"
            "How many cities are you visiting?",
            [
                {"content_type": "text", "title": "2 cities",  "payload": "MCITY_2"},
                {"content_type": "text", "title": "3 cities",  "payload": "MCITY_3"},
                {"content_type": "text", "title": "4 cities",  "payload": "MCITY_4"},
                {"content_type": "text", "title": "5+ cities", "payload": "MCITY_5"},
            ],
        )

    if raw_u.startswith("MCITY_"):
        state["adv_step"] = None
        cities = raw_u[6:]
        return (
            f"Multi-city trip: {cities} cities.\n\n"
            "Let's start with Leg 1.\n\n"
            "Which city is your first destination?",
            [],
        )

    # Honeymoon / anniversary / romantic
    if raw_u == "ROMANTIC_BOOKING" or any(kw in en_lower for kw in _romantic_kw):
        state["adv_step"] = None
        return (
            "Romantic & Honeymoon Packages\n\nHow romantic!\n\n"
            "Let me find the perfect romantic escape.\n\n"
            "Destination preference?",
            [
                {"content_type": "text", "title": "Beach / Island",    "payload": "ROM_BEACH"},
                {"content_type": "text", "title": "Mountains",         "payload": "ROM_MOUNTAINS"},
                {"content_type": "text", "title": "City Luxury",       "payload": "ROM_CITY"},
                {"content_type": "text", "title": "Tropical / Exotic", "payload": "ROM_TROPICAL"},
                {"content_type": "text", "title": "Heritage / Palace", "payload": "ROM_PALACE"},
                {"content_type": "text", "title": "I have a place",    "payload": "ACTION_BOOK"},
            ],
        )

    if raw_u.startswith("ROM_"):
        state["adv_step"] = None
        rom_dest = {"ROM_BEACH": "Beach/Island destinations", "ROM_MOUNTAINS": "Mountain retreats",
                    "ROM_CITY": "City luxury stays", "ROM_TROPICAL": "Tropical destinations",
                    "ROM_PALACE": "Heritage palace hotels"}.get(raw_u, "Romantic destinations")
        return (
            f"{rom_dest}\n\n"
            "Let me find the best romantic hotels for you.\n"
            "Bot automatically suggests Romance Package add-on.\n\n"
            "When are you planning to travel?",
            _checkin_buttons(),
        )

    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# PART U — SEASONAL & EVENT BOOKING FLOWS
# ─────────────────────────────────────────────────────────────────────────────

_SEASONAL_KEYWORDS = {
    "new year": "nye", "new years eve": "nye", "nye": "nye", "31 december": "nye",
    "christmas": "xmas", "xmas": "xmas", "festive": "xmas",
    "valentine": "valentines", "romantic": "valentines",
    "ramadan": "ramadan", "iftar": "ramadan", "suhoor": "ramadan",
    "eid": "eid",
    "summer": "summer", "monsoon": "monsoon", "off-peak": "monsoon", "off peak": "monsoon",
    "conference": "conference", "trade show": "conference", "ces": "conference",
    "convention": "conference", "summit": "conference",
    "match": "sports", "champions league": "sports", "world cup": "sports",
    "grand prix": "sports", "formula 1": "sports", "f1": "sports",
    "festival": "festival", "oktoberfest": "festival", "carnival": "festival",
}


def _handle_seasonal_flow(
    sender_id: str, state: dict, user_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle seasonal, event, and festival booking flows (Part U)."""
    raw_u = user_message.strip().upper()

    # Detect seasonal event from NL
    detected_event = None
    for kw, event_type in _SEASONAL_KEYWORDS.items():
        if kw in en_lower:
            detected_event = event_type
            break

    # Button-triggered sub-flows
    if raw_u == "NYE_BOOK" or detected_event == "nye":
        city = state.get("city", "Dubai")
        return (
            f"🎆 New Year's Eve in {city} — Amazing Choice!\n\n"
            "December 31st is our most popular night of the year.\n"
            "NYE packages include:\n"
            "✅ Gala dinner\n"
            "✅ Fireworks view room\n"
            "✅ Champagne on arrival\n"
            "✅ Midnight countdown event\n\n"
            "⚠️ NYE rooms sell out fast — prices are 2–5× higher.\n"
            "Shall I search NYE packages for you?\n\n"
            "Which city for New Year's Eve?",
            [
                {"content_type": "text", "title": "Dubai", "payload": "NYE_DUBAI"},
                {"content_type": "text", "title": "London", "payload": "NYE_LONDON"},
                {"content_type": "text", "title": "Paris", "payload": "NYE_PARIS"},
                {"content_type": "text", "title": "New York", "payload": "NYE_NYC"},
                {"content_type": "text", "title": "Sydney", "payload": "NYE_SYDNEY"},
                {"content_type": "text", "title": "Other City", "payload": "ACTION_BOOK"},
            ],
        )

    if raw_u in ("NYE_DUBAI", "NYE_LONDON", "NYE_PARIS", "NYE_NYC", "NYE_SYDNEY"):
        city_map = {
            "NYE_DUBAI": "Dubai", "NYE_LONDON": "London", "NYE_PARIS": "Paris",
            "NYE_NYC": "New York", "NYE_SYDNEY": "Sydney",
        }
        city = city_map[raw_u]
        state["city"] = city
        state["_seasonal_event"] = "nye"
        state["step"] = "checkin"
        return (
            f"🎆 NYE Package Search — {city}\n\n"
            "I'll filter for hotels with fireworks view, gala dinner,\n"
            "and countdown event packages.\n\n"
            "Check-in date (usually Dec 30 or 31)?",
            [
                {"content_type": "text", "title": "30 Dec", "payload": "CHECKIN_2026-12-30"},
                {"content_type": "text", "title": "31 Dec", "payload": "CHECKIN_2026-12-31"},
            ],
        )

    if detected_event == "xmas":
        city = state.get("city", "")
        return (
            "🎄 Christmas Holiday Booking!\n\n"
            "I'll search for family-friendly hotels with Christmas packages:\n"
            "✅ Santa meet & greet for children\n"
            "✅ Christmas Day lunch/dinner included\n"
            "✅ Festive decorations & activities\n"
            "✅ Family suite options\n\n"
            "Which city for Christmas?",
            [
                {"content_type": "text", "title": "London", "payload": "XMAS_LONDON"},
                {"content_type": "text", "title": "Paris",  "payload": "XMAS_PARIS"},
                {"content_type": "text", "title": "Vienna", "payload": "XMAS_VIENNA"},
                {"content_type": "text", "title": "Other",  "payload": "ACTION_BOOK"},
            ],
        )

    if raw_u.startswith("XMAS_"):
        city = raw_u[5:].capitalize()
        state["city"] = city
        state["_seasonal_event"] = "xmas"
        state["step"] = "checkin"
        return (
            f"🎄 Christmas stay in {city} — magical choice!\n\n"
            "Check-in date (typically 22–24 December)?",
            [
                {"content_type": "text", "title": "22 Dec", "payload": "CHECKIN_2026-12-22"},
                {"content_type": "text", "title": "23 Dec", "payload": "CHECKIN_2026-12-23"},
                {"content_type": "text", "title": "24 Dec", "payload": "CHECKIN_2026-12-24"},
            ],
        )

    if detected_event == "valentines":
        return (
            "💕 Valentine's Day Special!\n\n"
            "I'll find the most romantic hotels with packages including:\n"
            "🌹 Rose petals on bed\n"
            "🍾 Champagne on arrival\n"
            "🛁 Rose petal bath\n"
            "🕯️ Candle-lit private dinner\n"
            "💌 Personalised welcome message\n\n"
            "Which city for your romantic escape?",
            [
                {"content_type": "text", "title": "Paris",     "payload": "VAL_PARIS"},
                {"content_type": "text", "title": "Rome",      "payload": "VAL_ROME"},
                {"content_type": "text", "title": "Maldives",  "payload": "VAL_MALDIVES"},
                {"content_type": "text", "title": "Santorini", "payload": "VAL_SANTORINI"},
                {"content_type": "text", "title": "Other City","payload": "ACTION_BOOK"},
            ],
        )

    if raw_u.startswith("VAL_"):
        city_map = {"VAL_PARIS": "Paris", "VAL_ROME": "Rome", "VAL_MALDIVES": "Maldives", "VAL_SANTORINI": "Santorini"}
        city = city_map.get(raw_u, raw_u[4:].capitalize())
        state["city"] = city
        state["_seasonal_event"] = "valentines"
        state["step"] = "checkin"
        return (
            f"💕 Romantic Package Search — {city}\n\n"
            "Would you like to add a surprise message for your partner\n"
            "displayed when they open the room door?\n\n"
            "When is your romantic getaway?",
            [
                {"content_type": "text", "title": "Feb 14 (Valentine's Day)", "payload": "CHECKIN_2027-02-14"},
                {"content_type": "text", "title": "Weekend before",           "payload": "CHECKIN_2027-02-13"},
                {"content_type": "text", "title": "Pick another date",        "payload": "ACTION_BOOK"},
            ],
        )

    if detected_event == "ramadan":
        return (
            "🌙 Ramadan Mubarak!\n\n"
            "I'll filter hotels with Ramadan amenities:\n"
            "✅ 100% Halal-certified kitchen\n"
            "✅ Suhoor (pre-dawn) room service\n"
            "✅ Iftar buffet\n"
            "✅ Prayer mat, Quran & Qibla in every room\n"
            "✅ Nearby mosques listed\n\n"
            "Which city for Ramadan?",
            [
                {"content_type": "text", "title": "Istanbul",  "payload": "RAM_ISTANBUL"},
                {"content_type": "text", "title": "Dubai",     "payload": "RAM_DUBAI"},
                {"content_type": "text", "title": "Kuala Lumpur","payload": "RAM_KL"},
                {"content_type": "text", "title": "Other City","payload": "ACTION_BOOK"},
            ],
        )

    if raw_u.startswith("RAM_"):
        city_map = {"RAM_ISTANBUL": "Istanbul", "RAM_DUBAI": "Dubai", "RAM_KL": "Kuala Lumpur"}
        city = city_map.get(raw_u, raw_u[4:].capitalize())
        state["city"] = city
        state["_halal_filter"] = True
        state["step"] = "checkin"
        return (f"🌙 Halal & Ramadan-friendly hotels in {city}.\nWhat are your dates?", [])

    if detected_event == "eid":
        return (
            "🌙 Eid Mubarak! Let me find special Eid packages for you.\n\n"
            "Which city are you celebrating in?",
            [
                {"content_type": "text", "title": "Dubai",   "payload": "RAM_DUBAI"},
                {"content_type": "text", "title": "Istanbul","payload": "RAM_ISTANBUL"},
                {"content_type": "text", "title": "London",  "payload": "ACTION_BOOK"},
            ],
        )

    if detected_event == "conference":
        return (
            "💼 Conference Booking!\n\n"
            "I'll prioritise:\n"
            "📍 Hotels close to the venue\n"
            "🌐 Verified fast Wi-Fi (speed tested)\n"
            "💼 Business centre & meeting rooms\n"
            "🚌 Shuttle to conference venue\n"
            "🧾 Business invoice for expense claim\n\n"
            "Which conference/city?",
            [
                {"content_type": "text", "title": "Las Vegas (CES/NAB)", "payload": "CONF_LAS_VEGAS"},
                {"content_type": "text", "title": "Barcelona (MWC)",     "payload": "CONF_BARCELONA"},
                {"content_type": "text", "title": "Dubai (GITEX)",       "payload": "CONF_DUBAI"},
                {"content_type": "text", "title": "Other city",          "payload": "ACTION_BOOK"},
            ],
        )

    if raw_u.startswith("CONF_"):
        city_map = {"CONF_LAS_VEGAS": "Las Vegas", "CONF_BARCELONA": "Barcelona", "CONF_DUBAI": "Dubai"}
        city = city_map.get(raw_u, raw_u[5:].replace("_", " ").title())
        state["city"] = city
        state["_conference_mode"] = True
        state["step"] = "checkin"
        return (
            f"💼 Conference hotels in {city}.\n\n"
            "I'll filter for fast Wi-Fi, business centre,\n"
            "and proximity to the convention centre.\n\n"
            "What are your conference dates?",
            [],
        )

    if detected_event == "sports":
        return (
            "⚽ Sports Event Booking!\n\n"
            "I'll find hotels with:\n"
            "📍 Closest distance to the stadium\n"
            "🚌 Match-day shuttle service\n"
            "📺 Fan zone / live match screening\n"
            "🍺 Pre/post match packages\n\n"
            "Which sport / event and city?",
            [
                {"content_type": "text", "title": "Football Match",   "payload": "SPORT_FOOTBALL"},
                {"content_type": "text", "title": "Cricket",          "payload": "SPORT_CRICKET"},
                {"content_type": "text", "title": "Formula 1",        "payload": "SPORT_F1"},
                {"content_type": "text", "title": "Other sport",      "payload": "ACTION_BOOK"},
            ],
        )

    if detected_event == "monsoon":
        return (
            "🌴 Off-Peak / Monsoon Deals!\n\n"
            "Monsoon season offers incredible savings — up to 75% off peak rates.\n"
            "Perfect for: spa retreats, yoga, green landscapes, budget travel.\n\n"
            "Which destination interests you?",
            [
                {"content_type": "text", "title": "Goa (India)",      "payload": "MONSOON_GOA"},
                {"content_type": "text", "title": "Bali (Indonesia)", "payload": "MONSOON_BALI"},
                {"content_type": "text", "title": "Kerala (India)",   "payload": "MONSOON_KERALA"},
                {"content_type": "text", "title": "Other",            "payload": "ACTION_BOOK"},
            ],
        )

    if raw_u.startswith("MONSOON_"):
        city_map = {"MONSOON_GOA": "Goa", "MONSOON_BALI": "Bali", "MONSOON_KERALA": "Kerala"}
        city = city_map.get(raw_u, raw_u[8:].capitalize())
        state["city"] = city
        state["step"]  = "checkin"
        return (
            f"🌿 Off-peak / monsoon deals in {city}.\n"
            "You can save up to 75% vs peak season!\n\n"
            "What are your dates?",
            [],
        )

    if detected_event == "summer" and any(dest in en_lower for dest in ["maldives", "bali", "phuket", "ibiza"]):
        return (
            "☀️ Summer Peak Season Warning!\n\n"
            "⚠️ Summer is peak season for your chosen destination.\n"
            "Hotels fill up fast and prices are highest in July–August.\n\n"
            "🔒 Tip: Book now to lock in today's rate.\n"
            "Prices typically rise 25–40% closer to the date.\n\n"
            "Shall I search available hotels?",
            [
                {"content_type": "text", "title": "Yes, search now",    "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "Show me off-peak dates", "payload": "MONSOON_BALI"},
                {"content_type": "text", "title": "Set price alert",    "payload": "PRICE_ALERT"},
            ],
        )

    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# PART V — CORPORATE & BUSINESS TRAVEL MODULE
# ─────────────────────────────────────────────────────────────────────────────

_CORP_KEYWORDS = {
    "corporate account", "set up corporate", "business account", "company account",
    "business travel", "corporate booking", "gst invoice", "vat invoice",
    "expense report", "concur", "sap", "travel policy", "corporate rate",
    "team booking", "book for team", "book for colleague",
    "multi-city trip", "multi city trip", "travel report", "policy violation",
    "carbon offset",
}


def _handle_corporate_flow(
    sender_id: str, state: dict, user_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle corporate & business travel module (Part V)."""
    raw_u = user_message.strip().upper()

    # Corporate account setup
    if raw_u == "CORP_SETUP" or any(kw in en_lower for kw in (
        "set up corporate", "corporate account", "company account", "bookbot business"
    )):
        state["corp_step"] = "company_name"
        return (
            "🏢 Welcome to BookBot Business!\n\n"
            "Setting up a corporate account gives your company:\n"
            "✅ Negotiated corporate rates (up to 30% off)\n"
            "✅ Centralised invoicing & billing\n"
            "✅ Travel policy enforcement\n"
            "✅ Multi-traveler management dashboard\n"
            "✅ Monthly consolidated invoice\n"
            "✅ Expense report integration (SAP/Concur/QuickBooks)\n"
            "✅ Carbon offset reporting\n\n"
            "What is your company name?",
            [{"content_type": "text", "title": "Cancel", "payload": "RESTART"}],
        )

    corp_step = state.get("corp_step")

    if corp_step == "company_name":
        state["corp_name"] = user_message.strip()
        state["corp_step"] = "domain"
        return (
            f"Great! And your company email domain?\n"
            "(e.g. yourcompany.com — we'll verify all employees)",
            [],
        )

    if corp_step == "domain":
        domain = user_message.strip().lower()
        if "." not in domain:
            return ("Please enter a valid domain, e.g. yourcompany.com", [])
        state["corp_domain"] = domain
        state["corp_step"] = "traveler_count"
        return (
            f"Perfect — domain: {domain}\n\n"
            "How many employees will travel monthly?",
            [
                {"content_type": "text", "title": "1–5 travelers",     "payload": "CORP_SIZE_S"},
                {"content_type": "text", "title": "6–20 travelers",    "payload": "CORP_SIZE_M"},
                {"content_type": "text", "title": "21–100 travelers",  "payload": "CORP_SIZE_L"},
                {"content_type": "text", "title": "100+ (Enterprise)", "payload": "CORP_SIZE_E"},
            ],
        )

    if raw_u.startswith("CORP_SIZE_"):
        size_map = {"CORP_SIZE_S": "1–5", "CORP_SIZE_M": "6–20",
                    "CORP_SIZE_L": "21–100", "CORP_SIZE_E": "100+"}
        size = size_map.get(raw_u, "1–5")
        name = state.get("corp_name", "your company")
        domain = state.get("corp_domain", "")
        state.pop("corp_step", None)
        return (
            f"🎉 Corporate account created for {name}!\n\n"
            f"Company: {name}\n"
            f"Domain : {domain}\n"
            f"Size   : {size} monthly travelers\n\n"
            "A verification email has been sent to your domain admin.\n"
            "Once verified, your corporate rates activate within 24 hours.\n\n"
            "Your dedicated business support line: +1-800-BOOKBOT\n\n"
            "What would you like to do next?",
            [
                {"content_type": "text", "title": "Book Business Travel", "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "Travel Report",        "payload": "CORP_REPORT"},
                {"content_type": "text", "title": "Team Booking",         "payload": "CORP_TEAM"},
                {"content_type": "text", "title": "Main Menu",            "payload": "RESTART"},
            ],
        )

    # Corporate reporting
    if raw_u == "CORP_REPORT" or any(kw in en_lower for kw in (
        "travel report", "travel spending", "travel expense", "monthly report",
        "policy violation", "carbon offset"
    )):
        return (
            "📊 Corporate Travel Report\n\n"
            "Choose your report type:",
            [
                {"content_type": "text", "title": "This Month's Spend",  "payload": "CORP_RPT_MONTH"},
                {"content_type": "text", "title": "Last Month",           "payload": "CORP_RPT_LAST"},
                {"content_type": "text", "title": "Policy Violations",    "payload": "CORP_RPT_VIOLATIONS"},
                {"content_type": "text", "title": "Carbon Footprint",     "payload": "CORP_RPT_CARBON"},
                {"content_type": "text", "title": "Download PDF",         "payload": "CORP_RPT_PDF"},
                {"content_type": "text", "title": "Email to CFO",         "payload": "CORP_RPT_EMAIL"},
            ],
        )

    if raw_u == "CORP_RPT_MONTH":
        return (
            "📊 Corporate Travel Report — This Month\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Request your full report from your corporate dashboard.\n\n"
            "The report includes:\n"
            "✅ Total spend by city\n"
            "✅ Spend by traveler\n"
            "✅ Policy compliance rate\n"
            "✅ CO₂ footprint\n"
            "✅ Savings vs public rate\n\n"
            "Shall I email this to your finance team?",
            [
                {"content_type": "text", "title": "Email to Finance", "payload": "CORP_RPT_EMAIL"},
                {"content_type": "text", "title": "Download PDF",     "payload": "CORP_RPT_PDF"},
                {"content_type": "text", "title": "Main Menu",        "payload": "RESTART"},
            ],
        )

    if raw_u in ("CORP_RPT_PDF", "CORP_RPT_EMAIL"):
        action = "emailed to your finance team" if raw_u == "CORP_RPT_EMAIL" else "ready for download"
        return (
            f"✅ Your corporate travel report is {action}.\n\n"
            "Need anything else?",
            [
                {"content_type": "text", "title": "Book Business Travel", "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "Main Menu",            "payload": "RESTART"},
            ],
        )

    if raw_u == "CORP_RPT_CARBON" or "carbon" in en_lower:
        return (
            "🌱 Carbon Footprint Report\n\n"
            "Your hotel stays generate an estimated CO₂ footprint.\n"
            "BookBot can offset your carbon emissions.\n\n"
            "Would you like to purchase carbon offsets\n"
            "for your company's hotel travel?",
            [
                {"content_type": "text", "title": "Yes — offset all travel", "payload": "CORP_CARBON_BUY"},
                {"content_type": "text", "title": "Tell me more",            "payload": "FAQ_CARBON"},
                {"content_type": "text", "title": "Not now",                 "payload": "RESTART"},
            ],
        )

    # Team booking
    if raw_u == "CORP_TEAM" or any(kw in en_lower for kw in (
        "team booking", "book for team", "multiple rooms for", "book colleague"
    )):
        state["corp_team_step"] = "room_count"
        return (
            "👥 Team Booking\n\n"
            "How many rooms do you need for your team?",
            [
                {"content_type": "text", "title": "2–3 rooms",  "payload": "CORP_TEAM_2"},
                {"content_type": "text", "title": "4–5 rooms",  "payload": "CORP_TEAM_5"},
                {"content_type": "text", "title": "6–10 rooms", "payload": "CORP_TEAM_10"},
                {"content_type": "text", "title": "10+ rooms",  "payload": "GROUP_BOOKING"},
            ],
        )

    if raw_u.startswith("CORP_TEAM_"):
        rooms = raw_u.split("_")[-1]
        state["corp_team_rooms"] = rooms
        state["corp_team_step"]  = "guest_list"
        return (
            f"👥 Team booking: {rooms} rooms\n\n"
            "Do you want to send confirmations to each team member directly?",
            [
                {"content_type": "text", "title": "Yes — email each person", "payload": "CORP_TEAM_EMAIL_ALL"},
                {"content_type": "text", "title": "All to me only",          "payload": "CORP_TEAM_EMAIL_ME"},
                {"content_type": "text", "title": "Enter guest list now",    "payload": "CORP_TEAM_LIST"},
            ],
        )

    if raw_u == "CORP_TEAM_LIST":
        state["corp_team_step"] = "parsing_list"
        return (
            "Please share the guest list in this format\n"
            "(one per line, up to 20 guests):\n\n"
            "Name | Email\n\n"
            "Example:\n"
            "John Smith | john@company.com\n"
            "Jane Doe | jane@company.com",
            [{"content_type": "text", "title": "Cancel", "payload": "RESTART"}],
        )

    if state.get("corp_team_step") == "parsing_list":
        lines = [l.strip() for l in user_message.strip().split("\n") if "|" in l]
        if not lines:
            return ("I could not parse the guest list. Use Name | Email format.", [])
        guests = []
        for ln in lines:
            parts = ln.split("|", 1)
            if len(parts) == 2:
                g_name = parts[0].strip()
                g_email = parts[1].strip()
                if re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', g_email):
                    guests.append(f"{g_name} <{g_email}>")
        state.pop("corp_team_step", None)
        if not guests:
            return ("Could not find valid email addresses. Please try again.", [])
        state["corp_guest_list"] = guests
        state["step"] = "city"
        return (
            f"✅ {len(guests)} guest(s) added:\n" +
            "\n".join(f"  • {g}" for g in guests[:5]) +
            ("\n  ..." if len(guests) > 5 else "") +
            "\n\nConfirmations will be sent to each person after booking.\n\n"
            "Now let's search for hotels — which city?",
            [],
        )

    # Multi-city trip
    if any(kw in en_lower for kw in ("multi-city", "multi city", "multiple cities", "several cities")):
        return (
            "✈️ Multi-City Business Trip!\n\n"
            "I can book hotels across multiple cities at once.\n"
            "Please list your itinerary:\n\n"
            "Format: City — arrival date — departure date\n\n"
            "Example:\n"
            "Bangalore — Mon Mar 16 — Wed Mar 18\n"
            "Mumbai — Thu Mar 19 — Fri Mar 20\n"
            "Delhi — Mon Mar 23 — Fri Mar 27",
            [
                {"content_type": "text", "title": "Enter my itinerary", "payload": "CORP_MULTICITY"},
                {"content_type": "text", "title": "Book one city only", "payload": "ACTION_BOOK"},
            ],
        )

    # Expense report
    if raw_u.startswith("CORP_INVOICE_") or any(kw in en_lower for kw in (
        "expense report", "invoice format", "sap concur", "gst invoice", "expense claim"
    )):
        ref = raw_u.replace("CORP_INVOICE_", "") if raw_u.startswith("CORP_INVOICE_") else ""
        ref_s = f"booking {ref}" if ref else "your booking"
        return (
            f"📋 Expense Documentation — {ref_s}\n\n"
            "I can send the invoice in multiple formats:\n\n"
            "1. PDF Invoice (company letterhead)\n"
            "2. CSV for SAP Concur upload\n"
            "3. XML for QuickBooks import\n"
            "4. JSON for custom ERP systems\n"
            "5. Email directly to your finance team\n\n"
            "The invoice includes:\n"
            "✅ Hotel name, address, VAT number\n"
            "✅ Rate breakdown per night\n"
            "✅ Taxes and service charges itemized\n"
            "✅ GST/VAT split\n"
            "✅ Employee ID / cost center",
            [
                {"content_type": "text", "title": "PDF Invoice",        "payload": f"DL_PDF_{ref}"},
                {"content_type": "text", "title": "SAP Concur CSV",     "payload": f"DL_GST_{ref}"},
                {"content_type": "text", "title": "Email Finance Team", "payload": f"EMAIL_RECEIPT_{ref}"},
                {"content_type": "text", "title": "All Formats (ZIP)",  "payload": f"DL_CORP_{ref}"},
            ],
        )

    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# PART W — ACCESSIBILITY & SPECIAL NEEDS MODULE
# ─────────────────────────────────────────────────────────────────────────────

_ACCESS_KEYWORDS = {
    "wheelchair": "wheelchair", "disabled": "wheelchair", "ada room": "wheelchair",
    "accessible room": "wheelchair", "roll-in shower": "wheelchair",
    "visual impairment": "visual", "visually impaired": "visual", "blind": "visual",
    "guide dog": "visual", "seeing eye dog": "visual",
    "deaf": "hearing", "hearing impairment": "hearing", "hard of hearing": "hearing",
    "visual alarm": "hearing",
    "insulin": "medical", "diabetic": "medical", "medical fridge": "medical",
    "medical needs": "medical", "nurse on call": "medical",
    "special needs": "any_access", "accessibility": "any_access",
}


def _handle_accessibility_flow(
    sender_id: str, state: dict, user_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle accessibility & special needs booking (Part W)."""
    raw_u = user_message.strip().upper()

    # Detect accessibility need from NL
    detected_need = None
    for kw, need_type in _ACCESS_KEYWORDS.items():
        if kw in en_lower:
            detected_need = need_type
            break

    if raw_u == "ACCESSIBLE_BOOKING" or (detected_need and not raw_u.startswith("HOTEL_")):
        pass  # continue to sub-flow below
    elif not detected_need:
        return None, []

    # wheelchair / accessible rooms
    if detected_need == "wheelchair" or raw_u == "ACCESS_WHEELCHAIR":
        state["_accessibility"] = ["wheelchair"]
        return (
            "♿ Wheelchair Accessible Room Search\n\n"
            "I'll filter for fully accessible properties with:\n"
            "✅ Roll-in shower (no lip/step)\n"
            "✅ Wide doorways (≥81 cm / 32 inches)\n"
            "✅ Lowered bed height available\n"
            "✅ Accessible bathroom with grab rails\n"
            "✅ Elevator access to all floors\n"
            "✅ Accessible pool (hydraulic lift)\n"
            "✅ Level / ramped entrance\n"
            "✅ Accessible parking\n\n"
            "Do you have any additional requirements?",
            [
                {"content_type": "text", "title": "Also need visual assist",   "payload": "ACCESS_VISUAL"},
                {"content_type": "text", "title": "Also need hearing assist",  "payload": "ACCESS_HEARING"},
                {"content_type": "text", "title": "Also need medical support", "payload": "ACCESS_MEDICAL"},
                {"content_type": "text", "title": "No — search hotels now",   "payload": "ACTION_BOOK"},
            ],
        )

    # visual impairment / guide dog
    if detected_need == "visual" or raw_u == "ACCESS_VISUAL":
        state["_accessibility"] = state.get("_accessibility", []) + ["visual_impairment"]
        return (
            "🦮 Visual Impairment & Guide Dog Assistance\n\n"
            "I'll filter for hotels with:\n"
            "✅ Guide dogs / service animals welcome (all areas)\n"
            "✅ Braille signage in corridors and rooms\n"
            "✅ Audio elevator announcements\n"
            "✅ Large-print menus available\n"
            "✅ Trained accessibility concierge\n"
            "✅ Orientation assistance on arrival\n"
            "✅ Dog relief area on premises\n\n"
            "Please note: Guide dogs are welcome free of charge\n"
            "— no pet deposit required.\n\n"
            "Which city are you visiting?",
            [
                {"content_type": "text", "title": "London",   "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "New York", "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "Other",    "payload": "ACTION_BOOK"},
            ],
        )

    # hearing impairment / deaf
    if detected_need == "hearing" or raw_u == "ACCESS_HEARING":
        state["_accessibility"] = state.get("_accessibility", []) + ["hearing_impairment"]
        return (
            "🔔 Deaf-Friendly Hotel Features\n\n"
            "I'll look for hotels with:\n"
            "✅ Visual fire alarm / strobe light system\n"
            "✅ Vibrating alarm clock (on request)\n"
            "✅ Closed captioning on all TVs\n"
            "✅ Video relay service for calls\n"
            "✅ SMS/text communication with front desk\n"
            "✅ Visual doorbell / knock alert\n"
            "✅ Text-based check-in option\n\n"
            "I'll also communicate all confirmations via text/chat only\n"
            "— no phone calls required.\n\n"
            "Which city and dates?",
            [
                {"content_type": "text", "title": "Search Hotels", "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "Add wheelchair needs", "payload": "ACCESS_WHEELCHAIR"},
            ],
        )

    # medical / health needs
    if detected_need == "medical" or raw_u == "ACCESS_MEDICAL":
        state["_accessibility"] = state.get("_accessibility", []) + ["medical"]
        return (
            "💊 Medical & Health Needs\n\n"
            "I'll filter hotels with:\n"
            "✅ In-room mini-fridge / medical fridge (0–4°C)\n"
            "✅ 24/7 nurse on call or medical concierge\n"
            "✅ Nearby pharmacy (within 500 m)\n"
            "✅ Nearest hospital / emergency room listed\n"
            "✅ Sharps disposal bin available\n"
            "✅ Diabetic-friendly meal options\n\n"
            "⚠️ I always recommend calling the hotel directly\n"
            "to confirm specific medical equipment availability.\n\n"
            "Which city?",
            [
                {"content_type": "text", "title": "Search Hotels", "payload": "ACTION_BOOK"},
            ],
        )

    # Generic accessibility query
    if detected_need == "any_access" or raw_u == "ACCESSIBLE_BOOKING":
        return (
            "♿ Accessibility & Special Needs\n\n"
            "Which type of accessibility support do you need?",
            [
                {"content_type": "text", "title": "♿ Wheelchair Access",    "payload": "ACCESS_WHEELCHAIR"},
                {"content_type": "text", "title": "🦮 Visual Impairment",   "payload": "ACCESS_VISUAL"},
                {"content_type": "text", "title": "🔔 Hearing Impairment",  "payload": "ACCESS_HEARING"},
                {"content_type": "text", "title": "💊 Medical Needs",       "payload": "ACCESS_MEDICAL"},
                {"content_type": "text", "title": "🐾 Service Animal",      "payload": "ACCESS_VISUAL"},
            ],
        )

    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# PART X — CRISIS & EMERGENCY HANDLING MODULE
# ─────────────────────────────────────────────────────────────────────────────

_CRISIS_KEYWORDS = {
    "emergency booking": "emergency", "flood": "emergency", "fire": "emergency",
    "earthquake": "emergency", "evacuation": "emergency", "disaster": "emergency",
    "urgent accommodation": "emergency", "tonight urgently": "emergency",
    "flight cancelled": "stranded", "flight canceled": "stranded",
    "stranded at airport": "stranded", "stuck at airport": "stranded",
    "my hotel is overbooked": "overbooking", "hotel overbooked": "overbooking",
    "hotel has no room": "overbooking", "walked from hotel": "overbooking",
    "medical emergency": "medical_emergency", "hospital near": "medical_emergency",
    "need hospital": "medical_emergency", "need a doctor": "medical_emergency",
    "lost passport": "lost_docs", "lost my passport": "lost_docs",
    "stolen passport": "lost_docs", "lost documents": "lost_docs",
    "can't check in without id": "lost_docs",
}

_EMBASSY_CONTACTS = {
    "indian": ("India", "📞 Emergency Passport Line\nHigh Commission of India, London: +44-20-7836-8484\nEmbassy of India, Washington DC: +1-202-939-7000\nIndian Consulate General (worldwide): hcilondon.in"),
    "british": ("UK", "📞 Her Majesty's Passport Office\nEmergency Travel Document: +44-300-222-0000\nFCO Global Response Centre: +44-20-7008-5000"),
    "american": ("USA", "📞 US Embassy Emergency Passport\nSmart Traveler Enrollment: step.state.gov\nEmergency: +1-888-407-4747 (from US) | +1-202-501-4444 (international)"),
    "australian": ("Australia", "📞 Australian Passport Emergency\nAustralian Embassy: smartraveller.gov.au\n24/7 consular emergency: 1300 555 135"),
    "canadian": ("Canada", "📞 Canadian Emergency Passport\n+1-800-387-3124 (Canada/US) | +1-613-996-8885 (international)"),
}


def _handle_crisis_flow(
    sender_id: str, state: dict, user_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle crisis and emergency situations (Part X)."""
    raw_u = user_message.strip().upper()

    # Detect crisis type from NL
    detected_crisis = None
    for kw, crisis_type in _CRISIS_KEYWORDS.items():
        if kw in en_lower:
            detected_crisis = crisis_type
            break

    # Emergency accommodation (X1)
    if detected_crisis == "emergency" or raw_u == "EMERGENCY_BOOK":
        state["step"] = "city"
        state["_emergency_mode"] = True
        return (
            "🚨 EMERGENCY BOOKING — Prioritising Your Request Now\n\n"
            "I'm searching for available rooms for immediate check-in.\n\n"
            "For your safety, please also contact:\n"
            "📞 Emergency services: 112 (EU) | 999 (UK) | 911 (US)\n"
            "📞 Flood/disaster helpline: your local authority\n\n"
            "Which is your current city or nearest city?",
            [
                {"content_type": "text", "title": "⚡ Search NOW", "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "Speak to Agent", "payload": "AGENT_HANDOFF"},
            ],
        )

    # Stranded traveler / flight cancelled (X2)
    if detected_crisis == "stranded" or raw_u == "STRANDED_TRAVELER":
        state["_emergency_mode"] = True
        return (
            "✈️ Flight Cancelled — I'll Find You a Hotel Fast!\n\n"
            "I'm sorry about your flight. Let me help you quickly.\n\n"
            "⚠️ Tip: Keep your airline cancellation notice —\n"
            "many airlines reimburse hotel costs up to £200/night.\n"
            "I can prepare an itemised receipt for your airline claim.\n\n"
            "What matters most right now?",
            [
                {"content_type": "text", "title": "🚌 Free hotel shuttle",  "payload": "STRAND_SHUTTLE"},
                {"content_type": "text", "title": "💰 Cheapest available",  "payload": "STRAND_CHEAP"},
                {"content_type": "text", "title": "🛎️ Comfortable option",  "payload": "STRAND_COMFORT"},
                {"content_type": "text", "title": "📄 Airline claim receipt","payload": "STRAND_RECEIPT"},
                {"content_type": "text", "title": "Speak to Agent",         "payload": "AGENT_HANDOFF"},
            ],
        )

    if raw_u in ("STRAND_SHUTTLE", "STRAND_CHEAP", "STRAND_COMFORT"):
        pref = {"STRAND_SHUTTLE": "free shuttle", "STRAND_CHEAP": "cheapest",
                "STRAND_COMFORT": "comfortable"}.get(raw_u, "")
        state["step"] = "city"
        state["_emergency_mode"] = True
        return (
            f"🚌 Searching for hotels with {pref} near you.\n\n"
            "Which airport are you at?\n"
            "(Type the airport name or city)",
            [],
        )

    if raw_u == "STRAND_RECEIPT":
        return (
            "📄 Airline Claim Receipt\n\n"
            "I can prepare an itemised receipt for your airline.\n"
            "Please provide your booking reference (or make a new booking first):",
            [
                {"content_type": "text", "title": "My Bookings", "payload": "MY_BOOKINGS"},
                {"content_type": "text", "title": "Book Hotel First", "payload": "ACTION_BOOK"},
            ],
        )

    # Overbooking recovery (X3)
    if detected_crisis == "overbooking" or raw_u == "OVERBOOK_RECOVERY":
        return (
            "😔 Overbooking — We Sincerely Apologise\n\n"
            "This should never happen, and we take full responsibility.\n\n"
            "Here's what we'll do for you RIGHT NOW:\n"
            "✅ Find a comparable or BETTER hotel nearby\n"
            "✅ Cost difference (if higher) is fully covered\n"
            "✅ Taxi to the new hotel — fully covered by BookBot\n"
            "✅ 5,000 loyalty points added for the inconvenience\n\n"
            "Please share your booking reference so I can find\n"
            "the best alternative immediately.",
            [
                {"content_type": "text", "title": "Enter Booking Reference", "payload": "LOOKUP_BOOKING"},
                {"content_type": "text", "title": "Speak to Agent NOW",      "payload": "AGENT_HANDOFF"},
                {"content_type": "text", "title": "Full Refund Instead",     "payload": "CANCEL_BOOKING"},
            ],
        )

    # Medical emergency (X4)
    if detected_crisis == "medical_emergency" or raw_u == "MEDICAL_EMERGENCY":
        # Try to find the user's current hotel city from active booking
        hotel_city = state.get("city", "")
        hotel_name = state.get("hotel_name", "your hotel")
        city_s = f" near {hotel_name}" if hotel_name else ""
        return (
            f"🚨 MEDICAL EMERGENCY — IMMEDIATE HELP{city_s}\n\n"
            "CALL EMERGENCY SERVICES NOW:\n"
            "📞 112 — universal emergency (EU, Asia, 100+ countries)\n"
            "📞 999 — emergency (UK)\n"
            "📞 911 — emergency (USA/Canada)\n"
            "📞 102 — ambulance (India)\n\n"
            "🏥 FINDING NEAREST HOSPITALS...\n\n"
            f"I've alerted the hotel reception about your emergency.\n\n"
            "Please tell the hotel receptionist to call an ambulance NOW\n"
            "or ask for the hotel doctor on call.\n\n"
            "What is your current city so I can find the nearest hospital?",
            [
                {"content_type": "text", "title": "📞 Call 112 Emergency",  "payload": "CALL_EMERGENCY"},
                {"content_type": "text", "title": "Speak to Agent",         "payload": "AGENT_HANDOFF"},
                {"content_type": "text", "title": "Find Hospital",          "payload": "FIND_HOSPITAL"},
            ],
        )

    if raw_u == "CALL_EMERGENCY":
        return (
            "🚨 EMERGENCY: Please call 112 immediately.\n\n"
            "If you cannot call, ask someone nearby to call for you\n"
            "or alert the hotel reception.\n\n"
            "Stay on the line with emergency services.",
            [{"content_type": "text", "title": "I need help booking", "payload": "AGENT_HANDOFF"}],
        )

    if raw_u == "FIND_HOSPITAL":
        city = state.get("city", "")
        city_s = f" in {city}" if city else ""
        return (
            f"🏥 Finding nearest hospitals{city_s}...\n\n"
            "For the most accurate results, please type your\n"
            "exact location or nearest landmark.",
            [{"content_type": "text", "title": "Speak to Agent", "payload": "AGENT_HANDOFF"}],
        )

    # Lost passport / documents (X5)
    if detected_crisis == "lost_docs" or raw_u == "LOST_PASSPORT":
        state["crisis_step"] = "nationality"
        return (
            "😟 Lost Passport — Step-by-Step Guide\n\n"
            "Don't panic — I'll guide you through this.\n\n"
            "STEP 1 — Report to police (required by all embassies)\n"
            "  → Go to nearest police station\n"
            "  → Ask for a 'lost passport report'\n"
            "  → Keep the report number\n\n"
            "STEP 2 — Contact your embassy\n"
            "  What is your nationality?",
            [
                {"content_type": "text", "title": "Indian",     "payload": "EMBASSY_INDIAN"},
                {"content_type": "text", "title": "British",    "payload": "EMBASSY_BRITISH"},
                {"content_type": "text", "title": "American",   "payload": "EMBASSY_AMERICAN"},
                {"content_type": "text", "title": "Australian", "payload": "EMBASSY_AUSTRALIAN"},
                {"content_type": "text", "title": "Canadian",   "payload": "EMBASSY_CANADIAN"},
                {"content_type": "text", "title": "Other",      "payload": "EMBASSY_OTHER"},
            ],
        )

    if raw_u.startswith("EMBASSY_"):
        nat_key = raw_u[8:].lower()
        if nat_key == "other":
            state["crisis_step"] = "nationality"
            return (
                "Please type your nationality (e.g. French, German, Japanese):",
                [],
            )
        entry = _EMBASSY_CONTACTS.get(nat_key)
        if entry:
            country, contact = entry
        else:
            country, contact = nat_key.capitalize(), "Please visit your country's embassy website for emergency passport contact details."
        state.pop("crisis_step", None)
        return (
            f"🏛️ {country} Emergency Passport Help\n\n"
            f"{contact}\n\n"
            "STEP 3 — Check in to hotel with alternate ID:\n"
            "✅ Driving licence (domestic)\n"
            "✅ Birth certificate + photo ID\n"
            "✅ Emergency travel document from embassy\n\n"
            "STEP 4 — I can help:\n"
            "📧 Send hotel your booking confirmation with photo\n"
            "📞 Contact hotel on your behalf to explain the situation\n"
            "📋 Prepare a guest verification letter",
            [
                {"content_type": "text", "title": "Contact Hotel for Me", "payload": "AGENT_HANDOFF"},
                {"content_type": "text", "title": "My Bookings",          "payload": "MY_BOOKINGS"},
                {"content_type": "text", "title": "Main Menu",            "payload": "RESTART"},
            ],
        )

    if state.get("crisis_step") == "nationality":
        nat_key = en_lower.strip().rstrip("n").lower()  # "indian" → "india"
        state.pop("crisis_step", None)
        entry = _EMBASSY_CONTACTS.get(nat_key) or _EMBASSY_CONTACTS.get(nat_key + "n")
        if entry:
            country, contact = entry
            return (
                f"🏛️ {country} Embassy Emergency Contact\n\n{contact}\n\n"
                "Is there anything else I can help with?",
                _MAIN_BUTTONS,
            )
        return (
            "I don't have that country's embassy details handy.\n"
            "Please search: '[your country] embassy emergency passport'\n"
            "or visit your country's foreign ministry website.",
            _MAIN_BUTTONS,
        )

    return None, []


# ─────────────────────────────────────────────────────────────────────────────
# PART Y — AI UPSELL & CROSS-SELL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _handle_upsell_flow(
    sender_id: str, state: dict, user_message: str, en_lower: str
) -> tuple[str | None, list]:
    """Handle intelligent upsell / cross-sell suggestions (Part Y)."""
    raw_u = user_message.strip().upper()

    # Y1 — Room upgrade offer
    if raw_u == "UPSELL_UPGRADE":
        hotel    = state.get("hotel_name", "your hotel")
        room     = state.get("room_name", "your current room")
        currency = state.get("selected_hotel", {}).get("currency", "")
        current_price = state.get("selected_room", {}).get("price_per_night", 0)
        upgrade_delta = round(current_price * 0.30, 0) if current_price else 3200
        return (
            f"🌟 Exclusive Upgrade Offer — {hotel}!\n\n"
            f"YOUR CURRENT ROOM:\n"
            f"  {room}\n\n"
            f"UPGRADE AVAILABLE:\n"
            f"  Premier Suite / Sea View Room\n"
            f"  🌊 Panoramic view\n"
            f"  🛁 Plunge pool / premium bath\n"
            f"  🍾 Complimentary champagne on arrival\n"
            f"  🛎️ Butler service included\n\n"
            f"Upgrade cost: Just {currency} {upgrade_delta:.0f}/night extra\n"
            f"(Limited time offer — expires in 12 hrs)\n\n"
            "Would you like to upgrade?",
            [
                {"content_type": "text", "title": f"✅ Upgrade for {currency} {upgrade_delta:.0f}/night",
                 "payload": "UPSELL_UPGRADE_CONFIRM"},
                {"content_type": "text", "title": "🖼️ Show photos first",    "payload": "UPSELL_UPGRADE_PHOTOS"},
                {"content_type": "text", "title": "❌ Keep my room",         "payload": "RESTART"},
            ],
        )

    if raw_u == "UPSELL_UPGRADE_CONFIRM":
        state["_room_upgraded"] = True
        return (
            "✅ Great! Your room has been upgraded.\n\n"
            "The charge will be added to your total.\n"
            "Looking forward to your stay!",
            _MAIN_BUTTONS,
        )

    if raw_u == "UPSELL_UPGRADE_PHOTOS":
        hotel = state.get("hotel_name", "the hotel")
        return (
            f"Here is the suite / upgraded room at {hotel}.\n\n"
            "The views are spectacular — would you like to upgrade?",
            [
                {"content_type": "text", "title": "✅ Yes — Upgrade",   "payload": "UPSELL_UPGRADE_CONFIRM"},
                {"content_type": "text", "title": "❌ No thanks",        "payload": "RESTART"},
            ],
        )

    # Y2 — Dining package upsell
    if raw_u == "UPSELL_DINING" or any(kw in en_lower for kw in (
        "add meal", "add breakfast", "add dinner", "half board", "full board",
        "meal plan upsell", "dining package"
    )):
        currency = state.get("selected_hotel", {}).get("currency", "")
        return (
            "🍽️ Add a Dining Package — Save vs À La Carte!\n\n"
            "HALF-BOARD (Breakfast + Dinner):\n"
            f"→ +{currency} 45/person/night (saves ~{currency} 60 vs paying separately)\n"
            "✅ Full breakfast (7–10 AM)\n"
            "✅ 4-course dinner at hotel restaurant\n"
            "✅ Kids eat free (under 10)\n\n"
            "BREAKFAST ONLY:\n"
            f"→ +{currency} 22/person/night\n"
            "✅ Award-winning hotel breakfast\n"
            "✅ Fresh options, coffee bar\n\n"
            "Which would you prefer?",
            [
                {"content_type": "text", "title": "🍷 Half-Board",        "payload": "MEAL_HB"},
                {"content_type": "text", "title": "☕ Breakfast Only",    "payload": "MEAL_BB"},
                {"content_type": "text", "title": "🚫 Stay room-only",    "payload": "MEAL_RO"},
                {"content_type": "text", "title": "🍽️ Show menu first",   "payload": "FAQ_RESTAURANT"},
            ],
        )

    # Y3 — Spa & wellness cross-sell (triggered for 3+ night bookings)
    if raw_u == "UPSELL_SPA" or (
        "spa" in en_lower and "add" in en_lower
        or "wellness" in en_lower and "package" in en_lower
        or "bali" in en_lower and any(kw in en_lower for kw in ("massage", "yoga", "retreat", "wellness"))
    ):
        nights = state.get("_nights", 1)
        currency = state.get("selected_hotel", {}).get("currency", "$")
        daily_massage = 35
        couples_pkg   = 80
        full_retreat  = 120
        return (
            f"🌿 Wellness & Spa Packages — {nights} Nights\n\n"
            "1. Daily 60-min Massage\n"
            f"   → +{currency}{daily_massage}/day ({currency}{daily_massage * nights} total)\n"
            "   🌺 Traditional technique | In-room or spa\n\n"
            "2. Couples Yoga + Massage Package\n"
            f"   → +{currency}{couples_pkg}/day ({currency}{couples_pkg * nights} total)\n"
            "   🧘‍♀️ Sunrise yoga + 90-min couples massage\n"
            "   🥤 Detox smoothies included\n\n"
            "3. Full Wellness Retreat\n"
            f"   → +{currency}{full_retreat}/day ({currency}{full_retreat * nights} total)\n"
            "   Yoga, meditation, massage, healthy meals,\n"
            "   sunset cruise + sound healing\n\n"
            "Limited spots available!",
            [
                {"content_type": "text", "title": "💆 Daily Massage",      "payload": "ADDON_SPA"},
                {"content_type": "text", "title": "👫 Couples Package",    "payload": "ADDON_ROMANCE"},
                {"content_type": "text", "title": "🌿 Full Retreat",       "payload": "ADDON_SPA"},
                {"content_type": "text", "title": "🙅 No thanks",          "payload": "RESTART"},
            ],
        )

    # Y4 — Airport transfer upsell (shown at booking confirmation)
    if raw_u == "UPSELL_TRANSFER" or (
        "add transfer" in en_lower or "airport transfer" in en_lower and "add" in en_lower
    ):
        currency = state.get("selected_hotel", {}).get("currency", "")
        return (
            "✈️ Add Airport Transfer?\n\n"
            "Don't forget you'll need to get from the airport to the hotel.\n\n"
            "Option A — Economy Car (up to 4 passengers):\n"
            f"  → {currency} 1,200 one-way | {currency} 2,200 return\n"
            "  🚗 Toyota Innova | A/C | Driver with name board\n\n"
            "Option B — Premium Car (up to 4 passengers):\n"
            f"  → {currency} 2,500 one-way | {currency} 4,500 return\n"
            "  🚘 Mercedes E-Class | Meet & Greet in arrivals\n\n"
            "Option C — Minibus (5–8 passengers):\n"
            f"  → {currency} 3,200 one-way | {currency} 5,800 return\n"
            "  🚌 For groups or large luggage\n\n"
            "ℹ️ All drivers track your flight live\n"
            "— they wait even if your flight is delayed.",
            [
                {"content_type": "text", "title": "🚗 Economy Transfer",    "payload": "ADDON_TRANSFER_BOTH"},
                {"content_type": "text", "title": "🚘 Premium Transfer",    "payload": "ADDON_TRANSFER_BOTH"},
                {"content_type": "text", "title": "🚌 Minibus",             "payload": "ADDON_TRANSFER_BOTH"},
                {"content_type": "text", "title": "✈️ Different airport",   "payload": "UPSELL_TRANSFER"},
                {"content_type": "text", "title": "🙅 I'll arrange my own", "payload": "RESTART"},
            ],
        )

    # Upsell trigger keywords
    if any(kw in en_lower for kw in (
        "upgrade room", "better room", "suite upgrade", "upsell",
        "add spa", "add wellness", "add massage",
        "add dining", "add meal plan",
        "airport pickup", "airport drop",
    )):
        if any(kw in en_lower for kw in ("upgrade", "suite", "better room")):
            return _handle_upsell_flow(sender_id, state, "UPSELL_UPGRADE", "upsell upgrade")
        if any(kw in en_lower for kw in ("spa", "massage", "wellness")):
            return _handle_upsell_flow(sender_id, state, "UPSELL_SPA", "upsell spa")
        if any(kw in en_lower for kw in ("dining", "meal plan", "breakfast", "half board")):
            return _handle_upsell_flow(sender_id, state, "UPSELL_DINING", "upsell dining")
        if any(kw in en_lower for kw in ("airport", "transfer", "pickup")):
            return _handle_upsell_flow(sender_id, state, "UPSELL_TRANSFER", "upsell transfer")

    return None, []



_TELEGRAM_COMMANDS = {
    "/start":       ("GET_STARTED",  "book"),
    "/book":        ("ACTION_BOOK",  "book"),
    "/mybookings":  ("MY_BOOKINGS",  "my bookings"),
    "/help":        ("ACTION_HELP",  "help"),
    "/cancel":      ("CANCEL_BOOKING", "cancel booking"),
    "/loyalty":     ("LOYALTY_MENU",   "loyalty"),
    "/language":    ("ACTION_CHANGE_LANG", "change language"),
    "/agent":       ("AGENT_HANDOFF",  "agent"),
    "/restart":     ("RESTART",        "restart"),
}

_SMS_MENU = (
    "BookBot – Reply with a number:\n"
    "1. Book a Hotel\n"
    "2. My Bookings\n"
    "3. Cancel a Booking\n"
    "4. Loyalty Rewards\n"
    "5. Help\n"
    "6. Speak to Agent\n"
    "0. Main Menu / Restart"
)

_SMS_MENU_MAP = {
    "1": "ACTION_BOOK", "2": "MY_BOOKINGS", "3": "CANCEL_BOOKING",
    "4": "LOYALTY_MENU", "5": "ACTION_HELP", "6": "AGENT_HANDOFF",
    "0": "RESTART",
}


def _resolve_telegram_command(user_message: str) -> str | None:
    """If message is a Telegram /command, return the equivalent payload string."""
    stripped = user_message.strip().lower()
    cmd = stripped.split()[0]  # handle /help@botname style
    if "@" in cmd:
        cmd = cmd.split("@")[0]
    entry = _TELEGRAM_COMMANDS.get(cmd)
    return entry[0] if entry else None


def _resolve_sms_menu(user_message: str) -> str | None:
    """If message is a single digit from the SMS menu, return equivalent payload."""
    msg = user_message.strip()
    return _SMS_MENU_MAP.get(msg)


# ─────────────────────────────────────────────────────────────────────────────

def _handle_faq(en_lower: str, state: dict) -> str | None:
    """Return a FAQ answer if the query matches, else None."""
    t = en_lower.strip()

    # Check-in / check-out times
    if any(kw in t for kw in ("check-in time", "checkin time", "when can i check in", "what time check in", "check in time")):
        return "Standard check-in is 2:00 PM. Early check-in can be requested (subject to availability). Tap Pre-Arrival Services to set up an early check-in request."

    if any(kw in t for kw in ("checkout time", "check-out time", "check out time", "when is checkout", "what time checkout")):
        return "Standard check-out is 12:00 PM (noon). Late check-out can be requested. Tap the Late Checkout option under In-Stay Services."

    # WiFi
    if any(kw in t for kw in ("wifi", "wi-fi", "internet", "free wifi")):
        return "Most hotels listed on BookBot offer free WiFi. It will be mentioned in the hotel amenities during your search. If you need to confirm, the hotel front desk can assist."

    # Parking
    if any(kw in t for kw in ("parking", "car park", "valet", "where to park")):
        return "Parking availability varies by hotel. It is listed in the hotel amenities during your search. Valet parking is available at most 5-star properties. Self-parking may require a fee."

    # Pool
    if any(kw in t for kw in ("pool", "swimming pool", "is there a pool")):
        return "Pool availability is shown in the hotel amenities during your search. Search with the Pool filter enabled to only see hotels with swimming pools."

    # Pets
    if any(kw in t for kw in ("pet policy", "are pets allowed", "can i bring pet")):
        return "Pet policy varies by hotel. I can filter for pet-friendly hotels. Just tell me you are travelling with a pet when searching. A pet fee may apply at check-in."

    # Cancellation policy
    if any(kw in t for kw in ("cancellation policy", "free cancellation", "refund policy", "can i cancel", "if i cancel")):
        return ("Most hotels offer free cancellation 72+ hours before check-in.\n"
                "After that, a cancellation fee (typically 1 night) applies.\n"
                "Non-refundable (Early Bird) rates cannot be cancelled.\n\n"
                "Your specific cancellation deadline is shown in your booking details.")

    # Payment
    if any(kw in t for kw in ("payment method", "how to pay", "accepted payment", "can i pay")):
        return ("Accepted payment methods:\n"
                "Credit/Debit Card (Visa, Mastercard, Amex, Rupay)\n"
                "UPI (GPay, PhonePe, Paytm)\n"
                "PayPal\n"
                "Pay at Hotel\n"
                "Voucher/Promo Code\n"
                "Loyalty Points\n\n"
                "Select your method at the payment step during booking.")

    # Breakfast
    if any(kw in t for kw in ("breakfast included", "is breakfast included", "free breakfast")):
        return "I offer several meal plan options during booking: Room Only, Bed & Breakfast, Half Board, and Full Board. Choose your preferred plan at the meal selection step."

    # Receipt / invoice
    if any(kw in t for kw in ("receipt", "invoice", "gst invoice", "tax invoice", "download receipt")):
        return ("Your booking confirmation includes:\n"
                "- Booking reference and summary\n"
                "- Detailed receipt on request\n"
                "- GST/VAT invoice for corporate bookings\n\n"
                "To get your receipt, go to My Bookings and select Download Receipt.")

    # Languages
    if any(kw in t for kw in ("languages supported", "what languages", "which language")):
        return ("BookBot supports 40 languages including English, Hindi, Arabic, Spanish, French, "
                "Portuguese, Bengali, Urdu, Japanese, Korean, and many more.\n\n"
                "Tap Change Language to switch at any time.")

    # Voice messages
    if any(kw in t for kw in ("voice message", "can i use voice", "speak to bot")):
        return "Yes! You can send voice messages — BookBot uses AI speech-to-text to understand your audio and will respond with text and voice."

    # Loyalty points
    if any(kw in t for kw in ("how do i earn points", "earn points", "loyalty points earn")):
        return ("Earn points on every booking:\n"
                "10 points per 100 currency units spent\n\n"
                "Tiers:\n"
                "Bronze (0–9,999 pts) — 5% bonus\n"
                "Silver (10k–49,999) — 10% bonus + upgrades\n"
                "Gold (50k–99,999) — 15% bonus + free breakfast\n"
                "Platinum (100k+) — 20% bonus + butler service\n\n"
                "Redeem 2 points = 1 unit of currency off your next booking.")

    # Refund
    if any(kw in t for kw in ("when refund", "refund time", "refund processing", "how long refund")):
        return "Refunds are processed within 5–7 business days after cancellation. The amount returns to your original payment method."

    # 24/7 support
    if any(kw in t for kw in ("24/7", "always open", "24 hour support", "open late")):
        return "BookBot is available 24/7 to help you search, book, and manage your hotel reservations. Live agent support is available during business hours."

    # Modify booking
    if any(kw in t for kw in ("how to modify", "can i change my booking")):
        return "Yes! You can modify your booking dates, room type, and meal plan. Tap Modify Booking in My Bookings or type 'modify booking' anytime."

    # How to book
    if any(kw in t for kw in ("how to book", "how do i book", "booking process")):
        return ("How to book with BookBot:\n"
                "1. Tap Book a Hotel or type your city\n"
                "2. Enter your check-in and check-out dates\n"
                "3. Select number of guests\n"
                "4. Choose from available hotels\n"
                "5. Select a room and meal plan\n"
                "6. Add any extras (airport transfer, spa, etc.)\n"
                "7. Enter your details\n"
                "8. Choose payment method\n"
                "9. Confirm — done!\n\n"
                "You will receive an instant email confirmation.")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SMART FILTERS — budget, stars, amenities
# ─────────────────────────────────────────────────────────────────────────────

def _parse_budget_intent(en_lower: str) -> float | None:
    """Extract budget amount from user input, return as float or None."""
    # Patterns like "under $100", "budget of 5000", "less than 200 per night"
    m = re.search(r'(?:under|below|less than|max|budget of|up to)\s*[$€£₹]?\s*(\d[\d,]*)', en_lower)
    if m:
        return float(m.group(1).replace(",", ""))
    m2 = re.search(r'[$€£₹]\s*(\d[\d,]*)\s*(?:per night|/night)?', en_lower)
    if m2:
        return float(m2.group(1).replace(",", ""))
    return None


def _parse_star_intent(en_lower: str) -> int | None:
    """Extract star rating from user input."""
    if "5 star" in en_lower or "five star" in en_lower or "luxury" in en_lower:
        return 5
    if "4 star" in en_lower or "four star" in en_lower:
        return 4
    if "3 star" in en_lower or "three star" in en_lower or "budget hotel" in en_lower:
        return 3
    m = re.search(r'(\d)\s*[-–]?\s*star', en_lower)
    if m:
        return int(m.group(1))
    return None


def _apply_hotel_filters(hotels: list, budget: float | None = None, stars: int | None = None,
                          amenities: list | None = None) -> list:
    """Filter hotel list by budget, stars, and amenities."""
    result = hotels
    if budget:
        result = [h for h in result if min(
            (r.get("price_per_night", 9999) for r in h.get("available_rooms", [{"price_per_night": 9999}])),
            default=9999,
        ) <= budget]
    if stars:
        result = [h for h in result if h.get("star_rating", 0) >= stars]
    if amenities:
        result = [h for h in result
                  if all(a.lower() in str(h.get("amenities", "")).lower() for a in amenities)]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# RETURNING GUEST PERSONALIZATION & SENTIMENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_negative_sentiment(text: str) -> bool:
    """Return True if the user message appears frustrated or angry."""
    _negative = {
        "terrible", "awful", "useless", "ridiculous", "unacceptable",
        "not working", "broken", "frustrated", "angry", "furious",
        "scam", "fraud", "cheated", "disgusted", "horrible", "worst",
        "30 minutes", "i've been waiting", "no response", "still waiting",
        "!!!", "help!!!",
    }
    t = text.lower()
    return any(kw in t for kw in _negative) or t.count("!") >= 3


def _get_returning_guest_greeting(state: dict, first_name: str) -> str | None:
    """If we have prior booking data, return a personalized welcome."""
    if state.get("loyalty_member_id") and state.get("loyalty_points", 0) > 0:
        pts = state["loyalty_points"]
        tier_emoji = "🥉" if pts < 10000 else ("🥈" if pts < 50000 else ("🥇" if pts < 100000 else "💎"))
        return (
            f"Welcome back, {first_name}! {tier_emoji}\n\n"
            f"Great to see you again. You have {pts:,} loyalty points.\n\n"
            "Ready for your next adventure?\n\n"
            "Tap Book a Hotel to start a new search."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():    return {
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
            # Delete Redis key BEFORE pre-creating state so _get_state() won't
            # restore a previous language and silently skip language selection.
            _redis_del_lang(sender_id)
            set_user_language(sender_id, "en")
            _user_states[sender_id] = {
                "lang_confirmed":      False,
                "awaiting_lang":       True,
                "lang_page":           1,
                "awaiting_type_input": False,
            }
            welcome_en = (
                "Welcome back to BookBot!\n\n"
                "Let's start fresh. Choose your language:"
            )
            audio_out = text_to_speech_bytes(_strip_for_tts(welcome_en), "en")
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": welcome_en, "buttons": _build_lang_buttons(1), "audio_b64": a64, "lang": "en"}

        # ── GET_STARTED: show welcome message + first page of language buttons ─
        if user_message.strip().upper() == "GET_STARTED":
            # Same as RESTART: wipe Redis + pre-create state so language selection
            # is always shown, even if the user had a previously saved language.
            _redis_del_lang(sender_id)
            set_user_language(sender_id, "en")
            _user_states[sender_id] = {
                "lang_confirmed":      False,
                "awaiting_lang":       True,
                "lang_page":           1,
                "awaiting_type_input": False,
            }
            welcome_en = (
                "Welcome to BookBot!\n\n"
                "I am your hotel booking assistant.\n"
                "I can search hotels, check availability, and make bookings.\n\n"
                "First, choose your language:"
            )
            audio_out = text_to_speech_bytes(_strip_for_tts(welcome_en), "en")
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": welcome_en, "buttons": _build_lang_buttons(1), "audio_b64": a64, "lang": "en"}

        # ── Step 2: Language selection flow ────────────────────────────────────
        if state["awaiting_lang"]:
            # User tapped "Type my language" and is now typing a language name
            if state.get("awaiting_type_input"):
                t_lower    = user_message.strip().lower()
                found_code = None
                for name, code in _LANG_BY_NAME.items():
                    if name in t_lower:
                        found_code = code
                        break
                if not found_code and t_lower in LANGUAGE_MENU:
                    found_code = LANGUAGE_MENU[t_lower]
                if not found_code:
                    try:
                        detected = detect_language(user_message)
                        if detected and detected != "en":
                            found_code = detected
                    except Exception:
                        pass
                if not found_code:
                    err_text = (
                        f"Sorry, I could not find '{user_message[:30]}'.\n"
                        "Please try again, or choose from the list:"
                    )
                    return {"text": err_text, "buttons": _build_lang_buttons(state["lang_page"]), "audio_b64": None, "lang": "en"}
                state["awaiting_type_input"] = False
                result = found_code
            else:
                result = _parse_lang_selection(user_message, state["lang_page"])

            # Navigation: move to a different language page
            if isinstance(result, tuple) and result[0] == "page":
                state["lang_page"] = result[1]
                menu_text = _build_lang_menu_text(result[1])
                buttons   = _build_lang_buttons(result[1])
                tts_text  = _strip_for_tts(menu_text)
                audio_out = text_to_speech_bytes(tts_text, "en")
                a64 = base64.b64encode(audio_out).decode() if audio_out else None
                return {"text": menu_text, "buttons": buttons, "audio_b64": a64, "lang": "en"}

            # "Type my language" button tapped — ask user to type
            if isinstance(result, tuple) and result[0] == "type_input":
                state["awaiting_type_input"] = True
                type_prompt = (
                    "Please type your language name.\n\n"
                    "Examples: Hindi, Spanish, Arabic, Tamil, French, Swahili"
                )
                return {"text": type_prompt, "buttons": [], "audio_b64": None, "lang": "en"}

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
            tts_text  = _strip_for_tts(body_text)
            audio_out = text_to_speech_bytes(tts_text, lang_choice)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": body_text, "buttons": _MAIN_BUTTONS, "audio_b64": a64, "lang": lang_choice}

        # ── Step 3: Normal conversation (language already confirmed) ───────────
        input_lang    = stt_lang or detect_language(user_message)
        english_input = translate_to_english(user_message, input_lang)
        en_lower      = english_input.strip().lower()

        # Handle quick-reply button payloads
        raw_upper = user_message.strip().upper()

        # ── Part P: Telegram /commands and SMS menu digit resolution ──────────
        _tg_payload = _resolve_telegram_command(user_message)
        if _tg_payload:
            raw_upper = _tg_payload
            user_message = _tg_payload
        else:
            _sms_payload = _resolve_sms_menu(user_message)
            if _sms_payload:
                raw_upper    = _sms_payload
                user_message = _sms_payload

        # ── Part O38: SQL injection / XSS / script injection block ────────────
        _INJECT_PATTERNS = [
            r"(?i)(select\s.+from\s|insert\s+into|drop\s+table|delete\s+from"
            r"|update\s+.+set\s|union\s+select|or\s+1=1|--\s|;\s*drop|<script"
            r"|javascript:|onerror=|onload=|eval\(|exec\(|system\()",
        ]
        import re as _re2
        if any(_re2.search(p, user_message) for p in _INJECT_PATTERNS):
            return {"text": "I did not understand that. How can I help you?",
                    "buttons": _MAIN_BUTTONS, "audio_b64": None, "lang": lang}

        # ── Part O45: Under-18 warning ────────────────────────────────────────
        _minor_re = _re2.search(
            r'\b(i am|i\'m|am)\s+(1[0-7]|[1-9])\s*(years?\s*old|yrs?\s*old|year\s*old)?\b|'
            r'\bi\'m\s+a\s+minor\b|\bminor\b|\bi\s+am\s+underage\b',
            user_message.lower()
        )
        if _minor_re:
            return {
                "text": (
                    "I am sorry, but hotel bookings require the primary guest to be 18 or over.\n\n"
                    "Please ask a parent or guardian to complete the booking on your behalf."
                ),
                "buttons": _MAIN_BUTTONS, "audio_b64": None, "lang": lang,
            }

        # ── Part O26-O31: Payment error payloads ──────────────────────────────
        if raw_upper == "PAY_CARD_DECLINED":
            return {
                "text": (
                    "Your card was declined. This can happen due to:\n"
                    "- Insufficient funds\n"
                    "- Incorrect card details\n"
                    "- Bank security block\n\n"
                    "Please try a different card or payment method."
                ),
                "buttons": [
                    {"content_type": "text", "title": "Try Again",         "payload": "PAY_CARD"},
                    {"content_type": "text", "title": "Pay at Hotel",      "payload": "PAY_AT_HOTEL"},
                    {"content_type": "text", "title": "UPI",               "payload": "PAY_UPI"},
                    {"content_type": "text", "title": "Speak to Agent",    "payload": "AGENT_HANDOFF"},
                ],
                "audio_b64": None, "lang": lang,
            }

        if raw_upper == "PAY_TIMEOUT":
            return {
                "text": (
                    "Your payment session timed out. No charge was made.\n\n"
                    "Please try again — your booking details are still saved."
                ),
                "buttons": [
                    {"content_type": "text", "title": "Try Payment Again", "payload": "PAY_CARD"},
                    {"content_type": "text", "title": "Pay at Hotel",      "payload": "PAY_AT_HOTEL"},
                ],
                "audio_b64": None, "lang": lang,
            }

        if raw_upper == "PAY_DOUBLE":
            return {
                "text": (
                    "It looks like you may have been charged twice.\n\n"
                    "Do not worry — our team will review this and issue a refund\n"
                    "for any duplicate charge within 24 hours.\n\n"
                    "Please contact us if you need urgent help."
                ),
                "buttons": [
                    {"content_type": "text", "title": "Speak to Agent", "payload": "AGENT_HANDOFF"},
                    {"content_type": "text", "title": "My Bookings",    "payload": "MY_BOOKINGS"},
                ],
                "audio_b64": None, "lang": lang,
            }

        if raw_upper == "VOUCHER_USED":
            return {
                "text": "That voucher has already been used. Each voucher can only be redeemed once.",
                "buttons": [
                    {"content_type": "text", "title": "Try Another Code", "payload": "PAY_VOUCHER"},
                    {"content_type": "text", "title": "Pay at Hotel",     "payload": "PAY_AT_HOTEL"},
                ],
                "audio_b64": None, "lang": lang,
            }

        # ── Part O39: Conversation stop/quit/reset keywords ───────────────────
        if raw_upper in ("STOP", "QUIT", "EXIT", "RESTART", "START_OVER", "START OVER"):
            _reset_booking_slots(state)
            state["step"] = None
            return {
                "text": "Okay, I have cleared your session. Tap below to start fresh.",
                "buttons": _MAIN_BUTTONS, "audio_b64": None, "lang": lang,
            }

        if raw_upper == "ACTION_CHANGE_LANG" or any(trigger in en_lower for trigger in _CHANGE_LANG_TRIGGERS) or "change language" in user_message.lower():
            state["lang_confirmed"]      = False
            state["awaiting_lang"]       = True
            state["lang_page"]           = 1
            state["awaiting_type_input"] = False
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
        elif raw_upper == "CALL_US":
            body_text = translate_to(
                "You can reach our 24/7 support line. Please contact the hotel directly "
                "or type 'agent' to be connected to a BookBot live agent.",
                lang
            )
            tts_text = _strip_for_tts(body_text)
            audio_out = text_to_speech_bytes(tts_text, lang)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": body_text, "buttons": _MAIN_BUTTONS, "audio_b64": a64, "lang": lang}

        # ── Sub-flow payloads: route before the booking state machine ──────────
        _SUB_FLOW_PREFIXES = (
            "LOYALTY_", "PRE_ARRIVAL", "EARLY_CI", "AIRPORT_", "ROOM_UPGRADE",
            "SPECIAL_", "DIET_", "DIETARY", "OCCASION_",
            "IN_STAY", "SPA_", "RESTAURANT_", "ROOM_SERVICE", "HK_",
            "HOUSEKEEPING", "LOCAL_TIPS", "COMP_", "LCO_", "LOST_",
            "MODIFY_", "MODSEL_", "MOD_", "EXTEND_",
            "AGENT_HANDOFF", "HANDOFF_", "LEAVE_MESSAGE",
            "GROUP_BOOKING", "CORP_BOOKING", "WEDDING_", "LONG_STAY",
            "GTYPE_", "LONGSTAY_", "ACCESSIBLE_", "ACC_", "PET_",
            "MCITY_", "ROMANTIC_", "ROM_", "WED_",
            "EARLY_CHECKIN", "SPA_BOOKING", "COMPLAINT",
            # Part U — Seasonal
            "NYE_", "XMAS_", "VAL_", "RAM_", "MONSOON_", "CONF_", "SPORT_",
            # Part V — Corporate
            "CORP_", "CORP_SETUP", "CORP_TEAM", "CORP_INVOICE_", "CORP_RPT",
            # Part W — Accessibility
            "ACCESS_", "ACCESSIBLE_BOOKING",
            # Part X — Crisis
            "EMERGENCY_", "EMERGENCY_BOOK", "STRANDED_", "STRANDED_TRAVELER",
            "STRAND_", "OVERBOOK_", "MEDICAL_EMERGENCY", "MEDICAL_EMERGENCY",
            "LOST_PASSPORT", "EMBASSY_", "CALL_EMERGENCY", "FIND_HOSPITAL",
            # Part Y — Upsell
            "UPSELL_", "UPGRADE_OFFER",
        )

        # ── Part G6/G7/G8: Booking reference action payloads ──────────────────
        if raw_upper.startswith("BOOKING_DETAIL_"):
            ref_bd = raw_upper[len("BOOKING_DETAIL_"):]
            bot_en, buttons = None, []
            if _DB_AVAILABLE:
                try:
                    bk = get_booking_by_ref(ref_bd)
                    if bk:
                        bot_en, buttons = _show_booking_detail(bk)
                except Exception as _e:
                    logger.error("BOOKING_DETAIL lookup: %s", _e)
            if not bot_en:
                bot_en  = f"Booking {ref_bd} was not found."
                buttons = _MAIN_BUTTONS
            response_text = translate_to(bot_en, lang) if lang != "en" else bot_en
            tts_text  = _strip_for_tts(response_text)
            audio_out = text_to_speech_bytes(tts_text, lang)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": response_text, "buttons": buttons, "audio_b64": a64, "lang": lang}

        if raw_upper.startswith("SHARE_") and "_" in raw_upper[6:]:
            ref_sh = raw_upper[6:]
            if not ref_sh.startswith("EMAIL_"):
                bot_en, buttons = _handle_share_booking(ref_sh)
            else:
                # SHARE_EMAIL_<ref>
                _share_ref = ref_sh[6:]
                state["share_ref"]  = _share_ref
                state["step"]       = "share_email"
                bot_en  = f"Please enter the email address to send booking {_share_ref} to:"
                buttons = [{"content_type": "text", "title": "Cancel", "payload": "RESTART"}]
            response_text = translate_to(bot_en, lang) if lang != "en" else bot_en
            tts_text  = _strip_for_tts(response_text)
            audio_out = text_to_speech_bytes(tts_text, lang)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": response_text, "buttons": buttons, "audio_b64": a64, "lang": lang}

        if raw_upper.startswith("RECEIPT_") or raw_upper.startswith("INVOICE_"):
            _inv  = raw_upper.startswith("INVOICE_")
            _pref = "INVOICE_" if _inv else "RECEIPT_"
            ref_rc = raw_upper[len(_pref):]
            bot_en, buttons = _handle_receipt(ref_rc, invoice=_inv)
            response_text = translate_to(bot_en, lang) if lang != "en" else bot_en
            tts_text  = _strip_for_tts(response_text)
            audio_out = text_to_speech_bytes(tts_text, lang)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": response_text, "buttons": buttons, "audio_b64": a64, "lang": lang}

        if raw_upper.startswith("DL_PDF_") or raw_upper.startswith("DL_GST_") or \
           raw_upper.startswith("DL_CORP_") or raw_upper.startswith("EMAIL_RECEIPT_"):
            _dl_ref = raw_upper.split("_", 2)[-1]
            _dl_url = f"https://bookbot.io/receipt/{_dl_ref}"
            bot_en  = (
                f"Your document for booking {_dl_ref} is ready.\n"
                f"Download here:\n{_dl_url}\n\n"
                "The link will expire in 24 hours."
            )
            buttons = [
                {"content_type": "text", "title": "My Bookings", "payload": "MY_BOOKINGS"},
                {"content_type": "text", "title": "Main Menu",   "payload": "RESTART"},
            ]
            response_text = translate_to(bot_en, lang) if lang != "en" else bot_en
            tts_text  = _strip_for_tts(response_text)
            audio_out = text_to_speech_bytes(tts_text, lang)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": response_text, "buttons": buttons, "audio_b64": a64, "lang": lang}

        if raw_upper.startswith("COPY_LINK_"):
            _cl_ref = raw_upper[len("COPY_LINK_"):]
            bot_en  = f"Your booking link:\nhttps://bookbot.io/view/{_cl_ref}"
            buttons = [{"content_type": "text", "title": "My Bookings", "payload": "MY_BOOKINGS"}]
            response_text = translate_to(bot_en, lang) if lang != "en" else bot_en
            tts_text  = _strip_for_tts(response_text)
            audio_out = text_to_speech_bytes(tts_text, lang)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": response_text, "buttons": buttons, "audio_b64": a64, "lang": lang}

        if raw_upper.startswith("REBOOK_"):
            _rb_ref = raw_upper[len("REBOOK_"):]
            bot_en, buttons = _handle_rebook(sender_id, state, _rb_ref)
            response_text = translate_to(bot_en, lang) if lang != "en" else bot_en
            tts_text  = _strip_for_tts(response_text)
            audio_out = text_to_speech_bytes(tts_text, lang)
            a64 = base64.b64encode(audio_out).decode() if audio_out else None
            return {"text": response_text, "buttons": buttons, "audio_b64": a64, "lang": lang}

        _always_booking = raw_upper.startswith(("HOTEL_", "ROOM_", "MEAL_", "CHECKIN_",
                                                 "CHECKOUT_", "GUESTS_", "CONFIRM_", "CANCEL_",
                                                 "SKIP_", "RESELECT_", "ADDON_", "PAY_",
                                                 "PAYMENT_")) or raw_upper in (
            "ACTION_BOOK", "MY_BOOKINGS", "LOOKUP_BOOKING",
        )

        # ── Booking flow ───────────────────────────────────────────────────────
        if _always_booking or not any(raw_upper == p or raw_upper.startswith(p)
                                       for p in _SUB_FLOW_PREFIXES):
            bot_en, buttons = _handle_booking_flow(sender_id, state, en_lower, user_message, lang)
        else:
            bot_en, buttons = None, []

        if bot_en is None:
            # Try sub-flow handlers in priority order
            for _handler in [
                lambda: _handle_crisis_flow(sender_id, state, user_message, en_lower),
                lambda: _handle_accessibility_flow(sender_id, state, user_message, en_lower),
                lambda: _handle_seasonal_flow(sender_id, state, user_message, en_lower),
                lambda: _handle_corporate_flow(sender_id, state, user_message, en_lower),
                lambda: _handle_upsell_flow(sender_id, state, user_message, en_lower),
                lambda: _handle_loyalty_flow(sender_id, state, user_message, en_lower),
                lambda: _handle_pre_arrival_flow(sender_id, state, user_message, en_lower),
                lambda: _handle_in_stay_flow(sender_id, state, user_message, en_lower),
                lambda: _handle_modification_flow(sender_id, state, user_message, en_lower),
                lambda: _handle_agent_handoff(sender_id, state, user_message, en_lower),
                lambda: _handle_advanced_booking_flow(sender_id, state, user_message, en_lower),
            ]:
                _r = _handler()
                if _r[0] is not None:
                    bot_en, buttons = _r
                    break

        if bot_en is None:
            # FAQ check
            faq_answer = _handle_faq(en_lower, state)
            if faq_answer:
                bot_en  = faq_answer
                buttons = _MAIN_BUTTONS

        if bot_en is None:
            # Negative sentiment — empathise before fallback
            if _detect_negative_sentiment(user_message):
                bot_en = (
                    "I am really sorry to hear that, and I completely understand your frustration.\n\n"
                    "Let me connect you with a live agent who can resolve this immediately."
                )
                buttons = [
                    {"content_type": "text", "title": "Speak to Agent", "payload": "AGENT_HANDOFF"},
                    {"content_type": "text", "title": "Try Again",      "payload": "RESTART"},
                ]
            else:
                # Returning guest personalisation
                first_name = (state.get("guest_name") or "").split()[0] if state.get("guest_name") else ""
                returning_msg = _get_returning_guest_greeting(state, first_name) if first_name else None
                if returning_msg and raw_upper in ("GET_STARTED", "RESTART", "MY_PROFILE"):
                    bot_en  = returning_msg
                    buttons = _MAIN_BUTTONS
                else:
                    # General conversational fallback — also try booking flow for NL input
                    if not _always_booking:
                        _bf_en, _bf_buttons = _handle_booking_flow(sender_id, state, en_lower, user_message, lang)
                        if _bf_en:
                            bot_en, buttons = _bf_en, _bf_buttons

                if bot_en is None:
                    bot_en = _human_response(en_lower) or (
                        "I am not sure I understood that. Here is what I can help with:\n\n"
                        "- Tap Book a Hotel to search for hotels\n"
                        "- Tap My Bookings to see your reservations\n"
                        "- Type 'early check-in', 'spa', 'modify booking' for extra services\n"
                        "- Type 'loyalty' to see your rewards\n"
                        "- Type 'agent' to speak to a human\n"
                        "- Tap Help for full feature list"
                    )
                    buttons = _MAIN_BUTTONS

        response_text = translate_to(bot_en, lang) if lang != "en" else bot_en
        tts_text      = _strip_for_tts(response_text)
        audio_out     = text_to_speech_bytes(tts_text, lang)
        a64 = base64.b64encode(audio_out).decode() if audio_out else None
        return {"text": response_text, "buttons": buttons, "audio_b64": a64, "lang": lang}

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        return {"text": "Sorry, something went wrong. Please try again.", "buttons": _MAIN_BUTTONS, "audio_b64": None, "lang": "en"}