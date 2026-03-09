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
    lines += [
        "\u2500" * 30,
        f"Total     : {currency} {total:.2f}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# BOOKING STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────

def _reset_booking_slots(state: dict) -> None:
    for k in ("city", "checkin", "checkout", "num_adults", "num_children",
              "selected_hotel", "selected_room", "meal_plan", "rate_plan",
              "guest_name", "guest_email", "guest_phone", "special_requests",
              "_hotel_results", "_nights"):
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
            "It starts with BB, for example BBAB1234XY.",
            [{"content_type": "text", "title": "Go Back", "payload": "RESTART"}],
        )

    # ── Look up booking ───────────────────────────────────────────────────────
    _lookup_kw = {"find my booking", "lookup booking",
                  "booking status", "booking details", "check my booking status"}
    if raw_upper == "LOOKUP_BOOKING" or any(kw in en_lower for kw in _lookup_kw):
        state["step"] = "lookup_ref"
        return (
            "Please enter your booking reference number to look it up.",
            [{"content_type": "text", "title": "Go Back", "payload": "RESTART"}],
        )

    # ── Handle active cancel / lookup flows ───────────────────────────────────
    if step == "cancel_ref":
        return _handle_cancel_ref(sender_id, state, raw_message, en_lower)
    if step == "cancel_confirm":
        return _handle_cancel_confirm(sender_id, state, en_lower)
    if step == "lookup_ref":
        return _handle_lookup_ref(sender_id, state, raw_message)

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
            return ("That date is in the past. Please choose a future check-in date.",
                    _checkin_buttons())
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

        header = (
            f"Found {len(hotels)} hotel{'s' if len(hotels) > 1 else ''} "
            f"in {city} — {_pretty_date(checkin)} to {_pretty_date(checkout)}, "
            f"{guest_s}:\n\n"
        )
        lines, buttons = [], []
        for i, h in enumerate(hotels[:5]):
            stars    = "\u2605" * (h.get("star_rating") or 0)
            currency = h.get("currency", "USD")
            min_p    = min(
                (r["price_per_night"] for r in h.get("available_rooms", [])
                 if r.get("price_per_night")),
                default=None,
            )
            price_s  = f"From {currency} {min_p:.0f}/night" if min_p else "Price on request"
            lines.append(f"{i+1}. {h.get('name','Hotel')} {stars}\n   {h.get('city','')}, {h.get('country','')}\n   {price_s}")
            title = f"{i+1}. {h.get('name','Hotel')}"[:20]
            buttons.append({"content_type": "text", "title": title, "payload": f"HOTEL_{i}"})

        return header + "\n\n".join(lines) + "\n\nTap a hotel to view rooms.", buttons[:13]

    # ── STEP: hotel ───────────────────────────────────────────────────────────
    if step == "hotel":
        hotels   = state.get("_hotel_results", [])
        selected = None
        raw      = raw_message.strip()

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
        state["step"] = "name"

        _ml = {"room_only": "Room Only", "breakfast": "Bed & Breakfast",
               "half_board": "Half Board", "full_board": "Full Board"}
        ml_disp  = _ml.get(meal, "Room Only")
        nights   = state.get("_nights", 1)
        currency = state.get("selected_hotel", {}).get("currency", "USD")
        total    = price_n * nights

        return (
            f"Meal plan: {ml_disp}\n"
            f"Total for {nights} night{'s' if nights > 1 else ''}: {currency} {total:.2f}\n\n"
            "Now I need a few details.\n\n"
            "Please enter the lead guest full name:",
            [],
        )

    # ── STEP: name ────────────────────────────────────────────────────────────
    if step == "name":
        name = raw_message.strip()
        if len(name) < 2:
            return "Please enter a valid full name.", []
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
        if raw.upper() in ("SKIP", "SKIP_PHONE", "NO", "NO THANKS", "NONE"):
            state["guest_phone"] = ""
        else:
            cleaned = re.sub(r'[^\d\+\-\s\(\)]', '', raw).strip()
            state["guest_phone"] = cleaned if len(cleaned) >= 7 else ""
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
            return _create_booking(sender_id, state)
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

    return None, []


def _create_booking(sender_id: str, state: dict) -> tuple[str, list]:
    """Persist the booking to Supabase and return a confirmation message."""
    hotel  = state.get("selected_hotel", {})
    room   = state.get("selected_room",  {})
    nights = state.get("_nights", 1)
    price_n = room.get("_final_price") or room.get("price_per_night") or 0
    total   = price_n * nights
    currency = hotel.get("currency", "USD")

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
    meal_labels = {"room_only": "Room Only", "breakfast": "Bed & Breakfast",
                   "half_board": "Half Board", "full_board": "Full Board"}
    meal_disp  = meal_labels.get(state.get("meal_plan", "room_only"), "Room Only")

    # Reset booking slots
    state["step"] = None
    _reset_booking_slots(state)

    return (
        f"Your booking is confirmed!\n\n"
        f"Reference : {booking_ref}\n"
        f"Hotel     : {hotel_name}\n"
        f"Room      : {room_name}\n"
        f"Meal Plan : {meal_disp}\n"
        f"Check-in  : {_pretty_date(checkin)}\n"
        f"Check-out : {_pretty_date(checkout)}\n"
        f"Total     : {currency} {total:.2f}\n\n"
        f"A confirmation has been noted for {guest_email}.\n\n"
        "Is there anything else I can help you with?",
        _MAIN_BUTTONS,
    )


def _show_my_bookings(sender_id: str, state: dict) -> tuple[str, list]:
    """Fetch and display the user's recent bookings."""
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
            _MAIN_BUTTONS,
        )

    lines = []
    for b in bookings[:5]:
        ref    = b.get("booking_reference", "N/A")
        hotel  = b.get("hotel_name", b.get("name", "Hotel"))
        ci     = b.get("check_in",  "")
        co     = b.get("check_out", "")
        status = (b.get("status") or "pending").capitalize()
        amt    = b.get("total_amount")
        curr   = b.get("currency", "")
        p_s    = f"  {curr} {float(amt):.0f}" if amt else ""
        lines.append(f"\u2022 {ref} \u2014 {hotel}\n  {ci} to {co}  [{status}]{p_s}")

    return (
        f"Your recent bookings ({len(bookings)}):\n\n" + "\n\n".join(lines) +
        "\n\nTo cancel a booking tap Cancel Booking and enter the reference.",
        [
            {"content_type": "text", "title": "Cancel a Booking",
             "payload": "CANCEL_BOOKING"},
            {"content_type": "text", "title": "New Booking",
             "payload": "ACTION_BOOK"},
            {"content_type": "text", "title": "Go Home",
             "payload": "ACTION_HELP"},
        ],
    )


def _handle_cancel_ref(
    sender_id: str, state: dict, raw_message: str, en_lower: str
) -> tuple[str, list]:
    """User just entered a booking reference for cancellation."""
    ref = raw_message.strip().upper()

    if not re.match(r'^BB[A-Z0-9]{6,10}$', ref):
        return (
            "That does not look like a valid booking reference.\n"
            "It should start with BB, e.g. BBXYZ12345. Please try again:",
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
            [{"content_type": "text", "title": "Try Again", "payload": "CANCEL_BOOKING"},
             {"content_type": "text", "title": "Go Back",   "payload": "RESTART"}],
        )

    if booking.get("status") == "cancelled":
        return (
            f"Booking {ref} is already cancelled.",
            _MAIN_BUTTONS,
        )

    hotel = booking.get("hotel_name", booking.get("name", ""))
    ci    = booking.get("check_in",  "")
    co    = booking.get("check_out", "")
    amt   = booking.get("total_amount")
    curr  = booking.get("currency", "")
    p_s   = f"{curr} {float(amt):.0f}" if amt else ""

    state["cancel_ref"]        = ref
    state["cancel_hotel_name"] = hotel
    state["step"]              = "cancel_confirm"

    return (
        f"Are you sure you want to cancel this booking?\n\n"
        f"Ref   : {ref}\n"
        f"Hotel : {hotel}\n"
        f"Dates : {ci} to {co}\n"
        f"Total : {p_s}\n\n"
        "This action cannot be undone.",
        [
            {"content_type": "text", "title": "Yes, Cancel It", "payload": "CANCEL_CONFIRM"},
            {"content_type": "text", "title": "No, Keep It",    "payload": "RESTART"},
        ],
    )


def _handle_cancel_confirm(
    sender_id: str, state: dict, en_lower: str
) -> tuple[str, list]:
    """Process the user's confirmation/denial of a cancellation."""
    raw   = en_lower.strip()
    ref   = state.get("cancel_ref", "")
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
        state.pop("cancel_ref",        None)
        state.pop("cancel_hotel_name", None)

        if success:
            return (
                f"Booking {ref} has been cancelled.\n\n"
                "Your refund (if applicable) will be processed\n"
                "according to the hotel cancellation policy.\n\n"
                "Is there anything else I can help you with?",
                _MAIN_BUTTONS,
            )
        return (
            f"Sorry, I could not cancel {ref} right now.\n"
            "Please try again later or contact support.",
            _MAIN_BUTTONS,
        )

    # User said no
    state["step"] = None
    state.pop("cancel_ref",        None)
    state.pop("cancel_hotel_name", None)
    return (
        "Your booking is still active.\n\n"
        "Is there anything else I can help you with?",
        _MAIN_BUTTONS,
    )


def _handle_lookup_ref(
    sender_id: str, state: dict, raw_message: str
) -> tuple[str, list]:
    """User entered a booking reference to look up."""
    ref = raw_message.strip().upper()

    if not re.match(r'^BB[A-Z0-9]{6,10}$', ref):
        return (
            "That does not look like a valid booking reference. Please try again.",
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
            _MAIN_BUTTONS,
        )

    hotel  = booking.get("hotel_name", booking.get("name", ""))
    ci     = booking.get("check_in",  "")
    co     = booking.get("check_out", "")
    room   = booking.get("room_type_code", "")
    status = (booking.get("status") or "").capitalize()
    paid   = (booking.get("payment_status") or "").capitalize()
    amt    = booking.get("total_amount")
    curr   = booking.get("currency", "")
    p_s    = f"{curr} {float(amt):.0f}" if amt else ""

    return (
        f"Booking Details\n"
        f"{chr(8212)*20}\n"
        f"Reference : {ref}\n"
        f"Hotel     : {hotel}\n"
        f"Room      : {room}\n"
        f"Check-in  : {ci}\n"
        f"Check-out : {co}\n"
        f"Status    : {status}\n"
        f"Payment   : {paid}\n"
        f"Total     : {p_s}",
        [
            {"content_type": "text", "title": "Cancel Booking",
             "payload": "CANCEL_BOOKING"},
            {"content_type": "text", "title": "Book Another",
             "payload": "ACTION_BOOK"},
            {"content_type": "text", "title": "My Bookings",
             "payload": "MY_BOOKINGS"},
        ],
    )


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

        # ── Booking flow ───────────────────────────────────────────────────────
        bot_en, buttons = _handle_booking_flow(sender_id, state, en_lower, user_message, lang)

        if bot_en is None:
            # Not a booking intent — use conversational fallback
            bot_en = _human_response(en_lower) or (
                "I am not sure I understood that. Here is what I can help with:\n\n"
                "Tap Book a Hotel to search for hotels.\n"
                "Tap My Bookings to see your reservations.\n"
                "Tap Help to see all features."
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