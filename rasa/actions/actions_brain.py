"""
rasa/actions/actions_brain.py
------------------------------
Custom Rasa actions — BRAIN module.

Module 1: Language Detection + Multilingual Greeting (COMPLETE)
  ActionGreetUser — detects user language, responds in that language,
                    sets the `language` slot, sends quick-reply buttons.

Do NOT modify existing action classes. Append new actions below.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path bootstrap — allows importing from services/ when actions run as a
# standalone server (rasa run actions).  The actions server starts from the
# rasa/ directory, so we walk two levels up to the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Multilingual greeting strings (mirrors services/language_service/main.py).
# Kept here as a local copy so ActionGreetUser never needs an HTTP call at
# greet time — zero latency, zero external dependency on the language service.
# ---------------------------------------------------------------------------
_GREETINGS: Dict[str, str] = {
    "en": "Hello! Welcome to BookBot. 👋 How can I help you today?",
    "ar": "مرحباً! أهلاً بك في BookBot. 👋 كيف يمكنني مساعدتك اليوم؟",
    "fr": "Bonjour! Bienvenue sur BookBot. 👋 Comment puis-je vous aider?",
    "es": "¡Hola! Bienvenido a BookBot. 👋 ¿Cómo puedo ayudarte hoy?",
    "de": "Hallo! Willkommen bei BookBot. 👋 Wie kann ich Ihnen heute helfen?",
    "hi": "नमस्ते! BookBot में आपका स्वागत है। 👋 आज मैं आपकी कैसे मदद कर सकता हूँ?",
    "zh": "您好！欢迎来到 BookBot。👋 今天我能帮您什么？",
    "ja": "こんにちは！BookBot へようこそ。👋 本日はどのようなご用件でしょうか？",
    "ko": "안녕하세요! BookBot 에 오신 것을 환영합니다. 👋 오늘 어떻게 도와드릴까요?",
    "pt": "Olá! Bem-vindo ao BookBot. 👋 Como posso ajudá-lo hoje?",
    "ru": "Здравствуйте! Добро пожаловать в BookBot. 👋 Чем могу помочь?",
    "id": "Halo! Selamat datang di BookBot. 👋 Bagaimana saya bisa membantu Anda?",
    "tr": "Merhaba! BookBot'a hoş geldiniz. 👋 Bugün size nasıl yardımcı olabilirim?",
    "th": "สวัสดี! ยินดีต้อนรับสู่ BookBot! 👋 วันนี้ฉันจะช่วยคุณได้อย่างไร?",
    "vi": "Xin chào! Chào mừng đến với BookBot. 👋 Tôi có thể giúp gì cho bạn hôm nay?",
    "te": "హలో! BookBot కి స్వాగతం. 👋 నేను మీకు ఎలా సహాయపడగలను?",
    "ta": "வணக்கம்! BookBot-க்கு வரவேற்கிறோம். 👋 இன்று நான் உங்களுக்கு எவ்வாறு உதவலாம்?",
}

# Quick-reply button definitions sent with every greeting.
# Payload prefix LANG_ is stripped by main.py before forwarding to the processor.
_LANG_BUTTONS: List[Dict[str, str]] = [
    {"content_type": "text", "title": "🇬🇧 English",  "payload": "LANG_en"},
    {"content_type": "text", "title": "🇸🇦 عربي",     "payload": "LANG_ar"},
    {"content_type": "text", "title": "🇫🇷 Français", "payload": "LANG_fr"},
    {"content_type": "text", "title": "🇪🇸 Español",  "payload": "LANG_es"},
    {"content_type": "text", "title": "🇩🇪 Deutsch",  "payload": "LANG_de"},
    {"content_type": "text", "title": "🇮🇳 हिन्दी",   "payload": "LANG_hi"},
    {"content_type": "text", "title": "🇨🇳 中文",      "payload": "LANG_zh"},
    {"content_type": "text", "title": "🇯🇵 日本語",    "payload": "LANG_ja"},
    {"content_type": "text", "title": "🇧🇷 Português", "payload": "LANG_pt"},
    {"content_type": "text", "title": "🇷🇺 Русский",  "payload": "LANG_ru"},
]

# Languages that are written right-to-left.
_RTL_LANGS = {"ar", "he", "fa", "ur", "yi"}


def _detect_lang(text: str) -> tuple[str, float]:
    """
    Detect the language of *text* using the project's language detector.

    Returns (language_code, confidence).  Falls back to ("en", 0.0) on any
    error so the greeting always succeeds.
    """
    if not text or not text.strip():
        return "en", 0.0
    try:
        from services.language_service.detector import detect
        result = detect(text)
        return result.get("language_code", "en"), result.get("confidence", 0.0)
    except Exception as exc:
        logger.warning("Language detection failed: %s — defaulting to 'en'", exc)
        return "en", 0.0


def _translate_greeting(text: str, target_lang: str) -> str:
    """
    Translate the English greeting to *target_lang* via the translator service.
    Returns the original English greeting on any error.
    """
    try:
        from services.language_service.translator import translate
        return translate(text, src="en", tgt=target_lang)
    except Exception as exc:
        logger.warning("Translation failed for lang=%s: %s", target_lang, exc)
        return text


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1 — ACTION: ActionGreetUser
# ═══════════════════════════════════════════════════════════════════════════

class ActionGreetUser(Action):
    """
    Multilingual greeting action — Module 1, fully complete.

    Flow:
      1. Check if `language` slot already set (returning user / language button tap).
      2. If not set: detect from the user's opening message text.
      3. If confidence < 0.75 (short greetings like "hi"): keep detected language
         but show language-selection quick replies so the user can confirm.
      4. Return greeting in the user's language.
      5. Set `language` slot so all subsequent actions know the preferred language.
    """

    def name(self) -> Text:
        return "action_greet_user"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # ── 1. Resolve language ──────────────────────────────────────────────
        lang = tracker.get_slot("language")
        confidence = 1.0

        if not lang:
            user_text = tracker.latest_message.get("text", "") or ""
            lang, confidence = _detect_lang(user_text)

        # Sanitise: strip region suffix (e.g. "zh-cn" → "zh")
        lang = lang.lower().split("-")[0].split("_")[0]

        # ── 2. Build greeting ────────────────────────────────────────────────
        if lang in _GREETINGS:
            greeting = _GREETINGS[lang]
        else:
            # Language known but no pre-written greeting → translate English one
            english = _GREETINGS["en"]
            greeting = _translate_greeting(english, lang)

        # ── 3. Build follow-up prompt ────────────────────────────────────────
        # Low confidence + Latin-script message could mean we misidentified.
        # Show language buttons so user can pick if needed; also offer "Book now".
        show_lang_buttons = confidence < 0.85

        if show_lang_buttons:
            dispatcher.utter_message(text=greeting, quick_replies=_LANG_BUTTONS)
        else:
            # High-confidence detection — show action buttons directly.
            action_buttons = [
                {"content_type": "text", "title": "🏨 Book a Hotel",  "payload": "ACTION_BOOK"},
                {"content_type": "text", "title": "📋 My Bookings",   "payload": "MY_BOOKINGS"},
                {"content_type": "text", "title": "🌐 Change Language", "payload": "ACTION_CHANGE_LANG"},
            ]
            dispatcher.utter_message(text=greeting, quick_replies=action_buttons)

        # ── 4. Persist language slot ─────────────────────────────────────────
        return [SlotSet("language", lang)]


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1 — ACTION: ActionChangeLanguage
# ═══════════════════════════════════════════════════════════════════════════

class ActionChangeLanguage(Action):
    """
    Handles ACTION_CHANGE_LANG postback and `change_language` intent.

    Sends the full language-selection quick-reply menu and clears the
    current language slot so the user can pick a new one.
    """

    def name(self) -> Text:
        return "action_change_language"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(
            text="Which language would you prefer? 🌐",
            quick_replies=_LANG_BUTTONS,
        )
        return [SlotSet("language", None)]


# ═══════════════════════════════════════════════════════════════════════════
# MODULE 1 — ACTION: ActionSessionStart
# ═══════════════════════════════════════════════════════════════════════════

class ActionSessionStart(Action):
    """
    Fires on every new conversation session (maps to action_session_start).

    Restores the `language` slot from the previous session if it was saved
    in Redis, so returning users are greeted in their chosen language.
    """

    def name(self) -> Text:
        return "action_session_start"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        sender = tracker.sender_id
        saved_lang: str | None = None

        try:
            from db.redis_client import sync_redis
            r = sync_redis()
            saved_lang = r.get(f"user:{sender}:lang")
        except Exception as exc:
            logger.debug("Could not load language from Redis: %s", exc)

        events: List[Dict[Text, Any]] = []
        if saved_lang:
            events.append(SlotSet("language", saved_lang))
            logger.info("Restored language=%s for sender=%s", saved_lang, sender)

        return events
