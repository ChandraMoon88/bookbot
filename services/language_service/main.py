"""
services/language_service/main.py
-----------------------------------
FastAPI language detection microservice.

POST /detect  → { language_code, greeting, is_rtl, tier }
Stores detected language in Redis: session:{session_id}:lang
"""

import os
from fastapi import FastAPI
from pydantic import BaseModel

from .detector import detect
from .translator import from_english

app = FastAPI(title="Language Service")

REDIS_URL = os.getenv("REDIS_URL", "")
_redis = None


def _get_redis():
    global _redis
    if _redis is None and REDIS_URL:
        try:
            import redis as rlib
            _redis = rlib.from_url(REDIS_URL, decode_responses=True)
        except Exception:
            pass
    return _redis


_GREETINGS = {
    "en": "Hello! Welcome to BookBot. How can I help you today?",
    "ar": "مرحباً! أهلاً بك في بوك بوت. كيف يمكنني مساعدتك اليوم؟",
    "fr": "Bonjour! Bienvenue chez BookBot. Comment puis-je vous aider?",
    "es": "¡Hola! Bienvenido a BookBot. ¿Cómo puedo ayudarte hoy?",
    "de": "Hallo! Willkommen bei BookBot. Wie kann ich Ihnen heute helfen?",
    "hi": "नमस्कार! BookBot में आपका स्वागत है। आज मैं आपकी कैसे मदद कर सकता हूँ?",
    "zh": "您好！欢迎来到BookBot。今天我能帮您什么？",
    "ja": "こんにちは！BookBotへようこそ。本日はどのようなご用件でしょうか？",
    "ko": "안녕하세요! BookBot에 오신 것을 환영합니다. 오늘 어떻게 도와드릴까요?",
    "pt": "Olá! Bem-vindo ao BookBot. Como posso ajudá-lo hoje?",
    "ru": "Здравствуйте! Добро пожаловать в BookBot. Чем могу помочь?",
    "te": "హలో! BookBot కి స్వాగతం. నేను మీకు ఎలా సహాయపడగలను?",
    "ta": "வணக்கம்! BookBot-க்கு வரவேற்கிறோம். இன்று நான் உங்களுக்கு எவ்வாறு உதவலாம்?",
}


class DetectRequest(BaseModel):
    text:       str
    session_id: str = ""


class DetectResponse(BaseModel):
    language_code: str
    confidence:    float
    is_rtl:        bool
    tier:          int
    greeting:      str


@app.post("/detect", response_model=DetectResponse)
def detect_language(req: DetectRequest):
    result = detect(req.text)
    code   = result["language_code"]

    # Persist in Redis
    if req.session_id:
        r = _get_redis()
        if r:
            try:
                r.setex(f"session:{req.session_id}:lang", 3600, code)
            except Exception:
                pass

    greeting = _GREETINGS.get(code)
    if not greeting:
        en_greeting = "Hello! Welcome to BookBot. How can I help you today?"
        try:
            greeting = from_english(en_greeting, code)
        except Exception:
            greeting = en_greeting

    return DetectResponse(
        language_code=code,
        confidence=result["confidence"],
        is_rtl=result["is_rtl"],
        tier=result["tier"],
        greeting=greeting,
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "language_service"}
