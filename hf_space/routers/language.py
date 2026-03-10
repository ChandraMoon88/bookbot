"""
hf_space/routers/language.py
------------------------------
Language detection and translation endpoints.

Imports from the already-completed services/language_service/ — no logic duplication.
"""

from __future__ import annotations

import logging

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from services.language_service.detector import detect as _detect
from services.language_service.translator import translate as _translate

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class DetectRequest(BaseModel):
    text: str
    psid: str | None = None


class DetectResponse(BaseModel):
    language:   str
    confidence: float
    tier:       str   # "1" | "2" | "3"
    rtl:        bool


class TranslateRequest(BaseModel):
    text:        str
    source_lang: str
    target_lang: str


class TranslateResponse(BaseModel):
    translated_text: str
    source_lang:     str
    target_lang:     str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/detect", response_model=DetectResponse)
async def detect_language(req: DetectRequest) -> DetectResponse:
    """
    Detect the language of incoming text.
    Delegates to services/language_service/detector.py (completed Module 1).
    """
    result = _detect(req.text)
    return DetectResponse(
        language=result["language"],
        confidence=result.get("confidence", 1.0),
        tier=str(result.get("tier", "1")),
        rtl=result.get("rtl", False),
    )


@router.post("/translate", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest) -> TranslateResponse:
    """
    Translate text between languages.
    Delegates to services/language_service/translator.py (Helsinki-NLP).
    """
    translated = await _translate(req.text, req.source_lang, req.target_lang)
    return TranslateResponse(
        translated_text=translated,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
    )
