"""
services/language_service/detector.py
--------------------------------------
Language detection for incoming text.

Tiers:
  Tier 1 (full support): en ar fr es de ja zh ko pt ru hi id tr th vi
  Tier 2 (partial):      bn ur pa te ta ml kn gu mr si ne sw
  Tier 3 (basic/fallback): all others → respond in English
"""

from __future__ import annotations

import logging
import os

from langdetect import detect_langs, DetectorFactory

logger = logging.getLogger(__name__)
DetectorFactory.seed = 0


# ─── TIER CLASSIFICATION ──────────────────────────────────────────────────────
TIER_1 = {"en", "ar", "fr", "es", "de", "ja", "zh", "ko", "pt", "ru", "hi", "id", "tr", "th", "vi"}
TIER_2 = {"bn", "ur", "pa", "te", "ta", "ml", "kn", "gu", "mr", "si", "ne", "sw", "ms", "fil"}
RTL    = {"ar", "he", "fa", "ur", "yi"}

# Unicode script ranges for fast non-Latin detection
_SCRIPT_MAP: list[tuple[int, int, str | None]] = [
    (0x0900, 0x097F, "hi"),   # Devanagari
    (0x0980, 0x09FF, "bn"),   # Bengali
    (0x0A00, 0x0A7F, "pa"),   # Gurmukhi
    (0x0A80, 0x0AFF, "gu"),   # Gujarati
    (0x0B00, 0x0B7F, "or"),   # Odia
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0C80, 0x0CFF, "kn"),   # Kannada
    (0x0D00, 0x0D7F, "ml"),   # Malayalam
    (0x0D80, 0x0DFF, "si"),   # Sinhala
    (0x0E00, 0x0E7F, "th"),   # Thai
    (0x0E80, 0x0EFF, "lo"),   # Lao
    (0x0F00, 0x0FFF, "bo"),   # Tibetan
    (0x1000, 0x109F, "my"),   # Myanmar
    (0x10A0, 0x10FF, "ka"),   # Georgian
    (0x1200, 0x137F, "am"),   # Ethiopic
    (0x1780, 0x17FF, "km"),   # Khmer
    (0x0530, 0x058F, "hy"),   # Armenian
    (0x0590, 0x05FF, "he"),   # Hebrew
    (0x0600, 0x06FF, "ar"),   # Arabic
    (0x3040, 0x30FF, "ja"),   # Hiragana+Katakana
    (0x4E00, 0x9FFF, "zh"),   # CJK Unified
    (0xAC00, 0xD7AF, "ko"),   # Hangul
    (0x0400, 0x04FF, None),   # Cyrillic — delegate to langdetect
]


def _script_detect(text: str) -> str | None:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for start, end, lang in _SCRIPT_MAP:
            if start <= cp <= end:
                if lang:
                    counts[lang] = counts.get(lang, 0) + 1
                break
    return max(counts, key=counts.__getitem__) if counts else None


def detect(text: str) -> dict:
    """
    Detect language of a user-typed message.

    Returns:
      {
        language_code: str,        # ISO 639-1
        confidence:    float,
        is_rtl:        bool,
        tier:          int,        # 1 / 2 / 3
      }
    """
    if not text or not text.strip():
        return {"language_code": "en", "confidence": 1.0, "is_rtl": False, "tier": 1}

    # 1. Script-range detection (zero-latency)
    code = _script_detect(text)

    # 2. langdetect for Latin / Cyrillic disambiguation
    if not code:
        try:
            langs = detect_langs(text)
            min_prob = 0.95 if len(text.strip()) <= 15 else 0.80
            if langs and langs[0].prob >= min_prob:
                code = langs[0].lang
                confidence = float(langs[0].prob)
            else:
                code, confidence = "en", 0.5
        except Exception:
            code, confidence = "en", 0.5
    else:
        confidence = 1.0

    tier = 1 if code in TIER_1 else (2 if code in TIER_2 else 3)

    return {
        "language_code": code,
        "confidence":    confidence,
        "is_rtl":        code in RTL,
        "tier":          tier,
    }
