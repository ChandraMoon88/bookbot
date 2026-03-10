"""
hf_space/models/whisper_model.py
----------------------------------
Faster-Whisper STT singleton.

Loaded ONCE at HF Space startup. Stays in memory to avoid cold-start.
Used for:
  - Messenger voice note transcription (audio attachment)
  - Voice Webview real-time transcription via /api/voice/transcribe
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton
_model = None

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")  # tiny|base|small|medium


async def load_whisper() -> None:
    """Load Faster-Whisper at startup. Model cached in HF Space filesystem."""
    global _model
    from faster_whisper import WhisperModel

    logger.info("Loading Faster-Whisper (%s)…", WHISPER_MODEL_SIZE)
    # cpu with int8 quantisation — suitable for HF Space CPU tier
    _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    logger.info("✅ Faster-Whisper loaded (size=%s)", WHISPER_MODEL_SIZE)


def transcribe_bytes(audio_bytes: bytes, language: str | None = None) -> dict:
    """
    Transcribe raw audio bytes (WAV/MP3/OGG/WEBM).

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio data.
    language : str | None
        BCP-47 language code hint (e.g. "ar", "en"). None = auto-detect.

    Returns
    -------
    {"transcript": str, "language": str, "confidence": float, "success": bool}
    """
    if _model is None:
        raise RuntimeError("Whisper model not loaded — call load_whisper() during startup")

    audio_io = io.BytesIO(audio_bytes)

    kwargs: dict = {"beam_size": 5}
    if language:
        kwargs["language"] = language

    segments, info = _model.transcribe(audio_io, **kwargs)

    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    confidence = float(info.language_probability)

    if confidence < 0.6 or not transcript:
        return {
            "success": False,
            "transcript": "",
            "language": info.language,
            "confidence": confidence,
            "message": "Could not understand audio. Please try again.",
        }

    return {
        "success": True,
        "transcript": transcript,
        "language": info.language,
        "confidence": confidence,
        "message": None,
    }
