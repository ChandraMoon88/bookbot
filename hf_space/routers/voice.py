"""
hf_space/routers/voice.py
--------------------------
Module — Voice Interaction

Handles:
  1. Incoming voice notes from Messenger (audio_url) → STT via Faster-Whisper
     → route result as text through normal flow
  2. Voice recording webview (voice_webview.html) → browser MediaRecorder →
     POST /api/voice/transcribe → return JSON transcript
  3. TTS: generate spoken reply with edge-tts → return audio URL

STT:  models/whisper_model.py (Faster-Whisper base, CPU int8)
TTS:  edge-tts (Microsoft Azure free tier, no API key needed)
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import uuid

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from hf_space.models.whisper_model import transcribe_bytes
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()

_TTS_VOICE_MAP = {
    "en": "en-US-JennyNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
    "ar": "ar-EG-SalmaNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "hi": "hi-IN-SwaraNeural",
    "th": "th-TH-PremwadeeNeural",
    "id": "id-ID-GadisNeural",
    "tr": "tr-TR-EmelNeural",
}
_DEFAULT_VOICE = "en-US-JennyNeural"


# ── STT via audio_url (Messenger voice notes) ─────────────────────────────────

async def handle_voice_message(psid: str, audio_url: str, lang: str) -> tuple[str | None, str | None]:
    """
    Download audio from Messenger CDN and transcribe.
    Returns (transcript, detected_language) or (None, None) on failure.
    """
    try:
        import httpx
        headers = {"Authorization": f"Bearer {os.environ.get('FACEBOOK_PAGE_ACCESS_TOKEN','')}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(audio_url, headers=headers)
            if resp.status_code != 200:
                log.error("audio_download_failed", status=resp.status_code)
                return None, None
            audio_bytes = resp.content
    except Exception as e:
        log.error("audio_download_error", error=str(e))
        return None, None

    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: transcribe_bytes(audio_bytes, language=lang if lang != "auto" else None)
    )

    if not result.get("success"):
        return None, None

    return result.get("transcript"), result.get("language", lang)


# ── TTS via edge-tts ──────────────────────────────────────────────────────────

async def generate_tts(text: str, lang: str = "en") -> str | None:
    """
    Generate TTS MP3 using edge-tts. Returns local file path or None.
    Caller is responsible for cleanup.
    """
    voice = _TTS_VOICE_MAP.get(lang[:2], _DEFAULT_VOICE)
    out_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.mp3")

    try:
        import edge_tts
        communicate = edge_tts.Communicate(text[:3000], voice)
        await communicate.save(out_path)
        return out_path
    except Exception as e:
        log.error("tts_generation_failed", error=str(e), lang=lang)
        return None


# ── API endpoints ──────────────────────────────────────────────────────────────

class TranscribeRequest(BaseModel):
    audio_base64: str    # Base64-encoded audio (WebM/OGG from browser)
    language: str = "auto"


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"
    psid: str = ""


@router.post("/transcribe")
async def transcribe_audio(req: TranscribeRequest) -> dict:
    """
    Called by voice_webview.html.
    Accepts base64 audio, returns JSON with transcript.
    """
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 audio")

    lang = req.language if req.language != "auto" else None

    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: transcribe_bytes(audio_bytes, language=lang)
    )

    if not result.get("success"):
        return {"success": False, "error": "Could not transcribe audio. Please make sure you spoke clearly."}

    return {
        "success":    True,
        "transcript": result.get("transcript", ""),
        "language":   result.get("language", "en"),
        "confidence": result.get("confidence", 0),
    }


@router.post("/tts")
async def text_to_speech(req: TTSRequest) -> dict:
    """Generate TTS and return base64-encoded MP3."""
    path = await generate_tts(req.text, req.lang)
    if not path:
        return {"success": False, "error": "TTS unavailable"}

    try:
        with open(path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        os.unlink(path)
        return {"success": True, "audio_base64": audio_b64, "format": "mp3"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/languages")
async def supported_languages() -> dict:
    return {"languages": list(_TTS_VOICE_MAP.keys())}
