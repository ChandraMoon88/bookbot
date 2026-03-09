"""
services/voice_service/stt.py
--------------------------------
Speech-to-text using faster-whisper (local) with a HuggingFace Space fallback.
"""

import io
import os
import logging

log = logging.getLogger(__name__)

HF_STT_URL  = os.environ.get("HF_STT_URL", "")
WHISPER_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")

_whisper_model = None


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
            log.info("Whisper %s loaded", WHISPER_SIZE)
        except Exception as exc:
            log.error("Could not load Whisper: %s", exc)
    return _whisper_model


def transcribe(audio_bytes: bytes, language: str = None) -> dict:
    """
    Transcribes audio bytes (WAV 16 kHz mono) and returns
    {text, language, confidence}.
    Falls back to HF Space if local model unavailable.
    """
    # Local model path
    model = _get_model()
    if model:
        try:
            import numpy as np
            import scipy.io.wavfile as wav_io
            sr, data = wav_io.read(io.BytesIO(audio_bytes))
            audio_np  = data.astype(np.float32) / 32768.0
            segs, info = model.transcribe(
                audio_np,
                beam_size=5,
                language=language,
                vad_filter=True,
            )
            text = " ".join(s.text for s in segs).strip()
            return {"text": text, "language": info.language, "confidence": info.language_probability}
        except Exception as exc:
            log.warning("Local Whisper failed: %s", exc)

    # HF Space fallback
    if HF_STT_URL:
        try:
            import urllib.request
            import json
            import base64
            payload = json.dumps({"audio_b64": base64.b64encode(audio_bytes).decode(), "language": language}).encode()
            req = urllib.request.Request(
                f"{HF_STT_URL}/transcribe",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            log.error("HF STT fallback failed: %s", exc)

    return {"text": "", "language": language or "en", "confidence": 0.0}
