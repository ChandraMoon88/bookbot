"""
services/voice_service/tts.py
--------------------------------
Text-to-speech using SpeechT5 (English) / MMS-TTS (multilingual) locally,
with a HuggingFace Space fallback.
Redis-caches audio by md5(text+lang) for 24 h.
"""

import os
import io
import hashlib
import logging
from typing import Optional

import redis

log = logging.getLogger(__name__)

REDIS_URL  = os.environ.get("REDIS_URL",   "redis://localhost:6379")
HF_TTS_URL = os.environ.get("HF_TTS_URL",  "")

_r = redis.from_url(REDIS_URL, decode_responses=False)

MMS_LANG_MIN = {
    "en": "eng", "hi": "hin", "te": "tel", "ta": "tam",
    "fr": "fra", "es": "spa", "ar": "ara", "de": "deu",
    "zh": "zho", "ja": "jpn", "ko": "kor", "pt": "por",
}


def synthesize(text: str, language: str = "en") -> bytes:
    """
    Returns MP3 audio bytes.
    Checks Redis cache first; generates and stores if not cached.
    """
    cache_key = "tts:" + hashlib.md5(f"{language}:{text}".encode()).hexdigest()
    cached    = _r.get(cache_key)
    if cached:
        return cached

    audio = _generate(text, language)
    if audio:
        _r.setex(cache_key, 86400, audio)
    return audio or b""


def _generate(text: str, language: str) -> Optional[bytes]:
    """Tries local model first, then HF Space."""
    # Local MMS-TTS (multilingual) -----------------------------------------
    lang_code = MMS_LANG_MIN.get(language[:2], "eng")
    if language[:2] == "en":
        audio = _speecht5(text)
        if audio:
            return audio

    audio = _mms_tts(text, lang_code)
    if audio:
        return audio

    # HF Space fallback -------------------------------------------------------
    if HF_TTS_URL:
        try:
            import urllib.request
            import json
            payload = json.dumps({"text": text, "language": language}).encode()
            req = urllib.request.Request(
                f"{HF_TTS_URL}/tts",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as exc:
            log.error("HF TTS fallback failed: %s", exc)

    return None


def _speecht5(text: str) -> Optional[bytes]:
    try:
        import torch
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset

        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        model     = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        vocoder   = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        ds        = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        xvec      = torch.tensor(ds[7306]["xvector"]).unsqueeze(0)

        inputs    = processor(text=text, return_tensors="pt")
        speech    = model.generate_speech(inputs["input_ids"], xvec, vocoder=vocoder)

        import scipy.io.wavfile as wav_io
        buf = io.BytesIO()
        wav_io.write(buf, 16000, speech.numpy())
        return buf.getvalue()
    except Exception as exc:
        log.debug("SpeechT5 unavailable: %s", exc)
        return None


def _mms_tts(text: str, lang_code: str) -> Optional[bytes]:
    try:
        from transformers import VitsModel, AutoTokenizer
        import torch
        model_id  = f"facebook/mms-tts-{lang_code}"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = VitsModel.from_pretrained(model_id)
        inputs    = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform.squeeze().numpy()
        import scipy.io.wavfile as wav_io
        buf = io.BytesIO()
        wav_io.write(buf, model.config.sampling_rate, output)
        return buf.getvalue()
    except Exception as exc:
        log.debug("MMS-TTS unavailable for %s: %s", lang_code, exc)
        return None
