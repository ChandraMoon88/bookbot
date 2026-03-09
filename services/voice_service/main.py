"""
services/voice_service/main.py
---------------------------------
FastAPI voice microservice (STT + TTS).
"""

import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from .audio_utils import convert_to_wav, denoise
from .stt import transcribe
from .tts import synthesize

app = FastAPI(title="Voice Service")


class TTSRequest(BaseModel):
    text:     str
    language: Optional[str] = "en"


@app.post("/stt")
async def speech_to_text(
    file:      UploadFile = File(...),
    language:  Optional[str] = None,
    denoise_audio: bool = True,
):
    raw = await file.read()
    fmt = (file.content_type or "audio/ogg").split("/")[-1].split(";")[0]

    wav = convert_to_wav(raw, src_format=fmt)
    if denoise_audio:
        wav = denoise(wav)

    result = transcribe(wav, language=language)
    return result


@app.post("/tts")
def text_to_speech(req: TTSRequest):
    audio = synthesize(req.text, req.language)
    return {"audio_b64": base64.b64encode(audio).decode(), "format": "wav"}


@app.get("/tts/stream")
def tts_stream(text: str, language: str = "en"):
    audio = synthesize(text, language)
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=speech.wav"},
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "voice_service"}
