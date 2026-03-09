"""
hf_spaces/stt/app.py
-----------------------
HuggingFace Space: Whisper Speech-to-Text service.
POST /transcribe  {"audio_b64": "...", "language": "en"}
               → {"text": "...", "language": "en", "confidence": 0.99}
"""

import os
import io
import base64
import logging
import threading

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

log = logging.getLogger(__name__)
app = FastAPI(title="STT Space")

WHISPER_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
_model       = None


def _load_model():
    global _model
    try:
        from faster_whisper import WhisperModel
        _model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
        log.info("Whisper %s loaded", WHISPER_SIZE)
    except Exception as exc:
        log.error("Whisper load failed: %s", exc)


@app.on_event("startup")
async def startup():
    threading.Thread(target=_load_model, daemon=True).start()


class TranscribeRequest(BaseModel):
    audio_b64: str
    language:  Optional[str] = None


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    if _model is None:
        return {"text": "", "language": "en", "confidence": 0.0, "status": "warming_up"}

    try:
        import numpy as np
        import scipy.io.wavfile as wav_io

        audio_bytes = base64.b64decode(req.audio_b64)
        sr, data    = wav_io.read(io.BytesIO(audio_bytes))
        audio_np    = data.astype(np.float32) / 32768.0

        segs, info = _model.transcribe(
            audio_np,
            beam_size=5,
            language=req.language,
            vad_filter=True,
        )
        text = " ".join(s.text for s in segs).strip()
        return {
            "text":       text,
            "language":   info.language,
            "confidence": round(info.language_probability, 3),
        }
    except Exception as exc:
        log.error("Transcription error: %s", exc)
        return {"text": "", "language": "en", "confidence": 0.0, "error": str(exc)}


@app.get("/health")
def health():
    return {"status": "ok", "model": f"whisper-{WHISPER_SIZE}", "loaded": _model is not None}


# Gradio UI
import gradio as gr

def gradio_transcribe(audio_path, language):
    if _model is None:
        return "Model loading, please wait..."
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    result = transcribe(TranscribeRequest(audio_b64=audio_b64, language=language or None))
    return result.get("text", "")

demo = gr.Interface(
    fn=gradio_transcribe,
    inputs=[gr.Audio(type="filepath", label="Upload Audio"), gr.Textbox(label="Language (optional)")],
    outputs=gr.Textbox(label="Transcription"),
    title=f"Whisper STT ({WHISPER_SIZE})",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
