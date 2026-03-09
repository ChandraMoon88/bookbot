"""
hf_spaces/embeddings/app.py
------------------------------
HuggingFace Space: LaBSE sentence embedding service.
POST /embed  {"texts": ["..."]}  → {"embeddings": [[float, ...]]}
POST /embed_one {"text": "..."}  → {"embedding": [float, ...]}
"""

import os
import logging
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger(__name__)
app = FastAPI(title="Embeddings Space")

_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("sentence-transformers/LaBSE")
            log.info("LaBSE model loaded")
        except Exception as exc:
            log.error("Could not load LaBSE: %s", exc)
    return _model


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedOneRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup():
    import threading
    threading.Thread(target=_get_model, daemon=True).start()


@app.post("/embed")
def embed(req: EmbedRequest):
    model = _get_model()
    if model is None:
        return {"embeddings": [], "error": "model_loading"}
    embeddings = model.encode(req.texts, normalize_embeddings=True).tolist()
    return {"embeddings": embeddings}


@app.post("/embed_one")
def embed_one(req: EmbedOneRequest):
    model = _get_model()
    if model is None:
        return {"embedding": [], "error": "model_loading"}
    vec = model.encode([req.text], normalize_embeddings=True)[0].tolist()
    return {"embedding": vec}


@app.get("/health")
def health():
    return {"status": "ok", "model": "LaBSE", "loaded": _model is not None}


# Gradio interface (for HF Space UI)
import gradio as gr

def gradio_embed(texts_str: str) -> str:
    texts = [t.strip() for t in texts_str.split("\n") if t.strip()]
    model = _get_model()
    if not model:
        return "Model loading..."
    vecs = model.encode(texts, normalize_embeddings=True)
    return str(vecs.tolist())

demo = gr.Interface(
    fn=gradio_embed,
    inputs=gr.Textbox(lines=5, label="Enter texts (one per line)"),
    outputs=gr.Textbox(label="Embeddings (JSON)"),
    title="LaBSE Embeddings",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
