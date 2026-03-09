"""
hf_spaces/llm/app.py
-----------------------
HuggingFace Space: LLM inference for RAG (Llama 3 via transformers).
Falls back to Groq API if local model OOMs.

POST /generate {"prompt": "...", "max_tokens": 256}
             → {"text": "...", "source": "local|groq"}
"""

import os
import logging
import threading
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger(__name__)
app = FastAPI(title="LLM Space")

MODEL_ID   = os.environ.get("LLM_MODEL_ID",    "meta-llama/Meta-Llama-3-8B-Instruct")
GROQ_KEY   = os.environ.get("GROQ_API_KEY",    "")
HF_TOKEN   = os.environ.get("HF_TOKEN",         "")
_pipeline  = None
_loading   = False


def _load_model():
    global _pipeline, _loading
    _loading = True
    try:
        import torch
        from transformers import pipeline as hf_pipeline, AutoTokenizer
        tok    = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        _pipeline = hf_pipeline(
            "text-generation",
            model=MODEL_ID,
            tokenizer=tok,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN,
        )
        log.info("LLM %s loaded", MODEL_ID)
    except Exception as exc:
        log.warning("Local LLM unavailable: %s (Groq fallback will be used)", exc)
    finally:
        _loading = False


@app.on_event("startup")
async def startup():
    threading.Thread(target=_load_model, daemon=True).start()


class GenerateRequest(BaseModel):
    prompt:     str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7


def _groq_generate(prompt: str, max_tokens: int, temperature: float) -> str:
    import json, urllib.request
    payload = json.dumps({
        "model":       "llama3-8b-8192",
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


@app.post("/generate")
def generate(req: GenerateRequest):
    if _loading:
        return {"text": "Model is warming up, please retry shortly.", "source": "loading"}

    if _pipeline is not None:
        try:
            outputs = _pipeline(
                req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=True,
                pad_token_id=_pipeline.tokenizer.eos_token_id,
            )
            text = outputs[0]["generated_text"][len(req.prompt):].strip()
            return {"text": text, "source": "local"}
        except Exception as exc:
            log.warning("Local generation failed: %s", exc)

    if GROQ_KEY:
        try:
            text = _groq_generate(req.prompt, req.max_tokens, req.temperature)
            return {"text": text, "source": "groq"}
        except Exception as exc:
            log.error("Groq fallback failed: %s", exc)

    return {"text": "I apologize, the AI service is temporarily unavailable.", "source": "fallback"}


@app.get("/health")
def health():
    return {"status": "ok", "loaded": _pipeline is not None, "loading": _loading,
            "groq_available": bool(GROQ_KEY)}


# Gradio UI
import gradio as gr

def gradio_generate(prompt_text, max_tokens, temperature):
    result = generate(GenerateRequest(prompt=prompt_text, max_tokens=int(max_tokens), temperature=float(temperature)))
    return result.get("text", "")

demo = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(lines=6, label="Prompt"),
        gr.Slider(64, 1024, value=256, step=64, label="Max Tokens"),
        gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Response"),
    title="Llama 3 RAG Generator",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
