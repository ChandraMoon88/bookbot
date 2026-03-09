"""
services/rag_service/main.py
------------------------------
FastAPI RAG pipeline microservice.

POST /faq  → { answer, source }
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .embedder import embed_one
from .qdrant_client import retrieve
from .llm_generator import generate

app = FastAPI(title="RAG Service")


class FAQRequest(BaseModel):
    question:   str
    hotel_id:   str
    language:   str = "en"
    top_k:      int = 3


@app.post("/faq")
async def answer_faq(req: FAQRequest):
    vector = await embed_one(req.question)
    chunks = retrieve(vector, req.hotel_id, req.top_k)
    result = await generate(req.question, chunks, req.language)
    return result


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag_service"}
