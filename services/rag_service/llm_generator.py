"""
services/rag_service/llm_generator.py
---------------------------------------
RAG answer generator using Llama 3 on HuggingFace Spaces (or Groq fallback).
"""

from __future__ import annotations

import logging
import os

import httpx

logger        = logging.getLogger(__name__)
HF_LLM_URL    = os.getenv("HF_LLM_URL", "")
HF_TOKEN      = os.getenv("HF_TOKEN", "")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")


SYSTEM_PROMPT = (
    "You are a helpful hotel concierge. Using ONLY the hotel information below, "
    "answer the guest's question in a friendly, concise tone. "
    "If the information is not available, politely say so."
)


async def generate(
    question: str,
    context_chunks: list[dict],
    target_language: str = "en",
) -> dict:
    """
    Generate an answer from retrieved FAQ chunks.

    Returns: { answer, source: 'rag' | 'direct' | 'fallback' }
    """
    if not context_chunks:
        return {"answer": "I do not have specific information on that. Please contact the hotel directly.", "source": "fallback"}

    context = "\n".join([c["payload"].get("answer", "") for c in context_chunks])

    # Try primary HF Space
    if HF_LLM_URL:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    f"{HF_LLM_URL}/generate",
                    json={
                        "question":       question,
                        "context_chunks": context_chunks,
                        "language":       target_language,
                    },
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning("HF LLM failed: %s — trying Groq fallback", e)

    # Groq fallback
    if GROQ_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={
                        "model": "llama3-8b-8192",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user",   "content": f"Hotel Information:\n{context}\n\nGuest Question: {question}"},
                        ],
                        "max_tokens": 300,
                        "temperature": 0.3,
                    },
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                )
                resp.raise_for_status()
                answer = resp.json()["choices"][0]["message"]["content"]
                return {"answer": answer, "source": "rag"}
        except Exception as e:
            logger.error("Groq fallback failed: %s", e)

    # Direct fallback: return top chunk answer
    best = context_chunks[0]["payload"].get("answer", "Please contact the hotel for details.")
    return {"answer": best, "source": "direct"}
