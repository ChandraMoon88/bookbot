"""
hf_space/models/llm_client.py
-------------------------------
Groq API client using free-tier Llama 3 70B.

CRITICAL: Use Groq API (NOT a local model) to avoid GPU memory issues on HF Space.
  model: "llama3-70b-8192"
  GROQ_API_KEY from environment variable.

Used for:
  - RAG answer generation when Qdrant score is 0.6–0.85
  - Fallback intent clarification
  - Concierge-style open-ended responses
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level client singleton
_client = None

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")


async def init_groq() -> None:
    """Initialise Groq client and verify API key at startup."""
    global _client
    from groq import Groq

    api_key = os.environ["GROQ_API_KEY"]
    _client = Groq(api_key=api_key)

    # Simple connectivity test
    test_resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
    )
    logger.info("✅ Groq API connected (model=%s)", GROQ_MODEL)


def get_groq_client():
    """Return the initialised Groq client singleton."""
    if _client is None:
        raise RuntimeError("Groq client not initialised — call init_groq() during startup")
    return _client


async def chat(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """
    Single-turn chat completion via Groq Llama 3.

    Returns the assistant's reply as a plain string.
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ── System prompts ─────────────────────────────────────────────────────────────

CONCIERGE_SYSTEM_PROMPT = """You are an expert hotel concierge AI assistant.
Answer questions about the hotel clearly and concisely in the user's language.
Use the provided FAQ context to answer accurately.
If unsure, say so politely — never make up information.
Keep answers under 200 words. Be warm and professional."""

FALLBACK_SYSTEM_PROMPT = """You are a friendly hotel booking assistant.
The user's intent is unclear. Ask one clarifying question to understand what they need.
Offer 2-3 specific options as suggestions. Be concise — under 100 words."""
