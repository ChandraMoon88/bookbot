"""
hf_space/models/llm_client.py
-------------------------------
Groq API client (OPTIONAL) — Llama 3 70B free tier.

If GROQ_API_KEY is not set, all chat() calls return None and callers
fall back to extractive / rule-based answers.  The app works fully
without Groq — it is an optional enhancement, not a hard dependency.

Used for:
  - RAG answer generation / rephrasing
  - Fallback intent clarification
  - Concierge-style open-ended responses
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level client singleton — None means Groq is unavailable
_client = None
_groq_available = False

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")


async def init_groq() -> None:
    """Initialise Groq client if GROQ_API_KEY is set. No-op otherwise."""
    global _client, _groq_available

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.info("GROQ_API_KEY not set — LLM responses disabled (extractive fallback active)")
        return

    try:
        from groq import Groq
        _client = Groq(api_key=api_key)
        # Quick connectivity test
        _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        _groq_available = True
        logger.info("Groq API connected (model=%s)", GROQ_MODEL)
    except Exception as exc:
        logger.warning("Groq init failed (%s) — extractive fallback active", exc)
        _client = None
        _groq_available = False


def get_groq_client():
    """Return the Groq client, or None if unavailable."""
    return _client


def is_available() -> bool:
    """True if Groq is initialised and ready."""
    return _groq_available


async def chat(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Single-turn chat completion via Groq Llama 3.

    Returns the assistant's reply as a plain string, or None if Groq is
    unavailable.  Callers must handle None and produce an extractive fallback.
    """
    if not _groq_available or _client is None:
        return None
    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("Groq chat failed: %s", exc)
        return None


# ── System prompts ─────────────────────────────────────────────────────────────

CONCIERGE_SYSTEM_PROMPT = """You are an expert hotel concierge AI assistant.
Answer questions about the hotel clearly and concisely in the user's language.
Use the provided FAQ context to answer accurately.
If unsure, say so politely — never make up information.
Keep answers under 200 words. Be warm and professional."""

FALLBACK_SYSTEM_PROMPT = """You are a friendly hotel booking assistant.
The user's intent is unclear. Ask one clarifying question to understand what they need.
Offer 2-3 specific options as suggestions. Be concise — under 100 words."""
