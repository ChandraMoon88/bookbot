"""
services/rag_service/embedder.py
----------------------------------
Text embedding using LaBSE.
Calls the HuggingFace Space 'hotel-booking-embeddings' for 768-dim vectors.
"""

from __future__ import annotations

import logging
import os

import httpx

logger   = logging.getLogger(__name__)
HF_EMBED_URL = os.getenv("HF_EMBED_URL", "")
HF_TOKEN     = os.getenv("HF_TOKEN", "")


async def embed(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of texts → LaBSE embedding vectors (768-dim).
    Works across all 109 LaBSE languages — no language-specific handling.
    """
    if not texts:
        return []

    if not HF_EMBED_URL:
        logger.warning("HF_EMBED_URL not set — returning zero vectors.")
        return [[0.0] * 768 for _ in texts]

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{HF_EMBED_URL}/embed",
            json={"texts": texts, "batch": True},
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
        )
        resp.raise_for_status()
        return resp.json()["vectors"]


async def embed_one(text: str) -> list[float]:
    vectors = await embed([text])
    return vectors[0] if vectors else [0.0] * 768
