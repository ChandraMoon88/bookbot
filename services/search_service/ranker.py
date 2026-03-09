"""
services/search_service/ranker.py
-----------------------------------
Re-ranks ES hotel hits using guest preference vectors (Qdrant/LaBSE).

Scoring formula:
  0.4 * ES_relevance + 0.3 * guest_rating + 0.2 * availability + 0.1 * personalization
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
HF_EMBED_URL   = os.getenv("HF_EMBED_URL", "")
HF_TOKEN       = os.getenv("HF_TOKEN", "")


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x ** 2 for x in a)) or 1e-9
    nb   = math.sqrt(sum(x ** 2 for x in b)) or 1e-9
    return dot / (na * nb)


def _get_user_pref_vector(guest_id: str) -> list[float] | None:
    """Fetch guest preference vector from Qdrant (if exists)."""
    if not QDRANT_URL or not guest_id:
        return None
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        results = qc.retrieve(
            collection_name="guest_preferences",
            ids=[guest_id],
            with_vectors=True,
        )
        if results:
            return results[0].vector
    except Exception as e:
        logger.warning("Qdrant preference fetch failed: %s", e)
    return None


def rank(
    es_hits: list[dict],
    guest_id: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Re-rank ES results into the best 5 hotels for the user.

    Returns list sorted by composite score (highest first).
    """
    if not es_hits:
        return []

    pref_vec = _get_user_pref_vector(guest_id or "") if guest_id else None
    max_es   = max((h.get("score", 0) for h in es_hits), default=1) or 1

    scored = []
    for h in es_hits:
        es_score     = (h.get("score", 0) / max_es) * 0.4
        rating_score = (h.get("average_rating", 0) / 5.0) * 0.3
        avail_score  = min(1.0, (h.get("available_rooms", 10) / 10.0)) * 0.2

        if pref_vec and h.get("embedding"):
            pers_score = max(0.0, _cosine(pref_vec, h["embedding"])) * 0.1
        else:
            pers_score = 0.05  # neutral when no pref vector

        composite = es_score + rating_score + avail_score + pers_score
        scored.append({**h, "_rank_score": composite})

    scored.sort(key=lambda x: x["_rank_score"], reverse=True)
    return scored[:top_k]
