"""
services/search_service/ranker.py
-----------------------------------
Re-ranks hotel search hits using sentence-transformers (all-MiniLM-L6-v2).
No external services required — runs on HF Space CPU.

Scoring formula:
  0.4 * semantic_similarity + 0.3 * star_rating/5 + 0.2 * search_score + 0.1 * has_price
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy load — model is already warm from processor.py background loader
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def rank_hotels(
    hits: list[dict],
    psid: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """
    Re-rank PostgreSQL FTS results by composite score.
    Falls back to star_rating sort if sentence-transformers unavailable.
    """
    if not hits:
        return []

    try:
        from sentence_transformers import util as st_util
        import torch

        model = _get_model()

        def _hotel_text(h: dict) -> str:
            amenities = h.get("amenities") or []
            return (
                f"{h.get('name', '')} {h.get('city', '')} {h.get('country', '')} "
                f"{h.get('description', '')[:200]} "
                f"{' '.join(str(a) for a in amenities[:8])}"
            ).strip()

        # Use city as implicit query context
        city = hits[0].get("city", "") if hits else ""
        query = f"hotel {city} comfortable great amenities"

        texts   = [_hotel_text(h) for h in hits]
        q_emb   = model.encode(query, convert_to_tensor=True)
        t_emb   = model.encode(texts,  convert_to_tensor=True)
        sem_scores = st_util.cos_sim(q_emb, t_emb)[0].tolist()

        scored = []
        for h, sem in zip(hits, sem_scores):
            stars     = float(h.get("star_rating", 3)) / 5.0
            search_sc = min(1.0, float(h.get("search_score", 0.5)))
            has_price = 1.0 if h.get("price_from_usd", 0) > 0 else 0.0
            composite = (0.4 * float(sem) + 0.3 * stars
                         + 0.2 * search_sc + 0.1 * has_price)
            scored.append((composite, h))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [h for _, h in scored[:top_k]]

    except Exception as exc:
        logger.warning("rank_hotels semantic re-rank failed (%s) — fallback", exc)
        # Fallback: sort by star_rating desc, price availability
        return sorted(
            hits,
            key=lambda h: (float(h.get("star_rating", 0)),
                           1.0 if h.get("price_from_usd", 0) > 0 else 0.0),
            reverse=True,
        )[:top_k]



def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x ** 2 for x in a)) or 1e-9
    nb   = math.sqrt(sum(x ** 2 for x in b)) or 1e-9
    return dot / (na * nb)


def _get_user_pref_vector(guest_id: str) -> list[float] | None:
    """
    Build a guest preference vector from their booking history in PostgreSQL.
    Averages the hotel embeddings of previous stays — no Qdrant needed.
    Returns None if no history exists or on any error.
    """
    if not guest_id:
        return None
    try:
        import os, psycopg2
        db_url = os.environ.get("DATABASE_URL", "")
        if not db_url:
            return None
        conn = psycopg2.connect(db_url, connect_timeout=3)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT h.name, h.city, h.stars
                    FROM bookings b
                    JOIN hotels h ON h.id = b.hotel_id
                    WHERE b.guest_id = %s
                      AND b.status = 'confirmed'
                    ORDER BY b.created_at DESC
                    LIMIT 10
                    """,
                    (guest_id,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return None

        # Build a simple preference text from booking history and encode it
        texts = [f"{name}, {city}, {stars} stars" for name, city, stars in rows]
        model = _get_model()
        vecs  = model.encode(texts, normalize_embeddings=True)
        avg   = vecs.mean(axis=0).tolist()
        return avg
    except Exception as e:
        logger.warning("pref_vector_failed: %s", e)
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
