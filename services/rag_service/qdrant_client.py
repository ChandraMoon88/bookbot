"""
services/rag_service/qdrant_client.py
----------------------------------------
FAQ retrieval — PostgreSQL + LaBSE cosine similarity (no Qdrant required).

Public API is intentionally compatible with the old Qdrant version so
callers (hotel_onboarding.py, rag.py) don't need to change signatures.

PostgreSQL `faqs` table schema:
    id          UUID / BIGSERIAL PRIMARY KEY
    hotel_id    TEXT NOT NULL
    question    TEXT NOT NULL
    answer      TEXT NOT NULL
    category    TEXT DEFAULT ''
    embedding   JSONB        -- 768-dim LaBSE float list, computed at upsert
"""

from __future__ import annotations

import json
import logging
import math
import os

logger = logger = logging.getLogger(__name__)

_DATABASE_URL = None


def _get_conn():
    global _DATABASE_URL
    import psycopg2
    if _DATABASE_URL is None:
        _DATABASE_URL = os.environ.get("DATABASE_URL", "")
    if not _DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    return psycopg2.connect(_DATABASE_URL, connect_timeout=5)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb  = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def retrieve(
    query_vector: list[float],
    hotel_id: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve top_k FAQ chunks for a given hotel closest to query_vector.
    Uses in-memory cosine similarity over FAQs fetched from PostgreSQL.
    """
    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, question, answer, category, embedding "
                    "FROM faqs WHERE hotel_id = %s LIMIT 200",
                    (hotel_id,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return []

        scored = []
        for row_id, question, answer, category, embedding in rows:
            if embedding:
                emb = json.loads(embedding) if isinstance(embedding, str) else embedding
                score = _cosine(query_vector, emb)
            else:
                score = 0.0
            scored.append({
                "score": score,
                "payload": {
                    "question": question,
                    "answer": answer,
                    "category": category or "",
                    "hotel_id": hotel_id,
                },
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    except Exception as e:
        logger.error("faq_retrieve_error: %s", e)
        return []


def upsert_faqs(hotel_id: str, faqs: list[dict], vectors: list[list[float]]) -> bool:
    """
    Insert / update FAQ entries in PostgreSQL, storing their LaBSE embeddings.

    faqs    : [{question, answer, category}]
    vectors : parallel list of 768-dim LaBSE embeddings
    """
    if not faqs:
        return False
    try:
        conn = _get_conn()
        try:
            with conn:
                with conn.cursor() as cur:
                    for faq, vec in zip(faqs, vectors):
                        cur.execute(
                            """
                            INSERT INTO faqs (hotel_id, question, answer, category, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (hotel_id, question) DO UPDATE
                              SET answer    = EXCLUDED.answer,
                                  category  = EXCLUDED.category,
                                  embedding = EXCLUDED.embedding
                            """,
                            (
                                hotel_id,
                                faq.get("question", ""),
                                faq.get("answer", ""),
                                faq.get("category", ""),
                                json.dumps(vec),
                            ),
                        )
        finally:
            conn.close()
        return True
    except Exception as e:
        logger.error("faq_upsert_error: %s", e)
        return False

