"""
services/rag_service/qdrant_client.py
----------------------------------------
Qdrant vector database client for hotel FAQ retrieval.
Collection: 'hotel_faqs'
"""

from __future__ import annotations

import logging
import os
import uuid as _uuid

logger        = logging.getLogger(__name__)
QDRANT_URL    = os.getenv("QDRANT_URL", "")
QDRANT_KEY    = os.getenv("QDRANT_API_KEY", "")
COLLECTION    = "hotel_faqs"
VECTOR_DIM    = 768


def _client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)


def retrieve(
    query_vector: list[float],
    hotel_id: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve top_k FAQ chunks for a given hotel that are closest to query_vector.
    ALWAYS filters by hotel_id to prevent cross-hotel leakage.
    """
    if not QDRANT_URL:
        return []

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        qc = _client()
        results = qc.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            query_filter=Filter(
                must=[FieldCondition(key="hotel_id", match=MatchValue(value=hotel_id))]
            ),
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "score":   r.score,
                "payload": r.payload,  # {question, answer, category, hotel_id}
            }
            for r in results
        ]
    except Exception as e:
        logger.error("Qdrant retrieve error: %s", e)
        return []


def upsert_faqs(hotel_id: str, faqs: list[dict], vectors: list[list[float]]) -> bool:
    """
    Batch-index hotel FAQ entries into Qdrant.
    faqs: [{question, answer, category}]
    vectors: parallel list of 768-dim embeddings
    """
    if not QDRANT_URL or not faqs:
        return False
    try:
        from qdrant_client.models import Distance, VectorParams, PointStruct
        qc = _client()

        # Ensure collection exists
        existing = [c.name for c in qc.get_collections().collections]
        if COLLECTION not in existing:
            qc.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

        points = [
            PointStruct(
                id=str(_uuid.uuid4()),
                vector=vec,
                payload={**faq, "hotel_id": hotel_id},
            )
            for faq, vec in zip(faqs, vectors)
        ]
        qc.upsert(collection_name=COLLECTION, points=points)
        return True
    except Exception as e:
        logger.error("Qdrant upsert error: %s", e)
        return False
