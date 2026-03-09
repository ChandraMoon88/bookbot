#!/usr/bin/env python3
"""
scripts/setup_qdrant_collection.py
------------------------------------
Creates the Qdrant collection for hotel FAQ embeddings (LaBSE 768-dim).

Usage:
    QDRANT_URL=https://... QDRANT_API_KEY=xxx python scripts/setup_qdrant_collection.py
"""

import os

QDRANT_URL     = os.environ.get("QDRANT_URL",     "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION     = "hotel_faqs"
VECTOR_SIZE    = 768   # LaBSE


def setup():
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, HnswConfigDiff
    except ImportError:
        raise SystemExit("qdrant-client is required: pip install qdrant-client")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    # Delete if exists
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION in collections:
        client.delete_collection(COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )
    print(f"Collection '{COLLECTION}' created (size={VECTOR_SIZE}, metric=COSINE)")


if __name__ == "__main__":
    setup()
