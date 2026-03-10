"""
hf_space/models/labse_model.py
--------------------------------
LaBSE (Language-agnostic BERT Sentence Embeddings) singleton.

Loaded ONCE at HF Space startup via the app.py lifespan context manager.
Stored as a module-level global to avoid cold-start timeouts on first message.

LaBSE is used for:
  - FAQ semantic search (embed questions before Qdrant lookup)
  - Hotel persona personalisation
  - Multi-lingual intent embedding
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton — set during lifespan startup
_model = None


async def load_labse() -> None:
    """
    Load LaBSE model at startup.
    Uses sentence-transformers — model loaded from HF Hub cache.
    """
    global _model
    from sentence_transformers import SentenceTransformer

    logger.info("Loading LaBSE model…")
    _model = SentenceTransformer("sentence-transformers/LaBSE")

    # Verify with test embed
    test_vec = _model.encode(["hello"])
    assert test_vec.shape == (1, 768), f"Unexpected LaBSE output shape: {test_vec.shape}"
    logger.info("✅ LaBSE model loaded (dim=768)")


def encode(texts: list[str], normalize: bool = True) -> np.ndarray:
    """
    Encode a list of texts into LaBSE embeddings.

    Returns: float32 ndarray of shape (len(texts), 768).
    Normalised by default for cosine-similarity Qdrant search.
    """
    if _model is None:
        raise RuntimeError("LaBSE model not loaded — call load_labse() during startup")
    vecs = _model.encode(texts, normalize_embeddings=normalize, convert_to_numpy=True)
    return vecs


def encode_single(text: str, normalize: bool = True) -> list[float]:
    """Encode a single string and return as a plain Python list (for Qdrant)."""
    return encode([text], normalize=normalize)[0].tolist()
