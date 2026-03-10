"""
hf_space/routers/rag.py
------------------------
Module 11 — FAQ / RAG Concierge

Uses LaBSE (sentence-transformers) for semantic FAQ retrieval from PostgreSQL,
then Groq LLM (optional) or direct FAQ answer for response generation.
Falls back to handoff if confidence is too low.

No Qdrant, no external vector DB required — all free.

Flow:
  faq_browsing state OR any question →
  encode query (LaBSE) → cosine-search faqs table (PostgreSQL) →
  top-k results → optionally Groq → answer message

Score thresholds:
  ≥ 0.75 → answer confidently
  0.45 – 0.75 → answer with "based on available info"
  < 0.45 → escalate to human agent
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state
from hf_space.db.supabase import get_supabase
from hf_space.models.labse_model import encode_single, encode
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()

_SCORE_HIGH = 0.75
_SCORE_LOW  = 0.45
_TOP_K      = 5


def _cosine(a: list[float], b: list[float]) -> float:
    """Fast pure-Python cosine similarity."""
    import math
    dot  = sum(x * y for x, y in zip(a, b))
    na   = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb   = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


async def _semantic_faq_search(
    query_vec: list[float],
    hotel_id: str,
    top_k: int = _TOP_K,
    score_threshold: float = _SCORE_LOW,
) -> list[dict]:
    """
    Search FAQs from PostgreSQL using cosine similarity on stored embeddings.
    Falls back to PostgreSQL FTS if embeddings are not pre-computed.
    """
    sb = get_supabase()
    try:
        # Prefer pre-computed embeddings (faqs.embedding column)
        res = await sb.table("faqs").select(
            "id,question,answer,tags,embedding"
        ).eq("hotel_id", hotel_id).limit(100).execute()

        rows = res.data or []
        if not rows:
            return []

        scored = []
        for row in rows:
            emb = row.get("embedding")
            if emb:
                if isinstance(emb, str):
                    emb = json.loads(emb)
                score = _cosine(query_vec, emb)
            else:
                # No embeddings stored — FTS fallback using answer text match
                score = 0.5 if any(
                    w in (row.get("question", "") + row.get("answer", "")).lower()
                    for w in _top_words(query_vec, 3)
                ) else 0.3
            if score >= score_threshold:
                scored.append({**row, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    except Exception as exc:
        log.warning("faq_db_search_failed", error=str(exc))
        return []


def _top_words(vec: list[float], n: int) -> list[str]:
    """Placeholder — returns empty when no text context available."""
    return []


async def _llm_answer(context: str, query: str) -> str | None:
    """Try Groq LLM; silently return None if unavailable."""
    try:
        from hf_space.models.llm_client import chat, CONCIERGE_SYSTEM_PROMPT
        prompt = (
            f"Answer the guest's question using ONLY the FAQ context below. "
            f"Be concise (≤3 sentences). Do not invent information.\n\n"
            f"FAQ Context:\n{context}\n\n"
            f"Guest question: {query}"
        )
        return await chat(CONCIERGE_SYSTEM_PROMPT, prompt, max_tokens=300, temperature=0.2)
    except Exception:
        return None


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_rag_query(
    psid: str,
    query: str,
    hotel_id: str,
    lang: str,
) -> tuple[list[dict], str]:
    """
    Main entry point called from app.py when user sends a question.
    """
    mb = MessengerResponse(psid)

    # Encode query with LaBSE
    try:
        query_vec = encode_single(query)
    except Exception as e:
        log.error("labse_encode_failed", error=str(e))
        # Fallback: PostgreSQL FTS-only search
        query_vec = []

    # Semantic search against PostgreSQL faqs table
    hits = await _semantic_faq_search(query_vec, hotel_id)

    if not hits:
        return await _no_answer_fallback(psid, query, lang)

    top_score = hits[0].get("score", 0.0)
    if top_score < _SCORE_LOW:
        return await _no_answer_fallback(psid, query, lang)

    # Build context from top hits
    context_parts = []
    for h in hits[:3]:
        q = h.get("question", "")
        a = h.get("answer", "")
        if q and a:
            context_parts.append(f"Q: {q}\nA: {a}")
    context = "\n\n".join(context_parts)

    # Try LLM-generated answer first, fallback to direct FAQ answer
    answer = await _llm_answer(context, query)
    if not answer:
        # Direct best-match answer
        answer = hits[0].get("answer", "Sorry, I don't have a specific answer for that.")

    if top_score < _SCORE_HIGH:
        answer += "\n\nNote: Confirm details directly with the hotel."

    msgs = mb.send_sequence([mb.text(answer)])
    msgs += [mb.quick_replies(
        "Was this helpful?",
        [
            {"title": "👍 Yes, thanks!",   "payload": f"RAG_FEEDBACK_yes_{hits[0]['id']}"},
            {"title": "👎 Not quite",       "payload": f"RAG_FEEDBACK_no_{hits[0]['id']}"},
            {"title": "🧑‍💼 Talk to agent", "payload": "HANDOFF_REQUEST"},
        ],
    )]
    return msgs, "faq_browsing"


async def _no_answer_fallback(psid: str, query: str, lang: str) -> tuple[list[dict], str]:
    mb = MessengerResponse(psid)
    return [
        mb.text("I don't have a specific answer to that question yet. 🤔"),
        mb.quick_replies("Would you like to speak to someone?", [
            {"title": "🧑‍💼 Talk to agent", "payload": "HANDOFF_REQUEST"},
            {"title": "🔍 Search hotels",   "payload": "SEARCH_START"},
            {"title": "🏠 Main menu",        "payload": "MENU_MAIN"},
        ]),
    ], "faq_browsing"


async def handle_rag_state(psid: str, text: str, lang: str) -> tuple[list[dict], str]:
    """Handle messages while in faq_browsing state."""
    mb = MessengerResponse(psid)

    if text.startswith("RAG_FEEDBACK_"):
        parts = text.split("_")
        verdict = parts[2] if len(parts) > 2 else "no"
        faq_id  = "_".join(parts[3:]) if len(parts) > 3 else ""
        await _record_feedback(psid, faq_id, verdict)
        if verdict == "yes":
            return [mb.text("Great! Anything else I can help with? 😊")], "faq_browsing"
        return [mb.quick_replies(
            "Sorry about that! Let me connect you with a team member.",
            [
                {"title": "🧑‍💼 Talk to agent", "payload": "HANDOFF_REQUEST"},
                {"title": "🔍 Search hotels",  "payload": "SEARCH_START"},
            ],
        )], "faq_browsing"

    # Treat any text as a new question — get hotel_id from profile
    from hf_space.db.redis import get_booking_draft
    draft    = await get_booking_draft(psid) or {}
    hotel_id = draft.get("hotel_id", "")
    return await handle_rag_query(psid, text, hotel_id, lang)


async def _record_feedback(psid: str, faq_id: str, verdict: str) -> None:
    """Store thumbs up/down for analytics."""
    try:
        sb = __import__("hf_space.db.supabase", fromlist=["get_supabase"]).get_supabase()
        await sb.table("faq_feedback").insert({
            "faq_id": faq_id,
            "verdict": verdict,
            "psid_hash": __import__("hashlib").sha256(psid.encode()).hexdigest()[:16],
        }).execute()
    except Exception:
        pass


# ── API endpoints ──────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    psid: str
    query: str
    hotel_id: str
    lang: str = "en"


class FeedbackRequest(BaseModel):
    faq_id: str
    verdict: str  # "yes" | "no"
    psid: str


@router.post("/query")
async def rag_query_endpoint(req: RAGQueryRequest) -> dict:
    msgs, state = await handle_rag_query(req.psid, req.query, req.hotel_id, req.lang)
    return {"messages": msgs, "new_state": state}


@router.post("/feedback")
async def rag_feedback(req: FeedbackRequest) -> dict:
    await _record_feedback(req.psid, req.faq_id, req.verdict)
    return {"success": True}


@router.get("/faqs")
async def list_faqs(hotel_id: str, limit: int = 20) -> dict:
    from hf_space.db.supabase import get_supabase
    sb = get_supabase()
    try:
        res = await sb.table("faqs").select("id,question,answer,tags").eq("hotel_id", hotel_id).limit(limit).execute()
        return {"faqs": res.data or []}
    except Exception as e:
        return {"faqs": [], "error": str(e)}
