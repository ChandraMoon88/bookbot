"""
hf_space/routers/rag.py
------------------------
Module 11 — FAQ / RAG Concierge

Uses LaBSE + Qdrant for semantic FAQ retrieval, then Groq LLM to generate
a grounded answer. Falls back to handoff if confidence is too low.

Flow:
  faq_browsing state OR any question in other states →
  encode query (LaBSE) → search Qdrant (hotel_faqs collection) →
  top-k results → build context → Groq → answer message

Score thresholds:
  ≥ 0.80 → answer confidently
  0.55 – 0.80 → answer with "based on available info"
  < 0.55 → escalate to human agent

POST /api/rag/query     — main semantic search + generation
POST /api/rag/feedback  — thumbs up/down on answers
GET  /api/rag/faqs      — list FAQs for a hotel (for admin)
"""

from __future__ import annotations

import os
from typing import Any

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state
from hf_space.models.labse_model import encode_single
from hf_space.models.llm_client import chat, CONCIERGE_SYSTEM_PROMPT
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()

_SCORE_HIGH = 0.80
_SCORE_LOW  = 0.55
_TOP_K      = 5


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_rag_query(
    psid: str,
    query: str,
    hotel_id: str,
    lang: str,
) -> tuple[list[dict], str]:
    """
    Main entry point called from app.py when user sends a question.
    Can be called from any state — preserves current state on return.
    """
    mb = MessengerResponse(psid)

    # Encode query
    try:
        query_vec = encode_single(query)
    except Exception as e:
        log.error("labse_encode_failed", error=str(e))
        return [mb.text("Let me look that up… one moment! 🔍")], "faq_browsing"

    # Qdrant search
    try:
        from hf_space.qdrant_client import get_qdrant_client
        qdrant = get_qdrant_client()
        hits = await qdrant.search(
            collection_name="hotel_faqs",
            query_vector=query_vec,
            query_filter={"must": [{"key": "hotel_id", "match": {"value": hotel_id}}]},
            limit=_TOP_K,
            score_threshold=_SCORE_LOW,
        )
    except Exception as e:
        log.error("qdrant_search_failed", error=str(e))
        hits = []

    if not hits:
        return await _no_answer_fallback(psid, query, lang)

    top_score = hits[0].score if hits else 0.0

    if top_score < _SCORE_LOW:
        return await _no_answer_fallback(psid, query, lang)

    # Build context from top hits
    context_parts = []
    for h in hits[:3]:
        payload = h.payload or {}
        q = payload.get("question", "")
        a = payload.get("answer", "")
        if q and a:
            context_parts.append(f"Q: {q}\nA: {a}")

    context = "\n\n".join(context_parts)

    # Confidence hedge
    if top_score >= _SCORE_HIGH:
        confidence_note = ""
    else:
        confidence_note = "\nNote: This information may not be fully up-to-date. Confirm with the hotel."

    # LLM generation
    system = CONCIERGE_SYSTEM_PROMPT
    prompt = (
        f"Answer the guest's question using ONLY the FAQ context below. "
        f"Be concise (≤3 sentences). Do not invent information.\n\n"
        f"FAQ Context:\n{context}\n\n"
        f"Guest question: {query}"
    )

    try:
        answer = await chat(system, prompt, max_tokens=300, temperature=0.2)
        answer = answer.strip() + confidence_note
    except Exception as e:
        log.error("groq_generation_failed", error=str(e))
        answer = context_parts[0].split("\nA: ")[-1] if context_parts else "Sorry, I couldn't find an answer."

    msgs = mb.send_sequence([mb.text(answer)])
    msgs += [mb.quick_replies(
        "Was this helpful?",
        [
            {"title": "👍 Yes, thanks!",     "payload": f"RAG_FEEDBACK_yes_{hits[0].id}"},
            {"title": "👎 Not quite",         "payload": f"RAG_FEEDBACK_no_{hits[0].id}"},
            {"title": "🧑‍💼 Talk to agent",   "payload": "HANDOFF_REQUEST"},
        ],
    )]

    return msgs, "faq_browsing"


async def _no_answer_fallback(psid: str, query: str, lang: str) -> tuple[list[dict], str]:
    """When no relevant FAQ found — offer handoff."""
    mb = MessengerResponse(psid)
    return [
        mb.text("I don't have a specific answer to that question yet. 🤔"),
        mb.quick_replies("Would you like to speak to someone?", [
            {"title": "🧑‍💼 Talk to agent",  "payload": "HANDOFF_REQUEST"},
            {"title": "🔍 Search hotels",    "payload": "SEARCH_START"},
            {"title": "🏠 Main menu",         "payload": "MENU_MAIN"},
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
