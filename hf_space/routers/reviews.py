"""
hf_space/routers/reviews.py
-----------------------------
Module 12 — Post-Stay Review Collection

Triggered automatically 2 hours after checkout via Celery scheduled task.
Also accessible via LEAVE_REVIEW postback.

Flow:
  [auto-trigger OR postback] → overall rating (quick replies 1–5 stars) →
  category ratings (cleanliness/staff/value/location) → text review →
  sentiment analysis → publish to Supabase → thank you + loyalty bonus

Sentiment: services/review_service/sentiment.py
Celery task: services/review_service/celery_tasks.py (sends the trigger message)

API:
  POST /api/reviews/submit  — submit full review
  GET  /api/reviews/{hotel_id} — recent reviews (for hotel admin)
  POST /api/reviews/sentiment — analyse sentiment of text
"""

from __future__ import annotations

import hashlib
import json

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()

_STAR_MAP = {"1": "⭐", "2": "⭐⭐", "3": "⭐⭐⭐", "4": "⭐⭐⭐⭐", "5": "⭐⭐⭐⭐⭐"}
_CATEGORIES = ["cleanliness", "staff", "value", "location"]


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_review_trigger(psid: str, booking_id: str, lang: str) -> tuple[list[dict], str]:
    """
    Auto-triggered 2h after checkout. Start review flow.
    """
    mb = MessengerResponse(psid)
    r  = get_redis()

    # Store booking_id for this review session
    await r.set(f"user:{psid}:review_booking_id", booking_id, ex=86400)
    await r.set(f"user:{psid}:review_step", "overall", ex=86400)

    await set_user_state(psid, "post_stay_review")

    hotel_name = await _get_hotel_name_from_booking(booking_id)

    return mb.send_sequence([
        mb.text(
            f"👋 Welcome back! Hope your stay at **{hotel_name}** was wonderful.\n\n"
            f"Your feedback helps other travellers and earns you **50 bonus points** 🎁"
        ),
        mb.quick_replies(
            f"How would you rate your overall stay at {hotel_name}?",
            [{"title": f"{_STAR_MAP[str(i)]} {i} star{'s' if i>1 else ''}",
              "payload": f"REVIEW_OVERALL_{i}"} for i in range(1, 6)],
        ),
    ]), "post_stay_review"


async def handle_review_input(psid: str, text: str, lang: str) -> tuple[list[dict], str]:
    """Dispatch within post_stay_review state."""
    mb = MessengerResponse(psid)
    r  = get_redis()

    if text.startswith("REVIEW_OVERALL_"):
        rating = int(text.replace("REVIEW_OVERALL_", ""))
        await r.set(f"user:{psid}:review_overall", rating, ex=86400)
        await r.set(f"user:{psid}:review_step", "category_0", ex=86400)
        return await _ask_category(psid, 0, lang)

    if text.startswith("REVIEW_CAT_"):
        parts = text.split("_")
        # Format: REVIEW_CAT_{category_index}_{rating}
        cat_idx = int(parts[2])
        rating  = int(parts[3])
        cat     = _CATEGORIES[cat_idx]
        await r.set(f"user:{psid}:review_cat_{cat}", rating, ex=86400)
        next_idx = cat_idx + 1
        if next_idx < len(_CATEGORIES):
            return await _ask_category(psid, next_idx, lang)
        # All categories done — ask for text
        await r.set(f"user:{psid}:review_step", "text", ex=86400)
        return [mb.quick_replies(
            "Would you like to add a written review? (Optional but appreciated!)",
            [
                {"title": "Yes, I'll write one ✏️", "payload": "REVIEW_WRITE_YES"},
                {"title": "Skip →",                  "payload": "REVIEW_SKIP_TEXT"},
            ],
        )], "post_stay_review"

    if text == "REVIEW_WRITE_YES":
        return [mb.text("Please share your thoughts about your stay (up to 500 characters):")], "post_stay_review"

    if text == "REVIEW_SKIP_TEXT":
        return await _save_review(psid, "", lang)

    if text == "REVIEW_SKIP":
        await set_user_state(psid, "greeting")
        return [mb.text("No problem! You can always leave a review later. 😊")], "greeting"

    # If it's text in review_text step — save it
    step = str(await r.get(f"user:{psid}:review_step") or "")
    if step == "text":
        review_text = text[:500]
        return await _save_review(psid, review_text, lang)

    return [mb.quick_replies(
        "Rate your stay:",
        [{"title": f"{_STAR_MAP[str(i)]} {i}", "payload": f"REVIEW_OVERALL_{i}"} for i in range(1, 6)],
    )], "post_stay_review"


async def _ask_category(psid: str, cat_idx: int, lang: str) -> tuple[list[dict], str]:
    mb  = MessengerResponse(psid)
    cat = _CATEGORIES[cat_idx]
    labels = {"cleanliness": "🧹 Cleanliness", "staff": "👤 Staff & service",
              "value": "💰 Value for money", "location": "📍 Location"}
    label = labels.get(cat, cat.title())

    return [mb.quick_replies(
        f"Rate: {label}",
        [{"title": f"{_STAR_MAP[str(i)]}",  "payload": f"REVIEW_CAT_{cat_idx}_{i}"} for i in range(1, 6)],
    )], "post_stay_review"


async def _save_review(psid: str, review_text: str, lang: str) -> tuple[list[dict], str]:
    """Persist review to Supabase and award loyalty points."""
    mb = MessengerResponse(psid)
    r  = get_redis()
    sb = get_supabase()

    booking_id = str(await r.get(f"user:{psid}:review_booking_id") or "")
    overall    = int(await r.get(f"user:{psid}:review_overall") or 0)
    cats       = {cat: int(await r.get(f"user:{psid}:review_cat_{cat}") or 0) for cat in _CATEGORIES}
    psid_hash  = hashlib.sha256(psid.encode()).hexdigest()

    # Sentiment analysis
    sentiment_score = 0.0
    if review_text:
        try:
            from services.review_service.sentiment import analyse_sentiment
            sentiment_score = await analyse_sentiment(review_text)
        except Exception:
            pass

    # Get hotel_id
    hotel_id = await _get_hotel_id_from_booking(booking_id)

    review_row = {
        "booking_id":     booking_id,
        "hotel_id":       hotel_id,
        "psid_hash":      psid_hash,
        "overall":        overall,
        "categories":     cats,
        "review_text":    review_text,
        "sentiment":      sentiment_score,
        "lang":           lang,
    }

    try:
        await sb.table("reviews").insert(review_row).execute()
    except Exception as e:
        log.error("review_save_failed", error=str(e))

    # Cleanup Redis keys
    for key in ["review_booking_id", "review_overall", "review_step"] + [f"review_cat_{c}" for c in _CATEGORIES]:
        await r.delete(f"user:{psid}:{key}")

    # Award loyalty points
    try:
        from hf_space.routers.loyalty import award_booking_points
        import asyncio
        asyncio.create_task(award_booking_points(psid, booking_id, 5.0))  # ~50 pts for $5 equivalent
    except Exception:
        pass

    await set_user_state(psid, "greeting")

    return [
        mb.text(
            f"🙏 Thank you for your review!\n\n"
            f"Overall: {_STAR_MAP.get(str(overall), '?')}\n"
            f"You've earned **50 bonus loyalty points**! 🎁"
        ),
        mb.quick_replies("Anything else?", [
            {"title": "🔍 Search hotels",  "payload": "SEARCH_START"},
            {"title": "💎 My points",      "payload": "LOYALTY_MENU"},
            {"title": "🏠 Main menu",       "payload": "MENU_MAIN"},
        ]),
    ], "greeting"


async def _get_hotel_name_from_booking(booking_id: str) -> str:
    sb = get_supabase()
    try:
        res = await sb.table("bookings").select("hotel_data:hotels(name)").eq("id", booking_id).single().execute()
        return (res.data or {}).get("hotel_data", {}).get("name", "your hotel")
    except Exception:
        return "your hotel"


async def _get_hotel_id_from_booking(booking_id: str) -> str:
    sb = get_supabase()
    try:
        res = await sb.table("bookings").select("hotel_id").eq("id", booking_id).single().execute()
        return (res.data or {}).get("hotel_id", "")
    except Exception:
        return ""


# ── API endpoints ──────────────────────────────────────────────────────────────

class ReviewSubmitRequest(BaseModel):
    psid:        str
    booking_id:  str
    overall:     int
    categories:  dict
    review_text: str = ""


class SentimentRequest(BaseModel):
    text: str


@router.post("/submit")
async def submit_review_api(req: ReviewSubmitRequest) -> dict:
    msgs, state = await _save_review(req.psid, req.review_text, "en")
    return {"success": True, "new_state": state}


@router.get("/{hotel_id}")
async def list_hotel_reviews(hotel_id: str, limit: int = 10) -> dict:
    sb = get_supabase()
    try:
        res = await sb.table("reviews") \
            .select("overall,categories,review_text,sentiment,created_at") \
            .eq("hotel_id", hotel_id) \
            .order("created_at", desc=True) \
            .limit(limit).execute()
        return {"reviews": res.data or []}
    except Exception as e:
        return {"reviews": [], "error": str(e)}


@router.post("/sentiment")
async def analyse_text_sentiment(req: SentimentRequest) -> dict:
    try:
        from services.review_service.sentiment import analyse_sentiment
        score = await analyse_sentiment(req.text)
        return {"score": score, "label": "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"}
    except Exception as e:
        return {"score": 0, "error": str(e)}
