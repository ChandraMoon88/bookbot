"""
services/review_service/main.py
---------------------------------
FastAPI review microservice.
Accepts guest reviews, runs sentiment analysis, stores in Supabase,
and queues a task to notify the hotel.
"""

import os
import json
import logging
from datetime import datetime, timezone
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .sentiment import analyse

log = logging.getLogger(__name__)
app = FastAPI(title="Review Service")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        "return=representation",
    }


def _insert(table: str, payload: dict) -> dict:
    url  = f"{SUPABASE_URL}/rest/v1/{table}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    return result[0] if isinstance(result, list) else result


class ReviewRequest(BaseModel):
    booking_id:  str
    guest_id:    str
    hotel_id:    str
    rating:      int   # 1-5
    text:        str
    language:    Optional[str] = "en"


@app.post("/review")
def submit_review(req: ReviewRequest):
    if not 1 <= req.rating <= 5:
        raise HTTPException(status_code=422, detail="Rating must be 1-5")

    sentiment = analyse(req.text)

    review = _insert("reviews", {
        "booking_id":  req.booking_id,
        "guest_id":    req.guest_id,
        "hotel_id":    req.hotel_id,
        "rating":      req.rating,
        "text":        req.text,
        "language":    req.language,
        "sentiment":   sentiment["sentiment"],
        "created_at":  datetime.now(timezone.utc).isoformat(),
    })

    return {"review_id": review.get("id"), "sentiment": sentiment}


@app.get("/reviews/{hotel_id}")
def list_reviews(hotel_id: str, limit: int = 20, offset: int = 0):
    url = (
        f"{SUPABASE_URL}/rest/v1/reviews"
        f"?hotel_id=eq.{hotel_id}&order=created_at.desc"
        f"&limit={limit}&offset={offset}"
    )
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


@app.get("/health")
def health():
    return {"status": "ok", "service": "review_service"}
