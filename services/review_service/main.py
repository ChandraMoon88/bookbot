"""
services/review_service/main.py
---------------------------------
FastAPI review microservice.
Accepts guest reviews, runs sentiment analysis, stores in Supabase,
and queues a task to notify the hotel.
"""

import os
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .sentiment import analyse

log = logging.getLogger(__name__)
app = FastAPI(title="Review Service")

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def _insert_review(payload: dict) -> dict:
    cols = list(payload.keys())
    phs  = ", ".join(f"%({c})s" for c in cols)
    sql  = f"INSERT INTO reviews ({', '.join(cols)}) VALUES ({phs}) RETURNING *"
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, payload)
            result = dict(cur.fetchone())
            conn.commit()
    return result


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

    review = _insert_review({
        "booking_id":  req.booking_id,
        "guest_id":    req.guest_id,
        "hotel_id":    req.hotel_id,
        "rating":      req.rating,
        "text":        req.text,
        "language":    req.language,
        "sentiment":   sentiment["sentiment"],
        "created_at":  datetime.now(timezone.utc),
    })

    return {"review_id": review.get("id"), "sentiment": sentiment}


@app.get("/reviews/{hotel_id}")
def list_reviews(hotel_id: str, limit: int = 20, offset: int = 0):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM reviews WHERE hotel_id=%s "
                "ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (hotel_id, limit, offset),
            )
            return [dict(r) for r in cur.fetchall()]


@app.get("/health")
def health():
    return {"status": "ok", "service": "review_service"}
