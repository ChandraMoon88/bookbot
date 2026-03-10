"""
hf_space/routers/analytics.py
-------------------------------
Analytics event ingestion and summary endpoints.

Events are written to PostgreSQL `analytics_events` table (via
services/analytics_service/kafka_consumer.produce_event — best-effort,
never blocks the booking flow).

Events captured:
  - message_received
  - state_transition
  - search_performed
  - hotel_viewed
  - booking_created
  - payment_completed
  - review_submitted
  - handoff_created
  - addon_added

All PSIDs stored hashed (SHA-256, first 16 chars).

API:
  POST /api/analytics/event      — ingest single event
  POST /api/analytics/events     — ingest batch of events
  GET  /api/analytics/summary    — today's KPIs (admin)
  GET  /api/analytics/funnel     — booking funnel conversion
"""

from __future__ import annotations

import hashlib
import os
from datetime import date, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Event model ────────────────────────────────────────────────────────────────

class AnalyticsEvent(BaseModel):
    psid:       str
    event_type: str
    properties: dict = {}
    timestamp:  str  = ""


class BatchEventsRequest(BaseModel):
    events: list[AnalyticsEvent]


# ── Internal helper ────────────────────────────────────────────────────────────

async def track_event(psid: str, event_type: str, properties: dict | None = None) -> None:
    """
    Fire-and-forget analytics tracking. Called from other routers.
    Never raises — analytics failures must not break the main flow.
    """
    try:
        psid_hash = hashlib.sha256(psid.encode()).hexdigest()[:16]
        row: dict[str, Any] = {
            "psid_hash":  psid_hash,
            "event_type": event_type,
            "properties": properties or {},
            "ts":         datetime.utcnow().isoformat(),
        }

        # ClickHouse via Kafka (best-effort)
        try:
            from services.analytics_service.kafka_consumer import produce_event
            await produce_event(row)
        except Exception:
            pass

        # Supabase fallback for simple deployments
        from hf_space.db.supabase import get_supabase
        sb = get_supabase()
        await sb.table("analytics_events").insert(row).execute()

    except Exception as e:
        log.debug("analytics_track_failed", error=str(e), event=event_type)


# ── Middleware to verify admin API key ─────────────────────────────────────────

def _check_admin_key(x_admin_key: str = Header(default="")) -> None:
    expected = os.environ.get("ANALYTICS_ADMIN_KEY", "")
    if expected and x_admin_key != expected:
        raise HTTPException(status_code=403, detail="Invalid admin key")


# ── API endpoints ──────────────────────────────────────────────────────────────

@router.post("/event")
async def ingest_event(event: AnalyticsEvent) -> dict:
    await track_event(event.psid, event.event_type, event.properties)
    return {"success": True}


@router.post("/events")
async def ingest_batch(req: BatchEventsRequest) -> dict:
    import asyncio
    tasks = [track_event(e.psid, e.event_type, e.properties) for e in req.events]
    await asyncio.gather(*tasks, return_exceptions=True)
    return {"success": True, "count": len(req.events)}


@router.get("/summary", dependencies=[Depends(_check_admin_key)])
async def get_daily_summary(target_date: str = "") -> dict:
    """Today's KPIs: messages, searches, bookings, revenue, handoffs."""
    from hf_space.db.supabase import get_supabase
    sb = get_supabase()

    d = target_date or date.today().isoformat()

    try:
        events_res = await sb.table("analytics_events") \
            .select("event_type") \
            .gte("ts", f"{d}T00:00:00") \
            .lt("ts", f"{d}T23:59:59") \
            .execute()
        events = events_res.data or []
    except Exception as e:
        return {"error": str(e)}

    from collections import Counter
    counts = Counter(e["event_type"] for e in events)

    # Revenue from bookings
    try:
        revenue_res = await sb.table("bookings") \
            .select("grand_total_usd") \
            .gte("created_at", f"{d}T00:00:00") \
            .eq("status", "confirmed") \
            .execute()
        revenue = sum((r.get("grand_total_usd") or 0) for r in (revenue_res.data or []))
    except Exception:
        revenue = 0

    return {
        "date":             d,
        "messages":         counts.get("message_received", 0),
        "searches":         counts.get("search_performed", 0),
        "hotels_viewed":    counts.get("hotel_viewed", 0),
        "bookings_created": counts.get("booking_created", 0),
        "payments":         counts.get("payment_completed", 0),
        "handoffs":         counts.get("handoff_created", 0),
        "reviews":          counts.get("review_submitted", 0),
        "revenue_usd":      revenue,
    }


@router.get("/funnel", dependencies=[Depends(_check_admin_key)])
async def get_booking_funnel(days: int = 7) -> dict:
    """
    Booking funnel conversion: search → hotel_viewed → room_selected → payment.
    """
    from hf_space.db.supabase import get_supabase
    from datetime import timedelta
    sb = get_supabase()

    start_date = (date.today() - timedelta(days=days)).isoformat()

    try:
        res = await sb.table("analytics_events") \
            .select("event_type,psid_hash") \
            .gte("ts", f"{start_date}T00:00:00") \
            .execute()
        events = res.data or []
    except Exception as e:
        return {"error": str(e)}

    from collections import defaultdict, Counter
    counts = Counter(e["event_type"] for e in events)
    unique_psids: dict[str, set] = defaultdict(set)
    for e in events:
        unique_psids[e["event_type"]].add(e["psid_hash"])

    steps = ["search_performed", "hotel_viewed", "room_selected", "payment_completed", "booking_created"]
    funnel = []
    for step in steps:
        funnel.append({
            "step":       step,
            "events":     counts.get(step, 0),
            "unique":     len(unique_psids.get(step, set())),
        })

    return {"funnel": funnel, "period_days": days}
