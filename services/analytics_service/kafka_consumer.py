"""
services/analytics_service/kafka_consumer.py
----------------------------------------------
Analytics event writer — PostgreSQL only (no Kafka, no ClickHouse).

Writes all events to the `analytics_events` table in the same Supabase
PostgreSQL database used by the rest of the app.  Zero external services
required.

The public API is intentionally identical to the old Kafka version so
callers in analytics.py don't need to change:

    from services.analytics_service.kafka_consumer import produce_event
    await produce_event({"type": "booking.created", ...})

Schema (run in Supabase SQL Editor):
    CREATE TABLE IF NOT EXISTS analytics_events (
        id          BIGSERIAL PRIMARY KEY,
        event_type  TEXT         NOT NULL,
        properties  JSONB        DEFAULT '{}',
        ts          TIMESTAMPTZ  DEFAULT NOW()
    );
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

_DATABASE_URL = None


def _get_connection():
    """Return a psycopg2 connection; lazy-init DATABASE_URL."""
    global _DATABASE_URL
    import psycopg2
    if _DATABASE_URL is None:
        _DATABASE_URL = os.environ.get("DATABASE_URL", "")
    if not _DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set — analytics events will be dropped")
    return psycopg2.connect(_DATABASE_URL, connect_timeout=5)


async def produce_event(row: dict[str, Any]) -> None:
    """
    Persist an analytics event to PostgreSQL.

    `row` should contain at minimum {"type": "..."} plus any extra fields.
    All non-standard fields are stored as JSON in the `properties` column.

    This function never raises — errors are logged and swallowed so that
    analytics failures never block the booking flow.
    """
    try:
        event_type = row.get("type") or row.get("event_type") or "unknown"
        # Strip the top-level 'type' key; keep everything else as properties
        props = {k: v for k, v in row.items() if k not in ("type", "event_type")}

        conn = _get_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO analytics_events (event_type, properties, ts) "
                        "VALUES (%s, %s, %s)",
                        (event_type, json.dumps(props), datetime.now(timezone.utc)),
                    )
        finally:
            conn.close()
    except Exception as exc:
        log.warning("analytics_event_failed event_type=%s error=%s", row.get("type"), exc)

