#!/usr/bin/env python3
"""
scripts/hotel_onboarding.py
-----------------------------
Onboards a new hotel partner:
  1. Inserts hotel record into Supabase
  2. Indexes hotel into Elasticsearch
  3. Embeds and upserts FAQ entries into Qdrant

Usage:
    python scripts/hotel_onboarding.py --file data/hotel_sample.json
"""

import argparse
import json
import os
import sys

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def insert_hotel(hotel: dict) -> dict:
    cols = list(hotel.keys())
    placeholders = ", ".join(f"%({c})s" for c in cols)
    col_list = ", ".join(cols)
    sql = f"INSERT INTO hotels ({col_list}) VALUES ({placeholders}) RETURNING *"
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, hotel)
            conn.commit()
            return dict(cur.fetchone())


def index_elasticsearch(hotel: dict) -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from services.search_service.elasticsearch_client import index_hotel
    index_hotel(hotel)
    print(f"  ✓ Elasticsearch indexed hotel '{hotel.get('name')}'")


def embed_faqs(hotel_id: str, faqs: list[dict]) -> None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from services.rag_service.qdrant_client import upsert_faqs
    upsert_faqs(hotel_id, faqs)
    print(f"  ✓ Qdrant upserted {len(faqs)} FAQ entries for hotel {hotel_id}")


def onboard(hotel_data: dict) -> None:
    print(f"\nOnboarding hotel: {hotel_data.get('name', '?')}")

    # 1. Database (PostgreSQL via DATABASE_URL)
    record = insert_hotel({k: v for k, v in hotel_data.items() if k != "faqs"})
    hotel_id = record.get("id")
    print(f"  ✓ DB inserted hotel id={hotel_id}")

    # 2. Elasticsearch
    index_elasticsearch({**hotel_data, "id": hotel_id})

    # 3. Qdrant FAQs
    faqs = hotel_data.get("faqs", [])
    if faqs:
        embed_faqs(hotel_id, faqs)
    else:
        print("  ℹ️  No FAQs provided; skipping Qdrant upsert")

    print(f"\n✅ Hotel '{hotel_data.get('name')}' onboarded successfully (id={hotel_id})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to hotel JSON file")
    args   = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        for hotel in data:
            onboard(hotel)
    else:
        onboard(data)
