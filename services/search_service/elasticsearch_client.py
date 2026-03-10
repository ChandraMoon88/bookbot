"""
services/search_service/elasticsearch_client.py
-------------------------------------------------
Hotel search using PostgreSQL full-text search — completely free,
no external service required. Uses the existing Supabase/PostgreSQL DB.

Search supports:
  - City name (ILIKE fuzzy match)
  - Hotel name (ILIKE + PostgreSQL tsvector FTS)
  - Country / region
  - Description keywords
  - Star rating, amenities filters
  - Availability check (date range × guests)

Input:  city/query, check_in, check_out, guests, filters
Output: { hits, total, facets }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


def _get_conn():
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    if "sslmode" not in url and "supabase.com" in url:
        sep = "&" if "?" in url else "?"
        url = url + sep + "sslmode=require"
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    num_guests: int = 1,
    filters: dict | None = None,
    language: str = "en",
) -> dict:
    """
    PostgreSQL full-text hotel search with availability check.

    Search strategy (in priority order):
      1. City ILIKE match  — "paris" matches city="Paris"
      2. PostgreSQL FTS    — plainto_tsquery on name+description+city+country
      3. ILIKE fallback    — %query% in name, description, country

    Returns {"hits": [...], "total": int, "facets": {}}
    """
    filters = filters or {}

    # Build WHERE clauses for active hotels
    base_where = ["h.is_active = TRUE"]
    base_params: list[Any] = []

    if "stars" in filters:
        base_where.append("h.star_rating = %s")
        base_params.append(int(filters["stars"]))

    if "amenities" in filters:
        for amenity in filters["amenities"]:
            base_where.append("h.amenities::text ILIKE %s")
            base_params.append(f"%{amenity}%")

    where = " AND ".join(base_where)
    q = city.strip()

    # Validate dates
    try:
        ci = datetime.strptime(check_in, "%Y-%m-%d").date()
        co = datetime.strptime(check_out, "%Y-%m-%d").date()
        nights = (co - ci).days
    except ValueError:
        return {"hits": [], "total": 0, "facets": {}}
    if nights <= 0:
        return {"hits": [], "total": 0, "facets": {}}

    _HOTEL_COLS = """
        h.id::text, h.name, h.description, h.city, h.country,
        h.star_rating, h.amenities, h.currency,
        h.check_in_time, h.check_out_time, h.thumbnail_url, h.policy
    """

    try:
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                # ── Pass 1: City ILIKE ────────────────────────────────────────
                cur.execute(
                    f"""
                    SELECT {_HOTEL_COLS}, 1.0::float AS search_score
                    FROM hotels h
                    WHERE {where} AND LOWER(h.city) ILIKE LOWER(%s)
                    LIMIT 30
                    """,
                    base_params + [f"%{q}%"],
                )
                city_rows = [dict(r) for r in cur.fetchall()]

                # ── Pass 2: PostgreSQL full-text search ───────────────────────
                fts_query = q.replace("'", " ")
                cur.execute(
                    f"""
                    SELECT {_HOTEL_COLS},
                        ts_rank(
                            to_tsvector('simple',
                                COALESCE(h.name,'') || ' ' ||
                                COALESCE(h.description,'') || ' ' ||
                                COALESCE(h.city,'') || ' ' ||
                                COALESCE(h.country,'')
                            ),
                            plainto_tsquery('simple', %s)
                        ) AS search_score
                    FROM hotels h
                    WHERE {where}
                      AND to_tsvector('simple',
                              COALESCE(h.name,'') || ' ' ||
                              COALESCE(h.description,'') || ' ' ||
                              COALESCE(h.city,'') || ' ' ||
                              COALESCE(h.country,'')
                          ) @@ plainto_tsquery('simple', %s)
                    ORDER BY search_score DESC
                    LIMIT 30
                    """,
                    base_params + [fts_query, fts_query],
                )
                fts_rows = [dict(r) for r in cur.fetchall()]

                # Merge, dedup by id (city results first)
                seen: set[str] = set()
                merged: list[dict] = []
                for row in city_rows + fts_rows:
                    if row["id"] not in seen:
                        seen.add(row["id"])
                        merged.append(row)

                # ── Pass 3: ILIKE fallback on name/description/country ────────
                if not merged:
                    cur.execute(
                        f"""
                        SELECT {_HOTEL_COLS}, 0.3::float AS search_score
                        FROM hotels h
                        WHERE {where}
                          AND (h.name ILIKE %s
                               OR h.description ILIKE %s
                               OR h.country ILIKE %s)
                        LIMIT 30
                        """,
                        base_params + [f"%{q}%", f"%{q}%", f"%{q}%"],
                    )
                    merged = [dict(r) for r in cur.fetchall()]

                if not merged:
                    return {"hits": [], "total": 0, "facets": {}}

                # ── Availability check ────────────────────────────────────────
                hotel_ids = [r["id"] for r in merged]
                ph = ",".join(["%s"] * len(hotel_ids))
                cur.execute(
                    f"""
                    SELECT
                        hotel_id::text,
                        room_type_code,
                        room_type_name,
                        MIN(available_count)                     AS min_avail,
                        MAX(max_adults)                          AS max_adults,
                        MAX(max_children)                        AS max_children,
                        (array_agg(rate_plans ORDER BY date))[1] AS rate_plans
                    FROM inventory
                    WHERE hotel_id::text IN ({ph})
                      AND date >= %s AND date < %s
                      AND available_count > 0
                      AND is_blackout = FALSE
                    GROUP BY hotel_id, room_type_code, room_type_name
                    HAVING COUNT(*) >= %s
                    """,
                    hotel_ids + [ci, co, nights],
                )
                rooms_rows = [dict(r) for r in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        logger.error("search_hotels (postgres) error: %s", exc, exc_info=True)
        return {"hits": [], "total": 0, "facets": {}}

    # Group rooms by hotel
    rooms_by_hotel: dict[str, list] = {}
    for row in rooms_rows:
        hid = str(row["hotel_id"])
        rate_plans = row.get("rate_plans") or {}
        if isinstance(rate_plans, str):
            try:
                rate_plans = json.loads(rate_plans)
            except Exception:
                rate_plans = {}
        prices = [
            float(v["price_per_night"])
            for v in rate_plans.values()
            if isinstance(v, dict) and v.get("price_per_night")
        ]
        rooms_by_hotel.setdefault(hid, []).append({
            "room_type_code":  row["room_type_code"],
            "room_type_name":  row["room_type_name"],
            "available_count": int(row.get("min_avail") or 0),
            "max_adults":      int(row.get("max_adults") or 2),
            "max_children":    int(row.get("max_children") or 0),
            "price_per_night": min(prices) if prices else None,
            "rate_plans":      rate_plans,
        })

    # Build final hits — only hotels with available rooms
    hits: list[dict] = []
    for h in merged:
        hid   = str(h["id"])
        rooms = rooms_by_hotel.get(hid, [])
        eligible = [
            r for r in rooms
            if r["max_adults"] >= num_guests and r["available_count"] > 0
        ]
        if not eligible:
            continue

        amenities = h.get("amenities") or []
        if isinstance(amenities, str):
            try:
                amenities = json.loads(amenities)
            except Exception:
                amenities = []

        prices = [r["price_per_night"] for r in eligible if r["price_per_night"]]
        hits.append({
            "id":              hid,
            "name":            h["name"],
            "description":     h.get("description", ""),
            "city":            h.get("city", ""),
            "country":         h.get("country", ""),
            "star_rating":     h.get("star_rating", 3),
            "amenities":       amenities,
            "currency":        h.get("currency", "USD"),
            "thumbnail_url":   h.get("thumbnail_url", ""),
            "check_in_time":   h.get("check_in_time", "14:00"),
            "check_out_time":  h.get("check_out_time", "12:00"),
            "policy":          h.get("policy", ""),
            "available_rooms": eligible,
            "price_from_usd":  min(prices) if prices else 0,
            "search_score":    float(h.get("search_score") or 0),
        })

    star_facets = sorted({h["star_rating"] for h in hits})
    city_facets = list({h["city"] for h in hits})

    return {
        "hits":   hits,
        "total":  len(hits),
        "facets": {"star_ratings": star_facets, "cities": city_facets},
    }


def index_hotel(hotel: dict) -> bool:
    """No-op — PostgreSQL FTS indexes automatically via tsvector."""
    return True



def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    num_guests: int = 1,
    filters: dict | None = None,
    language: str = "en",
) -> dict:
    """
    Full-text hotel search with geo filters.

    filters keys: stars (int), max_price (float), amenities (list), lat, lng
    """
    filters = filters or {}
    must_clauses: list[dict] = [
        {"match": {"city": {"query": city, "fuzziness": "AUTO"}}}
    ]
    filter_clauses: list[dict] = [
        {"term": {"active": True}}
    ]

    # Star rating
    if "stars" in filters:
        filter_clauses.append({"term": {"star_rating": filters["stars"]}})

    # Price range
    if "max_price" in filters:
        filter_clauses.append(
            {"range": {"base_price_usd": {"lte": filters["max_price"]}}}
        )

    # Amenities filter
    if "amenities" in filters:
        for amenity in filters["amenities"]:
            filter_clauses.append({"term": {"amenities": amenity}})

    # Geo-distance
    if filters.get("lat") and filters.get("lng"):
        filter_clauses.append({
            "geo_distance": {
                "distance":   "20km",
                "location":   {"lat": filters["lat"], "lon": filters["lng"]},
            }
        })

    query = {
        "query": {
            "bool": {
                "must":   must_clauses,
                "filter": filter_clauses,
            }
        },
        "aggs": {
            "stars":     {"terms": {"field": "star_rating"}},
            "amenities": {"terms": {"field": "amenities", "size": 20}},
        },
        "size": 20,
    }

    try:
        es   = _client()
        resp = es.search(index=INDEX, body=query)
        hits = [
            {**h["_source"], "score": h["_score"]}
            for h in resp["hits"]["hits"]
        ]
        facets = {
            "stars":     resp["aggregations"]["stars"]["buckets"],
            "amenities": resp["aggregations"]["amenities"]["buckets"],
        }
        return {"hits": hits, "total": resp["hits"]["total"]["value"], "facets": facets}
    except Exception as e:
        logger.error("Elasticsearch search error: %s", e)
        return {"hits": [], "total": 0, "facets": {}}


def index_hotel(hotel: dict) -> bool:
    """Index or re-index a hotel document in Elasticsearch."""
    try:
        es = _client()
        es.index(index=INDEX, id=hotel["id"], body=hotel)
        return True
    except Exception as e:
        logger.error("Elasticsearch index error: %s", e)
        return False
