"""
services/search_service/elasticsearch_client.py
-------------------------------------------------
Hotel search using Elasticsearch.

Input:  city, check_in, check_out, guests, filters
Output: { hits, total, facets }
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ES_URL  = os.getenv("ELASTICSEARCH_URL", "")
ES_USER = os.getenv("ELASTICSEARCH_USER", "")
ES_PASS = os.getenv("ELASTICSEARCH_PASS", "")
INDEX   = "hotels"


def _client():
    from elasticsearch import Elasticsearch
    return Elasticsearch(
        ES_URL,
        http_auth=(ES_USER, ES_PASS) if ES_USER else None,
        verify_certs=True,
    )


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
