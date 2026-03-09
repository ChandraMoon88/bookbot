#!/usr/bin/env python3
"""
scripts/setup_elasticsearch_index.py
--------------------------------------
Creates (or re-creates) the Elasticsearch hotel index with
the correct mappings and settings.

Usage:
    ES_URL=https://... python scripts/setup_elasticsearch_index.py
"""

import os
import json
import urllib.request

ES_URL   = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")
ES_USER  = os.environ.get("ELASTICSEARCH_USERNAME", "elastic")
ES_PASS  = os.environ.get("ELASTICSEARCH_PASSWORD", "")
INDEX    = "hotels"

MAPPING = {
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 1,
        "analysis": {
            "analyzer": {
                "hotel_analyzer": {
                    "type":      "custom",
                    "tokenizer": "standard",
                    "filter":    ["lowercase", "asciifolding", "stop"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "id":          {"type": "keyword"},
            "name":        {"type": "text",    "analyzer": "hotel_analyzer"},
            "city":        {"type": "text",    "analyzer": "hotel_analyzer",
                            "fields": {"keyword": {"type": "keyword"}}},
            "country":     {"type": "keyword"},
            "description": {"type": "text",   "analyzer": "hotel_analyzer"},
            "stars":       {"type": "integer"},
            "rating":      {"type": "float"},
            "price_per_night": {"type": "float"},
            "amenities":   {"type": "keyword"},
            "location": {
                "type":       "geo_point",
            },
            "is_active":   {"type": "boolean"},
            "images":      {"type": "keyword", "index": False},
            "updated_at":  {"type": "date"},
        }
    },
}


def _request(method: str, path: str, body: dict = None):
    url  = f"{ES_URL}{path}"
    data = json.dumps(body).encode() if body else None
    import base64
    creds   = base64.b64encode(f"{ES_USER}:{ES_PASS}".encode()).decode()
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Basic {creds}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def setup():
    # Delete existing index if present
    try:
        _request("DELETE", f"/{INDEX}")
        print(f"Deleted existing index '{INDEX}'")
    except Exception:
        pass

    result = _request("PUT", f"/{INDEX}", MAPPING)
    print(f"Index '{INDEX}' created:", result)


if __name__ == "__main__":
    setup()
