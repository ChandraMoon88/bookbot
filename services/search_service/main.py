"""
services/search_service/main.py
---------------------------------
FastAPI hotel search microservice.

POST /search  → ranked hotel list
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .elasticsearch_client import search_hotels as es_search
from .ranker import rank

app = FastAPI(title="Search Service")


class SearchRequest(BaseModel):
    city:       str
    check_in:   str
    check_out:  str
    num_adults: int = 1
    num_children: int = 0
    filters:    Optional[dict] = None
    guest_id:   Optional[str] = None
    language:   str = "en"


@app.post("/search")
def search(req: SearchRequest):
    result = es_search(
        city=req.city,
        check_in=req.check_in,
        check_out=req.check_out,
        num_guests=req.num_adults,
        filters=req.filters or {},
        language=req.language,
    )
    ranked = rank(result["hits"], guest_id=req.guest_id, top_k=5)
    return {"hotels": ranked, "total": result["total"], "facets": result["facets"]}


@app.get("/health")
def health():
    return {"status": "ok", "service": "search_service"}
