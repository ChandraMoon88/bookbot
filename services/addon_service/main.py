"""
services/addon_service/main.py
---------------------------------
FastAPI add-on recommendation microservice.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .recommender import recommend

app = FastAPI(title="Addon Service")


class RecommendRequest(BaseModel):
    purpose:     Optional[str] = "leisure"
    guests:      Optional[int] = 1
    nights:      Optional[int] = 1
    hotel_stars: Optional[int] = 3


@app.post("/recommend")
def recommend_addons(req: RecommendRequest):
    addons = recommend(
        purpose=req.purpose,
        guests=req.guests,
        nights=req.nights,
        hotel_stars=req.hotel_stars,
    )
    return {"addons": addons, "count": len(addons)}


@app.get("/catalog")
def catalog():
    from .recommender import ADDONS_CATALOG
    return {"catalog": ADDONS_CATALOG}


@app.get("/health")
def health():
    return {"status": "ok", "service": "addon_service"}
