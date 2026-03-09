"""
services/force_majeure_service/main.py
----------------------------------------
FastAPI force-majeure detection & relocation microservice.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .news_monitor import assess
from .relocation   import find_alternatives, build_relocation_message

app = FastAPI(title="Force Majeure Service")


class AssessRequest(BaseModel):
    city:         str
    country:      str
    country_iso2: Optional[str] = ""


class RelocationRequest(BaseModel):
    original_city:  str
    original_ref:   str
    check_in:       str
    check_out:      str
    guests:         Optional[int]   = 1
    budget:         Optional[float] = 500.0
    stars:          Optional[int]   = 3


@app.post("/assess")
def assess_risk(req: AssessRequest):
    return assess(req.city, req.country, req.country_iso2)


@app.post("/relocate")
def relocate(req: RelocationRequest):
    alternatives = find_alternatives(
        original_city=req.original_city,
        check_in=req.check_in,
        check_out=req.check_out,
        guests=req.guests,
        budget=req.budget,
        stars=req.stars,
    )
    message = build_relocation_message(alternatives, req.original_ref)
    return {"alternatives": alternatives, "message": message}


@app.get("/health")
def health():
    return {"status": "ok", "service": "force_majeure_service"}
