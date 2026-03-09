"""
services/group_service/main.py
--------------------------------
FastAPI group booking microservice.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .room_block         import create_block, pickup_room, get_block
from .deposit_scheduler  import schedule_deposits

app = FastAPI(title="Group Service")


class BlockRequest(BaseModel):
    hotel_id:        str
    room_type_id:    str
    rooms_reserved:  int
    date_from:       str  # YYYY-MM-DD
    date_to:         str  # YYYY-MM-DD
    group_name:      str
    pickup_deadline: str  # YYYY-MM-DD
    rate_per_night:  float
    organiser_email: str


class PickupRequest(BaseModel):
    block_id:   str
    booking_id: str


@app.post("/block")
def create(req: BlockRequest):
    if req.rooms_reserved < 1:
        raise HTTPException(status_code=422, detail="rooms_reserved must be >= 1")

    block    = create_block(**req.dict())
    total    = req.rooms_reserved * req.rate_per_night * _nights(req.date_from, req.date_to)
    schedule = schedule_deposits(
        block_id=block["id"],
        total_amount=total,
        date_from=req.date_from,
        organiser_email=req.organiser_email,
    )
    return {"block": block, "deposit_schedule": schedule}


@app.post("/pickup")
def pickup(req: PickupRequest):
    try:
        result = pickup_room(req.block_id, req.booking_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return result


@app.get("/block/{block_id}")
def fetch_block(block_id: str):
    try:
        return get_block(block_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "service": "group_service"}


def _nights(date_from: str, date_to: str) -> int:
    from datetime import datetime
    d1 = datetime.strptime(date_from, "%Y-%m-%d")
    d2 = datetime.strptime(date_to,   "%Y-%m-%d")
    return max((d2 - d1).days, 1)
