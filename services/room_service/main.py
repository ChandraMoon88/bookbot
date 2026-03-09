"""
services/room_service/main.py
-------------------------------
FastAPI room availability + soft lock microservice.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from .availability import check_availability
from .soft_lock    import acquire_lock, release_lock, refresh_lock

app = FastAPI(title="Room Service")


class AvailabilityRequest(BaseModel):
    room_type_id: str
    check_in:     str
    check_out:    str
    num_guests:   int = 1


class LockRequest(BaseModel):
    room_type_id:  str
    check_in:      str
    check_out:     str
    session_id:    str
    lock_minutes:  int = 15


@app.post("/availability")
def availability(req: AvailabilityRequest):
    return check_availability(req.room_type_id, req.check_in, req.check_out, req.num_guests)


@app.post("/lock")
def lock(req: LockRequest):
    return acquire_lock(req.room_type_id, req.check_in, req.check_out,
                        req.session_id, req.lock_minutes)


@app.post("/lock/release")
def unlock(req: LockRequest):
    release_lock(req.room_type_id, req.check_in, req.check_out, req.session_id)
    return {"released": True}


@app.post("/lock/refresh")
def refresh(req: LockRequest):
    ok = refresh_lock(req.room_type_id, req.check_in, req.check_out,
                      req.session_id, req.lock_minutes)
    return {"refreshed": ok}


@app.get("/health")
def health():
    return {"status": "ok", "service": "room_service"}
