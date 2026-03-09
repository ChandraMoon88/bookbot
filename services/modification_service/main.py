"""
services/modification_service/main.py
---------------------------------------
FastAPI booking modification microservice.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .policy_engine import evaluate
from .rebooking import modify_booking

app = FastAPI(title="Modification Service")


class ModifyRequest(BaseModel):
    booking_id:          str
    new_check_in:        str   # YYYY-MM-DD
    new_check_out:       str   # YYYY-MM-DD
    new_room_type:       str
    amount_per_night:    float
    current_check_in:    str   # Original check-in for policy evaluation
    reason:              Optional[str] = None
    agent_id:            Optional[str] = "bot"


@app.post("/modify")
def modify(req: ModifyRequest):
    policy = evaluate(
        check_in_str=req.current_check_in,
        amount_per_night=req.amount_per_night,
        reason=req.reason,
    )
    if not policy["allowed"]:
        raise HTTPException(
            status_code=409,
            detail={"error": "modification_not_allowed", "policy": policy},
        )

    result = modify_booking(
        booking_id=req.booking_id,
        new_check_in=req.new_check_in,
        new_check_out=req.new_check_out,
        new_room_type=req.new_room_type,
        fee_usd=policy["fee_usd"],
        agent_id=req.agent_id,
    )
    return {"booking": result, "policy": policy}


@app.post("/policy")
def check_policy(check_in: str, amount_per_night: float, reason: Optional[str] = None):
    return evaluate(check_in, amount_per_night, reason)


@app.get("/health")
def health():
    return {"status": "ok", "service": "modification_service"}
