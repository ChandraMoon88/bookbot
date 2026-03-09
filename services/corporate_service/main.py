"""
services/corporate_service/main.py
------------------------------------
FastAPI corporate booking microservice.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .rate_lookup     import get_corporate_rate
from .approval_engine import evaluate as check_approval

app = FastAPI(title="Corporate Service")


class RateRequest(BaseModel):
    corporate_account_id: str
    hotel_id:             str
    room_type_id:         str
    check_in:             str  # YYYY-MM-DD


class ApprovalRequest(BaseModel):
    corporate_account_id: str
    booking_id:           str
    amount_usd:           float
    approver_email:       str


@app.post("/rate")
def corporate_rate(req: RateRequest):
    return get_corporate_rate(
        corporate_account_id=req.corporate_account_id,
        hotel_id=req.hotel_id,
        room_type_id=req.room_type_id,
        check_in=req.check_in,
    )


@app.post("/approval")
def approval(req: ApprovalRequest):
    return check_approval(
        corporate_account_id=req.corporate_account_id,
        booking_id=req.booking_id,
        amount_usd=req.amount_usd,
        approver_email=req.approver_email,
    )


@app.get("/approval/{approval_id}")
def get_approval(approval_id: str):
    import os, json, urllib.request
    SUPA = os.environ.get("SUPABASE_URL", "")
    KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    url  = f"{SUPA}/rest/v1/approval_requests?id=eq.{approval_id}"
    req  = urllib.request.Request(url, headers={
        "apikey": KEY, "Authorization": f"Bearer {KEY}"
    })
    with urllib.request.urlopen(req) as resp:
        rows = json.loads(resp.read())
    if not rows:
        raise HTTPException(status_code=404, detail="Approval not found")
    return rows[0]


@app.get("/health")
def health():
    return {"status": "ok", "service": "corporate_service"}
