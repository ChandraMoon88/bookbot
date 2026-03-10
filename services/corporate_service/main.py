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
    import os, psycopg2, psycopg2.extras
    dsn = os.environ.get("DATABASE_URL", "")
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    with psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM approval_requests WHERE id = %s", (approval_id,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Approval not found")
    return dict(row)


@app.get("/health")
def health():
    return {"status": "ok", "service": "corporate_service"}
