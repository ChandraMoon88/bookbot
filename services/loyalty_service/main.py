"""
services/loyalty_service/main.py
----------------------------------
FastAPI loyalty microservice.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .points_engine  import award, redeem, get_balance
from .tier_engine    import evaluate as evaluate_tier, get_perks, next_tier_info
from .gamification   import update_leaderboard, top_guests, get_rank, award_badge, list_badges

app = FastAPI(title="Loyalty Service")


class AwardRequest(BaseModel):
    guest_id:   str
    amount_usd: float
    booking_id: str


class RedeemRequest(BaseModel):
    guest_id:          str
    points_to_redeem:  int
    booking_id:        str


class BadgeRequest(BaseModel):
    guest_id:    str
    badge_name:  str


@app.post("/award")
def award_points(req: AwardRequest):
    result = award(req.guest_id, req.amount_usd, req.booking_id)
    balance = get_balance(req.guest_id)
    tier    = evaluate_tier(balance["points_balance"])
    update_leaderboard(req.guest_id, balance["points_balance"])
    return {**result, "balance": balance, "tier": tier}


@app.post("/redeem")
def redeem_points(req: RedeemRequest):
    try:
        result = redeem(req.guest_id, req.points_to_redeem, req.booking_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    update_leaderboard(req.guest_id, result["remaining_balance"])
    return result


@app.get("/balance/{guest_id}")
def balance(guest_id: str):
    bal     = get_balance(guest_id)
    tier    = evaluate_tier(bal["points_balance"])
    next_t  = next_tier_info(bal["points_balance"])
    badges  = list_badges(guest_id)
    rank    = get_rank(guest_id)
    return {**bal, "tier": tier, "next_tier_info": next_t, "badges": badges, "leaderboard_rank": rank}


@app.get("/perks/{tier}")
def perks(tier: str):
    return get_perks(tier)


@app.post("/badge")
def badge(req: BadgeRequest):
    return award_badge(req.guest_id, req.badge_name)


@app.get("/leaderboard")
def leaderboard(n: int = 10):
    return {"leaderboard": top_guests(n)}


@app.get("/health")
def health():
    return {"status": "ok", "service": "loyalty_service"}
