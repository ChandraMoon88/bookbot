"""
hf_space/routers/loyalty.py
-----------------------------
Module 10 — Loyalty Programme & Gamification

Tiers: Bronze (0–999 pts) → Silver (1000–4999) → Gold (5000–14999) → Platinum (15000+)
Earn: 100 pts per $10 USD spent on bookings + bonus events (birthday, 5th stay, etc.)
Redeem: 1000 pts = $10 discount on next booking

Uses services/loyalty_service/ for tier + points logic.

Flow:
  LOYALTY_MENU postback → show balance card → quick replies
  LOYALTY_REDEEM → pick discount → lock discount credit in draft
  LOYALTY_HISTORY → list recent transactions
"""

from __future__ import annotations

import hashlib
import json

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse

log = structlog.get_logger(__name__)
router = APIRouter()

_TIER_THRESHOLDS = {"bronze": 0, "silver": 1000, "gold": 5000, "platinum": 15000}
_TIER_EMOJI      = {"bronze": "🥉", "silver": "🥈", "gold": "🥇", "platinum": "💎"}
_POINTS_PER_10   = 100  # 100 pts per $10 spent


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_loyalty_menu(psid: str, lang: str) -> tuple[list[dict], str]:
    """Show loyalty balance and options."""
    mb      = MessengerResponse(psid)
    account = await _get_or_create_account(psid)

    points  = account.get("points", 0)
    tier    = _calc_tier(points)
    next_t, pts_needed = _next_tier(points)
    redeemable_dollars = (points // 1000) * 10

    tier_bar = _tier_progress_bar(points, tier)

    balance_text = (
        f"{_TIER_EMOJI.get(tier,'✨')} **{tier.title()} Member**\n\n"
        f"Points balance: **{points:,} pts**\n"
        f"{tier_bar}\n"
        f"{'Next: ' + next_t.title() + ' (' + str(pts_needed) + ' pts away)' if next_t else '🏆 Maximum tier reached!'}\n\n"
        f"💰 Redeemable value: **${redeemable_dollars}** discount"
    )

    opts = [
        {"title": "💰 Redeem points",      "payload": "LOYALTY_REDEEM"},
        {"title": "📋 Points history",     "payload": "LOYALTY_HISTORY"},
        {"title": "🎯 How to earn more",   "payload": "LOYALTY_HOW_TO_EARN"},
        {"title": "🏠 Main menu",           "payload": "MENU_MAIN"},
    ]

    if redeemable_dollars == 0:
        opts = [o for o in opts if o["payload"] != "LOYALTY_REDEEM"]

    await set_user_state(psid, "loyalty_browsing")
    return mb.send_sequence([mb.text(balance_text)]) + [mb.quick_replies("What would you like to do?", opts[:4])], "loyalty_browsing"


async def handle_loyalty_input(psid: str, text: str, lang: str) -> tuple[list[dict], str]:
    """Dispatch within loyalty_browsing state."""
    if text == "LOYALTY_HISTORY":
        return await _show_history(psid, lang)
    if text == "LOYALTY_REDEEM":
        return await _handle_redeem(psid, lang)
    if text.startswith("LOYALTY_REDEEM_CONFIRM_"):
        amount = int(text.split("_")[-1])
        return await _confirm_redeem(psid, amount, lang)
    if text == "LOYALTY_HOW_TO_EARN":
        return await _show_earn_guide(psid, lang)
    return await handle_loyalty_menu(psid, lang)


async def _handle_redeem(psid: str, lang: str) -> tuple[list[dict], str]:
    mb      = MessengerResponse(psid)
    account = await _get_or_create_account(psid)
    points  = account.get("points", 0)

    redeemable_dollars = (points // 1000) * 10
    if redeemable_dollars <= 0:
        return [mb.text(f"You need at least 1000 points to redeem. You currently have {points} pts.")], "loyalty_browsing"

    # Offer redemption in $10 steps up to max $50 per booking
    max_redemption = min(redeemable_dollars, 50)
    options = []
    for amount in [10, 20, 30, 40, 50]:
        if amount <= max_redemption:
            pts_cost = amount * 100
            options.append({"title": f"${amount} off ({pts_cost} pts)", "payload": f"LOYALTY_REDEEM_CONFIRM_{amount}"})

    options.append({"title": "← Back",  "payload": "LOYALTY_MENU"})

    return [mb.quick_replies(
        f"You have {points:,} pts. Choose your discount:",
        options[:5],
    )], "loyalty_browsing"


async def _confirm_redeem(psid: str, amount: int, lang: str) -> tuple[list[dict], str]:
    """Lock the loyalty discount into the booking draft."""
    mb      = MessengerResponse(psid)
    account = await _get_or_create_account(psid)
    points  = account.get("points", 0)
    pts_cost = amount * 100

    if points < pts_cost:
        return [mb.text(f"Insufficient points. You need {pts_cost} pts but have {points} pts.")], "loyalty_browsing"

    # Store discount in booking draft
    from hf_space.db.redis import get_booking_draft, set_booking_draft
    draft = await get_booking_draft(psid) or {}
    draft["loyalty_discount_usd"] = amount
    draft["loyalty_pts_to_deduct"] = pts_cost
    await set_booking_draft(psid, draft)

    return [
        mb.text(f"✅ **${amount} loyalty discount** applied to your booking!\n(Will be deducted from your points after checkout.)"),
        mb.quick_replies("Would you like to proceed?", [
            {"title": "💳 Continue to payment", "payload": "PAYMENT_START"},
            {"title": "← Back",                  "payload": "LOYALTY_MENU"},
        ]),
    ], "loyalty_browsing"


async def _show_history(psid: str, lang: str) -> tuple[list[dict], str]:
    mb = MessengerResponse(psid)
    sb = get_supabase()
    psid_hash = hashlib.sha256(psid.encode()).hexdigest()

    try:
        res = await sb.table("loyalty_transactions") \
            .select("points,type,description,created_at") \
            .eq("psid_hash", psid_hash) \
            .order("created_at", desc=True) \
            .limit(5).execute()
        txns = res.data or []
    except Exception:
        txns = []

    if not txns:
        return [mb.text("No points history yet. Book a stay to start earning! 🌟")], "loyalty_browsing"

    lines = []
    for t in txns:
        sign = "+" if t["type"] == "earn" else "-"
        lines.append(f"{sign}{t['points']} pts — {t['description']}")

    return [mb.text("📋 Recent points activity:\n\n" + "\n".join(lines))], "loyalty_browsing"


async def _show_earn_guide(psid: str, lang: str) -> tuple[list[dict], str]:
    mb = MessengerResponse(psid)
    guide = (
        "🎯 How to earn points:\n\n"
        "🛏 Book a stay → 100 pts/$10 spent\n"
        "⭐ Leave a review → 50 pts\n"
        "🎂 Birthday bonus → 200 pts\n"
        "🏅 5th booking → 500 bonus pts\n"
        "👥 Refer a friend → 300 pts each\n\n"
        "Redeem: 1000 pts = $10 discount 💰"
    )
    return [mb.text(guide)], "loyalty_browsing"


async def award_booking_points(psid: str, booking_id: str, amount_usd: float) -> None:
    """Called after confirmed booking to award points. Fire-and-forget."""
    try:
        from services.loyalty_service.points_engine import calculate_booking_points
        points = await calculate_booking_points(amount_usd)
        sb = get_supabase()
        psid_hash = hashlib.sha256(psid.encode()).hexdigest()
        await sb.table("loyalty_transactions").insert({
            "psid_hash":  psid_hash,
            "booking_id": booking_id,
            "points":     points,
            "type":       "earn",
            "description": f"Booking #{booking_id[:8]}",
        }).execute()
        # Update total
        acc = await _get_or_create_account(psid)
        new_pts = acc.get("points", 0) + points
        await sb.table("loyalty_accounts").upsert({"psid_hash": psid_hash, "points": new_pts}).execute()
        log.info("points_awarded", psid_hash=psid_hash[:12], points=points)
    except Exception as e:
        log.error("points_award_failed", error=str(e))


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_or_create_account(psid: str) -> dict:
    sb = get_supabase()
    psid_hash = hashlib.sha256(psid.encode()).hexdigest()
    try:
        res = await sb.table("loyalty_accounts").select("*").eq("psid_hash", psid_hash).maybe_single().execute()
        if res.data:
            return res.data
        # Create account
        await sb.table("loyalty_accounts").insert({"psid_hash": psid_hash, "points": 0, "tier": "bronze"}).execute()
        return {"psid_hash": psid_hash, "points": 0, "tier": "bronze"}
    except Exception:
        return {"points": 0, "tier": "bronze"}


def _calc_tier(points: int) -> str:
    if points >= 15000:
        return "platinum"
    if points >= 5000:
        return "gold"
    if points >= 1000:
        return "silver"
    return "bronze"


def _next_tier(points: int) -> tuple[str, int]:
    for tier, threshold in [("silver", 1000), ("gold", 5000), ("platinum", 15000)]:
        if points < threshold:
            return tier, threshold - points
    return "", 0


def _tier_progress_bar(points: int, tier: str) -> str:
    thresholds = {"bronze": (0, 1000), "silver": (1000, 5000), "gold": (5000, 15000), "platinum": (15000, 15000)}
    low, high = thresholds.get(tier, (0, 1000))
    if high == low:
        return "█████ MAX"
    pct = min(1.0, (points - low) / (high - low))
    filled = int(pct * 10)
    return "█" * filled + "░" * (10 - filled) + f" {int(pct*100)}%"


# ── API endpoints ──────────────────────────────────────────────────────────────

@router.get("/balance/{psid}")
async def get_balance(psid: str) -> dict:
    account = await _get_or_create_account(psid)
    points  = account.get("points", 0)
    tier    = _calc_tier(points)
    return {"points": points, "tier": tier, "redeemable_usd": (points // 1000) * 10}


class AwardRequest(BaseModel):
    psid:       str
    booking_id: str
    amount_usd: float


@router.post("/award")
async def award_points_endpoint(req: AwardRequest) -> dict:
    await award_booking_points(req.psid, req.booking_id, req.amount_usd)
    return {"success": True}
