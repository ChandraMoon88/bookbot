"""
hf_space/routers/guest.py
--------------------------
Module 4 — Guest Information Collection.

Conversational form with max button use:
  Q1 Name       → free text, validated (min 2 parts, no numbers)
  Q2 Email      → free text, RFC-5322 + typo correction quick reply
  Q3 Phone      → free text, E.164 normalisation
  Q4 Trip purpose → ALL BUTTONS (8 options)
  Q5 Dietary    → ALL BUTTONS (7 options)
  Q6 Accessibility → ALL BUTTONS (6 options)
  
  Returning user shortcut: "Use saved details? 💾"
  Passport OCR shortcut: optional after name

Backend:
  POST /api/guest/validate
  POST /api/guest/ocr_passport
  POST /api/guest/save_profile
"""

from __future__ import annotations

import json
import re
import logging

import structlog
from fastapi import APIRouter
from pydantic import BaseModel

from hf_space.db.redis import get_redis, set_user_state, get_user_profile, set_user_profile, get_booking_draft, set_booking_draft
from hf_space.db.supabase import get_supabase
from render_webhook.messenger_builder import MessengerResponse
from services.guest_service.validator import validate_field

log = structlog.get_logger(__name__)
router = APIRouter()


# ── Quick reply option sets ────────────────────────────────────────────────────

_TRIP_PURPOSE = [
    {"title": "🏖 Leisure",          "payload": "PURPOSE_leisure"},
    {"title": "💍 Honeymoon",        "payload": "PURPOSE_honeymoon"},
    {"title": "👨‍👩‍👧 Family Trip",    "payload": "PURPOSE_family"},
    {"title": "💼 Business",         "payload": "PURPOSE_business"},
    {"title": "🎓 Study/Conference", "payload": "PURPOSE_study"},
    {"title": "🏥 Medical",          "payload": "PURPOSE_medical"},
]

_DIETARY = [
    {"title": "🚫 None",           "payload": "DIET_none"},
    {"title": "🥗 Vegetarian",     "payload": "DIET_vegetarian"},
    {"title": "🌱 Vegan",          "payload": "DIET_vegan"},
    {"title": "🕌 Halal",          "payload": "DIET_halal"},
    {"title": "✡️ Kosher",         "payload": "DIET_kosher"},
    {"title": "🚫🥜 Nut Allergy",  "payload": "DIET_nut_allergy"},
    {"title": "✏️ Other…",         "payload": "DIET_other"},
]

_ACCESSIBILITY = [
    {"title": "None needed ✅",       "payload": "ACCESS_none"},
    {"title": "Wheelchair ♿",        "payload": "ACCESS_wheelchair"},
    {"title": "Ground floor 🔑",     "payload": "ACCESS_ground_floor"},
    {"title": "Visual assist 👁️",    "payload": "ACCESS_visual"},
    {"title": "Hearing assist 👂",   "payload": "ACCESS_hearing"},
    {"title": "✏️ Specify…",         "payload": "ACCESS_other"},
]

# Form steps in order
_STEPS = ["name", "email", "phone", "trip_purpose", "dietary", "accessibility"]


# ── Conversation handlers ──────────────────────────────────────────────────────

async def handle_guest_input(
    psid: str, text: str, lang: str
) -> tuple[list[dict], str]:
    """
    Stateful conversational form handler.
    Reads user:{psid}:guest_form to track which field we're collecting.
    """
    r = get_redis()
    raw = await r.get(f"user:{psid}:guest_form")
    form: dict = json.loads(raw) if raw else {}

    # Check returning user shortcut on entry
    if not form and text == "GUEST_USE_SAVED":
        return await _use_saved_guest_data(psid, lang)

    if not form and text == "GUEST_UPDATE":
        # Start fresh form
        await r.set(f"user:{psid}:guest_form", json.dumps({}), ex=3600)
        return await _send_next_question(psid, {}, lang)

    # If no prior form state, trigger it fresh or show returning user shortcut
    if not form:
        return await _start_guest_form(psid, lang)

    # Determine current step = first missing field
    current_step = _get_current_step(form)
    if current_step is None:
        # All fields collected — proceed
        return await _complete_guest_form(psid, form, lang)

    # Validate / store the provided value
    result = await _process_field(psid, current_step, text, form, lang)
    return result


async def _start_guest_form(psid: str, lang: str) -> tuple[list[dict], str]:
    """Check for saved data; offer shortcut or start fresh."""
    profile = await get_user_profile(psid) or {}
    mb = MessengerResponse(psid)

    if profile.get("name") and profile.get("email"):
        name   = profile["name"]
        email  = profile.get("email", "")
        phone  = profile.get("phone", "")
        items = [
            {"title": "Name",  "subtitle": name},
            {"title": "Email", "subtitle": email},
            {"title": "Phone", "subtitle": phone or "—"},
        ]
        saved_list = mb.list_template("Your saved details:", items)
        msgs = mb.send_sequence([saved_list]) + [
            mb.quick_replies(
                "Use your saved details? 💾",
                [
                    {"title": "✅ Yes, use saved",  "payload": "GUEST_USE_SAVED"},
                    {"title": "✏️ Update details",  "payload": "GUEST_UPDATE"},
                ],
            )
        ]
        return msgs, "filling_guest_form"

    # No saved data — start fresh
    r = get_redis()
    await r.set(f"user:{psid}:guest_form", json.dumps({}), ex=3600)
    return await _send_next_question(psid, {}, lang)


async def _use_saved_guest_data(psid: str, lang: str) -> tuple[list[dict], str]:
    """Pre-fill guest form from saved profile."""
    profile = await get_user_profile(psid) or {}
    mb = MessengerResponse(psid)

    form = {
        "name":        profile.get("name", ""),
        "email":       profile.get("email", ""),
        "phone":       profile.get("phone", ""),
        "trip_purpose": "",
        "dietary":     "",
        "accessibility": "",
    }
    r = get_redis()
    await r.set(f"user:{psid}:guest_form", json.dumps(form), ex=3600)

    # Ask trip purpose (first missing button field)
    return [
        mb.quick_replies(
            f"Great, using your saved details! What brings you to this destination? ✈️",
            _TRIP_PURPOSE,
        )
    ], "filling_guest_form"


async def _process_field(
    psid: str, field: str, text: str, form: dict, lang: str
) -> tuple[list[dict], str]:
    """Validate and store a single form field. Return next question or error."""
    mb = MessengerResponse(psid)
    r  = get_redis()

    # ── name ───────────────────────────────────────────────────────────────────
    if field == "name":
        valid, normalized, error = _validate_name(text)
        if not valid:
            return [mb.text(f"❌ {error} Please try again:")], "filling_guest_form"
        form["name"] = normalized

    # ── email ──────────────────────────────────────────────────────────────────
    elif field == "email":
        valid, normalized, suggestion = _validate_email(text)
        if suggestion:
            email_stored = text
            r2 = get_redis()
            await r2.set(f"user:{psid}:email_suggestion", normalized, ex=300)
            form["_email_candidate"] = text
            await r.set(f"user:{psid}:guest_form", json.dumps(form), ex=3600)
            return [mb.quick_replies(
                f"Did you mean {normalized}? 🤔",
                [
                    {"title": f"Yes ✅", "payload": f"EMAIL_ACCEPT_SUGGESTION"},
                    {"title": "No, retype ✏️", "payload": "EMAIL_RETYPE"},
                ],
            )], "filling_guest_form"

        if not valid:
            return [mb.text(f"❌ That doesn't look like a valid email. Please try again:")], "filling_guest_form"

        form.pop("_email_candidate", None)
        form["email"] = normalized

    # ── email suggestion responses ─────────────────────────────────────────────
    elif text == "EMAIL_ACCEPT_SUGGESTION":
        r2 = get_redis()
        suggestion = await r2.get(f"user:{psid}:email_suggestion")
        form.pop("_email_candidate", None)
        form["email"] = suggestion or form.get("_email_candidate", "")

    elif text == "EMAIL_RETYPE":
        form.pop("_email_candidate", None)
        await r.set(f"user:{psid}:guest_form", json.dumps(form), ex=3600)
        return [mb.text("Please enter your email address again: 📧")], "filling_guest_form"

    # ── phone ──────────────────────────────────────────────────────────────────
    elif field == "phone":
        valid, normalized, error = _validate_phone(text)
        if not valid:
            return [mb.text(
                f"❌ {error}\nEnter your phone number with country code.\n"
                "e.g. +1 555 123 4567 (US) or +44 7700 900123 (UK):"
            )], "filling_guest_form"
        form["phone"] = normalized

    # ── button fields ──────────────────────────────────────────────────────────
    elif field == "trip_purpose":
        if not text.startswith("PURPOSE_"):
            return [mb.quick_replies("What brings you here? ✈️", _TRIP_PURPOSE)], "filling_guest_form"
        form["trip_purpose"] = text.replace("PURPOSE_", "")

    elif field == "dietary":
        if text == "DIET_other":
            form["dietary"] = "_asking_other"
            await r.set(f"user:{psid}:guest_form", json.dumps(form), ex=3600)
            return [mb.text("Please describe your dietary requirement:")], "filling_guest_form"
        if form.get("dietary") == "_asking_other":
            form["dietary"] = text  # free text
        elif not text.startswith("DIET_"):
            # Might be free text answer for "other"
            form["dietary"] = text
        else:
            form["dietary"] = text.replace("DIET_", "")

    elif field == "accessibility":
        if text == "ACCESS_other":
            form["accessibility"] = "_asking_other"
            await r.set(f"user:{psid}:guest_form", json.dumps(form), ex=3600)
            return [mb.text("Please describe your accessibility need:")], "filling_guest_form"
        if form.get("accessibility") == "_asking_other":
            form["accessibility"] = text
        elif not text.startswith("ACCESS_"):
            form["accessibility"] = text
        else:
            form["accessibility"] = text.replace("ACCESS_", "")

    # Save updated form
    await r.set(f"user:{psid}:guest_form", json.dumps(form), ex=3600)

    # Get next question
    next_step = _get_current_step(form)
    if next_step is None:
        return await _complete_guest_form(psid, form, lang)

    return await _send_next_question(psid, form, lang)


async def _send_next_question(
    psid: str, form: dict, lang: str
) -> tuple[list[dict], str]:
    """Send the next unanswered question in the form."""
    mb = MessengerResponse(psid)
    next_step = _get_current_step(form)
    draft = await get_booking_draft(psid) or {}
    city = draft.get("city", "your destination")

    if next_step == "name":
        return [mb.text("What's your name? 👤")], "filling_guest_form"

    if next_step == "email":
        return [mb.text("What email should I send your booking confirmation to? 📧")], "filling_guest_form"

    if next_step == "phone":
        return [mb.text(
            "Your phone number? (with country code) 📱\n"
            "e.g. +1 555 123 4567 (US) or +44 7700 900123 (UK)"
        )], "filling_guest_form"

    if next_step == "trip_purpose":
        return [mb.quick_replies(
            f"What brings you to {city}? ✈️", _TRIP_PURPOSE
        )], "filling_guest_form"

    if next_step == "dietary":
        return [mb.quick_replies("Any dietary requirements? 🍽️", _DIETARY)], "filling_guest_form"

    if next_step == "accessibility":
        return [mb.quick_replies("Any accessibility needs? ♿", _ACCESSIBILITY)], "filling_guest_form"

    return await _complete_guest_form(psid, form, lang)


async def _complete_guest_form(
    psid: str, form: dict, lang: str
) -> tuple[list[dict], str]:
    """All fields collected — save to Supabase/Redis and proceed to add-ons."""
    mb = MessengerResponse(psid)
    r  = get_redis()

    # Persist to Supabase guests table
    sb = get_supabase()
    guest_data = {
        "messenger_psid": psid,
        "full_name":      form.get("name", ""),
        "email":          form.get("email", ""),
        "phone":          form.get("phone", ""),
        "trip_purpose":   form.get("trip_purpose", ""),
        "dietary_needs":  form.get("dietary", ""),
        "accessibility":  form.get("accessibility", ""),
    }
    try:
        await sb.table("guests").upsert(guest_data, on_conflict="messenger_psid").execute()
    except Exception as exc:
        log.error("Guest save failed: %s", exc)

    # Update profile cache
    profile = await get_user_profile(psid) or {}
    profile["name"]  = form.get("name", profile.get("name"))
    profile["email"] = form.get("email", profile.get("email"))
    profile["phone"] = form.get("phone", profile.get("phone"))
    await set_user_profile(psid, profile)

    # Store in booking draft
    draft = await get_booking_draft(psid) or {}
    draft["guest"] = guest_data
    await set_booking_draft(psid, draft)

    # Clear form scratchpad
    await r.delete(f"user:{psid}:guest_form")
    await set_user_state(psid, "selecting_addons")

    msgs = mb.send_sequence([
        mb.text(f"Perfect, {form.get('name','').split()[0]}! Your details are saved. ✅"),
        mb.text("Let me suggest some extras to make your stay special 🎁"),
    ])
    # Trigger add-on flow
    from hf_space.routers.addons import handle_addons_start
    addon_msgs, addon_state = await handle_addons_start(psid, lang)
    return msgs + addon_msgs, addon_state


def _get_current_step(form: dict) -> str | None:
    """Return name of the next unanswered (or empty-string) form field."""
    for step in _STEPS:
        if not form.get(step) or form.get(step) == "_asking_other":
            return step
    return None  # all done


# ── Validation helpers ─────────────────────────────────────────────────────────

def _validate_name(text: str) -> tuple[bool, str, str]:
    """Returns (valid, normalized, error_msg)."""
    t = text.strip()
    # Remove numbers
    if any(c.isdigit() for c in t):
        return False, t, "Name should not contain numbers."
    parts = t.split()
    if len(parts) < 2:
        return False, t, "Please enter your full name (first and last)."
    # Title-case
    normalized = " ".join(p.capitalize() for p in parts)
    return True, normalized, ""


def _validate_email(text: str) -> tuple[bool, str, str | None]:
    """Returns (valid, normalized/corrected, suggestion_domain_or_None)."""
    email = text.strip().lower()
    # Basic RFC-5322-ish regex
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        return False, email, None

    # Common domain typos
    domain_fixes = {
        "gmail.con": "gmail.com",
        "gmail.co": "gmail.com",
        "gmial.com": "gmail.com",
        "hotmail.con": "hotmail.com",
        "yahoo.con": "yahoo.com",
        "outlool.com": "outlook.com",
        "outllok.com": "outlook.com",
    }
    local, domain = email.rsplit("@", 1)
    if domain in domain_fixes:
        corrected = f"{local}@{domain_fixes[domain]}"
        return True, corrected, corrected  # (valid but with suggestion)

    return True, email, None


def _validate_phone(text: str) -> tuple[bool, str, str]:
    """Returns (valid, e164_normalized, error_msg)."""
    try:
        import phonenumbers
        parsed = phonenumbers.parse(text.strip(), None)
        if phonenumbers.is_valid_number(parsed):
            normalized = phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
            return True, normalized, ""
        return False, text, "Phone number is not valid for the given country code."
    except ImportError:
        # phonenumbers library not available — basic check
        cleaned = re.sub(r"[\s\-\(\)]", "", text.strip())
        if re.match(r"^\+[1-9]\d{7,14}$", cleaned):
            return True, cleaned, ""
        return False, text, "Include your country code, e.g. +1 555 123 4567."
    except Exception:
        return False, text, "Could not parse phone number. Include country code."


# ── API endpoints ──────────────────────────────────────────────────────────────

class ValidateRequest(BaseModel):
    field:   str
    value:   str
    context: dict = {}


class OCRRequest(BaseModel):
    image_url: str
    psid:      str


class SaveProfileRequest(BaseModel):
    psid:         str
    full_name:    str | None = None
    email:        str | None = None
    phone:        str | None = None
    trip_purpose: str | None = None


@router.post("/validate")
async def validate_guest_field(req: ValidateRequest) -> dict:
    """POST /api/guest/validate — field-level validation with normalisation."""
    if req.field == "email":
        valid, normalized, suggestion = _validate_email(req.value)
        return {"valid": valid, "normalized": normalized, "suggestion": suggestion, "error": None if valid else "Invalid email"}
    if req.field == "name":
        valid, normalized, error = _validate_name(req.value)
        return {"valid": valid, "normalized": normalized, "error": error or None}
    if req.field == "phone":
        valid, normalized, error = _validate_phone(req.value)
        return {"valid": valid, "normalized": normalized, "error": error or None}
    return {"valid": True, "normalized": req.value, "error": None}


@router.post("/ocr_passport")
async def ocr_passport(req: OCRRequest) -> dict:
    """POST /api/guest/ocr_passport — MRZ extraction from passport image."""
    from services.guest_service.ocr import extract_mrz
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(req.image_url)
            resp.raise_for_status()
            image_bytes = resp.content
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    result = extract_mrz(image_bytes)
    return result


@router.post("/save_profile")
async def save_guest_profile(req: SaveProfileRequest) -> dict:
    """POST /api/guest/save_profile — upsert guest to Supabase + Redis."""
    sb = get_supabase()
    data = {k: v for k, v in {
        "messenger_psid": req.psid,
        "full_name":      req.full_name,
        "email":          req.email,
        "phone":          req.phone,
        "trip_purpose":   req.trip_purpose,
    }.items() if v is not None}

    try:
        await sb.table("guests").upsert(data, on_conflict="messenger_psid").execute()
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    profile = await get_user_profile(req.psid) or {}
    if req.full_name:
        profile["name"]  = req.full_name
    if req.email:
        profile["email"] = req.email
    if req.phone:
        profile["phone"] = req.phone
    await set_user_profile(req.psid, profile)
    return {"success": True}
