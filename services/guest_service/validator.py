"""
services/guest_service/validator.py
--------------------------------------
Input validation for guest information fields.
"""

from __future__ import annotations

import re
from datetime import date, datetime


def validate_email(value: str) -> dict:
    """Validate email address. Checks RFC-5322 pattern + common typos."""
    email = value.strip().lower()
    pattern = r'^[^@\s]+@[^@\s]+\.[^@\s]{2,}$'
    if not re.match(pattern, email):
        return {"valid": False, "error": "Invalid email format", "normalized": ""}

    # Common typo domains
    typos = {"gmial.com", "gmai.com", "gmail.con", "yahooo.com", "hotmial.com"}
    domain = email.split("@")[1]
    if domain in typos:
        return {"valid": False, "error": f"Did you mean to type '{domain}'?", "normalized": email}

    return {"valid": True, "normalized": email, "error": ""}


def validate_phone(value: str) -> dict:
    """Validate phone number and normalize to E.164 format."""
    cleaned = re.sub(r'[^\d\+]', '', value.strip())
    if len(cleaned) < 7:
        return {"valid": False, "error": "Phone number too short", "normalized": ""}
    try:
        import phonenumbers
        p = phonenumbers.parse(cleaned, None)
        if phonenumbers.is_valid_number(p):
            normalized = phonenumbers.format_number(p, phonenumbers.PhoneNumberFormat.E164)
            return {"valid": True, "normalized": normalized, "error": ""}
    except Exception:
        pass
    # Fallback: accept any cleaned number ≥10 digits
    if len(cleaned) >= 10:
        return {"valid": True, "normalized": cleaned, "error": ""}
    return {"valid": False, "error": "Invalid phone number", "normalized": ""}


def validate_name(value: str) -> dict:
    """Validate guest name."""
    name = value.strip()
    if len(name) < 2:
        return {"valid": False, "error": "Name must be at least 2 characters"}
    if re.match(r'^\d+$', name):
        return {"valid": False, "error": "Name cannot be all numbers"}
    return {"valid": True, "normalized": " ".join(w.capitalize() for w in name.split())}


def validate_passport(number: str, expiry: str, check_out: str) -> dict:
    """Validate passport: expiry must be > check_out + 6 months."""
    number = number.strip().upper()
    if len(number) < 6:
        return {"valid": False, "error": "Passport number too short"}
    try:
        exp   = datetime.strptime(expiry, "%Y-%m-%d").date()
        cout  = datetime.strptime(check_out, "%Y-%m-%d").date()
        from dateutil.relativedelta import relativedelta
        min_exp = cout + relativedelta(months=6)
        if exp < min_exp:
            return {"valid": False, "error": f"Passport must be valid until {min_exp} (6 months after check-out)"}
    except Exception:
        return {"valid": False, "error": "Invalid passport expiry date format (use YYYY-MM-DD)"}
    return {"valid": True, "normalized": number}
