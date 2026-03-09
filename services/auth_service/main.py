"""
services/auth_service/main.py
--------------------------------
Lightweight JWT-based auth microservice for hotel staff dashboard.
Issues short-lived access tokens + long-lived refresh tokens.
Passwords are hashed with bcrypt.  Never stores plain text.
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional

log = logging.getLogger(__name__)
app = FastAPI(title="Auth Service")

JWT_SECRET     = os.environ.get("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION")
ACCESS_EXPIRE  = int(os.environ.get("JWT_ACCESS_EXPIRE_MINUTES",   "60"))
REFRESH_EXPIRE = int(os.environ.get("JWT_REFRESH_EXPIRE_DAYS",     "30"))

bearer_scheme = HTTPBearer()


def _encode(payload: dict) -> str:
    try:
        import jwt
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    except ImportError:
        raise RuntimeError("PyJWT is required: pip install PyJWT")


def _decode(token: str) -> dict:
    try:
        import jwt
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def _hash_password(password: str) -> str:
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        raise RuntimeError("bcrypt is required: pip install bcrypt")


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        import bcrypt
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ImportError:
        raise RuntimeError("bcrypt is required")


class LoginRequest(BaseModel):
    email:    str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


def _get_staff_from_db(email: str) -> Optional[dict]:
    """Fetch staff record from Supabase."""
    SUPA = os.environ.get("SUPABASE_URL", "")
    KEY  = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    import urllib.request
    url  = f"{SUPA}/rest/v1/staff_users?email=eq.{email}&select=*&limit=1"
    req  = urllib.request.Request(url, headers={
        "apikey": KEY, "Authorization": f"Bearer {KEY}"
    })
    with urllib.request.urlopen(req) as resp:
        rows = json.loads(resp.read())
    return rows[0] if rows else None


@app.post("/login")
def login(req: LoginRequest):
    staff = _get_staff_from_db(req.email)
    if not staff or not _verify_password(req.password, staff["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    now = datetime.now(timezone.utc)
    access_payload = {
        "sub":   staff["id"],
        "email": staff["email"],
        "role":  staff.get("role", "staff"),
        "exp":   now + timedelta(minutes=ACCESS_EXPIRE),
        "iat":   now,
    }
    refresh_payload = {
        "sub":  staff["id"],
        "type": "refresh",
        "exp":  now + timedelta(days=REFRESH_EXPIRE),
        "iat":  now,
    }
    return {
        "access_token":  _encode(access_payload),
        "refresh_token": _encode(refresh_payload),
        "token_type":    "bearer",
        "expires_in":    ACCESS_EXPIRE * 60,
    }


@app.post("/refresh")
def refresh(req: RefreshRequest):
    payload = _decode(req.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")

    now = datetime.now(timezone.utc)
    new_access = {
        "sub": payload["sub"],
        "exp": now + timedelta(minutes=ACCESS_EXPIRE),
        "iat": now,
    }
    return {"access_token": _encode(new_access), "token_type": "bearer"}


@app.get("/me")
def me(creds: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    payload = _decode(creds.credentials)
    return {"sub": payload.get("sub"), "email": payload.get("email"), "role": payload.get("role")}


@app.get("/health")
def health():
    return {"status": "ok", "service": "auth_service"}
