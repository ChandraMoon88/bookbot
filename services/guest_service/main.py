"""
services/guest_service/main.py
---------------------------------
FastAPI guest validation microservice.
"""

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional

from .validator import validate_email, validate_phone, validate_name, validate_passport
from .ocr import extract_from_image

app = FastAPI(title="Guest Service")


class ValidateRequest(BaseModel):
    field_name: str
    value:      str
    context:    Optional[dict] = None


@app.post("/validate")
def validate(req: ValidateRequest):
    fn = req.field_name
    v  = req.value
    ctx = req.context or {}

    if fn == "email":
        return validate_email(v)
    if fn == "phone":
        return validate_phone(v)
    if fn == "name":
        return validate_name(v)
    if fn == "passport":
        return validate_passport(
            v,
            ctx.get("expiry", "2099-01-01"),
            ctx.get("check_out", "2099-01-01"),
        )
    return {"valid": True, "normalized": v, "error": ""}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    image_bytes = await file.read()
    fmt = (file.content_type or "jpeg").split("/")[-1]
    return extract_from_image(image_bytes, fmt)


@app.get("/health")
def health():
    return {"status": "ok", "service": "guest_service"}
