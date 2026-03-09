"""
services/confirmation_service/main.py
---------------------------------------
FastAPI booking confirmation microservice.
Generates PDF, QR code, and ICS calendar for a booking.
"""

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from .qr_generator import generate as generate_qr
from .pdf_generator import generate_and_upload as generate_pdf
from .calendar import generate as generate_ics

app = FastAPI(title="Confirmation Service")


class BookingInfo(BaseModel):
    ref:         str
    guest_name:  str
    hotel_name:  str
    hotel_id:    Optional[str] = None
    address:     Optional[str] = ""
    check_in:    str   # YYYY-MM-DD
    check_out:   str   # YYYY-MM-DD
    room_type:   Optional[str] = "Standard"
    guests:      Optional[int] = 1
    meal_plan:   Optional[str] = "Room Only"
    amount_usd:  Optional[float] = 0.0


@app.post("/confirm")
def confirm(booking: BookingInfo):
    """
    Returns {pdf_url, qr_base64, ics_text}.
    Caller (notification service) sends these to the guest.
    """
    booking_dict = booking.dict()
    hotel_dict   = {
        "name":    booking.hotel_name,
        "address": booking.address,
    }

    # QR code
    qr_bytes = generate_qr(booking.ref)

    # PDF (uploads to Supabase Storage)
    pdf_url = generate_pdf(booking_dict, hotel_dict, qr_bytes)

    # ICS
    ics_text = generate_ics(booking_dict)

    import base64
    return {
        "pdf_url":    pdf_url,
        "qr_base64":  base64.b64encode(qr_bytes).decode(),
        "ics_text":   ics_text,
    }


@app.get("/qr/{booking_ref}")
def qr_png(booking_ref: str):
    png = generate_qr(booking_ref)
    return Response(content=png, media_type="image/png")


@app.get("/ics/{booking_ref}")
def ics_file(booking_ref: str, guest_name: str = "Guest", hotel_name: str = "Hotel",
             check_in: str = "2025-01-01", check_out: str = "2025-01-02"):
    ics = generate_ics({
        "ref": booking_ref, "guest_name": guest_name, "hotel_name": hotel_name,
        "check_in": check_in, "check_out": check_out, "room_type": "Standard",
    })
    return Response(
        content=ics,
        media_type="text/calendar",
        headers={"Content-Disposition": f'attachment; filename="{booking_ref}.ics"'},
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "confirmation_service"}
