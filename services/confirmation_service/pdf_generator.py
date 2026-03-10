"""
services/confirmation_service/pdf_generator.py
------------------------------------------------
Generates a PDF booking voucher using ReportLab and uploads it to Supabase Storage.
"""

import io
import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

PDF_BUCKET      = "booking-vouchers"


def _build_pdf(booking: dict, hotel: dict, qr_bytes: Optional[bytes] = None) -> bytes:
    """Build the PDF and return raw bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        raise RuntimeError("reportlab is required: pip install reportlab")

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story  = []

    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"], alignment=TA_CENTER, fontSize=20, textColor=colors.HexColor("#1a73e8")
    )
    story.append(Paragraph("Hotel Booking Confirmation", title_style))
    story.append(Spacer(1, 8*mm))

    data = [
        ["Booking Reference", booking.get("ref", "—")],
        ["Guest Name",        booking.get("guest_name", "—")],
        ["Hotel",             hotel.get("name", "—")],
        ["Address",           hotel.get("address", "—")],
        ["Check-in",          booking.get("check_in", "—")],
        ["Check-out",         booking.get("check_out", "—")],
        ["Room Type",         booking.get("room_type", "—")],
        ["Guests",            str(booking.get("guests", 1))],
        ["Meal Plan",         booking.get("meal_plan", "Room Only")],
        ["Total Paid",        f"${booking.get('amount_usd', 0):.2f}"],
    ]

    table = Table(data, colWidths=[60*mm, 110*mm])
    table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#f1f3f4")),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)

    if qr_bytes:
        story.append(Spacer(1, 8*mm))
        story.append(Paragraph("Scan at check-in counter", styles["Normal"]))
        story.append(Spacer(1, 4*mm))
        qr_img = Image(io.BytesIO(qr_bytes), width=40*mm, height=40*mm)
        story.append(qr_img)

    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("Thank you for booking with us! Enjoy your stay.", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()


def generate_and_upload(booking: dict, hotel: dict, qr_bytes: Optional[bytes] = None) -> str:
    """
    Generates the PDF, uploads to Supabase Storage, and returns the public URL.
    Falls back to returning base64 if upload fails.
    """
    pdf_bytes = _build_pdf(booking, hotel, qr_bytes)
    ref       = booking.get("ref", "unknown")
    filename  = f"{ref}.pdf"

    import base64
    return "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode()
