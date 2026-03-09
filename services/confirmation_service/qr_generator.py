"""
services/confirmation_service/qr_generator.py
-----------------------------------------------
Generates a QR code PNG for the booking reference.
300×300 pixels, error correction level H (30% damage resilience).
"""

import io
import logging

log = logging.getLogger(__name__)


def generate(booking_ref: str, size: int = 300) -> bytes:
    """
    Returns raw PNG bytes for a QR code encoding the booking reference.
    """
    try:
        import qrcode
        from qrcode.constants import ERROR_CORRECT_H
    except ImportError:
        raise RuntimeError("qrcode is required: pip install qrcode[pil]")

    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(booking_ref)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img = img.resize((size, size))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
