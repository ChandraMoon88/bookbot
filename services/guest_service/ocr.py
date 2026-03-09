"""
services/guest_service/ocr.py
--------------------------------
Passport image OCR using Tesseract + OpenCV.
Extracts MRZ data and validates ISO 7501 checksums.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def _mrz_checksum(data: str) -> int:
    """ISO 7501 MRZ checksum."""
    weights = [7, 3, 1]
    total   = 0
    for i, ch in enumerate(data):
        if ch.isdigit():
            val = int(ch)
        elif ch.isalpha():
            val = ord(ch.upper()) - 55
        elif ch == "<":
            val = 0
        else:
            val = 0
        total += val * weights[i % 3]
    return total % 10


def _validate_mrz_check(field: str, check_char: str) -> bool:
    return str(_mrz_checksum(field)) == check_char


def extract_from_image(image_bytes: bytes, image_format: str = "jpeg") -> dict:
    """
    Extract guest info from passport image.
    Returns: { first_name, last_name, passport_number, expiry, confidence }
    """
    try:
        import cv2
        import numpy as np
        import pytesseract

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"confidence": 0.0, "error": "Could not decode image"}

        # Pre-process: grayscale → deskew → sharpen
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharp = cv2.filter2D(gray, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
        thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # OCR
        text = pytesseract.image_to_string(thresh, config="--psm 6")
        lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 44]

        if len(lines) < 2:
            return {"confidence": 0.3, "error": "MRZ lines not found in image"}

        mrz1, mrz2 = lines[-2], lines[-1]

        # Parse MRZ line 2
        passport_number = mrz2[0:9].replace("<", "").strip()
        birth_date      = mrz2[13:19]
        expiry_raw      = mrz2[19:25]
        check_exp       = mrz2[25]

        # Convert expiry YY-MM-DD
        year  = int(expiry_raw[:2])
        full_year = 2000 + year if year < 50 else 1900 + year
        expiry = f"{full_year}-{expiry_raw[2:4]}-{expiry_raw[4:6]}"

        # Validate expiry checksum
        check_ok = _validate_mrz_check(expiry_raw, check_exp)

        # Parse names from MRZ line 1
        name_field = mrz1[5:44]
        parts = name_field.split("<<", 1)
        last_name  = parts[0].replace("<", " ").strip().title()
        first_name = parts[1].replace("<", " ").strip().title() if len(parts) > 1 else ""

        confidence = 0.9 if check_ok else 0.6

        return {
            "first_name":       first_name,
            "last_name":        last_name,
            "passport_number":  passport_number,
            "expiry":           expiry,
            "confidence":       confidence,
        }

    except ImportError:
        logger.warning("OpenCV / Tesseract not installed — OCR unavailable")
        return {"confidence": 0.0, "error": "OCR dependencies not installed"}
    except Exception as e:
        logger.error("OCR error: %s", e)
        return {"confidence": 0.0, "error": str(e)}
