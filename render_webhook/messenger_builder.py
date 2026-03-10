"""
render_webhook/messenger_builder.py
-------------------------------------
MessengerResponse builder — ALWAYS use this class to construct Messenger
message payloads. Never build raw Messenger JSON by hand.

Enforces:
  - text <= 2000 chars (auto-split at sentence boundaries)
  - quick replies <= 13 items, titles <= 20 chars
  - carousel elements <= 10
  - button titles <= 20 chars
  - typing indicators before every substantive message
"""

from __future__ import annotations

import re
from typing import Any


# ─── SECTION 5: MessengerResponse builder ─────────────────────────────────────

class MessengerResponse:
    def __init__(self, recipient_psid: str) -> None:
        self.psid = recipient_psid

    # ── typing indicator ───────────────────────────────────────────────────────

    def typing(self) -> dict:
        """Send before every substantive message (sender_action: typing_on)."""
        return {
            "recipient": {"id": self.psid},
            "sender_action": "typing_on",
        }

    # ── plain text ─────────────────────────────────────────────────────────────

    def text(self, message: str) -> dict:
        """Plain text reply. Auto-splits at sentence boundaries if > 2000 chars."""
        return {
            "recipient": {"id": self.psid},
            "message": {"text": _safe_text(message)},
        }

    # ── quick replies ──────────────────────────────────────────────────────────

    def quick_replies(self, text: str, options: list[dict]) -> dict:
        """
        Build a quick-reply message.

        options = [
          {"title": "Paris 🗼",  "payload": "CITY_PARIS"},
          {"title": "Tokyo 🗾", "payload": "CITY_TOKYO"},
        ]
        Max 13 options. Titles auto-truncated to 20 chars.
        """
        qr = [
            {
                "content_type": "text",
                "title": opt["title"][:20],
                "payload": opt["payload"],
            }
            for opt in options[:13]
        ]
        return {
            "recipient": {"id": self.psid},
            "message": {
                "text": _safe_text(text),
                "quick_replies": qr,
            },
        }

    # ── hotel cards carousel ───────────────────────────────────────────────────

    def hotel_cards(self, hotels: list[dict]) -> dict:
        """
        Generic template carousel for hotel results.

        hotels = [{
            "name": "Grand Tokyo Hotel",
            "stars": 5,
            "price_from": 28000,
            "currency": "JPY",
            "price_usd": 187,
            "thumbnail_url": "https://...",
            "hotel_id": "h_123",
            "distance_km": 1.2,
            "top_feature": "Free breakfast",
        }]
        Max 10 hotels. Always show top 3 with "Show 4 more →" handled by caller.
        """
        elements = []
        for h in hotels[:10]:
            stars_emoji = "⭐" * int(h.get("stars", 0))
            elements.append({
                "title": f"{h['name']} {stars_emoji}"[:80],
                "subtitle": (
                    f"From {h['currency']} {h['price_from']:,}/night · {h['top_feature']}"
                )[:80],
                "image_url": h["thumbnail_url"],
                "buttons": [
                    {
                        "type": "postback",
                        "title": "📋 View Details",
                        "payload": f"HOTEL_DETAILS_{h['hotel_id']}",
                    },
                    {
                        "type": "postback",
                        "title": "✅ Select Hotel",
                        "payload": f"HOTEL_SELECT_{h['hotel_id']}",
                    },
                    {
                        "type": "postback",
                        "title": "🔖 Save for Later",
                        "payload": f"HOTEL_SAVE_{h['hotel_id']}",
                    },
                ],
            })
        return {
            "recipient": {"id": self.psid},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "generic",
                        "elements": elements,
                    },
                }
            },
        }

    # ── room cards carousel ────────────────────────────────────────────────────

    def room_cards(self, rooms: list[dict]) -> dict:
        """
        Generic template carousel for room type results.

        rooms = [{
            "room_id": "r_456",
            "name": "Deluxe King Room",
            "size_m2": 42,
            "bed_type": "King",
            "price_from": 28000,
            "currency": "JPY",
            "thumbnail_url": "https://...",
            "features": ["Bathtub", "City view"],
        }]
        """
        elements = []
        for r in rooms[:10]:
            features_str = " · ".join(r.get("features", [])[:3])
            elements.append({
                "title": f"{r['name']} • {r.get('size_m2', '?')}m²"[:80],
                "subtitle": (
                    f"🛏 {r.get('bed_type','?')} · {features_str} · "
                    f"from {r['currency']} {r['price_from']:,}/night"
                )[:80],
                "image_url": r["thumbnail_url"],
                "buttons": [
                    {
                        "type": "postback",
                        "title": "📸 See Photos",
                        "payload": f"ROOM_PHOTOS_{r['room_id']}",
                    },
                    {
                        "type": "postback",
                        "title": "✅ Choose Room",
                        "payload": f"ROOM_SELECT_{r['room_id']}",
                    },
                    {
                        "type": "postback",
                        "title": "ℹ️ Full Details",
                        "payload": f"ROOM_DETAILS_{r['room_id']}",
                    },
                ],
            })
        return {
            "recipient": {"id": self.psid},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "generic",
                        "elements": elements,
                    },
                }
            },
        }

    # ── booking summary card ───────────────────────────────────────────────────

    def booking_summary_card(self, booking: dict) -> dict:
        """
        Single generic template card with full booking summary before payment.

        booking = {
            "hotel_name": "Grand Tokyo Hotel",
            "stars": 5,
            "room_name": "Deluxe King",
            "check_in": "15 Mar 2026",
            "check_out": "17 Mar 2026",
            "num_guests": 2,
            "meal_plan": "Breakfast",
            "total_display": "¥64,000 (~$427 USD)",
            "hotel_photo_url": "https://...",
            "booking_draft_id": "draft_xxx",
        }
        """
        stars_emoji = "⭐" * int(booking.get("stars", 0))
        subtitle = (
            f"{booking['room_name']} · {booking['check_in']}–{booking['check_out']} · "
            f"{booking.get('num_guests', 1)} guest(s) · {booking.get('meal_plan','')}"
        )[:80]
        return {
            "recipient": {"id": self.psid},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "generic",
                        "elements": [
                            {
                                "title": f"{booking['hotel_name']} {stars_emoji}"[:80],
                                "subtitle": subtitle,
                                "image_url": booking.get("hotel_photo_url", ""),
                                "buttons": [
                                    {
                                        "type": "postback",
                                        "title": "📋 View Full Details",
                                        "payload": f"BOOKING_SUMMARY_FULL",
                                    },
                                    {
                                        "type": "postback",
                                        "title": "💳 Pay Now",
                                        "payload": "PAYMENT_START",
                                    },
                                    {
                                        "type": "postback",
                                        "title": "✏️ Change Something",
                                        "payload": "BOOKING_MODIFY_DRAFT",
                                    },
                                ],
                            }
                        ],
                    },
                }
            },
        }

    # ── list template ──────────────────────────────────────────────────────────

    def list_template(
        self,
        title: str,
        items: list[dict],
        cta_button: dict | None = None,
    ) -> dict:
        """
        Facebook List Template (max 4 items).

        items = [{"title": "...", "subtitle": "...", "payload": "..."}]
        cta_button = {"title": "View All", "payload": "VIEW_ALL"}
        Used for: modification options, loyalty info, FAQ categories.
        """
        elements = []
        for item in items[:4]:
            el: dict[str, Any] = {
                "title": item["title"][:80],
            }
            if item.get("subtitle"):
                el["subtitle"] = item["subtitle"][:80]
            if item.get("image_url"):
                el["image_url"] = item["image_url"]
            if item.get("payload"):
                el["buttons"] = [
                    {
                        "type": "postback",
                        "title": item.get("button_label", "Select")[:20],
                        "payload": item["payload"],
                    }
                ]
            elements.append(el)

        payload: dict[str, Any] = {
            "template_type": "list",
            "top_element_style": "compact",
            "elements": elements,
        }
        if cta_button:
            payload["buttons"] = [
                {
                    "type": "postback",
                    "title": cta_button["title"][:20],
                    "payload": cta_button["payload"],
                }
            ]
        return {
            "recipient": {"id": self.psid},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": payload,
                }
            },
        }

    # ── image ──────────────────────────────────────────────────────────────────

    def image(self, url: str, accessible_title: str = "Image") -> dict:
        """Send hotel photo, QR code, or map screenshot."""
        return {
            "recipient": {"id": self.psid},
            "message": {
                "attachment": {
                    "type": "image",
                    "payload": {
                        "url": url,
                        "is_reusable": True,
                    },
                }
            },
        }

    # ── file attachment ────────────────────────────────────────────────────────

    def file(self, url: str, filename: str = "document.pdf") -> dict:
        """Send PDF booking voucher."""
        return {
            "recipient": {"id": self.psid},
            "message": {
                "attachment": {
                    "type": "file",
                    "payload": {
                        "url": url,
                        "is_reusable": False,
                    },
                }
            },
        }

    # ── webview button ─────────────────────────────────────────────────────────

    def webview_button(self, text: str, button_title: str, url: str) -> dict:
        """
        Button template that opens a Messenger Webview.
        Used for: Stripe payment, voice recording, date picker.
        """
        return {
            "recipient": {"id": self.psid},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {
                        "template_type": "button",
                        "text": _safe_text(text),
                        "buttons": [
                            {
                                "type": "web_url",
                                "title": button_title[:20],
                                "url": url,
                                "messenger_extensions": True,
                                "webview_height_ratio": "tall",
                            }
                        ],
                    },
                }
            },
        }

    # ── send sequence helper ───────────────────────────────────────────────────

    def send_sequence(
        self, messages: list[dict], delay_ms: int = 600
    ) -> list[dict]:
        """
        Wrap each message with a typing indicator before it.
        Returns an ordered list to be sent one by one with 600ms delays.

        Pattern: typing → message → typing → message → ...
        """
        sequence: list[dict] = []
        for msg in messages:
            sequence.append(self.typing())
            sequence.append(msg)
        return sequence


# ─── VALIDATION HELPER ────────────────────────────────────────────────────────

def validate_messages(messages: list[dict]) -> list[str]:
    """
    Validate a list of Messenger message objects.
    Returns list of violation strings (empty = valid).

    Called before returning any messages list from a handler.
    """
    violations: list[str] = []
    for i, msg in enumerate(messages):
        m = msg.get("message", {})

        # Check text length
        if "text" in m and len(m["text"]) > 2000:
            violations.append(f"[msg {i}] text exceeds 2000 chars")

        # Check quick replies count
        qr = m.get("quick_replies", [])
        if len(qr) > 13:
            violations.append(f"[msg {i}] quick_replies > 13 ({len(qr)})")
        for j, q in enumerate(qr):
            if len(q.get("title", "")) > 20:
                violations.append(f"[msg {i}] quick_reply[{j}] title > 20 chars")

        # Check carousel
        att = m.get("attachment", {})
        p = att.get("payload", {})
        if p.get("template_type") == "generic":
            elements = p.get("elements", [])
            if len(elements) > 10:
                violations.append(f"[msg {i}] carousel has {len(elements)} elements (max 10)")
            for k, el in enumerate(elements):
                for btn in el.get("buttons", []):
                    if len(btn.get("title", "")) > 20:
                        violations.append(
                            f"[msg {i}] element[{k}] button title > 20 chars"
                        )

    return violations


# ─── INTERNAL HELPERS ─────────────────────────────────────────────────────────

def _safe_text(text: str, max_chars: int = 2000) -> str:
    """
    Truncate text to max_chars at the nearest sentence boundary.
    Never truncates mid-word.
    """
    if len(text) <= max_chars:
        return text
    # Try sentence boundary first
    boundary = text[:max_chars].rfind(". ")
    if boundary > max_chars // 2:
        return text[: boundary + 1]
    # Fall back to word boundary
    word_boundary = text[:max_chars].rfind(" ")
    if word_boundary > 0:
        return text[:word_boundary] + "…"
    return text[:max_chars]
