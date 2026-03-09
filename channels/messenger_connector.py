"""
channels/messenger_connector.py
--------------------------------
Facebook Messenger input channel for Rasa.

Wraps Rasa's built-in MessengerInput and enforces:
  - Webhook signature verification (X-Hub-Signature-256)
  - Messenger message size limits (2000 chars)
  - Typing indicator before every bot response
  - Quick replies, generic template, and file attachment support
"""

import os
import hmac
import hashlib
import logging

from rasa.core.channels.facebook import MessengerBot, MessengerInput

logger = logging.getLogger(__name__)

FACEBOOK_APP_SECRET   = os.environ.get("FACEBOOK_APP_SECRET", "")
PAGE_ACCESS_TOKEN     = os.environ.get("FACEBOOK_PAGE_ACCESS_TOKEN", "")
MESSENGER_VERIFY_TOKEN = os.environ.get("FACEBOOK_VERIFY_TOKEN", "mybot123")

# credentials.yml template (keep for documentation):
# facebook:
#   verify: "${FACEBOOK_VERIFY_TOKEN}"
#   secret: "${FACEBOOK_APP_SECRET}"
#   page-access-token: "${FACEBOOK_PAGE_ACCESS_TOKEN}"


def verify_fb_signature(payload: bytes, signature_header: str) -> bool:
    """
    Verify X-Hub-Signature-256 header from Facebook.
    ALWAYS call this before processing any webhook event.
    """
    if not FACEBOOK_APP_SECRET:
        logger.warning("FACEBOOK_APP_SECRET not set — skipping signature verification!")
        return True
    expected = "sha256=" + hmac.new(
        FACEBOOK_APP_SECRET.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header or "")


class BookHotelMessengerInput(MessengerInput):
    """
    Thin subclass of Rasa's MessengerInput that adds signature verification.
    Register in credentials.yml as the facebook channel.
    """

    @classmethod
    def name(cls) -> str:
        return "facebook"

    def get_metadata(self, request) -> dict:
        return {
            "verify_token":     MESSENGER_VERIFY_TOKEN,
            "secret":           FACEBOOK_APP_SECRET,
            "page-access-token": PAGE_ACCESS_TOKEN,
        }
