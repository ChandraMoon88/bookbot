#!/usr/bin/env python3
"""
scripts/setup_messenger_menu.py
---------------------------------
Sets up the Facebook Messenger Persistent Menu for the bot page.
Run once per page or whenever menu items change.

Usage:
    python scripts/setup_messenger_menu.py
"""

import os
import json
import urllib.request

PAGE_ACCESS_TOKEN = os.environ["PAGE_ACCESS_TOKEN"]
META_GRAPH_URL    = "https://graph.facebook.com/v18.0/me/messenger_profile"

MENU = {
    "persistent_menu": [
        {
            "locale":                 "default",
            "composer_input_disabled": False,
            "call_to_actions": [
                {"type": "postback", "title": "🏨 Book a Hotel",        "payload": "BOOK_HOTEL"},
                {"type": "postback", "title": "📋 My Bookings",          "payload": "MY_BOOKINGS"},
                {
                    "type":  "nested",
                    "title": "⚙️ Manage Booking",
                    "call_to_actions": [
                        {"type": "postback", "title": "✏️ Modify Booking",    "payload": "MODIFY_BOOKING"},
                        {"type": "postback", "title": "❌ Cancel Booking",     "payload": "CANCEL_BOOKING"},
                        {"type": "postback", "title": "📧 Resend Confirmation","payload": "RESEND_CONFIRM"},
                    ],
                },
                {"type": "postback", "title": "🌐 Change Language",    "payload": "CHANGE_LANGUAGE"},
                {"type": "postback", "title": "🤝 Talk to Agent",       "payload": "HUMAN_HANDOFF"},
            ],
        }
    ],
    "get_started": {"payload": "GET_STARTED"},
    "greeting": [
        {"locale": "default", "text": "Hello {{user_first_name}}! I'm your hotel booking assistant. How can I help you today?"},
        {"locale": "hi_IN",   "text": "नमस्ते {{user_first_name}}! मैं आपका होटल बुकिंग सहायक हूं।"},
    ],
}


def setup():
    data    = json.dumps(MENU).encode()
    headers = {
        "Content-Type":  "application/json",
    }
    url = f"{META_GRAPH_URL}?access_token={PAGE_ACCESS_TOKEN}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print("Messenger menu set:", result)


if __name__ == "__main__":
    setup()
