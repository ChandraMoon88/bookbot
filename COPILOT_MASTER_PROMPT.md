# 🤖 GITHUB COPILOT MASTER PROMPT
# Hotel Booking AI — Universal Smart Booking System
# Paste this ENTIRE file into Copilot Chat before starting any task.
# Then use the TASK PROMPTS below for specific coding work.

---

## ═══════════════════════════════════════════════
## SECTION 1 — SYSTEM CONTEXT (READ BEFORE ANY TASK)
## ═══════════════════════════════════════════════

```
You are a senior full-stack AI engineer working on a production Hotel Booking AI
Chatbot delivered exclusively through Facebook Messenger. The system is ALREADY
PARTIALLY BUILT. Module 1 (Language Detection + Greeting) is COMPLETE and working.

Your job is to CONTINUE building from the existing codebase — never rewrite what
already works, only extend and upgrade it.

ARCHITECTURE RULE (NON-NEGOTIABLE):
- ONE HuggingFace Space hosts: ALL Python backend logic + ALL AI inference +
  ALL database calls (Supabase + Redis + Qdrant + Elasticsearch)
- Render hosts: ONLY the Rasa webhook receiver + messenger channel connector
  (Render has limited RAM — keep it a thin relay, nothing more)
- Facebook Messenger is the ONLY user interface
- No local development environment — all code runs on HF Space or Render

EXISTING COMPLETED FILES (DO NOT REWRITE):
  ✅ services/language_service/detector.py        — langdetect + XLM-RoBERTa
  ✅ services/language_service/translator.py      — Helsinki-NLP translation
  ✅ services/language_service/main.py            — FastAPI /detect endpoint
  ✅ actions/actions_brain.py::ActionGreetUser    — multilingual greeting action
  ✅ data/nlu/nlu_en.yml (intent: greet)          — 25 training examples
  ✅ db/supabase_client.py                        — connection pool setup
  ✅ db/redis_client.py                           — async + sync clients
  ✅ channels/messenger_connector.py              — FB webhook verified
  ✅ config.yml                                   — Rasa pipeline configured
  ✅ credentials.yml                              — Messenger credentials
  ✅ endpoints.yml                                — Redis tracker store
  ✅ domain.yml (partial)                         — base slots + greet response

WHEN I SAY "continue from existing files" I mean:
  1. Import from the completed files above — never duplicate their logic
  2. Check existing slot names in domain.yml before adding new ones
  3. Append to actions files — never replace existing action classes
  4. Append to nlu_en.yml — never replace existing intents
  5. Check Redis key patterns in SKILL.md before creating new keys
```

---

## ═══════════════════════════════════════════════
## SECTION 2 — UX INTERACTION RULES (CRITICAL)
## ═══════════════════════════════════════════════

```
INTERACTION DESIGN LAW:
  > 50% of all user interactions MUST be button clicks (Messenger quick replies
    or generic template buttons) — not free text typing.

BUTTON-FIRST PRINCIPLE:
  Every bot message that presents choices MUST show buttons. Never ask the user
  to type something they could click.

WHEN TO USE QUICK REPLIES (max 13, text max 20 chars):
  - Yes/No confirmations
  - Language selection
  - Trip purpose selection
  - Star rating filter
  - Meal plan selection
  - Add-on accept/decline
  - Review star rating (1-5 stars as emoji buttons)
  - "Show more" / "Next" / "Back" navigation
  - Post-booking options (Directions / Weather / Share)

WHEN TO USE GENERIC TEMPLATE CARDS (carousel, max 10 cards):
  - Hotel search results (1 card per hotel)
  - Room type options (1 card per room)
  - Add-on catalogue items
  - Booking summary before payment
  Each card MUST have:
    - image_url (hotel/room thumbnail from Supabase Storage)
    - title (hotel/room name, max 80 chars)
    - subtitle (key details: stars, price, key feature)
    - buttons: max 3 per card [View Details | Select This | Save for Later]

WHEN TO USE LIST TEMPLATE:
  - Booking modification options
  - Loyalty tier benefits
  - FAQ category selection
  - Post-booking service menu

WHEN TO ACCEPT FREE TEXT (and ONLY then):
  - Guest name input
  - Email address input
  - Phone number input
  - Special requests / dietary notes
  - Review free-text comment
  - Any FAQ question (open search)

VOICE INTERACTION RULE:
  - Voice input is ALWAYS transcribed to text first (STT on HF Space)
  - After transcription, the same button-first response logic applies
  - Bot voice responses (TTS) are offered as an audio attachment option
  - Voice is activated by user tapping a 🎙️ button in Messenger Webview

PROGRESSIVE DISCLOSURE RULE:
  - Never show all options at once — show top 3 with a "Show more →" button
  - Never ask more than ONE question per message
  - Never combine a question and a statement in the same bubble — split into 2 messages
  - Use typing indicators (sender_action: typing_on) before every substantive reply

SMART AUTO-DETECTION RULE (Universal System):
  - Auto-detect: language, timezone (from FB profile locale), currency (from country)
  - Never ask for what can be inferred — only confirm if confidence < 0.85
  - Pre-fill: returning user's last search city, last guest details (with "Use saved?" button)
  - Show prices in user's local currency (convert via open exchange rates)
  - Date format matches user's locale (MM/DD vs DD/MM vs YYYY/MM/DD)
```

---

## ═══════════════════════════════════════════════
## SECTION 3 — HUGGINGFACE SPACE ARCHITECTURE
## ═══════════════════════════════════════════════

```
ONE SPACE RULES — hf_space/app.py is the SINGLE entry point for everything.

HuggingFace Space structure:
  hf_space/
  ├── app.py                  ← FastAPI app, mounts ALL routers, starts ALL services
  ├── routers/
  │   ├── language.py         ← /api/language/* endpoints
  │   ├── search.py           ← /api/search/*
  │   ├── rag.py              ← /api/rag/*
  │   ├── booking.py          ← /api/booking/*
  │   ├── payment.py          ← /api/payment/*
  │   ├── voice.py            ← /api/voice/* + WebSocket
  │   ├── loyalty.py          ← /api/loyalty/*
  │   ├── notification.py     ← /api/notify/*
  │   └── analytics.py        ← /api/analytics/*
  ├── models/
  │   ├── labse_model.py      ← LaBSE loaded ONCE at startup (module-level singleton)
  │   ├── whisper_model.py    ← Faster-Whisper loaded ONCE at startup
  │   └── llm_client.py       ← Groq API client (free tier) for Llama 3 — NOT local model
  ├── db/
  │   ├── supabase.py         ← Supabase client singleton
  │   └── redis.py            ← Redis client singleton (Upstash Redis — HTTP-based, HF compatible)
  └── requirements.txt

CRITICAL: Use Groq API (free tier) for LLM — NOT a local Llama model.
  Groq gives Llama 3 70B for free with fast inference.
  This avoids GPU memory issues on HF Space.
  GROQ_API_KEY from environment — model: "llama3-70b-8192"

CRITICAL: Use Upstash Redis (HTTP REST API) — NOT a TCP Redis connection.
  HF Spaces blocks outbound TCP on non-standard ports.
  Upstash Redis works over HTTPS — fully compatible.
  from upstash_redis import Redis
  redis = Redis(url=os.environ["UPSTASH_REDIS_URL"], token=os.environ["UPSTASH_REDIS_TOKEN"])

CRITICAL: All AI models load at HF Space STARTUP — not per-request.
  Use @asynccontextmanager lifespan in FastAPI app.py
  Store models as module-level globals: labse_model = None, whisper_model = None
  This prevents cold-start timeouts on first user message.

HF Space startup sequence (app.py lifespan):
  1. Connect Supabase (verify with SELECT 1)
  2. Connect Upstash Redis (verify with PING)
  3. Load LaBSE model → verify with test embed
  4. Load Faster-Whisper model → keep in memory
  5. Test Groq API → verify connection
  6. Connect Qdrant Cloud → verify collection exists
  7. Connect Elasticsearch → verify index exists
  8. Log "✅ All systems ready" to HF Space logs
  9. FastAPI begins accepting requests

RENDER (THIN WEBHOOK ONLY):
  render_webhook/
  ├── main.py       ← FastAPI, receives Messenger webhook, forwards to HF Space
  ├── rasa_relay.py ← Rasa action server that calls HF Space endpoints
  └── requirements.txt  ← ONLY: fastapi, uvicorn, rasa, httpx (nothing heavy)

  Render main.py logic:
    POST /webhooks/messenger/webhook
      → verify FB signature
      → extract sender_psid + message text/postback
      → POST to HF_SPACE_URL/api/process_message
      → receive response (text + buttons + cards)
      → format for Messenger Graph API
      → POST reply to graph.facebook.com/v18.0/me/messages
      → return 200 OK immediately (< 200ms, before FB 5s timeout)
```

---

## ═══════════════════════════════════════════════
## SECTION 4 — UNIVERSAL SYSTEM RULES
## ═══════════════════════════════════════════════

```
This system serves ANYONE from ANYWHERE automatically. These rules are mandatory:

1. AUTO-LOCALE DETECTION
   On first message from new user:
   - Call GET https://graph.facebook.com/v18.0/{psid}?fields=locale,timezone,name
   - Parse locale: "en_US" → language="en", country="US", currency="USD"
   - Parse locale: "ar_SA" → language="ar", country="SA", currency="SAR", rtl=True
   - Never ask "what language?" unless confidence < 0.75
   - Store in Upstash Redis: user:{psid}:profile for 30 days

2. AUTO-CURRENCY
   - All prices stored in USD in Supabase
   - Display converted to user's local currency using Open Exchange Rates (free API)
   - Cache rates in Redis: fx:{currency} EX 3600 (refresh hourly)
   - Show: "¥28,000 / night (~$187 USD)" for transparency

3. AUTO-TIMEZONE
   - All Supabase datetimes stored as UTC
   - Display times converted to user's Facebook timezone
   - Date pickers show in user's local date format

4. RETURNING USER INTELLIGENCE
   On every message, before responding:
   - Check Redis user:{psid}:profile for existing profile
   - If returning: load last_search_city, last_booking, preferred_language, tier
   - If booking in progress: resume from last state (show "Continue booking?" button)
   - Pre-fill forms with saved data, show "Use saved details?" button

5. GLOBAL PAYMENT SUPPORT
   Stripe supports 135+ currencies — always charge in user's local currency if supported
   Fallback: charge in USD with conversion shown
   Show accepted payment methods icons as quick-reply buttons:
   💳 Card | 🍎 Apple Pay | 🤖 Google Pay | 🏦 Bank Transfer

6. ACCESSIBILITY (non-negotiable)
   - RTL languages (ar, he, fa, ur): all text labels marked with dir="rtl" in templates
   - All button labels max 20 chars (Messenger limit, also aids screen readers)
   - All image attachments include accessible_title field
   - Audio responses always offered alongside text (never audio-only)
   - Simple language mode: user can say "speak simply" → bot uses shorter sentences

7. GRACEFUL DEGRADATION
   If HF Space is cold-starting (first request after idle):
   - Render webhook immediately replies: "One moment, connecting you to our AI... 🔄"
   - Polls HF Space health endpoint every 2s for up to 30s
   - Once ready, sends the actual response
   - If HF Space fails after 30s: send fallback menu with pre-built button options

8. SESSION CONTINUITY
   User can switch devices, come back days later:
   - All state in Upstash Redis keyed by Facebook PSID (not session UUID)
   - Incomplete bookings preserved 48 hours with "Resume booking?" offer
   - Completed bookings accessible anytime via "My Bookings" menu button
```

---

## ═══════════════════════════════════════════════
## SECTION 5 — MESSENGER RESPONSE BUILDER
## ═══════════════════════════════════════════════

```
ALWAYS use this builder pattern — never construct raw Messenger JSON manually.

File: render_webhook/messenger_builder.py
This file must be created and used by ALL response-sending code.

class MessengerResponse:
    def __init__(self, recipient_psid: str):
        self.psid = recipient_psid

    def typing(self) -> dict:
        """Send before every substantive message"""
        return {"recipient": {"id": self.psid}, "sender_action": "typing_on"}

    def text(self, message: str) -> dict:
        """Plain text — auto-split if > 2000 chars"""
        # Split at sentence boundaries if > 2000 chars
        # Never truncate mid-word
        return {"recipient": {"id": self.psid}, "message": {"text": message[:2000]}}

    def quick_replies(self, text: str, options: list[dict]) -> dict:
        """
        options = [
          {"title": "Paris 🗼", "payload": "CITY_PARIS"},
          {"title": "Tokyo 🗾", "payload": "CITY_TOKYO"},
        ]
        Max 13 options. Title max 20 chars.
        Auto-truncate titles to 20 chars with ellipsis.
        """
        qr = [
            {
                "content_type": "text",
                "title": opt["title"][:20],
                "payload": opt["payload"]
            }
            for opt in options[:13]
        ]
        return {
            "recipient": {"id": self.psid},
            "message": {"text": text, "quick_replies": qr}
        }

    def hotel_cards(self, hotels: list[dict]) -> dict:
        """
        hotels = [{
            "name": "Grand Tokyo Hotel",
            "stars": 5,
            "price_from": 28000,
            "currency": "JPY",
            "price_usd": 187,
            "thumbnail_url": "https://...",
            "hotel_id": "h_123",
            "distance_km": 1.2,
            "top_feature": "Free breakfast"
        }]
        Max 10 hotels. Always show top 3 first with "Show 4 more →" button.
        """
        elements = []
        for h in hotels[:10]:
            stars_emoji = "⭐" * h["stars"]
            elements.append({
                "title": f"{h['name']} {stars_emoji}"[:80],
                "subtitle": f"From {h['currency']} {h['price_from']:,}/night · {h['top_feature']}"[:80],
                "image_url": h["thumbnail_url"],
                "buttons": [
                    {"type": "postback", "title": "📋 View Details", "payload": f"HOTEL_DETAILS_{h['hotel_id']}"},
                    {"type": "postback", "title": "✅ Select Hotel", "payload": f"HOTEL_SELECT_{h['hotel_id']}"},
                    {"type": "postback", "title": "🔖 Save for Later", "payload": f"HOTEL_SAVE_{h['hotel_id']}"},
                ]
            })
        return {
            "recipient": {"id": self.psid},
            "message": {
                "attachment": {
                    "type": "template",
                    "payload": {"template_type": "generic", "elements": elements}
                }
            }
        }

    def room_cards(self, rooms: list[dict]) -> dict:
        """Same pattern as hotel_cards but for room types"""
        # buttons: View Photos | Select Room | Compare
        ...

    def booking_summary_card(self, booking: dict) -> dict:
        """
        Single card with full booking summary before payment.
        Button: Confirm & Pay | Modify | Cancel
        """
        ...

    def list_template(self, title: str, items: list[dict], cta_button: dict = None) -> dict:
        """
        items = [{"title": "...", "subtitle": "...", "payload": "..."}]
        Max 4 items. Optional global CTA button at bottom.
        Used for: modification options, loyalty info, FAQ categories
        """
        ...

    def image(self, url: str, caption: str = None) -> dict:
        """Send hotel photo, QR code, or map screenshot"""
        payload = {"url": url, "is_reusable": True}
        msg = {"attachment": {"type": "image", "payload": payload}}
        if caption:
            # Send caption as separate text message after image
            pass
        return {"recipient": {"id": self.psid}, "message": msg}

    def file(self, url: str, filename: str) -> dict:
        """Send PDF booking voucher"""
        return {
            "recipient": {"id": self.psid},
            "message": {
                "attachment": {
                    "type": "file",
                    "payload": {"url": url, "is_reusable": False}
                }
            }
        }

    def send_sequence(self, messages: list[dict], delay_ms: int = 600) -> list[dict]:
        """
        Send multiple messages in sequence with typing indicators.
        Always: typing → message → typing → message
        delay_ms between messages prevents Messenger from reordering them.
        Returns ordered list to be sent one by one.
        """
        sequence = []
        for msg in messages:
            sequence.append(self.typing())
            sequence.append(msg)
        return sequence

# USAGE EXAMPLE in render_webhook/main.py:
async def send_to_messenger(messages: list[dict], access_token: str):
    async with httpx.AsyncClient() as client:
        for msg in messages:
            await client.post(
                f"https://graph.facebook.com/v18.0/me/messages?access_token={access_token}",
                json=msg
            )
            if msg.get("sender_action") != "typing_on":
                await asyncio.sleep(0.6)   # 600ms between messages
```

---

## ═══════════════════════════════════════════════
## SECTION 6 — COMPLETE TASK PROMPTS
## ═══════════════════════════════════════════════

### COPY ONE PROMPT AT A TIME INTO COPILOT CHAT

---

### 🔵 TASK PROMPT 1 — HF SPACE UNIFIED APP SETUP

```
TASK: Create the unified HuggingFace Space entry point.

Context: Module 1 is done. I need the single hf_space/app.py that:
1. Loads ALL AI models at startup (LaBSE, Faster-Whisper, Groq client)
2. Connects to ALL databases (Supabase, Upstash Redis, Qdrant, Elasticsearch)
3. Mounts ALL routers (language, search, rag, booking, payment, voice, loyalty)
4. Exposes GET /health that returns {"status": "ready", "models": [...], "dbs": [...]}
5. Exposes POST /api/process_message — the MAIN endpoint called by Render webhook

The /api/process_message endpoint:
  Input: {
    "psid": "facebook_page_scoped_id",
    "message_type": "text" | "postback" | "audio",
    "text": "user message or postback payload",
    "audio_url": "messenger audio url (if voice message)",
    "fb_locale": "en_US",          # from FB profile
    "fb_timezone": -5,             # from FB profile
    "timestamp": 1234567890
  }
  
  Processing flow:
    1. Get/create user profile from Upstash Redis (user:{psid}:profile)
    2. If audio_url: call whisper_model.transcribe(audio_url) first
    3. If first message: auto-detect language from fb_locale, fetch FB profile
    4. Route message to correct handler based on user's current_state in Redis
    5. Return: {
         "messages": [array of Messenger-formatted message objects],
         "new_state": "searching" | "selecting_room" | "filling_form" | "paying" | etc.
       }

Use existing: db/supabase_client.py and db/redis_client.py (import them)
UPSTASH_REDIS_URL and UPSTASH_REDIS_TOKEN replace standard Redis env vars.
Use fastapi lifespan for model loading, NOT @app.on_event (deprecated).
```

---

### 🔵 TASK PROMPT 2 — MODULE 2: HOTEL SEARCH WITH BUTTON-FIRST UX

```
TASK: Build Module 2 — Hotel Search & Discovery with full button-first UX.

Continuing from existing: ActionGreetUser is done, language is detected.
Build: hf_space/routers/search.py + render_webhook state handler for search.

CONVERSATION FLOW (button-first design):
  
  STEP 1 — Destination (after greeting)
  Bot sends TWO messages:
    Message 1: "Where would you like to stay? 🌍"
    Message 2: Quick replies showing:
      [🗼 Paris] [🗾 Tokyo] [🏙️ Dubai] [🌴 Bali] [🎭 London] [🗽 New York] [✏️ Type city...]
    If user clicks "Type city..." → bot asks for free text
    If user types directly → accept free text too

  STEP 2 — Dates
  Bot sends:
    Message 1: "When are you checking in? 📅"
    Message 2: Quick replies:
      [Tonight] [Tomorrow] [This weekend] [Next week] [Next month] [📅 Pick date]
    "Pick date" opens Messenger Webview with a mini date-picker
    After check-in: same pattern for check-out

  STEP 3 — Guests
  Bot sends:
    Message 1: "How many guests? 👥"  
    Message 2: Quick replies: [1 Guest] [2 Guests] [3 Guests] [4 Guests] [4+ Guests]
    If "4+ Guests": ask for number (free text)

  STEP 4 — Optional filters (show as collapsible)
  Bot sends:
    Message 1: "Any preferences? (optional)"
    Message 2: Quick replies:
      [⭐ 3 Stars] [⭐⭐ 4 Stars] [⭐⭐⭐ 5 Stars] [🍳 Breakfast incl.] 
      [🏊 Pool] [💼 Business] [Skip filters →]

  STEP 5 — Results (hotel cards carousel)
  Show top 3 hotel cards first.
  After cards, send quick replies:
    [Show 4 more →] [Change filters 🔧] [New search 🔄] [Sort by price ↕]

SEARCH BACKEND (hf_space/routers/search.py):
  POST /api/search/hotels
    Input: {psid, city, check_in, check_out, num_guests, filters{stars, max_price, amenities}}
    
    1. Auto-detect currency from user profile → convert prices for display
    2. Auto-detect timezone → validate dates are in the future for user's timezone
    3. Call Elasticsearch with bool query + geo-distance + availability filter
    4. Call ranker.py: score = 0.4*ES + 0.3*rating + 0.2*availability + 0.1*personalization
    5. For returning users: fetch preference vector from Qdrant, boost personalization
    6. Return top 10 hotels formatted for hotel_cards() builder
    
  GET /api/search/suggest?query={partial_city}&lang={lang}
    Returns city autocomplete suggestions (from pre-loaded city list in Redis)
    Used by Webview mini search input

  SMART FEATURES:
    - If city not found in ES: suggest closest alternative ("Did you mean Dubai? 🤔")
    - If no availability for dates: show "sold out" message + suggest ±3 days alternatives
    - If < 3 results: automatically expand star rating filter by ±1 and notify user
    - Cache search results in Redis: search:{city}:{check_in}:{check_out} EX 300 (5 min)

State after search: set Redis user:{psid}:state = "viewing_hotels"
                   set Redis user:{psid}:last_search = {city, dates, guests}
```

---

### 🔵 TASK PROMPT 3 — MODULE 3: ROOM SELECTION WITH SMART CARDS

```
TASK: Build Module 3 — Room Selection.

Triggered when user clicks "✅ Select Hotel" button from hotel card (postback: HOTEL_SELECT_{id})

CONVERSATION FLOW:

  TRIGGER: postback payload = "HOTEL_SELECT_{hotel_id}"
  
  Bot immediately sends sequence:
    Message 1: typing...
    Message 2: "Great choice! Let me check available rooms at {hotel_name} 🏨"
    Message 3: typing...
    Message 4: Room cards carousel (top 3 rooms)
    Message 5: Quick replies: [Show all rooms] [Back to hotels ←]

  ROOM CARD FORMAT (per card):
    Image: room thumbnail from Supabase Storage
    Title: "Deluxe King Room • 42m²"
    Subtitle: "🛏 King bed • 🛁 Bathtub • 🌆 City view • from ¥28,000/night"
    Buttons:
      [📸 See Photos]    → postback: ROOM_PHOTOS_{room_id}
      [✅ Choose Room]   → postback: ROOM_SELECT_{room_id}
      [ℹ️ Full Details]  → postback: ROOM_DETAILS_{room_id}

  ON "See Photos" (postback: ROOM_PHOTOS_{room_id}):
    Send up to 5 room images one by one
    After last image: quick replies [✅ Choose This Room] [← Other Rooms]

  ON "Full Details" (postback: ROOM_DETAILS_{room_id}):
    Send list template with:
      - Size, bed type, max guests
      - Amenities (wifi, minibar, AC, safe, etc.)
      - Cancellation policy for this room
    Then quick replies: [✅ Choose Room] [← Back to Rooms]

  ON "Choose Room" (postback: ROOM_SELECT_{room_id}):
    STEP 1 — Rate plan selection
    Message: "Which rate plan suits you? 🍽️"
    Quick replies:
      [🛏 Room Only  ¥28,000]
      [☕ With Breakfast  ¥32,000]
      [🌮 Half Board  ¥38,000]
      [🍽 Full Board  ¥45,000]
    
    STEP 2 — After rate selected: ACQUIRE SOFT LOCK immediately
      Call hf_space POST /api/rooms/soft_lock
      If lock fails (race condition): "Room just got reserved by someone else!
        Let me find the next best option..." → show next available room

    STEP 3 — Show price summary before proceeding
    Message: "📋 Your selection:"
    List template:
      Hotel: Grand Tokyo Hotel ⭐⭐⭐⭐⭐
      Room: Deluxe King Room • City View
      Plan: Breakfast Included
      Check-in: Fri, 15 Mar 2026
      Check-out: Sun, 17 Mar 2026
      Total: ¥64,000 (~$427 USD)
    Quick replies: [✅ Looks Good!] [✏️ Change Room] [❌ Start Over]

BACKEND (hf_space/routers/rooms.py):
  GET /api/rooms/{hotel_id}?check_in=&check_out=&guests=
    Returns available rooms with real-time availability from Supabase
    
  POST /api/rooms/soft_lock
    Input: {room_type_id, date_range, psid, lock_minutes=15}
    Upstash Redis SET with NX flag (atomic)
    Lua script for multi-date atomic lock
    Returns: {locked: bool, expires_at, lock_id}
    
  POST /api/rooms/refresh_lock
    Called every 5 min during payment to extend lock
    Input: {lock_id, psid}

State: user:{psid}:state = "rate_selected"
       user:{psid}:booking_draft = {hotel_id, room_type_id, rate_plan, check_in, check_out, total}
```

---

### 🔵 TASK PROMPT 4 — MODULE 4: GUEST FORM (CONVERSATIONAL, BUTTON-ASSISTED)

```
TASK: Build Module 4 — Guest Information Collection, conversational with max button use.

NEVER show a web form. Collect everything through chat conversation.
One question per message. Validate immediately. Show friendly error, re-ask once.

CONVERSATION FLOW:

  Triggered after rate_selected state confirmed.

  Q1 — Name
  "What's your name? 👤"
  [Free text — no buttons here]
  Validation: min 2 chars each part, no numbers, strip whitespace
  
  Q2 — Email  
  "What email should I send your booking confirmation to? 📧"
  [Free text]
  Validation: RFC-5322 regex, detect common typos (gmail.con → gmail.com)
  On typo: "Did you mean {corrected}? 🤔"
  Quick replies: [Yes, use {corrected}] [No, let me retype]
  
  Q3 — Phone
  "Your phone number? (with country code) 📱"
  [Free text]
  Auto-detect country code from user's FB locale if possible
  Show hint: "e.g. +1 555 123 4567 (US) or +44 7700 900123 (UK)"
  Validation: phonenumbers library → E.164 format
  
  Q4 — Trip Purpose (ALL BUTTONS)
  "What brings you to {city}? ✈️"
  Quick replies:
    [🏖 Leisure] [💍 Honeymoon] [👨‍👩‍👧 Family Trip]
    [💼 Business] [🎓 Study/Conference] [🏥 Medical]
  
  Q5 — Dietary needs (ALL BUTTONS)
  "Any dietary requirements? 🍽️"
  Quick replies:
    [🚫 None] [🥗 Vegetarian] [🌱 Vegan] [🕌 Halal]
    [✡️ Kosher] [🚫🥜 Nut Allergy] [✏️ Other...]
  
  Q6 — Accessibility (ALL BUTTONS)
  "Any accessibility needs? ♿"
  Quick replies:
    [None needed ✅] [Wheelchair access ♿] [Ground floor room 🔑]
    [Visual assistance 👁️] [Hearing assistance 👂] [✏️ Specify...]

  RETURNING USER SHORTCUT:
  If user has booked before, show at start:
    Message: "Use your saved details? 💾"
    List template showing saved: Name, Email, Phone
    Quick replies: [✅ Yes, use saved] [✏️ Update details]
    
  PASSPORT OCR SHORTCUT (optional):
    After name collected:
    "Would you like to scan your passport for faster check-in? 📷"
    Quick replies: [📷 Scan Passport] [Skip for now →]
    If scan: open Messenger camera / image picker
    Process OCR in hf_space, pre-fill remaining fields

BACKEND (hf_space/routers/guest.py):
  POST /api/guest/validate
    Input: {field: "email", value: "test@gmail.con", context: {}}
    Returns: {valid: bool, normalized: "test@gmail.com", suggestion: "gmail.com", error: null}
    
  POST /api/guest/ocr_passport
    Input: {image_url: "messenger CDN URL", psid}
    Fetch image from Messenger CDN → pre-process → Tesseract MRZ
    Returns extracted fields with confidence scores
    
  POST /api/guest/save_profile
    Input: all validated guest fields + psid
    Upsert Supabase guests table WHERE messenger_psid = psid
    Update Redis user:{psid}:profile with name, email, phone

State: user:{psid}:state = "guest_complete"
       user:{psid}:guest_data = {validated guest fields JSON}
```

---

### 🔵 TASK PROMPT 5 — MODULE 5: ADD-ONS WITH SMART RECOMMENDATIONS

```
TASK: Build Module 5 — Add-Ons & Upsell, personalised and button-driven.

Triggered after guest_complete state.

CONVERSATION FLOW:

  SMART OPENER (personalised by trip_purpose):
  
  IF honeymoon:
    Message 1: "Since it's a special occasion 💍, I've picked these for you:"
    Message 2: Generic cards (3 items):
      Card 1: 🥂 Champagne Welcome — $45
              "Chilled champagne + chocolates waiting in your room"
              [Add to Booking ✅] [Tell me more 💬]
      Card 2: 🛁 Rose Petal Turndown — $35
              "Romantic rose petal room setup at evening turndown"
              [Add to Booking ✅] [Tell me more 💬]
      Card 3: 💆 Couples Spa 60min — $120
              "Private couples treatment at the hotel spa"
              [Add to Booking ✅] [Tell me more 💬]
    After cards:
    Quick replies: [See all add-ons 📋] [Skip add-ons →] [Add all 3! 🎁]
    
  IF family:
    Show: Extra bed, Kids club, Airport family transfer, Babysitting
    
  IF business:
    Show: Late checkout (2pm), Meeting room (2h), Airport sedan transfer, Laundry

  ON "Tell me more":
    Send 2-3 line description from addons table
    Quick replies: [Add this ✅] [No thanks ❌]
    
  ON "See all add-ons":
    Show categories as list template:
      🧖 Spa & Wellness
      🍽️ Dining & Drinks  
      🚗 Transport
      🎯 Activities
      🛏️ Room Extras
    User taps category → carousel of items in that category

  CART SUMMARY (after each add):
    Message: "✅ Added! Your extras so far:"
    List template showing cart items with prices
    Quick replies: [Continue adding 🛍️] [Proceed to payment →]

BACKEND (hf_space/routers/addons.py):
  GET /api/addons/recommend
    Input: {hotel_id, trip_purpose, num_children, psid}
    Query Supabase addons by hotel + trip_purpose_tags match
    Return top 3 with score multiplier applied
    
  POST /api/addons/cart/add
    Input: {psid, addon_id}
    Append to Redis user:{psid}:addon_cart (JSON list)
    
  POST /api/addons/cart/remove
    Input: {psid, addon_id}
    Remove from Redis list
    
  GET /api/addons/cart/{psid}
    Returns full cart with live prices and subtotal

State: user:{psid}:state = "addons_complete"
       user:{psid}:addon_cart = [{addon_id, name, price, currency}]
```

---

### 🔵 TASK PROMPT 6 — MODULE 6: PAYMENT FLOW (SECURE, BUTTON-DRIVEN)

```
TASK: Build Module 6 — Payment, fully secure with Stripe, button-driven UX.

CRITICAL SECURITY: Card data NEVER touches Render or HF Space.
Stripe.js tokenizes on client side inside a Messenger Webview popup.

CONVERSATION FLOW:

  STEP 1 — Full booking summary BEFORE payment
  Bot sends sequence:
    Message 1: "📋 Let's review your booking before payment:"
    Message 2: Generic template (1 card — booking summary card):
      Image: hotel main photo
      Title: "Grand Tokyo Hotel ⭐⭐⭐⭐⭐"
      Subtitle: "Deluxe King · 15-17 Mar · 2 guests · Breakfast"
      Button: [📋 View Full Details]   → postback BOOKING_SUMMARY_FULL
    Message 3: Text: "💰 Total: ¥64,000 (~$427 USD) for 2 nights"
    Message 4: Quick replies:
      [💳 Pay Now] [✏️ Change something] [❌ Cancel]
      
  IF "Change something":
    Quick replies:
      [📅 Change Dates] [🛏 Change Room] [👤 Edit Guest Info] [🛍 Edit Add-ons]
    Route to appropriate state based on selection

  STEP 2 — Payment method selection
  After "Pay Now":
    Message: "Choose payment method 💳"
    Quick replies:
      [💳 Credit / Debit Card]
      [🍎 Apple Pay]
      [🤖 Google Pay]
      
  STEP 3 — Stripe Webview (for card payment)
  Send Messenger button template:
    Message: "Tap below to enter your card securely 🔒"
    Button: [Open Secure Payment →] (web_url type, opens Webview)
    URL: https://{hf_space_url}/pay_webview?booking_draft_id={id}&psid={psid}
    
  hf_space/static/pay_webview.html:
    - Loads Stripe.js
    - Shows hotel name + amount (from booking_draft in Redis)
    - Stripe Elements card input (tokenizes client-side)
    - On submit: POST stripe_token to HF Space /api/payment/process
    - On success: closes Webview, sends postback PAYMENT_SUCCESS to Render webhook
    - On fail: shows error in Webview, allows retry (max 3 attempts)
    
  STEP 4 — Fraud check + charge (server-side in HF Space)
  POST /api/payment/process:
    1. fraud_check.py: calculate risk score
    2. If risk >= 71: block, message user "Payment flagged. Contact support."
    3. If risk 31-70: require Stripe 3DS (return client_secret to Webview)
    4. If risk < 31: charge immediately
    5. On Stripe success: 
       - Save to Supabase payments table
       - Generate booking_reference: HTL-{YEAR}-{6-char random uppercase}
       - Update bookings table status='confirmed'
       - Publish Kafka 'payment.completed' event
    6. Return {success: true, booking_reference: "HTL-2026-XYZABC"}

  STEP 5 — Post-payment (after PAYMENT_SUCCESS postback received by Render)
  Bot sends sequence (all within 30 seconds):
    Message 1: "🎉 Booking confirmed! Reference: HTL-2026-XYZABC"
    Message 2: QR code image (generated in HF Space, stored in Supabase Storage)
    Message 3: "📄 Your booking voucher is ready:"
    Message 4: File attachment — PDF voucher from Supabase Storage
    Message 5: Quick replies:
      [📅 Add to Calendar] [🗺 Get Directions] [🌤 Weather Forecast]
      [🏨 Hotel Contact] [📤 Share Booking] [🏠 Back to Menu]
    
  CONFIRMATION EMAIL: trigger async via hf_space/routers/notification.py

BACKEND (hf_space/routers/payment.py):
  POST /api/payment/process
    Full fraud check + Stripe charge as described above
    
  POST /api/payment/webhook (Stripe webhooks)
    Verify HMAC signature FIRST — reject if invalid
    Handle: payment_intent.succeeded, payment_intent.payment_failed
    
  GET /api/payment/pay_webview
    Render static HTML page with Stripe Elements
    Loads booking draft from Redis by booking_draft_id
    Shows amount in user's currency
```

---

### 🔵 TASK PROMPT 7 — MODULES 8 & 9: MODIFY + CANCEL WITH SMART POLICY

```
TASK: Build Modules 8 & 9 — Booking Modification and Cancellation.

Triggered from "My Bookings" menu or user saying "change/cancel my booking".

MY BOOKINGS FLOW:
  User taps "📋 My Bookings" from Persistent Menu
  
  If no bookings: "You don't have any bookings yet. Start searching? 🔍"
                  Quick replies: [🔍 Search Hotels] [🏠 Main Menu]
  
  If 1 booking: Show it directly as a card
  
  If multiple: Show list template with up to 4 recent bookings
    Each item: "Grand Tokyo · 15 Mar - 17 Mar · HTL-2026-XYZ"
    Button: [View & Manage]

  BOOKING DETAIL VIEW (single booking card):
    Card buttons:
      [✏️ Modify Booking] → postback BOOKING_MODIFY_{booking_id}
      [❌ Cancel Booking] → postback BOOKING_CANCEL_{booking_id}
      [📞 Contact Hotel]  → postback BOOKING_CONTACT_{hotel_id}

MODIFICATION FLOW:

  ON postback BOOKING_MODIFY_{booking_id}:
    1. Fetch booking from Supabase
    2. Call policy_engine.py → get allowed changes
    3. Message: "What would you like to change? ✏️"
    4. Quick replies (show only allowed options based on policy):
       [📅 Change Dates] [🛏 Change Room Type] [👥 Change Guest Count]
       [🍽 Change Meal Plan] [🛍 Edit Add-ons] [← Back]

    ON "Change Dates":
      Show current dates, ask for new check-in (same button flow as Module 2 Step 2)
      After new dates: call rebooking.py
      If price_diff > 0: "New dates cost ¥12,000 more. Proceed?"
                         Quick replies: [✅ Pay Difference ¥12,000] [❌ Keep Original Dates]
      If price_diff < 0: "New dates are ¥8,000 cheaper! You'll get a refund."
                         Quick replies: [✅ Confirm Change] [❌ Keep Original]
      If not available: "Sorry, {room_type} is sold out for new dates."
                         Quick replies: [🔄 Try Other Dates] [🛏 Try Different Room]

    ON fee applicable:
      ALWAYS show: "⚠️ A modification fee of ${fee} applies for changes within {days} days."
      Quick replies: [✅ Accept Fee & Continue] [❌ Keep Booking As Is]

CANCELLATION FLOW:

  ON postback BOOKING_CANCEL_{booking_id}:
    1. Call refund_engine.py → calculate refund FIRST
    2. Bot ALWAYS shows refund amount BEFORE asking to confirm:
    
    Full refund case:
      "Your booking qualifies for a full refund ✅
       💰 Refund: ¥64,000 → back to your card in 5-10 days
       Reference: HTL-2026-XYZ"
      Quick replies: [✅ Confirm Cancellation] [❌ Keep Booking]
      
    Partial refund case:
      "⚠️ Cancellation within {days} days:
       💰 Refund: ¥32,000 (50%) — ¥32,000 is non-refundable"
      Quick replies: [✅ Confirm & Get ¥32,000 Back] [❌ Keep Booking]
      
    No refund case:
      "⚠️ Non-refundable booking:
       💰 Refund: ¥0 (0%) — rate plan is non-refundable"
      Quick replies: [✅ Cancel Anyway (No Refund)] [❌ Keep Booking]
    
    ON confirm: ask reason (for analytics)
      Quick replies: [🗓 Date changed] [✈️ Flight issue] [💸 Price concern]
                     [😷 Health/Emergency] [🔄 Found better hotel] [❌ Skip]

BACKEND (hf_space/routers/modification.py + cancellation.py):
  Already defined in SKILL.md — implement policy_engine, rebooking, refund_engine, stripe_refund.
  All policy rules fetch from Supabase hotels.cancellation_policy JSONB column.
```

---

### 🔵 TASK PROMPT 8 — MODULE 11: FAQ RAG WITH SMART ROUTING

```
TASK: Build Module 11 — FAQ & RAG pipeline with smart question routing.

Triggered when: user asks a question about a specific hotel (any language).
Context needed: user must have a hotel_id in their current session.

CONVERSATION FLOW:

  FAQ CATEGORY SHORTCUT (proactive, after hotel selection):
  After user selects a hotel (before room selection):
    Bot sends: "Got any questions about {hotel_name}? I can help! 💬"
    List template with FAQ categories:
      🏊 Facilities (pool, gym, spa)
      🍽️ Dining (restaurants, breakfast, room service)
      🚗 Transport (airport transfer, parking)
      🛎️ Services (check-in, concierge, late checkout)
      💳 Policies (cancellation, pets, children)
    Global button: [No questions, continue booking →]
    
  ON category tap OR free text question:
    1. Embed question via LaBSE (HF Space model — already loaded)
    2. Search Qdrant hotel_faqs with hotel_id filter, top_k=3
    3. If Qdrant score > 0.85: return direct answer (no LLM needed, fast)
    4. If 0.6 < score < 0.85: send to Groq Llama 3 with context
    5. If score < 0.6: "I don't have that info, but here's the hotel's contact:"
                       + contact buttons
    6. Translate answer to user's detected language via translator.py
    
  ANSWER FORMAT:
    Message 1: Answer text (in user's language)
    Message 2: Quick replies:
      [🙋 Another question] [✅ Continue booking] [📞 Ask hotel directly]
    
  VOICE QUESTION SUPPORT:
    If user sends voice message while browsing hotel:
      1. Transcribe via Whisper (already loaded in HF Space)
      2. Show transcription: "You asked: '{transcript}'"
      3. Process through RAG as normal
      4. Offer TTS response: quick reply [🔊 Hear answer]

BACKEND (hf_space/routers/rag.py):
  POST /api/rag/ask
    Input: {question, hotel_id, psid, language}
    1. embed question: labse_model.encode([question])
    2. qdrant_client.search(collection="hotel_faqs", vector=embedding, 
                             filter={"hotel_id": hotel_id}, limit=3)
    3. If best_score > 0.85: return top result payload.answer directly
    4. Else: groq_client.chat.completions.create(
               model="llama3-70b-8192",
               messages=[{"role": "system", "content": concierge_prompt},
                          {"role": "user", "content": f"Context: {chunks}\n\nQuestion: {question}"}]
             )
    5. translator.translate(answer, target_lang=language) if language != "en"
    6. Return {answer, source: "direct"|"rag"|"fallback", confidence}
    
  POST /api/rag/onboard_hotel
    Input: {hotel_id, faqs: [{question, answer, category}]}
    Batch embed all FAQs with LaBSE
    Upsert to Qdrant hotel_faqs collection with hotel_id filter
    Called by hotel_onboarding.py script
```

---

### 🔵 TASK PROMPT 9 — MODULE 10: LOYALTY WITH GAMIFICATION BUTTONS

```
TASK: Build Module 10 — Loyalty & Gamification, fully surfaced through Messenger buttons.

LOYALTY CHECK FLOW (triggered by "My Rewards" menu or asking about points):

  Bot sends sequence:
    Message 1: "🏆 Your Loyalty Status"
    Message 2: Generic template card:
      Image: tier badge image from Supabase Storage (bronze/silver/gold/platinum/black)
      Title: "Ahmed Al-Rashidi • Gold Member"
      Subtitle: "⭐ 14,392 points · {points_needed} to Platinum"
      Buttons:
        [🎁 Redeem Points] → postback LOYALTY_REDEEM
        [📊 Points History] → postback LOYALTY_HISTORY
        [🎖 My Badges]    → postback LOYALTY_BADGES
    Message 3: Progress bar text:
      "Gold ████████░░ Platinum — 3,608 pts to go!"
    Message 4: Quick replies:
      [🎁 Redeem Points] [👥 Refer a Friend] [📜 Tier Benefits] [🏠 Main Menu]

  ON "Tier Benefits":
    List template with benefits for current + next tier
    Global button: [How to earn more points →]
    
  ON "My Badges" (postback LOYALTY_BADGES):
    Carousel showing earned badges as cards
    Each card: badge image, name, how it was earned, date
    Unearned badges shown as "locked" with earn conditions
    Quick replies: [🏠 Main Menu] [🔍 Search Hotels]

  ON "Refer a Friend":
    Message: "Share this link and earn 200 points when they book! 🎉"
    Send referral link + quick reply: [📋 Copy Link] [📤 Share]

  PROACTIVE LOYALTY (after booking confirmed):
  Automatically send:
    "🎉 You earned 842 points for this booking!
     New balance: 14,392 points (Gold tier)
     You're 3,608 points from Platinum — 2 more bookings could get you there! 🚀"
    Quick replies: [🏆 View Rewards] [🏠 Main Menu]

  TIER UPGRADE NOTIFICATION:
  On Kafka 'loyalty.tier_upgraded' event:
    Send proactive message to user:
    "🌟 CONGRATULATIONS! You've reached {new_tier} status!
     {tier_specific_welcome_message}"
    Send tier badge image
    Quick replies: [🎁 See New Benefits] [🏠 Main Menu]

BACKEND (hf_space/routers/loyalty.py):
  POST /api/loyalty/award
    Input: {psid, booking_id, booking_total, currency}
    Calculate points via points_engine.py
    Update Supabase loyalty_accounts + loyalty_transactions
    Check tier upgrade via tier_engine.py
    Award badges via gamification.py
    Return {points_earned, new_balance, tier_changed, badges_awarded}
    
  GET /api/loyalty/profile/{psid}
    Returns full loyalty profile for display
    
  POST /api/loyalty/redeem
    Input: {psid, points_to_redeem, booking_id}
    Validate points balance
    Apply discount to booking
    Update loyalty_accounts
```

---

### 🔵 TASK PROMPT 10 — MODULE 13: HUMAN HANDOFF + MODULE 12: REVIEWS

```
TASK: Build Module 13 (Human Handoff via Chatwoot) and Module 12 (Post-Stay Reviews).

HANDOFF FLOW:

  TRIGGERS (any of these):
  - User types: "talk to human", "real agent", "help me please" (intent: intent_human_handoff)
  - FallbackClassifier fires 2 consecutive times
  - User sends angry sentiment detected by sentiment.py (score < 0.2)
  - User asks same question 3 times without booking progressing

  Bot sends:
    Message 1: "I'll connect you with a live agent right away! 🙋"
    Message 2: "Estimated wait: ~{wait_time} minutes"
    Message 3: Quick replies: 
      [📞 Connect Now] [💬 Keep trying with AI] [📩 Email instead]
    
  ON "Connect Now":
    1. Call /api/handoff/create with full conversation history
    2. Create Chatwoot conversation with language-matched agent queue
    3. Post all Messenger messages as initial Chatwoot note
    4. Set Redis user:{psid}:state = "handoff_active"
    5. Set Redis user:{psid}:handoff_active = "true"
    6. Message: "✅ Connected! Reference: CHAT-{id}
                 Agent {name} will be with you shortly.
                 I'll be quiet until they're done helping you."
    
  WHILE handoff_active:
    All incoming Messenger messages relay to Chatwoot conversation via API
    All Chatwoot agent messages relay back to Messenger
    Bot does NOT process any NLU during handoff
    
  ON Chatwoot resolved webhook:
    1. Clear Redis user:{psid}:handoff_active
    2. Set user:{psid}:state = "post_handoff"
    3. Message: "Welcome back! Your issue has been resolved. 😊"
    4. Quick replies: [🔍 Search Hotels] [📋 My Bookings] [🏠 Main Menu]

REVIEW FLOW (automated, 24h after checkout):

  Celery beat task triggers:
    1. Query Supabase: bookings where check_out + 24h <= now AND review_requested = false
    2. For each: call /api/review/send_request
    
  Review Request Message:
    Message 1: "How was your stay at {hotel_name}? We'd love your feedback! 🌟"
    Message 2: "Rate your overall experience:"
    Quick replies (star ratings):
      [⭐ Poor] [⭐⭐ Fair] [⭐⭐⭐ Good] [⭐⭐⭐⭐ Great] [⭐⭐⭐⭐⭐ Excellent!]
    [Skip review →]
    
  AFTER STAR RATING (e.g., user clicks ⭐⭐⭐⭐ Great):
    Message: "Thanks! Rate specific aspects:"
    Multiple quick reply sequences (one per aspect):
      "🛏️ Room quality?"  → [⭐ Poor] [⭐⭐⭐ OK] [⭐⭐⭐⭐⭐ Great]
      "🍽️ Dining?"        → [⭐ Poor] [⭐⭐⭐ OK] [⭐⭐⭐⭐⭐ Great] [N/A]
      "🤝 Staff service?" → [⭐ Poor] [⭐⭐⭐ OK] [⭐⭐⭐⭐⭐ Great]
      "📍 Location?"      → [⭐ Poor] [⭐⭐⭐ OK] [⭐⭐⭐⭐⭐ Great]
    
    Then: "Any comments? (optional)"
    Quick replies: [Skip ✓] OR user types free text
    
  AFTER REVIEW SUBMITTED:
    1. Run sentiment.py on review text (if provided)
    2. If negative (score < 0.3): Kafka 'review.negative_alert' → hotel partner
    3. Award 50 loyalty points: "🎉 +50 loyalty points for your review!"
    4. Message: "Thanks {name}! Your feedback helps millions of travelers. 🙏"
    Quick replies: [🔍 Search Hotels] [🏆 My Rewards] [🏠 Main Menu]

BACKEND (hf_space/routers/reviews.py + handoff.py):
  POST /api/review/submit
    Input: {psid, booking_id, overall_score, aspect_scores, review_text}
    Insert into Supabase reviews table
    Run sentiment.py
    Award points via loyalty service
    
  POST /api/handoff/create
    Create Chatwoot conversation
    Post history
    Return {conversation_id, agent_name, wait_time}
    
  POST /api/handoff/relay_message
    Input: {chatwoot_conversation_id, message_text, from_agent: bool}
    If from_agent: forward to Messenger via Graph API
    If from_user: forward to Chatwoot via API
```

---

### 🔵 TASK PROMPT 11 — VOICE INTERACTION COMPLETE SETUP

```
TASK: Add full voice interaction support within the single HuggingFace Space.

Voice is delivered via Messenger's native audio message support + a Webview for recording.

HOW MESSENGER VOICE WORKS:
  Option A — User sends voice note in Messenger:
    Messenger delivers audio as: {"message": {"attachments": [{"type": "audio", "payload": {"url": "..."}}]}}
    Render webhook detects audio attachment → sends audio_url to HF Space /api/process_message
    HF Space fetches audio from Messenger CDN → feeds to Faster-Whisper → processes as text
    
  Option B — User taps 🎙️ button to open voice Webview:
    Opens hf_space/static/voice_webview.html
    MediaRecorder API captures audio in browser
    Sends audio blob to HF Space /api/voice/transcribe in real-time
    Returns transcript → shown in webview → auto-submitted to main flow

VOICE INPUT (hf_space/routers/voice.py):
  POST /api/voice/transcribe
    Input: audio file (multipart) OR {audio_url: "messenger CDN URL"}
    
    If audio_url: 
      async download with httpx (Messenger CDN requires no auth)
      Convert to 16kHz mono WAV using pydub in-memory (NO ffmpeg subprocess — use pydub.AudioSegment)
      
    transcribe with faster_whisper model (loaded at startup)
    segments, info = whisper_model.transcribe(wav_bytes, beam_size=5, language=None)
    
    If info.language_probability < 0.6:
      Return {success: false, message: "Could not understand audio. Please try again."}
    
    Store detected language in Upstash Redis (override if different from profile lang)
    Return {transcript, language, confidence}

VOICE OUTPUT (TTS via Groq/external):
  DO NOT load local TTS model (too heavy for HF Space).
  Use: Edge-TTS (Microsoft Azure free tier, no API key needed)
    import edge_tts
    voice map by language:
      en → en-US-JennyNeural (female) or en-US-GuyNeural (male)
      ar → ar-SA-ZariyahNeural
      fr → fr-FR-DeniseNeural
      es → es-ES-ElviraNeural
      zh → zh-CN-XiaoxiaoNeural
      ja → ja-JP-NanamiNeural
      hi → hi-IN-SwaraNeural
      (add all 15 Tier 1 languages)
    
    async def text_to_speech(text: str, language: str, gender: str = "female") -> bytes:
      voice = VOICE_MAP.get(f"{language}_{gender}", "en-US-JennyNeural")
      communicate = edge_tts.Communicate(text, voice)
      audio_bytes = b""
      async for chunk in communicate.stream():
          if chunk["type"] == "audio":
              audio_bytes += chunk["data"]
      return audio_bytes  # MP3 bytes
      
  Cache TTS in Upstash Redis:
    key: tts:{lang}:{gender}:{sha256(text)[:16]} 
    Store base64-encoded MP3, EX 86400
    Check cache before generating

VOICE UX RULE:
  - Never send ONLY audio — always include text first
  - After text response, add [🔊 Hear this] quick reply
  - When user clicks: generate TTS → upload to Supabase Storage → send as Messenger audio attachment
  - Voice input always shows transcript back to user: "I heard: '{transcript}'"

POST /api/voice/speak
  Input: {text, language, gender, psid}
  Check TTS cache
  Generate via edge_tts if cache miss
  Upload MP3 to Supabase Storage bucket "tts-audio" with 1-hour expiry
  Return {audio_url} to Render webhook which sends as Messenger audio attachment
```

---

### 🔵 TASK PROMPT 12 — RENDER WEBHOOK (THIN RELAY ONLY)

```
TASK: Build the Render webhook — a thin relay with NO heavy processing.

Render has limited RAM. This service does EXACTLY 3 things:
  1. Receive Facebook Messenger webhook events
  2. Forward to HF Space for processing
  3. Send HF Space response back to Messenger

File: render_webhook/main.py

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
import httpx, hmac, hashlib, asyncio, os

app = FastAPI()
HF_SPACE_URL = os.environ["HF_SPACE_URL"]  # full HF Space URL
PAGE_ACCESS_TOKEN = os.environ["FACEBOOK_PAGE_ACCESS_TOKEN"]
APP_SECRET = os.environ["FACEBOOK_APP_SECRET"]
VERIFY_TOKEN = os.environ["FACEBOOK_VERIFY_TOKEN"]
GRAPH_API = "https://graph.facebook.com/v18.0/me/messages"

@app.get("/webhooks/messenger/webhook")
async def verify_webhook(request: Request):
    """Facebook webhook verification — one-time setup"""
    params = dict(request.query_params)
    if params.get("hub.mode") == "subscribe" and params.get("hub.verify_token") == VERIFY_TOKEN:
        return int(params["hub.challenge"])
    raise HTTPException(status_code=403)

@app.post("/webhooks/messenger/webhook")  
async def receive_message(request: Request, background_tasks: BackgroundTasks):
    """
    Receive Messenger events.
    MUST return 200 OK within 200ms (before FB's 5-second timeout).
    All processing happens in background task.
    """
    # Verify Facebook signature
    signature = request.headers.get("X-Hub-Signature-256", "")
    body = await request.body()
    expected = "sha256=" + hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    data = await request.json()
    background_tasks.add_task(process_webhook_event, data)
    return {"status": "ok"}  # Return immediately

async def process_webhook_event(data: dict):
    """Run in background — no time pressure"""
    for entry in data.get("entry", []):
        for messaging in entry.get("messaging", []):
            psid = messaging["sender"]["id"]
            
            # Build HF Space request payload
            hf_payload = {
                "psid": psid,
                "message_type": "text",
                "timestamp": messaging.get("timestamp", 0)
            }
            
            if "message" in messaging:
                msg = messaging["message"]
                if msg.get("text"):
                    hf_payload["message_type"] = "text"
                    hf_payload["text"] = msg["text"]
                elif msg.get("attachments"):
                    att = msg["attachments"][0]
                    if att["type"] == "audio":
                        hf_payload["message_type"] = "audio"
                        hf_payload["audio_url"] = att["payload"]["url"]
                    elif att["type"] == "image":
                        hf_payload["message_type"] = "image"
                        hf_payload["image_url"] = att["payload"]["url"]
                        
            elif "postback" in messaging:
                hf_payload["message_type"] = "postback"
                hf_payload["text"] = messaging["postback"]["payload"]
            else:
                return  # Skip delivery receipts, read receipts
            
            # Send typing indicator immediately (shows user bot is working)
            await send_to_messenger([{
                "recipient": {"id": psid},
                "sender_action": "typing_on"
            }])
            
            # Call HF Space
            try:
                async with httpx.AsyncClient(timeout=55.0) as client:
                    resp = await client.post(f"{HF_SPACE_URL}/api/process_message", json=hf_payload)
                    result = resp.json()
                    messages = result.get("messages", [])
            except httpx.TimeoutException:
                messages = [{
                    "recipient": {"id": psid},
                    "message": {"text": "Sorry, I'm experiencing a delay. Please try again in a moment. 🔄"}
                }]
            
            # Send all messages to Messenger in sequence
            await send_to_messenger(messages)

async def send_to_messenger(messages: list):
    async with httpx.AsyncClient(timeout=10.0) as client:
        for msg in messages:
            try:
                await client.post(
                    f"{GRAPH_API}?access_token={PAGE_ACCESS_TOKEN}",
                    json=msg
                )
                if msg.get("sender_action") != "typing_on":
                    await asyncio.sleep(0.6)
            except Exception as e:
                print(f"Messenger send error: {e}")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "service": "messenger_relay"}

# requirements.txt for Render (MINIMAL — low RAM):
# fastapi==0.111.0
# uvicorn==0.29.0
# httpx==0.27.0
# python-multipart==0.0.9
# (nothing else — no ML libraries, no database drivers)
```

---

### 🔵 TASK PROMPT 13 — CICD + ENVIRONMENT SETUP

```
TASK: Create complete CI/CD pipeline and environment configuration.

File: .github/workflows/deploy.yml
Triggers:
  - Push to main → deploy HF Space + Render webhook
  - Push to develop → run tests only, no deploy
  - PR to main → run tests + lint, block merge if fail

Jobs:
  1. lint:
     - ruff check . (Python linting)
     - ruff format --check .
     
  2. test:
     - pytest tests/ --cov=hf_space --cov-report=xml --cov-fail-under=70
     - Test against real Supabase test project (SUPABASE_URL_TEST secret)
     - Test against Stripe test mode (STRIPE_SECRET_KEY_TEST secret)
     
  3. deploy_hf_space (on main only, after test passes):
     - Uses HuggingFace CLI to push to Space
     - huggingface-cli upload {HF_USERNAME}/hotel-booking-ai ./hf_space .
     - Wait for Space build to succeed (poll /health endpoint)
     
  4. deploy_render (on main only, after hf_space deploy):
     - Trigger Render deploy hook (POST to Render deploy URL)
     - Wait for health check at Render webhook /health
     - Run smoke test: send test Messenger message, verify response

File: .env.example (ALL variables documented):
  # HuggingFace Space
  HF_SPACE_URL=https://{username}-hotel-booking-ai.hf.space
  HF_TOKEN=hf_xxxxxxxxxxxx
  
  # Upstash Redis (HTTPS-based, HF Space compatible)
  UPSTASH_REDIS_URL=https://xxxx.upstash.io
  UPSTASH_REDIS_TOKEN=xxxxxxxxxxxx
  
  # Supabase
  SUPABASE_URL=https://xxxx.supabase.co
  SUPABASE_ANON_KEY=eyJ...
  SUPABASE_SERVICE_ROLE_KEY=eyJ...   # Server-side only
  SUPABASE_DATABASE_URL=postgresql://...
  
  # Facebook / Messenger
  FACEBOOK_PAGE_ACCESS_TOKEN=EAAxxxx
  FACEBOOK_APP_SECRET=xxxx
  FACEBOOK_VERIFY_TOKEN=your_chosen_string
  WHATSAPP_PHONE_NUMBER_ID=xxxx
  
  # Groq (free LLM API)
  GROQ_API_KEY=gsk_xxxx
  
  # Stripe
  STRIPE_SECRET_KEY=sk_live_xxxx         # sk_test_ for development
  STRIPE_PUBLISHABLE_KEY=pk_live_xxxx
  STRIPE_WEBHOOK_SECRET=whsec_xxxx
  
  # Qdrant Cloud
  QDRANT_URL=https://xxxx.qdrant.io
  QDRANT_API_KEY=xxxx
  
  # Elasticsearch (Bonsai.io free tier)
  ELASTICSEARCH_URL=https://xxxx.bonsai.io
  ELASTICSEARCH_USER=xxxx
  ELASTICSEARCH_PASS=xxxx
  
  # Notifications
  SENDGRID_API_KEY=SG.xxxx
  
  # Chatwoot
  CHATWOOT_API_URL=https://your-chatwoot.onrender.com
  CHATWOOT_API_TOKEN=xxxx
  CHATWOOT_ACCOUNT_ID=1
  
  # Kafka (Confluent Cloud free tier)
  KAFKA_BOOTSTRAP_SERVERS=pkc-xxxx.confluent.cloud:9092
  KAFKA_API_KEY=xxxx
  KAFKA_API_SECRET=xxxx
  
  # ClickHouse Cloud
  CLICKHOUSE_URL=https://xxxx.clickhouse.cloud:8443
  CLICKHOUSE_USER=default
  CLICKHOUSE_PASSWORD=xxxx
  
  # Open Exchange Rates (free tier)
  OPEN_EXCHANGE_RATES_APP_ID=xxxx
  
  # HF Space URL (used by Render)
  HF_SPACE_URL=https://{hf_username}-hotel-booking-ai.hf.space
```

---

## ═══════════════════════════════════════════════
## SECTION 7 — CODE QUALITY RULES (ALWAYS APPLY)
## ═══════════════════════════════════════════════

```
When Copilot generates ANY code for this project, enforce these rules:

1. ASYNC EVERYWHERE
   All FastAPI endpoints and database calls MUST be async.
   Use httpx.AsyncClient (never requests).
   Use asyncpg or supabase-py async client.
   Never use time.sleep() — use asyncio.sleep().

2. TYPE HINTS ALWAYS
   Every function signature must have complete type hints.
   Return types must be specified.
   Use Pydantic models for all FastAPI request/response bodies.

3. ERROR HANDLING PATTERN
   Every service call must have try/except with:
     - Specific exception type (not bare Exception)
     - Structured error response: {"error": "...", "code": "...", "fallback": "..."}
     - Fallback behavior (never let an error crash the conversation)
   
4. LOGGING STANDARD
   import structlog
   log = structlog.get_logger()
   Every endpoint logs: psid (hashed), intent, response_time_ms, success/fail
   Never log: full card numbers, passwords, passport numbers, raw personal data

5. ENVIRONMENT VARIABLES
   Never hardcode ANY value that could differ between environments.
   Never hardcode model names — always from os.environ with defaults.
   Use pydantic-settings for config management.

6. MESSENGER RESPONSE VALIDATION
   Before returning ANY messages list:
   - Verify no text > 2000 chars
   - Verify no carousel > 10 cards  
   - Verify no quick replies > 13
   - Verify all button titles <= 20 chars
   Use: from render_webhook.messenger_builder import validate_messages

7. IDEMPOTENCY
   All payment operations: idempotency_key = f"booking-{booking_id}"
   All notification sends: check Redis "sent:{type}:{booking_id}" before sending
   All Celery tasks: check idempotency key in Upstash Redis before executing

8. STATE MACHINE
   User conversation state MUST always be one of:
   new | greeting | searching | viewing_hotels | selecting_room |
   choosing_rate | filling_guest_form | selecting_addons | reviewing_booking |
   paying | booking_confirmed | modifying | cancelling | faq_browsing |
   handoff_active | post_stay_review | loyalty_browsing
   
   Every state transition must update: Upstash Redis user:{psid}:state
   Invalid state transitions must log warning and reset to appropriate state.
```

---

## ═══════════════════════════════════════════════
## HOW TO USE THIS FILE WITH GITHUB COPILOT
## ═══════════════════════════════════════════════

```
STEP 1 — Attach context before every session:
  Open this file in VS Code.
  In Copilot Chat, type:
  "@workspace Using COPILOT_MASTER_PROMPT.md as full context, [paste task prompt]"

STEP 2 — For new module work, paste ONE task prompt at a time:
  Copy the exact text from "TASK PROMPT X" section above.
  Paste into Copilot Chat.
  Review generated code against SKILL.md rules before accepting.

STEP 3 — For debugging / upgrading existing code:
  "Using COPILOT_MASTER_PROMPT.md context, review [filename] and:
   1. Check it follows all Code Quality Rules from Section 7
   2. Ensure all Messenger responses use MessengerResponse builder
   3. Verify button-first UX law (>50% interactions via buttons)
   4. Confirm state transitions update Upstash Redis correctly"

STEP 4 — For adding a feature not in this file:
  "Using COPILOT_MASTER_PROMPT.md context, add [feature].
   Constraints:
   - Must run in HF Space (not Render)
   - Must use Upstash Redis for state (not local Redis)
   - Must return Messenger-formatted message objects
   - Must follow button-first UX law
   - Must handle users from any country/language automatically"
```
