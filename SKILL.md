# SKILL.md — Hotel Booking AI Chatbot System
# GitHub Copilot Context File
# Read this entire file before generating ANY code for this project.

---

## 🧭 WHAT THIS PROJECT IS

A production-grade **Hotel Booking AI Chatbot** delivered via **Facebook Messenger**.
Users can search hotels, book rooms, manage reservations, and interact in 15+ languages — all through conversation.

**The system has 4 logical parts:**
- **BRAIN** — AI Engine, NLP, language detection, RAG, voice, handoff
- **BOOK** — Room selection, guest collection, payment, confirmation
- **MANAGE** — Modifications, cancellations, group/corporate bookings, overbooking
- **GROW** — Loyalty, reviews, notifications, analytics, DevOps

---

## 🏗️ MANDATORY TECHNOLOGY STACK

> Copilot MUST use ONLY these technologies. Never suggest alternatives unless explicitly asked.

| Layer | Technology | Purpose |
|---|---|---|
| **Chat Channel** | Facebook Messenger (Meta Graph API v18+) | Primary user interface |
| **Bot Framework** | Rasa Open Source 3.x | NLU + dialogue management |
| **AI/ML Hosting** | HuggingFace Spaces (Gradio/FastAPI) | LaBSE embeddings, Whisper STT, Llama 3 inference |
| **Backend Services** | Render.com (Web Services + Background Workers) | All FastAPI microservices |
| **Primary Database** | Supabase (PostgreSQL) | All persistent data — bookings, guests, hotels, rooms, loyalty |
| **Cache & Session Store** | Redis (Render Redis or Upstash Redis) | Sessions, soft locks, leaderboards, rate limits, TTS cache |
| **Message Queue** | Kafka (Confluent Cloud free tier) | Async events between services |
| **Vector Database** | Qdrant Cloud (free tier) | Hotel FAQ embeddings for RAG |
| **Search** | Elasticsearch (Elastic Cloud free trial OR Bonsai.io) | Hotel search with geo-filters |
| **File Storage** | Supabase Storage (S3-compatible) | PDFs, QR codes, hotel images |
| **Payments** | Stripe | Card tokenisation, PaymentIntents, refunds |
| **Email** | SendGrid (free 100/day) | Transactional emails |
| **SMS/WhatsApp** | Meta WhatsApp Business Cloud API | Booking alerts, OTP |
| **Live Agent** | Chatwoot (self-hosted on Render) | Human handoff |
| **Analytics** | ClickHouse Cloud (free tier) | Booking events, NLU logs |
| **CI/CD** | GitHub Actions | Lint, test, deploy to Render |
| **Monitoring** | Render native metrics + Sentry | Error tracking |

---

## 📡 FACEBOOK MESSENGER INTEGRATION — CRITICAL RULES

### Messenger Entry Point
All user messages arrive via Messenger Webhook → Rasa Connector → Rasa Core.

```
User (Messenger) → Meta Graph API Webhook
  → POST https://{render-rasa-url}/webhooks/messenger/webhook
  → Rasa processes → returns text/quick_replies/cards
  → Rasa POSTs reply back to Graph API
    POST https://graph.facebook.com/v18.0/me/messages
    Headers: { Authorization: Bearer {PAGE_ACCESS_TOKEN} }
```

### File: `channels/messenger_connector.py`
```python
# ALWAYS implement this connector — Rasa's built-in Messenger connector
# Input channel: Facebook Messenger webhook events
# Register in credentials.yml

from rasa.core.channels.facebook import MessengerBot, MessengerInput

# credentials.yml must have:
# facebook:
#   verify: "MESSENGER_VERIFY_TOKEN"     # your chosen verify string
#   secret: "FACEBOOK_APP_SECRET"        # from Meta app dashboard
#   page-access-token: "PAGE_ACCESS_TOKEN"  # from Meta Page dashboard

# All replies MUST use Messenger-compatible formats:
# - Text: { "text": "..." }
# - Quick Replies: { "text": "...", "quick_replies": [{"content_type":"text","title":"Yes","payload":"YES"}] }
# - Generic Template (hotel cards): { "attachment": { "type": "template", "payload": { "template_type": "generic", "elements": [...] } } }
# - Image: { "attachment": { "type": "image", "payload": { "url": "..." } } }
# - File (PDF): { "attachment": { "type": "file", "payload": { "url": "..." } } }
```

### Messenger Message Size Limits
- Text messages: max **2000 characters** — split longer responses
- Generic template: max **10 cards** per carousel
- Quick replies: max **13 buttons** per message
- Button text: max **20 characters**
- Always send typing indicator before bot response: `sender_action: "typing_on"`

### Persistent Menu (set once via Graph API)
```python
# File: scripts/setup_messenger_menu.py
# Run once after deployment to configure Messenger persistent menu
import requests

menu_payload = {
    "persistent_menu": [{
        "locale": "default",
        "composer_input_disabled": False,
        "call_to_actions": [
            {"type": "postback", "title": "🔍 Search Hotels", "payload": "SEARCH_HOTELS"},
            {"type": "postback", "title": "📋 My Bookings", "payload": "MY_BOOKINGS"},
            {"type": "postback", "title": "💬 Talk to Agent", "payload": "HUMAN_HANDOFF"},
            {"type": "postback", "title": "🌐 Change Language", "payload": "CHANGE_LANGUAGE"},
        ]
    }]
}
```

---

## 🗄️ SUPABASE DATABASE — CRITICAL RULES

### Connection Pattern
```python
# File: db/supabase_client.py
# ALWAYS use this pattern — never raw psycopg2 without connection pooling

import os
from supabase import create_client, Client
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Supabase client for CRUD operations
supabase: Client = create_client(
    os.environ["SUPABASE_URL"],       # https://xxxx.supabase.co
    os.environ["SUPABASE_ANON_KEY"]   # or SERVICE_ROLE_KEY for server-side
)

# SQLAlchemy engine for complex queries and transactions
DATABASE_URL = os.environ["SUPABASE_DATABASE_URL"]  # postgresql://postgres:password@host:5432/postgres
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,   # reconnect on stale connections
    pool_recycle=300      # recycle every 5 minutes
)
```

### Complete Database Schema
```sql
-- ============================================================
-- SUPABASE SCHEMA — Run in Supabase SQL Editor
-- Enable Row Level Security (RLS) on all tables
-- ============================================================

-- HOTELS
CREATE TABLE hotels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    country_code CHAR(2) NOT NULL,
    address TEXT,
    lat DECIMAL(9,6),
    lng DECIMAL(9,6),
    star_rating SMALLINT CHECK (star_rating BETWEEN 1 AND 5),
    amenities JSONB DEFAULT '[]',
    cancellation_policy JSONB,  -- {free_days: 7, partial_percent: 50, no_refund_days: 1}
    partner_id UUID REFERENCES hotel_partners(id),
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ROOM TYPES
CREATE TABLE room_types (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotel_id UUID REFERENCES hotels(id) ON DELETE CASCADE,
    name TEXT NOT NULL,           -- "Deluxe King", "Family Suite"
    description TEXT,
    size_sqm SMALLINT,
    max_adults SMALLINT DEFAULT 2,
    max_children SMALLINT DEFAULT 0,
    amenities JSONB DEFAULT '[]', -- ["wifi","minibar","balcony"]
    images JSONB DEFAULT '[]',    -- array of Supabase Storage URLs
    base_price_usd DECIMAL(10,2),
    active BOOLEAN DEFAULT true
);

-- ROOM RATES (rate plans per room)
CREATE TABLE room_rates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    room_type_id UUID REFERENCES room_types(id) ON DELETE CASCADE,
    rate_plan TEXT NOT NULL,      -- "room_only","breakfast","half_board","full_board"
    price_per_night DECIMAL(10,2) NOT NULL,
    currency CHAR(3) NOT NULL DEFAULT 'USD',
    meal_includes JSONB DEFAULT '[]',
    minimum_stay SMALLINT DEFAULT 1,
    valid_from DATE,
    valid_until DATE,
    non_refundable BOOLEAN DEFAULT false
);

-- ROOM AVAILABILITY (one row per room_type per date)
CREATE TABLE room_availability (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    room_type_id UUID REFERENCES room_types(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    available_count SMALLINT NOT NULL DEFAULT 0,
    UNIQUE(room_type_id, date)
);

-- GUESTS
CREATE TABLE guests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    messenger_psid TEXT UNIQUE,   -- Facebook Page-Scoped ID — PRIMARY KEY for Messenger users
    email TEXT UNIQUE,
    phone TEXT,
    first_name TEXT,
    last_name TEXT,
    nationality CHAR(2),
    passport_number TEXT,
    passport_expiry DATE,
    preferred_language CHAR(5) DEFAULT 'en',
    preferred_channel TEXT DEFAULT 'messenger',  -- messenger/whatsapp/email
    gdpr_consent BOOLEAN DEFAULT false,
    gdpr_consent_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- BOOKINGS
CREATE TABLE bookings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    booking_reference TEXT UNIQUE NOT NULL, -- HTL-2026-XYZABC
    guest_id UUID REFERENCES guests(id),
    hotel_id UUID REFERENCES hotels(id),
    room_type_id UUID REFERENCES room_types(id),
    rate_plan TEXT NOT NULL,
    check_in DATE NOT NULL,
    check_out DATE NOT NULL,
    num_adults SMALLINT NOT NULL,
    num_children SMALLINT DEFAULT 0,
    trip_purpose TEXT,             -- business/leisure/honeymoon/family
    dietary_needs TEXT,
    accessibility_needs TEXT,
    addons JSONB DEFAULT '[]',     -- [{addon_id, name, price, currency}]
    subtotal DECIMAL(10,2),
    taxes DECIMAL(10,2),
    total_amount DECIMAL(10,2) NOT NULL,
    currency CHAR(3) NOT NULL,
    status TEXT DEFAULT 'pending', -- pending/confirmed/modified/cancelled/completed
    payment_status TEXT DEFAULT 'unpaid', -- unpaid/paid/refunded/partial_refund
    force_majeure_flag BOOLEAN DEFAULT false,
    review_requested BOOLEAN DEFAULT false,
    review_requested_at TIMESTAMPTZ,
    modification_history JSONB DEFAULT '[]',
    cancellation_reason TEXT,
    cancelled_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- PAYMENTS
CREATE TABLE payments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    booking_id UUID REFERENCES bookings(id),
    gateway TEXT NOT NULL,        -- stripe/paypal/apple_pay
    transaction_id TEXT UNIQUE,   -- Stripe PaymentIntent ID
    amount DECIMAL(10,2) NOT NULL,
    currency CHAR(3) NOT NULL,
    status TEXT NOT NULL,         -- succeeded/failed/refunded
    risk_score SMALLINT,
    stripe_radar_score SMALLINT,
    refund_id TEXT,
    refund_amount DECIMAL(10,2),
    refund_status TEXT,
    refund_initiated_at TIMESTAMPTZ,
    error_code TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ADDONS
CREATE TABLE addons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotel_id UUID REFERENCES hotels(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,                -- spa/dining/transport/activities
    price DECIMAL(10,2) NOT NULL,
    currency CHAR(3) NOT NULL,
    trip_purpose_tags JSONB DEFAULT '[]', -- ["honeymoon","family"]
    recommended_rank SMALLINT DEFAULT 50,
    available BOOLEAN DEFAULT true
);

-- LOYALTY
CREATE TABLE loyalty_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    guest_id UUID UNIQUE REFERENCES guests(id),
    total_points INT DEFAULT 0,
    points_ytd INT DEFAULT 0,     -- resets Jan 1 each year
    tier TEXT DEFAULT 'bronze',   -- bronze/silver/gold/platinum/black
    badges JSONB DEFAULT '[]',
    streak_months INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- LOYALTY TRANSACTIONS
CREATE TABLE loyalty_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    guest_id UUID REFERENCES guests(id),
    booking_id UUID REFERENCES bookings(id),
    action_type TEXT NOT NULL,    -- booking/review/referral/checkin/redemption
    points_delta INT NOT NULL,    -- positive = earned, negative = spent
    multiplier DECIMAL(4,2) DEFAULT 1.0,
    balance_after INT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- REVIEWS
CREATE TABLE reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    booking_id UUID UNIQUE REFERENCES bookings(id),
    guest_id UUID REFERENCES guests(id),
    hotel_id UUID REFERENCES hotels(id),
    overall_score SMALLINT CHECK (overall_score BETWEEN 1 AND 5),
    room_score SMALLINT,
    food_score SMALLINT,
    staff_score SMALLINT,
    location_score SMALLINT,
    review_text TEXT,
    sentiment TEXT,               -- positive/neutral/negative
    sentiment_score DECIMAL(4,3),
    language TEXT DEFAULT 'en',
    alert_sent BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- HOTEL PARTNERS
CREATE TABLE hotel_partners (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    contact_email TEXT,
    commission_percent DECIMAL(4,2) DEFAULT 15.0,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- CORPORATE ACCOUNTS
CREATE TABLE corporate_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name TEXT NOT NULL,
    auto_approve_threshold DECIMAL(10,2) DEFAULT 500.00,
    currency CHAR(3) DEFAULT 'USD',
    billing_email TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- CORPORATE RATES
CREATE TABLE corporate_rates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES corporate_accounts(id),
    hotel_id UUID REFERENCES hotels(id),
    room_type_id UUID REFERENCES room_types(id),
    rate_per_night DECIMAL(10,2) NOT NULL,
    currency CHAR(3) NOT NULL,
    includes_json JSONB DEFAULT '[]', -- ["breakfast","wifi"]
    valid_from DATE,
    valid_until DATE
);

-- GROUP BOOKINGS
CREATE TABLE room_blocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotel_id UUID REFERENCES hotels(id),
    room_type_id UUID REFERENCES room_types(id),
    count SMALLINT NOT NULL,
    check_in DATE NOT NULL,
    check_out DATE NOT NULL,
    coordinator_guest_id UUID REFERENCES guests(id),
    status TEXT DEFAULT 'held',   -- held/confirmed/released/expired
    deposit_schedule JSONB DEFAULT '[]',
    expires_at TIMESTAMPTZ NOT NULL,
    group_booking_url TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- APPROVAL REQUESTS (corporate)
CREATE TABLE approval_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    booking_id UUID REFERENCES bookings(id),
    company_id UUID REFERENCES corporate_accounts(id),
    employee_id UUID REFERENCES guests(id),
    approver_email TEXT NOT NULL,
    status TEXT DEFAULT 'pending', -- pending/approved/rejected/expired
    total_amount DECIMAL(10,2),
    currency CHAR(3),
    created_at TIMESTAMPTZ DEFAULT now(),
    resolved_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ
);

-- SESSION DATA (Rasa tracker store backup)
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT UNIQUE NOT NULL,
    messenger_psid TEXT,
    language CHAR(5),
    guest_id UUID REFERENCES guests(id),
    booking_id UUID REFERENCES bookings(id),
    handoff_active BOOLEAN DEFAULT false,
    chatwoot_conversation_id TEXT,
    started_at TIMESTAMPTZ DEFAULT now(),
    last_active_at TIMESTAMPTZ DEFAULT now()
);

-- COMPLIANCE REQUESTS (GDPR)
CREATE TABLE compliance_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    guest_id UUID REFERENCES guests(id),
    request_type TEXT NOT NULL,   -- export/erase
    status TEXT DEFAULT 'pending',
    request_date TIMESTAMPTZ DEFAULT now(),
    completed_date TIMESTAMPTZ,
    requester_ip TEXT,
    data_categories_erased JSONB
);

-- INDEXES
CREATE INDEX idx_bookings_guest ON bookings(guest_id);
CREATE INDEX idx_bookings_hotel ON bookings(hotel_id);
CREATE INDEX idx_bookings_status ON bookings(status);
CREATE INDEX idx_bookings_checkin ON bookings(check_in);
CREATE INDEX idx_room_availability_date ON room_availability(room_type_id, date);
CREATE INDEX idx_guests_psid ON guests(messenger_psid);
CREATE INDEX idx_payments_booking ON payments(booking_id);
CREATE INDEX idx_reviews_hotel ON reviews(hotel_id);

-- ROW LEVEL SECURITY (enable on all tables)
ALTER TABLE guests ENABLE ROW LEVEL SECURITY;
ALTER TABLE bookings ENABLE ROW LEVEL SECURITY;
ALTER TABLE payments ENABLE ROW LEVEL SECURITY;
ALTER TABLE loyalty_accounts ENABLE ROW LEVEL SECURITY;
-- Service role key bypasses RLS — use for server-side operations
```

---

## 🔴 REDIS — CRITICAL RULES

### Connection Pattern
```python
# File: db/redis_client.py
# ALWAYS use this Redis client — used for sessions, locks, caches, rate limits

import os
import redis.asyncio as aioredis
import redis

# Async client (for FastAPI)
async_redis = aioredis.from_url(
    os.environ["REDIS_URL"],  # redis://default:password@host:port
    encoding="utf-8",
    decode_responses=True,
    max_connections=20
)

# Sync client (for Celery tasks, Rasa actions)
sync_redis = redis.from_url(
    os.environ["REDIS_URL"],
    encoding="utf-8",
    decode_responses=True,
    max_connections=20
)
```

### Redis Key Naming Convention — ALWAYS USE THESE EXACT KEY PATTERNS
```
# Session / Language
session:{session_id}:lang          → detected language code, EX 3600
session:{session_id}:user_id       → mapped guest UUID, EX 3600
session:{session_id}:booking_state → JSON of in-progress booking, EX 1800

# Room Soft Locks
soft_lock:{room_type_id}:{date}    → session_id holding lock, EX {lock_minutes*60}

# Rate Limiting
user:{user_id}:payment_attempts:1h → INCR counter, EX 3600
user:{user_id}:login_attempts      → INCR counter, EX 900 (15min lockout)

# TTS Audio Cache
tts_cache:{lang}:{text_hash}       → MP3 bytes (base64), EX 86400

# Leaderboard
leaderboard:monthly:{YYYYMM}       → ZADD sorted set, score=points, member=user_id

# Booking Month Streak
user:{user_id}:booking_months      → ZADD sorted set, score=monthstamp

# Rasa Tracker Store (async)
rasa_tracker:{sender_id}           → JSON tracker state, EX 86400

# Token Blacklist (logout)
blacklist:{refresh_token_hash}     → "1", EX 2592000 (30 days)

# Force Majeure Flags
force_majeure:{country_code}       → JSON event data, EX 43200 (12h)
```

---

## 🤗 HUGGINGFACE SPACES — CRITICAL RULES

HuggingFace Spaces hosts all AI/ML inference endpoints. Deploy each as a **separate Space**.

### Space 1: LaBSE Embeddings
```python
# Space name: hotel-booking-embeddings
# File: app.py (Gradio + FastAPI)
# Hardware: CPU Basic (free)

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()
model = SentenceTransformer('sentence-transformers/LaBSE')  # Load once at startup

@app.post("/embed")
async def embed(payload: dict):
    # Input: {"texts": ["Is there a pool?"], "batch": true}
    # Output: {"vectors": [[0.023, -0.114, ...]], "dim": 768}
    texts = payload.get("texts", [])
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    return {"vectors": vectors, "dim": 768}

# Mount FastAPI on Gradio for HuggingFace Spaces compatibility
import gradio as gr
demo = gr.Interface(fn=lambda x: embed({"texts": [x]}), inputs="text", outputs="json")
app = gr.mount_gradio_app(app, demo, path="/ui")
```

### Space 2: Whisper STT
```python
# Space name: hotel-booking-stt
# File: app.py
# Hardware: CPU Upgrade (or T4 GPU for production)

from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

@app.post("/transcribe")
async def transcribe(audio_file: UploadFile):
    # Input: multipart audio file (WAV 16kHz mono)
    # Output: {"transcript": "...", "language": "en", "confidence": 0.94}
    wav_bytes = await audio_file.read()
    segments, info = model.transcribe(wav_bytes, beam_size=5)
    transcript = " ".join([s.text for s in segments])
    return {
        "transcript": transcript,
        "language": info.language,
        "confidence": float(np.exp(info.language_probability))
    }
```

### Space 3: Llama 3 RAG Generator
```python
# Space name: hotel-booking-llm
# File: app.py
# Hardware: T4 GPU (Spaces GPU grant) OR use Groq API as fallback

@app.post("/generate")
async def generate(payload: dict):
    # Input: {"question": "...", "context_chunks": [...], "language": "en"}
    # Output: {"answer": "...", "source": "rag"}
    question = payload["question"]
    chunks = payload["context_chunks"]
    context = "\n".join([c["payload"]["answer"] for c in chunks])
    
    prompt = f"""You are a helpful hotel concierge. Using ONLY the hotel information below, answer the guest's question in a friendly, concise tone. If not available, say so politely.

Hotel Information:
{context}

Guest Question: {question}

Answer:"""
    # Use transformers pipeline or Groq API as fallback
    # Return answer + translate if target_language != 'en'
```

### Calling HuggingFace Spaces from Render Services
```python
# File: services/hf_client.py
# ALWAYS use this client to call HF Spaces endpoints

import httpx
import os

HF_EMBED_URL = os.environ["HF_EMBED_URL"]   # https://user-hotel-booking-embeddings.hf.space
HF_STT_URL = os.environ["HF_STT_URL"]
HF_LLM_URL = os.environ["HF_LLM_URL"]
HF_TOKEN = os.environ["HF_TOKEN"]           # HuggingFace API token (for private spaces)

async def get_embedding(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{HF_EMBED_URL}/embed",
            json={"texts": texts},
            headers={"Authorization": f"Bearer {HF_TOKEN}"}
        )
        resp.raise_for_status()
        return resp.json()["vectors"]

async def transcribe_audio(wav_bytes: bytes) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{HF_STT_URL}/transcribe",
            files={"audio_file": ("audio.wav", wav_bytes, "audio/wav")},
            headers={"Authorization": f"Bearer {HF_TOKEN}"}
        )
        return resp.json()

async def generate_answer(question: str, chunks: list, language: str) -> dict:
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(
            f"{HF_LLM_URL}/generate",
            json={"question": question, "context_chunks": chunks, "language": language},
            headers={"Authorization": f"Bearer {HF_TOKEN}"}
        )
        return resp.json()
```

---

## 🚀 RENDER DEPLOYMENT — CRITICAL RULES

### Services on Render
Every microservice is a separate Render **Web Service** unless noted.

```
render.yaml (root of project):
services:
  - name: rasa-core           # Rasa action server + core
    type: web
    runtime: docker
    plan: standard
    envVars: [from .env.brain]
    
  - name: language-service    # FastAPI language detection
    type: web
    runtime: python
    buildCommand: pip install -r services/language_service/requirements.txt
    startCommand: uvicorn services.language_service.main:app --host 0.0.0.0 --port 10000
    
  - name: search-service      # FastAPI hotel search
    type: web
    runtime: python
    
  - name: rag-service         # FastAPI RAG pipeline (calls HF Spaces)
    type: web
    runtime: python
    
  - name: booking-service     # FastAPI booking core
    type: web
    runtime: python
    
  - name: payment-service     # FastAPI payment (Stripe)
    type: web
    runtime: python
    
  - name: notification-service # FastAPI email/WhatsApp/push
    type: web
    runtime: python
    
  - name: loyalty-service     # FastAPI loyalty + gamification
    type: web
    runtime: python
    
  - name: analytics-consumer  # Kafka consumer → ClickHouse
    type: worker               # Background worker, not web
    runtime: python
    
  - name: celery-worker       # Celery for async tasks
    type: worker
    runtime: python
    startCommand: celery -A tasks.celery_app worker --loglevel=info
    
  - name: celery-beat         # Celery scheduler
    type: worker
    runtime: python
    startCommand: celery -A tasks.celery_app beat --loglevel=info

  - name: redis               # Render managed Redis
    type: redis
    plan: free
    ipAllowList: []           # internal only
```

### Render Environment Variables (set in Render Dashboard)
```bash
# All services get these base vars
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...  # server-side only
SUPABASE_DATABASE_URL=postgresql://postgres:password@db.xxxx.supabase.co:5432/postgres
REDIS_URL=redis://default:password@oregon-redis.render.com:6379

# BRAIN service vars
HF_EMBED_URL=https://user-hotel-booking-embeddings.hf.space
HF_STT_URL=https://user-hotel-booking-stt.hf.space
HF_LLM_URL=https://user-hotel-booking-llm.hf.space
HF_TOKEN=hf_xxxxxxxxxxxx
QDRANT_URL=https://xxxx.qdrant.io
QDRANT_API_KEY=xxxx
ELASTICSEARCH_URL=https://xxxx.bonsai.io
ELASTICSEARCH_USER=xxxx
ELASTICSEARCH_PASS=xxxx
CHATWOOT_API_URL=https://your-chatwoot.render.com
CHATWOOT_API_TOKEN=xxxx
CHATWOOT_ACCOUNT_ID=1

# BOOK service vars  
STRIPE_SECRET_KEY=sk_live_xxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxx
STRIPE_PUBLISHABLE_KEY=pk_live_xxxx

# Messenger vars
FACEBOOK_PAGE_ACCESS_TOKEN=EAAxxxx
FACEBOOK_APP_SECRET=xxxx
FACEBOOK_VERIFY_TOKEN=your_chosen_verify_string  # you pick this

# Meta WhatsApp
WHATSAPP_PHONE_NUMBER_ID=xxxx
WHATSAPP_BUSINESS_ACCOUNT_ID=xxxx

# Notification
SENDGRID_API_KEY=SG.xxxx

# Kafka
KAFKA_BOOTSTRAP_SERVERS=pkc-xxxx.confluent.cloud:9092
KAFKA_API_KEY=xxxx
KAFKA_API_SECRET=xxxx

# Analytics
CLICKHOUSE_URL=https://xxxx.clickhouse.cloud
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=xxxx
```

---

## 🧠 PART 1 — BRAIN: AI ENGINE

### Module 1 — Language Detection

**File: `services/language_service/detector.py`**
```python
# Input: text (str), session_id (str)
# Output: { language_code, confidence, is_rtl, tier }
# Rules:
# - Use langdetect as primary (< 10ms)
# - XLM-RoBERTa as fallback if confidence < 0.85
# - Empty/emoji-only messages → default 'en'
# - RTL languages: ar, he, fa, ur, yi → is_rtl: True
# - Tier 1 (full support): en, ar, fr, es, de, ja, zh, ko, pt, ru, hi, id, tr, th, vi
# - Tier 2 (partial): other common languages
# - Tier 3 (basic): fallback to EN responses
```

**File: `services/language_service/main.py`**
```python
# FastAPI app
# POST /detect → { language_code, greeting, user_context }
# Stores in Redis: SET session:{session_id}:lang '{code}' EX 3600
# Loads returning user from Supabase if session maps to known guest
```

**File: `services/language_service/translator.py`**
```python
# Translation layer for all multilingual responses
# Primary: Helsinki-NLP/opus-mt-{src}-{tgt} models on HuggingFace Spaces
# Fallback: LibreTranslate self-hosted on Render (free tier)
# Cache translations in Redis: SET trans:{src}:{tgt}:{text_hash} EX 86400
```

### Module 2 — Hotel Search

**File: `services/search_service/elasticsearch_client.py`**
```python
# Input: city, check_in, check_out, guests, filters{stars, max_price, amenities}
# Output: { hits: [{hotel_id, name, stars, price_from, distance_km, amenities, thumbnail_url}], total, facets }
# Rules:
# - Use Elasticsearch python client 8.x
# - Bool query: must match city, filter date range, filter stars
# - Geo-distance filter if lat/lng provided
# - Multi-language analyzer matching session language
# - Aggregation for facets (star rating, price, amenities)
# - Store hotel data in ES at partner onboarding
```

**File: `services/search_service/ranker.py`**
```python
# Input: ES hits + user preference vector from Qdrant (returning user)
# Scoring: 0.4*ES_relevance + 0.3*guest_rating + 0.2*availability + 0.1*personalization
# Output: top 5 ranked hotels with display-ready data
```

### Module 11 — FAQ RAG Pipeline

**File: `services/rag_service/embedder.py`**
```python
# Input: text (str, any language)
# Output: vector (768-dim float list)
# Calls HF_EMBED_URL via hf_client.get_embedding()
# LaBSE handles all 109 languages — NO language-specific handling needed
```

**File: `services/rag_service/qdrant_client.py`**
```python
# Input: query_vector, hotel_id, top_k=3
# Output: [{score, payload:{question, answer, category, hotel_id}}]
# Collection name: 'hotel_faqs'
# Filter: hotel_id == provided hotel_id (ALWAYS filter by hotel)
# Onboarding: batch index all FAQ entries for new hotel at setup
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
```

**File: `services/rag_service/llm_generator.py`**
```python
# Input: question, retrieved_chunks, target_language
# Output: { answer, source: 'rag' }
# Calls HF_LLM_URL via hf_client.generate_answer()
# Prompt: "You are a helpful hotel concierge. Using ONLY the information below..."
# If target_language != 'en': pass through translator.py
# Fallback: return top Qdrant chunk answer directly if LLM unavailable
```

### Module 13 — Human Handoff (Chatwoot)

**File: `services/handoff_service/main.py`**
```python
# POST /handoff
# Input: { session_id, conversation_history, user_profile, reason }
# Output: { chatwoot_conversation_id, agent_name, estimated_wait }
# Creates Chatwoot conversation via POST /api/v1/accounts/{id}/conversations
# Posts full history as initial message so agent has context
# Assigns to language-matched agent queue
# Webhook: POST /handoff/resolved → re-enable bot when agent closes
```

### Module 14 — Voice Interaction

**File: `services/voice_service/audio_utils.py`**
```python
# Input: audio_bytes, source_format (webm/ogg/wav/mp3)
# Output: wav_bytes (16kHz mono PCM WAV)
# Uses Pydub + FFmpeg: convert → normalize (-23 LUFS) → WebRTC VAD
# Reject audio > 60 seconds
```

**File: `services/voice_service/stt.py`**
```python
# Input: wav_bytes
# Output: { transcript, language, confidence }
# Calls HF_STT_URL via hf_client.transcribe_audio()
# Confidence < 0.6 → request re-speak
```

**File: `services/voice_service/tts.py`**
```python
# Input: text, language_code, voice_gender, speed
# Output: { audio_bytes, format: 'mp3', duration_seconds }
# Primary: Coqui XTTS-v2 (17 languages) on HF Space
# Fallback: eSpeak-NG for other languages
# Cache common phrases in Redis (tts_cache:{lang}:{hash})
```

**File: `services/voice_service/main.py`**
```python
# WebSocket endpoint: wss://{render-url}/voice/ws/{session_id}
# Messenger does NOT support native voice WebSocket —
# Voice widget is embedded in a Messenger Webview button
# Flow: binary audio chunks → VAD silence detect → STT → Rasa → TTS → stream back
```

---

## 🏨 PART 2 — BOOK: BOOKING FLOW

### Module 3 — Room Selection

**File: `services/room_service/availability.py`**
```python
# Input: hotel_id, room_type_id, check_in, check_out, num_guests
# Output: { available, available_count, min_price_per_night, total_price, currency }
# Query Supabase room_availability table
# Check Redis cache first (90-day pre-loaded window)
# available_count = MIN of all daily counts across stay (bottleneck night)
# Verify room capacity >= requested guests
```

**File: `services/room_service/soft_lock.py`**
```python
# Input: room_type_id, date_range, session_id, lock_duration_minutes=15
# Output: { locked, expires_at, lock_id }
# Redis key: soft_lock:{room_type_id}:{date}
# Lua script for atomic multi-date lock (all-or-nothing)
# /refresh endpoint called every 5 min from frontend during payment
```

### Module 4 — Guest Information

**File: `data/forms/booking_form.yml`**
```yaml
# Required slots (Rasa Form):
# guest_first_name, guest_last_name, guest_email, guest_phone
# num_adults, num_children, trip_purpose, dietary_needs, accessibility_needs
# Optional: passport_number, passport_expiry, passport_country
# ALL utter_ask_{slot} responses must exist in domain.yml in Tier 1 languages
```

**File: `services/guest_service/validator.py`**
```python
# Input: field_name, value, context
# Output: { valid, normalized_value, error }
# email: RFC-5322 regex + common typo detection (gmail.con etc.)
# phone: phonenumbers library → E.164 format (+1234567890)
# passport_expiry: must be > check_out + 6 months
# guest_name: min 2 chars, no all-numeric, strip whitespace
```

**File: `services/guest_service/ocr.py`**
```python
# Input: image_bytes, image_format
# Output: { first_name, last_name, passport_number, expiry, confidence }
# Pre-process: OpenCV deskew + sharpen + grayscale
# OCR: Tesseract with MRZ language data
# Validate: ISO 7501 checksum on MRZ line 2
# Low confidence fields flagged for user confirmation
```

### Module 5 — Add-Ons & Upsell

**File: `services/addon_service/recommender.py`**
```python
# Input: trip_purpose, num_children, num_adults, hotel_id
# Output: [ {addon_id, name, price, currency, recommended} ]
# Honeymoon → champagne, spa, romantic dinner (highest rank)
# Family → crib, extra bed, kids club, babysitting (highest rank)
# Business → late checkout, meeting room, airport transfer (highest rank)
# Query Supabase addons table, apply trip_purpose score multiplier
# Present TOP 3 recommendations; user can say 'show all' for full catalogue
```

### Module 6 — Payment (Stripe)

**File: `services/payment_service/fraud_check.py`**
```python
# Input: payment_metadata, user_history
# Output: { risk_score 0-100, action: allow/require_3ds/block, reason }
# Rules:
# - amount > 10x user average → flag
# - new account + high value → flag  
# - card_country != user_country → +score
# - Redis velocity: > 5 payment attempts/hour → block
# - Stripe Radar: forward metadata for ML scoring
# - 0-30: allow, 31-70: require_3ds, 71-100: block
```

**File: `services/payment_service/stripe_adapter.py`**
```python
# Input: stripe_token, amount, currency, booking_id, require_3ds
# Output: { success, transaction_id, amount_charged, currency, receipt_url }
# CRITICAL: Card data NEVER passes through Rasa or backend — Stripe.js client-side only
# PaymentIntent with idempotency_key = f'booking-{booking_id}'
# 3DS: return client_secret → Stripe.js handles auth UI on Messenger Webview
# Webhook /payment/webhook: verify HMAC → process payment.succeeded/failed
# On success: publish Kafka 'payment.completed' event
```

**File: `services/payment_service/main.py`**
```python
# POST /pay: adapter pattern for stripe/paypal/apple_pay
# Pre-payment: fraud_check.py (abort if risk >= 71)
# Post-payment: write to Supabase payments table
# Return result within 10-second timeout to Rasa
```

### Module 7 — Confirmation

**File: `services/confirmation_service/pdf_generator.py`**
```python
# Input: full booking_data dict
# Output: { pdf_bytes, storage_url, pages }
# Use ReportLab for PDF generation
# Store PDF in Supabase Storage bucket: 'booking-vouchers'
# Key: {booking_reference}.pdf
# Embed QR code as inline image
```

**File: `services/confirmation_service/qr_generator.py`**
```python
# Input: booking_reference, check_in_url
# Output: { qr_png_bytes, payload }
# qrcode library, 300x300, error correction H
# Payload: JSON with booking_ref + check_in deep link
```

**File: `services/confirmation_service/calendar.py`**
```python
# Input: hotel details + booking dates
# Output: ics_bytes
# icalendar library: check_in 15:00, check_out 11:00
# VALARM: 24h before check_in reminder
# Attach to email + offer as download link in Messenger chat
```

---

## ⚙️ PART 3 — MANAGE: LIFECYCLE

### Module 8 — Modification

**File: `services/modification_service/policy_engine.py`**
```python
# Input: booking_id, change_type, change_timestamp, new_values
# Output: { allowed, fee, fee_currency, policy_tier, explanation }
# Fetch hotel cancellation_policy from Supabase
# Tiers: > 7 days = free, 3-7 days = fee, < 3 days = restricted, < 24h = locked
# Non-refundable rate plan = no date change EVER
# Return human-readable explanation in user's language
```

**File: `services/modification_service/rebooking.py`**
```python
# Input: booking_id, new_check_in, new_check_out
# Output: { available, price_diff, price_diff_currency, action_required }
# Release old soft lock → check new availability → calculate price_diff
# price_diff > 0: charge additional via Stripe
# price_diff < 0: partial refund via stripe_refund.py
# Update Supabase bookings table, append to modification_history JSONB
```

### Module 9 — Cancellation

**File: `services/cancellation_service/refund_engine.py`**
```python
# Input: booking_id, cancellation_timestamp
# Output: { refund_amount, currency, refund_percent, timeline_days, policy_applied }
# ALWAYS show refund amount BEFORE asking for cancellation confirmation
# Fetch policy from Supabase hotel_policies
# Force majeure override: if Redis force_majeure:{country} exists → 100% refund
```

**File: `services/cancellation_service/stripe_refund.py`**
```python
# Input: transaction_id, refund_amount, reason
# Output: { refund_id, status, timeline, amount_refunded }
# stripe.Refund.create() with idempotency key
# Update Supabase payments table
# Publish Kafka 'booking.refunded' event
```

### Module 15 — Group Booking

**File: `services/group_service/room_block.py`**
```python
# Input: hotel_id, room_type_id, requested_count, check_in, check_out, coordinator_id
# Output: { block_id, rooms_held, expires_at, group_booking_url }
# Verify availability across all nights (bottleneck check)
# Insert room_blocks record with 48h expiry
# Reduce room_availability counts for blocked dates
# Publish Kafka 'room_block.created'
```

**File: `services/group_service/deposit_scheduler.py`**
```python
# Input: total_amount, currency, check_in_date, contract_type
# Output: { schedule: [{due_date, amount, percent}] }
# Standard: 30% now, 40% at 60 days, 30% at 14 days
# Create Celery periodic tasks for each milestone
# 7-day reminder emails to coordinator before each milestone
```

### Module 16 — Corporate

**File: `services/corporate_service/rate_lookup.py`**
```python
# Input: company_id, hotel_id, room_type_id, check_in, check_out
# Output: { rate_type, price_per_night, currency, includes, company_name }
# Query Supabase corporate_rates table with date validity check
# Parse includes_json → auto-add amenities without charging
```

**File: `services/corporate_service/approval_engine.py`**
```python
# Input: booking_total, currency, cost_center, employee_id, company_id
# Output: { approved, pending, approver_email, request_id, expires_at }
# Fetch auto_approve_threshold from corporate_accounts
# If below threshold: approved immediately
# If above: create approval_requests record, email approver via SendGrid
# Webhook POST /corporate/approve/{request_id}
# Auto-expire 24h via Celery task
```

### Module 17 — Overbooking & Force Majeure

**File: `services/force_majeure_service/news_monitor.py`**
```python
# Celery task: runs every 6 hours
# Queries NewsAPI free tier for travel bans + natural disasters
# NLP: keyword + country extraction
# Cross-reference with Supabase active bookings
# If match: set Redis force_majeure:{country_code}
# Push to Chatwoot human review queue before mass auto-refund
```

**File: `services/force_majeure_service/relocation.py`**
```python
# Input: original_booking, search_radius_km=5
# Output: { alternatives: [{hotel_id, name, distance_km, price_diff}] }
# Get hotel coords from Supabase
# Call search_service with geo-distance filter
# Filter: alternative price <= original + 20%
# Soft-lock alternative for 2 hours
# Expand radius 5→10→20km if no matches
```

---

## 📊 PART 4 — GROW: ENGAGEMENT

### Module 10 — Loyalty

**File: `services/loyalty_service/points_engine.py`**
```python
# Input: booking_total, currency, user_tier, action_type
# Output: { points_earned, multiplier, breakdown, new_balance }
# Base rate: 1 point per $1 USD
# Tier multipliers: Silver 1.25x, Gold 1.5x, Platinum 2x, Black 3x
# Bonuses: review=50, referral=200, first_booking=500
# Convert all currencies via open exchange rate
# Update Supabase loyalty_accounts + insert loyalty_transactions
```

**File: `services/loyalty_service/tier_engine.py`**
```python
# Input: user_id, total_points_ytd
# Output: { current_tier, next_tier, points_needed, tier_changed }
# Thresholds: Bronze 0, Silver 2000, Gold 5000, Platinum 15000, Black 50000
# Tier upgrade → Kafka 'loyalty.tier_upgraded' event
# Year-end Celery task: check requalification, drop one tier if not met
# Black tier: manual promotion only
```

**File: `services/loyalty_service/gamification.py`**
```python
# Input: user_id, trigger_event
# Output: { badges_newly_awarded, streak_count, leaderboard_rank, total_badges }
# Badges: first_booking, globetrotter (5+ countries), streak (3+ months)
# Redis ZADD for streaks and leaderboard
# Leaderboard: monthly sorted set, refresh on each booking
```

### Module 12 — Post-Stay Review

**File: `services/review_service/celery_tasks.py`**
```python
# Celery beat: every hour, check bookings where checkout + 24h <= now AND review_requested=False
# Route: WhatsApp if available, else email via SendGrid
# Set booking.review_requested=True to prevent duplicates
# One follow-up if no review in 48h
```

**File: `services/review_service/sentiment.py`**
```python
# Input: review_text, language
# Output: { sentiment, score, aspects:{room, food, staff, location} }
# English: VADER (fast, no model)
# Other: translate to EN first, then VADER
# Negative (score < 0.3): Kafka 'review.negative_alert' event
# Hotel partner receives alert within 5 minutes
```

### Notification Engine

**File: `services/notification_service/email_sender.py`**
```python
# Input: recipient_email, template_name, context, attachments
# Output: { sent, message_id, recipient }
# SendGrid API (not SMTP) for Render compatibility
# Jinja2 templates in templates/ folder
# Premailer for CSS inlining (Gmail compatibility)
# Always send HTML + plain text parts
# Attach PDF voucher + ICS calendar for booking confirmations
```

**File: `services/notification_service/whatsapp_sender.py`**
```python
# Input: wa_phone, template_name, template_params
# Output: { wa_message_id, status, timestamp }
# Meta WhatsApp Business Cloud API (free tier)
# POST https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages
# Required templates (pre-approve in Meta Business Manager):
#   booking_confirmation, modification_alert, cancellation_confirmation, 
#   review_request, otp
# Webhook /whatsapp/webhook: delivery receipts + user replies → Rasa
```

**File: `services/notification_service/push_sender.py`**
```python
# Input: device_token, platform, title, body, data
# Output: { delivered, platform, apns_id/fcm_id }
# Android: Firebase FCM v1 API
# iOS: APNs via PyAPNs2 + .p8 key
# Web: pywebpush + VAPID keys
# Deep link data: { action: 'open_booking', booking_id: '...' }
```

---

## 🗂️ RASA CONFIGURATION

### File: `config.yml`
```yaml
language: en
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
    constrain_similarities: true
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
  - name: FallbackClassifier
    threshold: 0.7
    ambiguity_threshold: 0.1
  - name: DucklingEntityExtractor
    url: http://duckling:8000     # Duckling container on Render
    dimensions: [time, duration, number, amount-of-money, distance]
    locale: en_US

policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
    max_history: 10
    epochs: 100
  - name: RulePolicy
  - name: UnexpecTEDIntentPolicy
    max_history: 5
    epochs: 100
```

### File: `credentials.yml`
```yaml
facebook:
  verify: "${FACEBOOK_VERIFY_TOKEN}"
  secret: "${FACEBOOK_APP_SECRET}"
  page-access-token: "${FACEBOOK_PAGE_ACCESS_TOKEN}"

rest:
  # REST channel for testing only
```

### File: `endpoints.yml`
```yaml
action_endpoint:
  url: "http://localhost:5055/webhook"

tracker_store:
  type: redis
  url: "${REDIS_URL}"
  db: 0
  key_prefix: "rasa_tracker"

lock_store:
  type: redis
  url: "${REDIS_URL}"
  db: 1

event_broker:
  type: kafka
  topic: rasa_events
  security_protocol: SASL_SSL
  sasl_mechanism: PLAIN
  sasl_plain_username: "${KAFKA_API_KEY}"
  sasl_plain_password: "${KAFKA_API_SECRET}"
  bootstrap_servers: "${KAFKA_BOOTSTRAP_SERVERS}"
```

### Rasa Domain Key Slots
```yaml
# File: domain.yml (slots section — ALL must be present)
slots:
  # Language & Session
  detected_language: {type: text, mappings: [{type: custom}]}
  session_id: {type: text, mappings: [{type: custom}]}
  
  # Hotel Search
  city: {type: text, mappings: [{type: from_entity, entity: city}]}
  check_in_date: {type: text, mappings: [{type: from_entity, entity: time}]}
  check_out_date: {type: text, mappings: [{type: from_entity, entity: time}]}
  num_adults: {type: float, mappings: [{type: from_entity, entity: number}]}
  num_children: {type: float, mappings: [{type: custom}]}
  search_results: {type: text, mappings: [{type: custom}]}
  
  # Hotel & Room
  selected_hotel_id: {type: text, mappings: [{type: custom}]}
  selected_room_type_id: {type: text, mappings: [{type: custom}]}
  selected_rate_plan: {type: text, mappings: [{type: custom}]}
  soft_lock_id: {type: text, mappings: [{type: custom}]}
  
  # Guest
  guest_first_name: {type: text, mappings: [{type: from_text, conditions: [{active_loop: booking_form}]}]}
  guest_last_name: {type: text, mappings: [{type: from_text, conditions: [{active_loop: booking_form}]}]}
  guest_email: {type: text, mappings: [{type: from_text, conditions: [{active_loop: booking_form}]}]}
  guest_phone: {type: text, mappings: [{type: from_text, conditions: [{active_loop: booking_form}]}]}
  trip_purpose: {type: text, mappings: [{type: from_entity, entity: trip_purpose}]}
  dietary_needs: {type: text, mappings: [{type: from_text, conditions: [{active_loop: booking_form}]}]}
  accessibility_needs: {type: text, mappings: [{type: from_text, conditions: [{active_loop: booking_form}]}]}
  
  # Booking & Payment
  addon_cart: {type: list, mappings: [{type: custom}]}
  booking_reference: {type: text, mappings: [{type: custom}]}
  payment_status: {type: text, mappings: [{type: custom}]}
  
  # Handoff
  handoff_active: {type: bool, mappings: [{type: custom}]}
  chatwoot_conversation_id: {type: text, mappings: [{type: custom}]}
```

### Rasa Intents — MINIMUM EXAMPLES PER INTENT
```
# All intents must have minimum 20 training examples in nlu_en.yml
greet: 25+ examples (Hi, Hello, Hey, Good morning, etc.)
search_hotel: 50+ examples (Find me, Book a room, I need accommodation...)
inform_location: 20+ examples (In Paris, Tokyo, Near the airport...)
inform_dates: 20+ examples (Next weekend, December 25, For 3 nights...)
select_option: 20+ examples (First one, Option 2, The deluxe room...)
confirm: 20+ examples (Yes, Correct, That's right, Go ahead...)
deny: 20+ examples (No, Wrong, That's not right...)
ask_faq: 30+ examples (Do you have wifi, Is breakfast included...)
intent_human_handoff: 20+ examples (Talk to human, Real agent, Human please...)
cancel_booking: 20+ examples (Cancel my booking, I want to cancel...)
modify_booking: 20+ examples (Change my dates, Modify reservation...)
loyalty_check: 20+ examples (How many points, My rewards, My tier...)
```

---

## 🧪 TESTING REQUIREMENTS

### File: `tests/test_messenger_webhook.py`
```python
# Test Facebook Messenger webhook verification (GET request with hub.challenge)
# Test message receive → Rasa routing
# Test postback payloads (menu items)
# Test quick reply responses formatted correctly
# All replies max 2000 chars
```

### File: `tests/test_booking_flow.py`
```python
# Integration test: full booking flow from search to confirmation
# Mock Supabase (use Supabase test project)
# Mock Stripe (use Stripe test mode sk_test_)
# Mock HF Spaces (use httpx MockTransport)
# Verify: soft lock acquired, payment recorded, PDF generated, email sent
```

### File: `tests/test_load.py` (Locust)
```python
# 10,000 concurrent users, 30 minutes
# Mix: 60% search, 30% full booking, 10% cancel/modify
# SLAs: search < 500ms P99, payment < 3s P99, NLU < 200ms P99
```

---

## 📁 COMPLETE FILE STRUCTURE

```
hotel-booking-ai/
├── .github/workflows/
│   └── ci.yml                         # GitHub Actions CI/CD → Render deploy
├── rasa/
│   ├── config.yml
│   ├── credentials.yml
│   ├── endpoints.yml
│   ├── domain.yml
│   ├── data/
│   │   ├── nlu/
│   │   │   ├── nlu_en.yml
│   │   │   ├── nlu_ar.yml
│   │   │   └── nlu_fr.yml             # + 12 more tier 1 languages
│   │   ├── stories/
│   │   │   ├── booking_stories.yml
│   │   │   ├── manage_stories.yml
│   │   │   └── grow_stories.yml
│   │   ├── rules/
│   │   │   ├── core_rules.yml
│   │   │   └── fallback_rules.yml
│   │   └── forms/
│   │       └── booking_form.yml
│   └── actions/
│       ├── actions_brain.py            # ActionGreetUser, ActionSearchHotels, ActionFAQ
│       ├── actions_book.py             # ActionShowRooms, ActionUpsellAddons, ActionSendConfirmation
│       ├── actions_manage.py           # ActionModifyBooking, ActionCancelBooking
│       └── actions_grow.py             # ActionShowLoyalty, ActionReviewFlow
├── services/
│   ├── language_service/
│   │   ├── detector.py
│   │   ├── translator.py
│   │   └── main.py
│   ├── search_service/
│   │   ├── elasticsearch_client.py
│   │   ├── ranker.py
│   │   └── main.py
│   ├── rag_service/
│   │   ├── embedder.py
│   │   ├── qdrant_client.py
│   │   ├── llm_generator.py
│   │   └── main.py
│   ├── voice_service/
│   │   ├── audio_utils.py
│   │   ├── stt.py
│   │   ├── tts.py
│   │   └── main.py
│   ├── handoff_service/
│   │   └── main.py
│   ├── room_service/
│   │   ├── availability.py
│   │   ├── soft_lock.py
│   │   └── main.py
│   ├── guest_service/
│   │   ├── validator.py
│   │   ├── ocr.py
│   │   └── main.py
│   ├── addon_service/
│   │   ├── recommender.py
│   │   └── main.py
│   ├── payment_service/
│   │   ├── fraud_check.py
│   │   ├── stripe_adapter.py
│   │   └── main.py
│   ├── confirmation_service/
│   │   ├── pdf_generator.py
│   │   ├── qr_generator.py
│   │   ├── calendar.py
│   │   └── main.py
│   ├── modification_service/
│   │   ├── policy_engine.py
│   │   ├── rebooking.py
│   │   └── main.py
│   ├── cancellation_service/
│   │   ├── refund_engine.py
│   │   ├── stripe_refund.py
│   │   └── main.py
│   ├── group_service/
│   │   ├── room_block.py
│   │   ├── deposit_scheduler.py
│   │   └── main.py
│   ├── corporate_service/
│   │   ├── rate_lookup.py
│   │   ├── approval_engine.py
│   │   └── main.py
│   ├── force_majeure_service/
│   │   ├── news_monitor.py
│   │   ├── relocation.py
│   │   └── main.py
│   ├── loyalty_service/
│   │   ├── points_engine.py
│   │   ├── tier_engine.py
│   │   ├── gamification.py
│   │   └── main.py
│   ├── review_service/
│   │   ├── celery_tasks.py
│   │   ├── sentiment.py
│   │   └── main.py
│   ├── notification_service/
│   │   ├── email_sender.py
│   │   ├── whatsapp_sender.py
│   │   ├── push_sender.py
│   │   └── main.py
│   ├── analytics_service/
│   │   ├── kafka_consumer.py
│   │   └── clickhouse_schema.sql
│   ├── auth_service/
│   │   └── main.py
│   └── maps_service/
│       └── nominatim.py
├── db/
│   ├── supabase_client.py
│   ├── redis_client.py
│   └── schema.sql                     # Full Supabase schema (above)
├── channels/
│   └── messenger_connector.py
├── tasks/
│   ├── celery_app.py                  # Celery app config with Kafka broker
│   └── scheduled_tasks.py            # Review requests, loyalty year-end, group expiry
├── hf_spaces/
│   ├── embeddings/
│   │   └── app.py                     # Deploy to HF as 'hotel-booking-embeddings'
│   ├── stt/
│   │   └── app.py                     # Deploy to HF as 'hotel-booking-stt'
│   └── llm/
│       └── app.py                     # Deploy to HF as 'hotel-booking-llm'
├── scripts/
│   ├── setup_messenger_menu.py        # One-time Messenger persistent menu setup
│   ├── setup_elasticsearch_index.py   # Create ES hotel index with mappings
│   ├── setup_qdrant_collection.py     # Create Qdrant hotel_faqs collection
│   └── hotel_onboarding.py           # Onboard new hotel (ES + Qdrant indexing)
├── templates/
│   ├── booking_confirmation.html
│   ├── modification_alert.html
│   ├── cancellation_confirmation.html
│   ├── review_request.html
│   └── post_stay_review_request.html
├── tests/
│   ├── test_messenger_webhook.py
│   ├── test_booking_flow.py
│   ├── test_payment.py
│   ├── test_rag.py
│   └── test_load.py
├── render.yaml                        # Render deployment config
├── docker-compose.yml                 # Local development
├── .env.example                       # All required env vars with descriptions
└── requirements.txt                   # Root requirements
```

---

## 🔐 SECURITY RULES — NEVER VIOLATE

1. **Card data**: NEVER in Rasa, NEVER in backend. Stripe.js tokenizes client-side only.
2. **Secrets**: NEVER in code, NEVER in git. Always from `os.environ[...]`.
3. **Messenger webhook**: ALWAYS verify Facebook signature (X-Hub-Signature-256 header).
4. **Stripe webhooks**: ALWAYS verify Stripe HMAC signature before processing.
5. **WhatsApp webhooks**: ALWAYS verify hub.verify_token on GET requests.
6. **JWT**: Access tokens expire in 15 minutes. Refresh tokens in Redis blacklist on logout.
7. **Payment rate limit**: max 5 payment attempts per user per hour (Redis INCR).
8. **Supabase**: Use SERVICE_ROLE_KEY only server-side. Use ANON_KEY for public reads.
9. **GDPR**: Erasure requests must complete within 30 days. Anonymize, don't just delete bookings.
10. **Idempotency**: ALL Stripe charges use `idempotency_key = f'booking-{booking_id}'`.

---

## ⚡ KAFKA EVENTS — CANONICAL LIST

```python
# Topic naming: {entity}.{action}
# Always include: event_id (UUID), timestamp, version: "1.0"

KAFKA_TOPICS = {
    # Payment flow
    "payment.completed": {"booking_id", "amount", "currency", "transaction_id"},
    "payment.failed": {"booking_id", "error_code", "risk_score"},

    # Booking lifecycle  
    "booking.confirmed": {"booking_id", "hotel_id", "guest_id", "check_in"},
    "booking.modified": {"booking_id", "change_type", "old_values", "new_values"},
    "booking.cancelled": {"booking_id", "refund_amount", "reason"},
    "booking.refunded": {"booking_id", "refund_id", "amount"},
    "booking.completed": {"booking_id", "check_out"},

    # Rooms
    "room_block.created": {"block_id", "hotel_id", "count", "coordinator_id"},

    # Loyalty
    "loyalty.points_awarded": {"guest_id", "points", "balance", "action_type"},
    "loyalty.tier_upgraded": {"guest_id", "old_tier", "new_tier"},

    # Reviews
    "review.submitted": {"booking_id", "hotel_id", "score", "sentiment"},
    "review.negative_alert": {"booking_id", "hotel_id", "score", "text_preview"},

    # Corporate
    "corporate.booking.approved": {"booking_id", "approver_email", "company_id"},

    # Force Majeure
    "force_majeure.detected": {"country", "event_type", "affected_bookings", "confidence"},

    # Sessions (from Rasa)
    "rasa_events": "raw Rasa tracker events"
}
```

---

## 🚦 IMPLEMENTATION ORDER

Build in this exact order — each phase depends on the previous.

**Phase 1 — Foundation (Week 1-2)**
1. Supabase: Run schema.sql, verify all tables
2. Redis: Test connection, verify key patterns work
3. Rasa: Basic config.yml + credentials.yml + Messenger webhook verified
4. HF Space 1: LaBSE embeddings deployed and callable from Render

**Phase 2 — BRAIN (Week 3-4)**
5. language_service: detector.py + translator.py + main.py
6. Rasa NLU: nlu_en.yml with all intents + domain.yml with all slots
7. search_service: elasticsearch_client.py + ranker.py (requires hotel data seeded)
8. rag_service: embedder.py + qdrant_client.py + llm_generator.py (requires HF LLM Space)

**Phase 3 — BOOK (Week 5-6)**
9. room_service: availability.py + soft_lock.py
10. guest_service: validator.py (OCR optional later)
11. payment_service: fraud_check.py + stripe_adapter.py (test mode first)
12. confirmation_service: pdf_generator.py + qr_generator.py + calendar.py
13. notification_service: email_sender.py (SendGrid)
14. Full booking E2E test in Messenger

**Phase 4 — MANAGE (Week 7-8)**
15. modification_service + cancellation_service
16. handoff_service (Chatwoot setup)
17. auth_service (JWT for staff dashboard)
18. group_service + corporate_service

**Phase 5 — GROW (Week 9-10)**
19. loyalty_service (points + tier + gamification)
20. review_service (Celery tasks + sentiment)
21. notification_service: WhatsApp + push
22. analytics_service: Kafka consumer → ClickHouse
23. force_majeure_service

**Phase 6 — Hardening (Week 11-12)**
24. Load testing (Locust)
25. GDPR/privacy_service
26. Mobile app offline storage
27. HF Space 2 (Whisper STT) for voice
28. Full CI/CD pipeline on GitHub Actions

---

## ❌ ANTI-PATTERNS — NEVER DO THESE

- Never use raw `psycopg2` without connection pooling in async FastAPI
- Never call HuggingFace Spaces synchronously from Rasa actions (always async)
- Never store card numbers, CVVs, or full PANs anywhere
- Never bypass Redis soft lock and book directly against DB
- Never send WhatsApp messages without pre-approved templates
- Never let fraud_check.py block payment flow with synchronous external API call >5s
- Never hardcode Redis keys as plain strings — always use the constants above
- Never use `datetime.now()` for financial calculations — always use `datetime.utcnow()` or timezone-aware
- Never commit `.env` files — `.env.example` only
- Never reply in Messenger with messages > 2000 characters
- Never process Stripe webhooks without HMAC verification
- Never use Supabase ANON_KEY for server-side writes to sensitive tables
