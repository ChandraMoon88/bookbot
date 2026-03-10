-- ============================================================
-- BOOKBOT — HOTEL BOOKING AI CHATBOT SYSTEM
-- Optimized, Normalized & Fully Indexed PostgreSQL Schema
-- Version: 2.0 | March 2026 | Production-Ready
--
-- WHAT CHANGED FROM v1.0:
--  ✦ Normalized: removed JSONB modification_history from bookings
--  ✦ Normalized: removed redundant oauth_provider cols from users
--  ✦ Normalized: removed redundant hotel_id from room_rates/availability
--    (kept as denorm col WITH COMMENT for query perf — see note)
--  ✦ Added: platform_identities (Messenger PSID, WhatsApp, Telegram)
--  ✦ Added: vouchers + voucher_redemptions
--  ✦ Added: price_alerts + room_waitlist
--  ✦ Added: support_tickets + ticket_messages
--  ✦ Added: hotel_services catalog
--  ✦ Added: service_bookings (spa, restaurant, room service, housekeeping)
--  ✦ Added: in_stay_requests (complaints, housekeeping, lost & found)
--  ✦ Added: lost_found_reports
--  ✦ Added: seasonal_pricing_rules
--  ✦ Added: overbooking_incidents
--  ✦ Added: hotel_accessibility_features
--  ✦ Added: corporate_travel_policies
--  ✦ Added: booking_sessions_log (abandoned booking analytics)
--  ✦ Added: agent_sessions (Chatwoot / live agent)
--  ✦ Added: missing composite & partial indexes
--  ✦ Added: updated_at triggers for all new tables
--  ✦ Added: RLS policies for all new user-facing tables
-- ============================================================

-- ============================================================
-- EXTENSIONS
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- for fuzzy city/name search


-- ============================================================
-- 1. USERS & AUTH
-- ============================================================

CREATE TABLE users (
  id                         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email                      TEXT UNIQUE,
  phone                      TEXT UNIQUE,
  first_name                 TEXT,
  last_name                  TEXT,
  preferred_language         TEXT DEFAULT 'en',           -- ISO 639-1
  is_rtl                     BOOLEAN DEFAULT FALSE,
  language_tier              SMALLINT DEFAULT 1,          -- 1=Tier1, 2=Tier2, 3=Tier3
  voice_gender_preference    TEXT DEFAULT 'female',
  voice_speed                NUMERIC(3,2) DEFAULT 1.0,
  nationality                TEXT,                        -- ISO 3166-1 alpha-2
  country_code               TEXT,
  timezone                   TEXT,
  password_hash              TEXT,
  -- NOTE: oauth_provider/oauth_provider_id removed — live in oauth_accounts
  is_email_verified          BOOLEAN DEFAULT FALSE,
  is_phone_verified          BOOLEAN DEFAULT FALSE,
  is_active                  BOOLEAN DEFAULT TRUE,
  gdpr_consent               BOOLEAN DEFAULT FALSE,
  gdpr_consent_at            TIMESTAMPTZ,
  marketing_consent          BOOLEAN DEFAULT FALSE,
  ccpa_opt_out               BOOLEAN DEFAULT FALSE,
  data_deletion_requested    BOOLEAN DEFAULT FALSE,
  last_login_at              TIMESTAMPTZ,
  failed_login_attempts      SMALLINT DEFAULT 0,
  locked_until               TIMESTAMPTZ,
  profile_photo_url          TEXT,
  created_at                 TIMESTAMPTZ DEFAULT NOW(),
  updated_at                 TIMESTAMPTZ DEFAULT NOW()
);

-- Platform identity mapping: one user can have PSID (Messenger),
-- WhatsApp number, Telegram chat_id, etc.
CREATE TABLE platform_identities (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id          UUID REFERENCES users(id) ON DELETE CASCADE,
  platform         TEXT NOT NULL,             -- 'messenger'|'whatsapp'|'telegram'|'web'|'sms'
  platform_user_id TEXT NOT NULL,             -- PSID, WA number, Telegram chat_id
  language         TEXT DEFAULT 'en',         -- last-known language for this platform session
  is_rtl           BOOLEAN DEFAULT FALSE,
  metadata         JSONB DEFAULT '{}',        -- platform-specific data (e.g. page_id)
  last_seen_at     TIMESTAMPTZ DEFAULT NOW(),
  created_at       TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (platform, platform_user_id)
);

CREATE TABLE oauth_accounts (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  provider          TEXT NOT NULL,            -- 'google'|'facebook'|'apple'
  provider_user_id  TEXT NOT NULL,
  access_token      TEXT,
  refresh_token     TEXT,
  token_expires_at  TIMESTAMPTZ,
  profile_data      JSONB DEFAULT '{}',
  created_at        TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (provider, provider_user_id)
);

CREATE TABLE auth_token_blacklist (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  token_hash       TEXT NOT NULL UNIQUE,
  user_id          UUID REFERENCES users(id) ON DELETE CASCADE,
  blacklisted_at   TIMESTAMPTZ DEFAULT NOW(),
  expires_at       TIMESTAMPTZ NOT NULL
);


-- ============================================================
-- 2. CHAT SESSIONS & CONVERSATIONS
-- ============================================================

CREATE TABLE chat_sessions (
  id                        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id                TEXT UNIQUE NOT NULL,          -- matches Redis key
  user_id                   UUID REFERENCES users(id) ON DELETE SET NULL,
  platform_identity_id      UUID REFERENCES platform_identities(id) ON DELETE SET NULL,
  detected_language         TEXT DEFAULT 'en',
  language_confidence       NUMERIC(4,3),
  is_rtl                    BOOLEAN DEFAULT FALSE,
  language_tier             SMALLINT DEFAULT 1,
  channel                   TEXT DEFAULT 'web',           -- 'web'|'mobile'|'whatsapp'|'messenger'|'telegram'|'voice'|'sms'
  is_voice_session          BOOLEAN DEFAULT FALSE,
  device_type               TEXT,
  user_agent                TEXT,
  ip_address                INET,
  country_from_ip           TEXT,
  handoff_active            BOOLEAN DEFAULT FALSE,
  chatwoot_conversation_id  TEXT,
  total_turns               SMALLINT DEFAULT 0,
  booking_completed         BOOLEAN DEFAULT FALSE,
  drop_off_intent           TEXT,
  session_started_at        TIMESTAMPTZ DEFAULT NOW(),
  session_ended_at          TIMESTAMPTZ,
  created_at                TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE conversation_messages (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id          UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role                TEXT NOT NULL,              -- 'user'|'bot'|'agent'
  text                TEXT NOT NULL,
  intent              TEXT,
  intent_confidence   NUMERIC(4,3),
  entities            JSONB DEFAULT '{}',
  language            TEXT DEFAULT 'en',
  is_voice            BOOLEAN DEFAULT FALSE,
  audio_url           TEXT,
  turn_number         SMALLINT,
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE nlu_confidence_log (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id      UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
  intent          TEXT NOT NULL,
  confidence      NUMERIC(4,3),
  language        TEXT,
  entities        JSONB DEFAULT '[]',
  was_fallback    BOOLEAN DEFAULT FALSE,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Persisted booking flow state for abandoned booking recovery
-- Redis is source of truth; this is the DB checkpoint/analytics copy
CREATE TABLE booking_sessions_log (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id       UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
  user_id          UUID REFERENCES users(id) ON DELETE SET NULL,
  step             TEXT NOT NULL,             -- 'AWAITING_CITY' | ... | 'CONFIRMED'
  snapshot         JSONB NOT NULL,            -- full booking_session JSON snapshot
  booking_type     TEXT DEFAULT 'STANDARD',   -- 'STANDARD'|'GROUP'|'CORPORATE'|'WEDDING'|'LONG_STAY'
  city             TEXT,
  hotel_id         UUID,
  checkin          DATE,
  checkout         DATE,
  total_amount     NUMERIC(10,2),
  currency         TEXT,
  was_abandoned    BOOLEAN DEFAULT FALSE,
  recovery_sent_at TIMESTAMPTZ,
  completed_at     TIMESTAMPTZ,
  created_at       TIMESTAMPTZ DEFAULT NOW(),
  updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE handoff_requests (
  id                          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id                  UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
  user_id                     UUID REFERENCES users(id) ON DELETE SET NULL,
  reason                      TEXT,           -- 'fallback_twice'|'explicit_request'|'frustration_detected'
  booking_intent              TEXT,
  slots_filled                JSONB DEFAULT '{}',
  conversation_history        JSONB DEFAULT '[]',
  chatwoot_conversation_id    TEXT,
  assigned_agent_name         TEXT,
  assigned_agent_language     TEXT,
  estimated_wait_minutes      SMALLINT,
  status                      TEXT DEFAULT 'pending',     -- 'pending'|'assigned'|'resolved'|'timeout'
  resolved_at                 TIMESTAMPTZ,
  created_at                  TIMESTAMPTZ DEFAULT NOW()
);

-- Live agent session tracking (each handoff conversation)
CREATE TABLE agent_sessions (
  id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  handoff_request_id   UUID NOT NULL REFERENCES handoff_requests(id) ON DELETE CASCADE,
  agent_id             TEXT NOT NULL,          -- internal agent identifier (e.g. AGT-012)
  agent_name           TEXT,
  agent_language       TEXT,
  started_at           TIMESTAMPTZ DEFAULT NOW(),
  ended_at             TIMESTAMPTZ,
  resolution_type      TEXT,                   -- 'resolved'|'escalated'|'transferred'
  csat_score           SMALLINT,               -- 1–5 post-handoff rating
  created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Support tickets (created from agent handoff, or direct complaint)
CREATE TABLE support_tickets (
  id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  ticket_ref           TEXT UNIQUE NOT NULL,   -- 'TKT-20260310-008'
  user_id              UUID REFERENCES users(id) ON DELETE SET NULL,
  session_id           UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
  booking_id           UUID,                   -- FK added after bookings table
  handoff_request_id   UUID REFERENCES handoff_requests(id) ON DELETE SET NULL,
  category             TEXT,                   -- 'billing'|'complaint'|'modification'|'general'
  priority             TEXT DEFAULT 'normal',  -- 'low'|'normal'|'high'|'urgent'
  subject              TEXT,
  description          TEXT,
  status               TEXT DEFAULT 'open',    -- 'open'|'in_progress'|'resolved'|'closed'
  assigned_agent_id    TEXT,
  resolved_at          TIMESTAMPTZ,
  first_response_at    TIMESTAMPTZ,
  created_at           TIMESTAMPTZ DEFAULT NOW(),
  updated_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE support_ticket_messages (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  ticket_id   UUID NOT NULL REFERENCES support_tickets(id) ON DELETE CASCADE,
  sender_type TEXT NOT NULL,              -- 'user'|'agent'|'system'
  sender_id   TEXT,
  message     TEXT NOT NULL,
  attachments JSONB DEFAULT '[]',
  created_at  TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- 3. HOTELS & ROOMS
-- ============================================================

CREATE TABLE hotel_partners (
  id                        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name                      TEXT NOT NULL,
  legal_name                TEXT,
  email                     TEXT NOT NULL,
  phone                     TEXT,
  address                   TEXT,
  city                      TEXT NOT NULL,
  state_province            TEXT,
  country                   TEXT NOT NULL,            -- ISO 3166-1 alpha-2
  postal_code               TEXT,
  lat                       NUMERIC(10,7),
  lng                       NUMERIC(10,7),
  star_rating               SMALLINT CHECK (star_rating BETWEEN 1 AND 5),
  logo_url                  TEXT,
  thumbnail_url             TEXT,
  description               TEXT,
  description_translations  JSONB DEFAULT '{}',      -- {fr:"...", ar:"..."}
  amenities                 TEXT[] DEFAULT '{}',
  check_in_time             TIME DEFAULT '15:00',
  check_out_time            TIME DEFAULT '11:00',
  currency                  TEXT DEFAULT 'USD',
  timezone                  TEXT,
  is_active                 BOOLEAN DEFAULT TRUE,
  is_pet_friendly           BOOLEAN DEFAULT FALSE,
  is_halal_certified        BOOLEAN DEFAULT FALSE,
  is_accessible             BOOLEAN DEFAULT FALSE,   -- quick flag; detail in hotel_accessibility
  partner_contract_type     TEXT DEFAULT 'standard', -- 'standard'|'corporate'|'mice'
  elasticsearch_id          TEXT,
  qdrant_indexed            BOOLEAN DEFAULT FALSE,
  partner_portal_user_id    UUID REFERENCES users(id) ON DELETE SET NULL,
  onboarded_at              TIMESTAMPTZ,
  created_at                TIMESTAMPTZ DEFAULT NOW(),
  updated_at                TIMESTAMPTZ DEFAULT NOW()
);

-- Granular accessibility features per hotel (replaces scattered boolean columns)
CREATE TABLE hotel_accessibility_features (
  id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id                UUID NOT NULL REFERENCES hotel_partners(id) ON DELETE CASCADE,
  wheelchair_accessible   BOOLEAN DEFAULT FALSE,
  roll_in_shower          BOOLEAN DEFAULT FALSE,
  wide_doorways           BOOLEAN DEFAULT FALSE,      -- >= 81 cm / 32 in
  lowered_bed_available   BOOLEAN DEFAULT FALSE,
  grab_rails              BOOLEAN DEFAULT FALSE,
  accessible_pool_lift    BOOLEAN DEFAULT FALSE,
  visual_fire_alarm       BOOLEAN DEFAULT FALSE,      -- strobe lights
  vibrating_alarm_clock   BOOLEAN DEFAULT FALSE,
  closed_captions_tv      BOOLEAN DEFAULT FALSE,
  braille_signage         BOOLEAN DEFAULT FALSE,
  audio_elevator          BOOLEAN DEFAULT FALSE,
  large_print_menu        BOOLEAN DEFAULT FALSE,
  guide_dogs_welcome      BOOLEAN DEFAULT TRUE,
  service_animals_welcome BOOLEAN DEFAULT TRUE,
  accessible_parking      BOOLEAN DEFAULT FALSE,
  ground_floor_rooms      BOOLEAN DEFAULT FALSE,
  accessibility_concierge BOOLEAN DEFAULT FALSE,
  notes                   TEXT,
  last_verified_at        DATE,
  created_at              TIMESTAMPTZ DEFAULT NOW(),
  updated_at              TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (hotel_id)
);

-- Hotel service catalog (spa, restaurants, room service, etc.)
CREATE TABLE hotel_services (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id         UUID NOT NULL REFERENCES hotel_partners(id) ON DELETE CASCADE,
  service_type     TEXT NOT NULL,              -- 'spa'|'restaurant'|'room_service'|'housekeeping'|'concierge'|'gym'|'pool'|'transfer'
  service_name     TEXT NOT NULL,
  description      TEXT,
  operating_hours  JSONB DEFAULT '{}',         -- {mon:"08:00-22:00", ...}
  is_available     BOOLEAN DEFAULT TRUE,
  booking_required BOOLEAN DEFAULT FALSE,
  min_advance_hours SMALLINT DEFAULT 0,
  max_capacity     SMALLINT,
  created_at       TIMESTAMPTZ DEFAULT NOW(),
  updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Nearby places (airports, hospitals, attractions) — pre-geocoded
CREATE TABLE hotel_nearby_places (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id      UUID NOT NULL REFERENCES hotel_partners(id) ON DELETE CASCADE,
  place_type    TEXT NOT NULL,                -- 'airport'|'hospital'|'attraction'|'restaurant'|'pharmacy'
  name          TEXT NOT NULL,
  address       TEXT,
  lat           NUMERIC(10,7),
  lng           NUMERIC(10,7),
  distance_km   NUMERIC(6,3),
  travel_mins   SMALLINT,
  travel_mode   TEXT DEFAULT 'driving',       -- 'walking'|'driving'|'transit'
  phone         TEXT,
  is_emergency  BOOLEAN DEFAULT FALSE,        -- TRUE for ER hospitals
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE hotel_faqs (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id          UUID NOT NULL REFERENCES hotel_partners(id) ON DELETE CASCADE,
  question          TEXT NOT NULL,
  answer            TEXT NOT NULL,
  category          TEXT,                     -- 'amenities'|'policies'|'dining'|'transport'|'accessibility'
  language          TEXT DEFAULT 'en',
  qdrant_vector_id  TEXT,
  is_active         BOOLEAN DEFAULT TRUE,
  created_at        TIMESTAMPTZ DEFAULT NOW(),
  updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE hotel_policies (
  id                         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id                   UUID NOT NULL REFERENCES hotel_partners(id) ON DELETE CASCADE,
  policy_type                TEXT NOT NULL,           -- 'cancellation'|'modification'
  free_cancel_days           SMALLINT DEFAULT 7,
  partial_refund_days        SMALLINT DEFAULT 3,
  partial_refund_percent     NUMERIC(5,2) DEFAULT 50.00,
  no_refund_days             SMALLINT DEFAULT 1,
  modification_free_days     SMALLINT DEFAULT 7,
  modification_fee_days      SMALLINT DEFAULT 3,
  modification_fee_amount    NUMERIC(10,2) DEFAULT 0,
  modification_fee_currency  TEXT DEFAULT 'USD',
  modification_fee_type      TEXT DEFAULT 'flat',     -- 'flat'|'percent'
  effective_from             DATE,
  effective_until            DATE,
  created_at                 TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE room_types (
  id                        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id                  UUID NOT NULL REFERENCES hotel_partners(id) ON DELETE CASCADE,
  name                      TEXT NOT NULL,
  name_translations         JSONB DEFAULT '{}',
  description               TEXT,
  description_translations  JSONB DEFAULT '{}',
  size_sqm                  SMALLINT,
  max_adults                SMALLINT DEFAULT 2,
  max_children              SMALLINT DEFAULT 0,
  max_occupancy             SMALLINT DEFAULT 2,
  bed_type                  TEXT,                     -- 'king'|'queen'|'twin'|'double'|'bunk'
  bed_count                 SMALLINT DEFAULT 1,
  floor_level               TEXT,
  has_balcony               BOOLEAN DEFAULT FALSE,
  has_sea_view              BOOLEAN DEFAULT FALSE,
  has_city_view             BOOLEAN DEFAULT FALSE,
  is_accessible             BOOLEAN DEFAULT FALSE,
  pet_allowed               BOOLEAN DEFAULT FALSE,
  amenities                 TEXT[] DEFAULT '{}',
  photos                    TEXT[] DEFAULT '{}',
  thumbnail_url             TEXT,
  is_active                 BOOLEAN DEFAULT TRUE,
  created_at                TIMESTAMPTZ DEFAULT NOW(),
  updated_at                TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE room_rates (
  id                        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  room_type_id              UUID NOT NULL REFERENCES room_types(id) ON DELETE CASCADE,
  -- hotel_id is denormalized here intentionally for query performance
  -- (avoids extra join in availability searches which run millions of times/day)
  hotel_id                  UUID NOT NULL REFERENCES hotel_partners(id),
  rate_plan                 TEXT NOT NULL,             -- 'room_only'|'breakfast'|'half_board'|'full_board'|'all_inclusive'
  meal_plan                 TEXT,
  base_price_per_night      NUMERIC(10,2) NOT NULL,
  currency                  TEXT NOT NULL DEFAULT 'USD',
  price_modifier_percent    NUMERIC(5,2) DEFAULT 0,
  min_stay_nights           SMALLINT DEFAULT 1,
  max_stay_nights           SMALLINT,
  is_refundable             BOOLEAN DEFAULT TRUE,
  cancellation_policy_id    UUID REFERENCES hotel_policies(id),
  advance_purchase_days     SMALLINT DEFAULT 0,
  is_early_bird             BOOLEAN DEFAULT FALSE,     -- non-refundable early rate
  valid_from                DATE,
  valid_until               DATE,
  is_active                 BOOLEAN DEFAULT TRUE,
  created_at                TIMESTAMPTZ DEFAULT NOW()
);

-- Seasonal pricing multipliers (NYE, Eid, Christmas, peak, etc.)
CREATE TABLE seasonal_pricing_rules (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id         UUID REFERENCES hotel_partners(id) ON DELETE CASCADE, -- NULL = global rule
  city             TEXT,                               -- NULL = all cities
  country          TEXT,
  event_name       TEXT NOT NULL,                     -- 'New Year Eve'|'Christmas'|'Eid'|'Diwali'|'CES'
  event_type       TEXT NOT NULL,                     -- 'holiday'|'conference'|'festival'|'sports'|'peak_season'
  date_from        DATE NOT NULL,
  date_to          DATE NOT NULL,
  price_multiplier NUMERIC(5,3) DEFAULT 1.0,          -- 1.5 = 50% more expensive
  min_stay_nights  SMALLINT DEFAULT 1,
  notes            TEXT,
  is_active        BOOLEAN DEFAULT TRUE,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE room_availability (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  room_type_id     UUID NOT NULL REFERENCES room_types(id) ON DELETE CASCADE,
  -- hotel_id denormalized for hot availability queries
  hotel_id         UUID NOT NULL REFERENCES hotel_partners(id),
  date             DATE NOT NULL,
  total_rooms      SMALLINT NOT NULL,
  available_count  SMALLINT NOT NULL CHECK (available_count >= 0),
  base_price       NUMERIC(10,2),
  seasonal_price   NUMERIC(10,2),                     -- overrides base_price if set
  currency         TEXT DEFAULT 'USD',
  is_blackout      BOOLEAN DEFAULT FALSE,
  updated_at       TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (room_type_id, date)
);

CREATE TABLE addons (
  id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id             UUID NOT NULL REFERENCES hotel_partners(id) ON DELETE CASCADE,
  name                 TEXT NOT NULL,
  name_translations    JSONB DEFAULT '{}',
  description          TEXT,
  category             TEXT NOT NULL,                 -- 'spa'|'dining'|'transport'|'activities'|'room_enhancement'|'childcare'|'accessibility'
  price                NUMERIC(10,2) NOT NULL,
  currency             TEXT NOT NULL DEFAULT 'USD',
  unit                 TEXT DEFAULT 'per_booking',    -- 'per_person'|'per_booking'|'per_night'
  recommended_rank     SMALLINT DEFAULT 0,
  trip_purpose_tags    TEXT[] DEFAULT '{}',           -- ['honeymoon','family','business']
  min_advance_hours    SMALLINT DEFAULT 24,
  is_available         BOOLEAN DEFAULT TRUE,
  image_url            TEXT,
  created_at           TIMESTAMPTZ DEFAULT NOW(),
  updated_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Vouchers / promo codes
CREATE TABLE vouchers (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code                TEXT NOT NULL UNIQUE,
  description         TEXT,
  discount_type       TEXT NOT NULL,              -- 'percent'|'flat'|'free_night'
  discount_value      NUMERIC(10,2) NOT NULL,     -- percent (0–100) or flat amount
  max_discount_amount NUMERIC(10,2),              -- cap for percent discounts
  min_booking_amount  NUMERIC(10,2) DEFAULT 0,
  currency            TEXT DEFAULT 'USD',
  valid_from          TIMESTAMPTZ DEFAULT NOW(),
  valid_until         TIMESTAMPTZ,
  max_uses            INTEGER,                    -- NULL = unlimited
  uses_per_user       SMALLINT DEFAULT 1,
  total_used          INTEGER DEFAULT 0,
  applicable_hotels   UUID[] DEFAULT '{}',        -- empty = all hotels
  applicable_channels TEXT[] DEFAULT '{}',        -- empty = all channels
  applicable_days     TEXT[] DEFAULT '{}',        -- ['friday','saturday'] empty=all
  new_users_only      BOOLEAN DEFAULT FALSE,
  is_active           BOOLEAN DEFAULT TRUE,
  created_by          UUID REFERENCES users(id) ON DELETE SET NULL,
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Room waitlist for sold-out dates
CREATE TABLE room_waitlist (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id          UUID REFERENCES users(id) ON DELETE CASCADE,
  hotel_id         UUID NOT NULL REFERENCES hotel_partners(id),
  room_type_id     UUID NOT NULL REFERENCES room_types(id),
  checkin          DATE NOT NULL,
  checkout         DATE NOT NULL,
  num_adults       SMALLINT DEFAULT 1,
  num_children     SMALLINT DEFAULT 0,
  notify_channel   TEXT DEFAULT 'messenger',      -- 'messenger'|'whatsapp'|'email'|'sms'
  notified_at      TIMESTAMPTZ,
  status           TEXT DEFAULT 'waiting',        -- 'waiting'|'notified'|'booked'|'expired'
  expires_at       TIMESTAMPTZ DEFAULT NOW() + INTERVAL '30 days',
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Price alerts ("notify me if price drops")
CREATE TABLE price_alerts (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id          UUID REFERENCES users(id) ON DELETE CASCADE,
  hotel_id         UUID NOT NULL REFERENCES hotel_partners(id),
  room_type_id     UUID REFERENCES room_types(id),  -- NULL = any room
  checkin          DATE NOT NULL,
  checkout         DATE NOT NULL,
  target_price     NUMERIC(10,2) NOT NULL,          -- alert when price drops below this
  currency         TEXT DEFAULT 'USD',
  notify_channel   TEXT DEFAULT 'messenger',
  notified_at      TIMESTAMPTZ,
  status           TEXT DEFAULT 'active',           -- 'active'|'triggered'|'expired'|'cancelled'
  expires_at       TIMESTAMPTZ DEFAULT NOW() + INTERVAL '60 days',
  created_at       TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- 4. BOOKINGS & GUESTS
-- ============================================================

CREATE TABLE bookings (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_reference     TEXT UNIQUE NOT NULL,          -- 'BB-MUM-20260320-X7K2'
  user_id               UUID REFERENCES users(id) ON DELETE SET NULL,
  hotel_id              UUID NOT NULL REFERENCES hotel_partners(id),
  room_type_id          UUID NOT NULL REFERENCES room_types(id),
  rate_plan_id          UUID REFERENCES room_rates(id),
  session_id            UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,

  -- Stay
  check_in              DATE NOT NULL,
  check_out             DATE NOT NULL,
  nights                INTEGER GENERATED ALWAYS AS (check_out - check_in) STORED,
  num_adults            SMALLINT NOT NULL DEFAULT 1,
  num_children          SMALLINT DEFAULT 0,
  num_rooms             SMALLINT DEFAULT 1,
  trip_purpose          TEXT,                          -- 'honeymoon'|'family'|'business'|'leisure'
  rate_plan             TEXT,
  meal_plan             TEXT,
  booking_type          TEXT DEFAULT 'STANDARD',       -- 'STANDARD'|'GROUP'|'CORPORATE'|'WEDDING'|'LONG_STAY'|'EMERGENCY'

  -- Pricing
  base_amount           NUMERIC(10,2) NOT NULL,
  addons_amount         NUMERIC(10,2) DEFAULT 0,
  tax_amount            NUMERIC(10,2) DEFAULT 0,
  discount_amount       NUMERIC(10,2) DEFAULT 0,
  total_amount          NUMERIC(10,2) NOT NULL,
  currency              TEXT NOT NULL DEFAULT 'USD',
  total_amount_usd      NUMERIC(10,2),
  fx_rate_used          NUMERIC(12,6),

  -- Status
  status                TEXT DEFAULT 'pending',
  -- 'pending'|'confirmed'|'modified'|'cancelled'|'cancelled_penalty'
  -- 'checked_in'|'checked_out'|'completed'|'no_show'|'waitlisted'
  payment_status        TEXT DEFAULT 'unpaid',         -- 'unpaid'|'paid'|'partially_refunded'|'refunded'

  -- Room soft lock
  soft_lock_id          TEXT,
  soft_lock_expires_at  TIMESTAMPTZ,

  -- Special booking types
  is_group_booking      BOOLEAN DEFAULT FALSE,
  group_booking_id      UUID,                          -- FK added after group_bookings
  is_corporate          BOOLEAN DEFAULT FALSE,
  corporate_account_id  UUID,                          -- FK added after corporate_accounts
  is_last_minute        BOOLEAN DEFAULT FALSE,
  is_early_bird         BOOLEAN DEFAULT FALSE,

  -- Force majeure
  force_majeure_flag    BOOLEAN DEFAULT FALSE,
  force_majeure_event_id UUID,                         -- FK added after force_majeure_events

  -- Voucher used
  voucher_id            UUID REFERENCES vouchers(id),
  voucher_discount      NUMERIC(10,2) DEFAULT 0,

  -- Review
  review_requested      BOOLEAN DEFAULT FALSE,
  review_requested_at   TIMESTAMPTZ,

  -- Confirmation artifacts
  pdf_voucher_url       TEXT,
  qr_code_url           TEXT,
  ics_calendar_url      TEXT,

  -- Guest extras
  special_requests      TEXT,
  dietary_needs         TEXT,
  accessibility_needs   TEXT,
  special_occasion      TEXT,                          -- 'birthday'|'anniversary'|'honeymoon'|'proposal'

  -- Cancellation
  cancellation_reason   TEXT,
  cancelled_at          TIMESTAMPTZ,

  -- Mobile check-in
  checkin_completed     BOOLEAN DEFAULT FALSE,
  checkin_completed_at  TIMESTAMPTZ,
  digital_key_token     TEXT,

  -- Timestamps
  confirmed_at          TIMESTAMPTZ,
  completed_at          TIMESTAMPTZ,
  created_at            TIMESTAMPTZ DEFAULT NOW(),
  updated_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE guests (
  id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id         UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id            UUID REFERENCES users(id) ON DELETE SET NULL,
  is_primary         BOOLEAN DEFAULT FALSE,
  first_name         TEXT NOT NULL,
  last_name          TEXT NOT NULL,
  email              TEXT,
  phone              TEXT,
  date_of_birth      DATE,
  nationality        TEXT,
  passport_number    TEXT,
  passport_expiry    DATE,
  passport_country   TEXT,
  passport_scan_url  TEXT,
  ocr_confidence     NUMERIC(4,3),
  gender             TEXT,
  dietary_needs      TEXT,
  accessibility_needs TEXT,
  is_child           BOOLEAN DEFAULT FALSE,
  child_age          SMALLINT,
  created_at         TIMESTAMPTZ DEFAULT NOW(),
  updated_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE booking_addons (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id      UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  addon_id        UUID NOT NULL REFERENCES addons(id),
  quantity        SMALLINT DEFAULT 1,
  price_at_time   NUMERIC(10,2) NOT NULL,
  currency        TEXT NOT NULL,
  status          TEXT DEFAULT 'confirmed',            -- 'confirmed'|'cancelled'
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE voucher_redemptions (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  voucher_id      UUID NOT NULL REFERENCES vouchers(id),
  booking_id      UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
  discount_amount NUMERIC(10,2) NOT NULL,
  currency        TEXT NOT NULL,
  redeemed_at     TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (voucher_id, booking_id)
);


-- ============================================================
-- 5. IN-STAY SERVICES
-- ============================================================

-- All service bookings: spa appointments, restaurant reservations,
-- room service orders, housekeeping requests — unified table
CREATE TABLE service_bookings (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id          UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  hotel_service_id    UUID REFERENCES hotel_services(id),
  user_id             UUID REFERENCES users(id) ON DELETE SET NULL,
  service_type        TEXT NOT NULL,                   -- 'spa'|'restaurant'|'room_service'|'housekeeping'|'transfer'|'concierge'
  service_name        TEXT NOT NULL,
  details             JSONB DEFAULT '{}',              -- structured details per service type
  -- Examples:
  -- spa: {treatment:"Swedish", duration_min:60}
  -- restaurant: {restaurant:"Threesixtyone", guests:2, occasion:"anniversary"}
  -- room_service: {items:[{name:"Club Sandwich",qty:1,price:750}], room:"714"}
  -- housekeeping: {request_type:"fresh_towels", do_not_disturb_until:"14:00"}
  -- transfer: {from:"Airport", flight:"AI101", arrives:"11:30", vehicle:"sedan"}
  scheduled_at        TIMESTAMPTZ,
  guests_count        SMALLINT DEFAULT 1,
  special_notes       TEXT,
  amount              NUMERIC(10,2),
  currency            TEXT DEFAULT 'USD',
  is_complimentary    BOOLEAN DEFAULT FALSE,
  status              TEXT DEFAULT 'requested',        -- 'requested'|'confirmed'|'in_progress'|'completed'|'cancelled'
  confirmed_at        TIMESTAMPTZ,
  completed_at        TIMESTAMPTZ,
  hotel_ref           TEXT,                            -- hotel PMS reference number
  created_at          TIMESTAMPTZ DEFAULT NOW(),
  updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- In-stay complaints and special requests
CREATE TABLE in_stay_requests (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id       UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id          UUID REFERENCES users(id) ON DELETE SET NULL,
  request_ref      TEXT UNIQUE NOT NULL,               -- 'COMP-20260321-004'
  request_type     TEXT NOT NULL,                      -- 'complaint'|'maintenance'|'request'|'feedback'
  category         TEXT,                               -- 'ac_heating'|'noise'|'cleanliness'|'tv_wifi'|'plumbing'|'bed_furniture'|'other'
  description      TEXT NOT NULL,
  urgency          TEXT DEFAULT 'normal',              -- 'low'|'normal'|'high'|'emergency'
  room_change_requested BOOLEAN DEFAULT FALSE,
  hotel_notified_at TIMESTAMPTZ,
  resolution_notes TEXT,
  status           TEXT DEFAULT 'open',                -- 'open'|'in_progress'|'resolved'|'escalated'
  resolved_at      TIMESTAMPTZ,
  created_at       TIMESTAMPTZ DEFAULT NOW(),
  updated_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE lost_found_reports (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  report_ref       TEXT UNIQUE NOT NULL,               -- 'LOST-20260325-001'
  booking_id       UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id          UUID REFERENCES users(id) ON DELETE SET NULL,
  items_description TEXT NOT NULL,
  hotel_notified_at TIMESTAMPTZ,
  found_status     TEXT DEFAULT 'searching',           -- 'searching'|'found'|'not_found'|'shipped'
  ship_to_address  TEXT,
  tracking_number  TEXT,
  notes            TEXT,
  created_at       TIMESTAMPTZ DEFAULT NOW(),
  updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Overbooking incidents: when hotel cannot accommodate a confirmed guest
CREATE TABLE overbooking_incidents (
  id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id             UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  original_hotel_id      UUID NOT NULL REFERENCES hotel_partners(id),
  replacement_hotel_id   UUID REFERENCES hotel_partners(id),
  replacement_booking_id UUID REFERENCES bookings(id),
  price_diff_covered     NUMERIC(10,2) DEFAULT 0,       -- we cover cost diff
  currency               TEXT DEFAULT 'USD',
  taxi_covered           BOOLEAN DEFAULT FALSE,
  loyalty_bonus_points   INTEGER DEFAULT 0,
  status                 TEXT DEFAULT 'pending',         -- 'pending'|'resolved'|'refunded'
  notes                  TEXT,
  created_at             TIMESTAMPTZ DEFAULT NOW(),
  updated_at             TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- 6. PAYMENTS
-- ============================================================

CREATE TABLE payments (
  id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id           UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id              UUID REFERENCES users(id) ON DELETE SET NULL,
  amount               NUMERIC(10,2) NOT NULL,
  currency             TEXT NOT NULL,
  amount_usd           NUMERIC(10,2),
  gateway              TEXT NOT NULL,                   -- 'stripe'|'razorpay'|'paypal'|'upi'|'netbanking'|'crypto'|'bizum'|'split'|'corporate'
  transaction_id       TEXT,
  payment_intent_id    TEXT,
  payment_method_type  TEXT,                            -- 'card'|'upi'|'netbanking'|'paypal'|'crypto'|'bizum'
  card_last4           TEXT,
  card_brand           TEXT,
  card_country         TEXT,
  upi_id               TEXT,                            -- for UPI payments
  crypto_currency      TEXT,                            -- 'BTC'|'ETH'|'USDT'|'USDC'
  crypto_wallet        TEXT,
  status               TEXT DEFAULT 'pending',
  risk_score           SMALLINT,
  fraud_action         TEXT,
  stripe_radar_score   SMALLINT,
  requires_3ds         BOOLEAN DEFAULT FALSE,
  three_ds_completed   BOOLEAN DEFAULT FALSE,
  idempotency_key      TEXT UNIQUE,
  receipt_url          TEXT,
  refund_id            TEXT,
  refund_amount        NUMERIC(10,2),
  refund_status        TEXT,
  refund_initiated_at  TIMESTAMPTZ,
  refund_reason        TEXT,
  metadata             JSONB DEFAULT '{}',
  webhook_verified     BOOLEAN DEFAULT FALSE,
  created_at           TIMESTAMPTZ DEFAULT NOW(),
  updated_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE payment_attempts (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id          UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id             UUID REFERENCES users(id) ON DELETE SET NULL,
  gateway             TEXT NOT NULL,
  amount              NUMERIC(10,2),
  currency            TEXT,
  error_code          TEXT,
  error_message       TEXT,
  risk_score          SMALLINT,
  blocked_by          TEXT,                             -- 'fraud_engine'|'stripe_radar'|'velocity_check'
  ip_address          INET,
  device_fingerprint  TEXT,
  card_country        TEXT,
  attempted_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Split payment tracking
CREATE TABLE split_payment_requests (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id      UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  total_amount    NUMERIC(10,2) NOT NULL,
  currency        TEXT NOT NULL,
  num_splits      SMALLINT NOT NULL DEFAULT 2,
  status          TEXT DEFAULT 'pending',               -- 'pending'|'partially_paid'|'completed'|'expired'
  expires_at      TIMESTAMPTZ DEFAULT NOW() + INTERVAL '24 hours',
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE split_payment_parts (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  split_request_id UUID NOT NULL REFERENCES split_payment_requests(id) ON DELETE CASCADE,
  part_number      SMALLINT NOT NULL,
  amount           NUMERIC(10,2) NOT NULL,
  currency         TEXT NOT NULL,
  payment_link     TEXT,
  payment_id       UUID REFERENCES payments(id),
  status           TEXT DEFAULT 'pending',              -- 'pending'|'paid'|'expired'
  paid_at          TIMESTAMPTZ,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- 7. BOOKING LIFECYCLE
-- ============================================================

CREATE TABLE booking_modifications (
  id                         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id                 UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id                    UUID REFERENCES users(id) ON DELETE SET NULL,
  change_type                TEXT NOT NULL,             -- 'dates'|'room_type'|'guest_count'|'addon'|'meal_plan'
  old_values                 JSONB NOT NULL,
  new_values                 JSONB NOT NULL,
  policy_tier                TEXT,                      -- 'free'|'fee'|'restricted'|'locked'
  days_until_checkin_at_request INTEGER,
  modification_fee           NUMERIC(10,2) DEFAULT 0,
  modification_fee_currency  TEXT,
  price_diff                 NUMERIC(10,2),
  price_diff_currency        TEXT,
  status                     TEXT DEFAULT 'pending',    -- 'pending'|'confirmed'|'rejected'
  confirmed_at               TIMESTAMPTZ,
  created_at                 TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE cancellations (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id            UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id               UUID REFERENCES users(id) ON DELETE SET NULL,
  reason                TEXT,                           -- 'change_of_plans'|'found_better_price'|'force_majeure'|'overbooking'|'other'
  reason_text           TEXT,
  days_until_checkin    SMALLINT,
  policy_applied        TEXT,                           -- 'full_refund'|'partial_refund'|'no_refund'|'force_majeure_full'
  refund_percent        NUMERIC(5,2),
  refund_amount         NUMERIC(10,2),
  currency              TEXT,
  refund_timeline_days  SMALLINT,
  initiated_by          TEXT DEFAULT 'guest',           -- 'guest'|'hotel'|'system'|'admin'
  created_at            TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- 8. GROUP & CORPORATE BOOKINGS
-- ============================================================

CREATE TABLE room_blocks (
  id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id               UUID NOT NULL REFERENCES hotel_partners(id),
  room_type_id           UUID NOT NULL REFERENCES room_types(id),
  rooms_held             SMALLINT NOT NULL,
  check_in               DATE NOT NULL,
  check_out              DATE NOT NULL,
  coordinator_session_id UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
  coordinator_user_id    UUID REFERENCES users(id) ON DELETE SET NULL,
  group_booking_id       UUID,
  status                 TEXT DEFAULT 'held',           -- 'held'|'confirmed'|'at_risk'|'released'|'expired'
  group_booking_url      TEXT,
  expires_at             TIMESTAMPTZ,
  created_at             TIMESTAMPTZ DEFAULT NOW(),
  updated_at             TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE group_bookings (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  room_block_id         UUID REFERENCES room_blocks(id),
  coordinator_user_id   UUID REFERENCES users(id) ON DELETE SET NULL,
  group_name            TEXT,
  event_type            TEXT,                           -- 'wedding'|'conference'|'corporate_retreat'|'tour_group'|'mice'
  num_rooms             SMALLINT NOT NULL,
  total_amount          NUMERIC(12,2),
  currency              TEXT DEFAULT 'USD',
  deposit_schedule      JSONB DEFAULT '[]',
  contract_type         TEXT DEFAULT 'standard',        -- 'standard'|'mice'
  status                TEXT DEFAULT 'pending',
  meeting_room_required BOOLEAN DEFAULT FALSE,
  meeting_room_capacity SMALLINT,
  notes                 TEXT,
  created_at            TIMESTAMPTZ DEFAULT NOW(),
  updated_at            TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE bookings ADD CONSTRAINT fk_bookings_group
  FOREIGN KEY (group_booking_id) REFERENCES group_bookings(id) ON DELETE SET NULL;
ALTER TABLE room_blocks ADD CONSTRAINT fk_room_blocks_group
  FOREIGN KEY (group_booking_id) REFERENCES group_bookings(id) ON DELETE SET NULL;

CREATE TABLE group_deposit_payments (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  group_booking_id    UUID NOT NULL REFERENCES group_bookings(id) ON DELETE CASCADE,
  installment_number  SMALLINT NOT NULL,
  due_date            DATE NOT NULL,
  amount              NUMERIC(10,2) NOT NULL,
  percent_of_total    NUMERIC(5,2),
  currency            TEXT NOT NULL,
  status              TEXT DEFAULT 'pending',           -- 'pending'|'paid'|'overdue'|'waived'
  payment_id          UUID REFERENCES payments(id),
  reminder_sent_at    TIMESTAMPTZ,
  paid_at             TIMESTAMPTZ,
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE corporate_accounts (
  id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  company_name             TEXT NOT NULL,
  company_domain           TEXT,
  billing_email            TEXT,
  billing_address          TEXT,
  tax_id                   TEXT,
  account_manager_email    TEXT,
  auto_approve_threshold   NUMERIC(10,2) DEFAULT 500,
  currency                 TEXT DEFAULT 'USD',
  invoice_frequency        TEXT DEFAULT 'monthly',      -- 'per_booking'|'weekly'|'monthly'
  expense_integration      TEXT,                        -- 'sap_concur'|'quickbooks'|'none'
  is_active                BOOLEAN DEFAULT TRUE,
  created_at               TIMESTAMPTZ DEFAULT NOW(),
  updated_at               TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE bookings ADD CONSTRAINT fk_bookings_corporate
  FOREIGN KEY (corporate_account_id) REFERENCES corporate_accounts(id) ON DELETE SET NULL;

CREATE TABLE corporate_employees (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  corporate_account_id  UUID NOT NULL REFERENCES corporate_accounts(id) ON DELETE CASCADE,
  user_id               UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  employee_id           TEXT,
  cost_center           TEXT,
  department            TEXT,
  role                  TEXT DEFAULT 'employee',        -- 'employee'|'manager'|'admin'
  spending_limit        NUMERIC(10,2),
  is_active             BOOLEAN DEFAULT TRUE,
  created_at            TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (corporate_account_id, user_id)
);

-- Formal travel policy per corporate account
CREATE TABLE corporate_travel_policies (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  corporate_account_id  UUID NOT NULL REFERENCES corporate_accounts(id) ON DELETE CASCADE,
  max_nightly_rate      NUMERIC(10,2),                 -- 0 = no limit
  max_star_rating       SMALLINT DEFAULT 5,
  preferred_districts   TEXT[] DEFAULT '{}',            -- CBD, airport, etc.
  require_manager_approval_above NUMERIC(10,2),         -- NULL = never require
  blackout_hotels       UUID[] DEFAULT '{}',
  preferred_hotels      UUID[] DEFAULT '{}',
  allowed_rate_plans    TEXT[] DEFAULT '{}',            -- empty = all
  carbon_offset_required BOOLEAN DEFAULT FALSE,
  advance_booking_days  SMALLINT DEFAULT 0,             -- must book N days ahead
  expense_category_mapping JSONB DEFAULT '{}',         -- {hotel:"T&E", meals:"Meals"}
  is_active             BOOLEAN DEFAULT TRUE,
  created_at            TIMESTAMPTZ DEFAULT NOW(),
  updated_at            TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (corporate_account_id)
);

CREATE TABLE corporate_rates (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  corporate_account_id  UUID NOT NULL REFERENCES corporate_accounts(id) ON DELETE CASCADE,
  hotel_id              UUID NOT NULL REFERENCES hotel_partners(id),
  room_type_id          UUID REFERENCES room_types(id),
  negotiated_rate_per_night NUMERIC(10,2) NOT NULL,
  currency              TEXT NOT NULL DEFAULT 'USD',
  includes              JSONB DEFAULT '[]',             -- ['breakfast','wifi','parking']
  valid_from            DATE NOT NULL,
  valid_until           DATE NOT NULL,
  is_active             BOOLEAN DEFAULT TRUE,
  created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE rate_blackout_dates (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  corporate_rate_id   UUID REFERENCES corporate_rates(id) ON DELETE CASCADE,
  hotel_id            UUID NOT NULL REFERENCES hotel_partners(id),
  blackout_date       DATE NOT NULL,
  reason              TEXT,
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE approval_requests (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id            UUID REFERENCES bookings(id) ON DELETE SET NULL,
  corporate_account_id  UUID NOT NULL REFERENCES corporate_accounts(id),
  employee_id           UUID NOT NULL REFERENCES corporate_employees(id),
  approver_user_id      UUID REFERENCES users(id) ON DELETE SET NULL,
  approver_email        TEXT,
  booking_total         NUMERIC(10,2) NOT NULL,
  currency              TEXT NOT NULL,
  cost_center           TEXT,
  status                TEXT DEFAULT 'pending',         -- 'pending'|'approved'|'rejected'|'expired'
  approval_token        TEXT UNIQUE,
  decided_at            TIMESTAMPTZ,
  expires_at            TIMESTAMPTZ,
  notes                 TEXT,
  created_at            TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- 9. FORCE MAJEURE & OVERBOOKING
-- ============================================================

CREATE TABLE force_majeure_events (
  id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  country_code            TEXT NOT NULL,
  event_type              TEXT NOT NULL,               -- 'travel_ban'|'natural_disaster'|'political_unrest'|'pandemic'|'flood'|'earthquake'
  event_description       TEXT,
  detected_by             TEXT DEFAULT 'news_monitor',
  confidence              NUMERIC(4,3),
  affected_date_from      DATE,
  affected_date_until     DATE,
  news_source_urls        TEXT[] DEFAULT '{}',
  affected_bookings_count INTEGER DEFAULT 0,
  status                  TEXT DEFAULT 'pending_review',-- 'pending_review'|'confirmed'|'dismissed'
  reviewed_by_user_id     UUID REFERENCES users(id) ON DELETE SET NULL,
  reviewed_at             TIMESTAMPTZ,
  auto_refund_triggered   BOOLEAN DEFAULT FALSE,
  created_at              TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE bookings ADD CONSTRAINT fk_bookings_force_majeure
  FOREIGN KEY (force_majeure_event_id) REFERENCES force_majeure_events(id) ON DELETE SET NULL;
ALTER TABLE support_tickets ADD CONSTRAINT fk_support_tickets_booking
  FOREIGN KEY (booking_id) REFERENCES bookings(id) ON DELETE SET NULL;


-- ============================================================
-- 10. LOYALTY & GAMIFICATION
-- ============================================================

CREATE TABLE loyalty_accounts (
  id                              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id                         UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
  total_points                    INTEGER DEFAULT 0,
  available_points                INTEGER DEFAULT 0,
  points_ytd                      INTEGER DEFAULT 0,
  tier                            TEXT DEFAULT 'bronze', -- 'bronze'|'silver'|'gold'|'platinum'|'black'
  tier_expiry                     DATE,
  total_bookings                  SMALLINT DEFAULT 0,
  total_nights                    INTEGER DEFAULT 0,
  unique_countries                TEXT[] DEFAULT '{}',
  consecutive_booking_months      SMALLINT DEFAULT 0,
  streak_last_updated             DATE,
  leaderboard_rank                INTEGER,
  leaderboard_points_this_month   INTEGER DEFAULT 0,
  annual_summary_url              TEXT,
  created_at                      TIMESTAMPTZ DEFAULT NOW(),
  updated_at                      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE loyalty_transactions (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id               UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  loyalty_account_id    UUID NOT NULL REFERENCES loyalty_accounts(id) ON DELETE CASCADE,
  booking_id            UUID REFERENCES bookings(id) ON DELETE SET NULL,
  action_type           TEXT NOT NULL,                  -- 'booking'|'review'|'referral'|'checkin'|'first_booking'|'redemption'|'expiry'|'adjustment'|'overbooking_compensation'
  points_earned         INTEGER DEFAULT 0,
  points_spent          INTEGER DEFAULT 0,
  points_expired        INTEGER DEFAULT 0,
  multiplier            NUMERIC(4,2) DEFAULT 1.0,
  base_points           INTEGER,
  bonus_points          INTEGER DEFAULT 0,
  running_balance       INTEGER,
  description           TEXT,
  created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE badge_definitions (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  badge_key         TEXT NOT NULL UNIQUE,
  name              TEXT NOT NULL,
  description       TEXT,
  icon_url          TEXT,
  trigger_event     TEXT NOT NULL,
  condition_type    TEXT,
  condition_value   INTEGER,
  points_bonus      INTEGER DEFAULT 0,
  is_active         BOOLEAN DEFAULT TRUE,
  created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE user_badges (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  badge_id    UUID NOT NULL REFERENCES badge_definitions(id),
  booking_id  UUID REFERENCES bookings(id) ON DELETE SET NULL,
  awarded_at  TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (user_id, badge_id)
);

CREATE TABLE referrals (
  id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  referrer_user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  referred_user_id         UUID REFERENCES users(id) ON DELETE SET NULL,
  referral_code            TEXT NOT NULL UNIQUE,
  status                   TEXT DEFAULT 'pending',       -- 'pending'|'converted'|'expired'
  converted_booking_id     UUID REFERENCES bookings(id) ON DELETE SET NULL,
  referrer_points_awarded  INTEGER DEFAULT 0,
  referred_discount_pct    NUMERIC(5,2) DEFAULT 0,
  created_at               TIMESTAMPTZ DEFAULT NOW(),
  converted_at             TIMESTAMPTZ
);


-- ============================================================
-- 11. REVIEWS
-- ============================================================

CREATE TABLE reviews (
  id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id           UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id              UUID REFERENCES users(id) ON DELETE SET NULL,
  hotel_id             UUID NOT NULL REFERENCES hotel_partners(id),
  overall_rating       SMALLINT CHECK (overall_rating BETWEEN 1 AND 5),
  review_text          TEXT,
  language             TEXT DEFAULT 'en',
  sentiment            TEXT,                            -- 'positive'|'neutral'|'negative'
  sentiment_score      NUMERIC(4,3),
  aspect_scores        JSONB DEFAULT '{}',              -- {room:0.9, food:0.7, staff:0.95}
  is_negative_alert    BOOLEAN DEFAULT FALSE,
  hotel_responded_at   TIMESTAMPTZ,
  hotel_response_text  TEXT,
  is_published         BOOLEAN DEFAULT FALSE,
  channel              TEXT DEFAULT 'chat',
  follow_up_sent       BOOLEAN DEFAULT FALSE,
  follow_up_sent_at    TIMESTAMPTZ,
  created_at           TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- 12. NOTIFICATIONS
-- ============================================================

CREATE TABLE notification_preferences (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id             UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
  preferred_channel   TEXT DEFAULT 'email',
  email_enabled       BOOLEAN DEFAULT TRUE,
  sms_enabled         BOOLEAN DEFAULT FALSE,
  whatsapp_enabled    BOOLEAN DEFAULT FALSE,
  push_enabled        BOOLEAN DEFAULT TRUE,
  messenger_enabled   BOOLEAN DEFAULT TRUE,
  marketing_emails    BOOLEAN DEFAULT FALSE,
  booking_reminders   BOOLEAN DEFAULT TRUE,
  review_requests     BOOLEAN DEFAULT TRUE,
  loyalty_updates     BOOLEAN DEFAULT TRUE,
  promotional_offers  BOOLEAN DEFAULT FALSE,
  price_alerts        BOOLEAN DEFAULT TRUE,
  pre_arrival_tips    BOOLEAN DEFAULT TRUE,
  weather_alerts      BOOLEAN DEFAULT TRUE,
  created_at          TIMESTAMPTZ DEFAULT NOW(),
  updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE device_push_tokens (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  token         TEXT NOT NULL,
  platform      TEXT NOT NULL,                          -- 'android'|'ios'|'web'
  device_model  TEXT,
  app_version   TEXT,
  is_active     BOOLEAN DEFAULT TRUE,
  last_used_at  TIMESTAMPTZ,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (user_id, token)
);

CREATE TABLE notification_logs (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id               UUID REFERENCES users(id) ON DELETE SET NULL,
  booking_id            UUID REFERENCES bookings(id) ON DELETE SET NULL,
  notification_type     TEXT NOT NULL,
  -- 'booking_confirmation'|'modification_alert'|'cancellation_confirmation'
  -- 'review_request'|'payment_receipt'|'loyalty_upgrade'|'otp'
  -- 'force_majeure_alert'|'checkin_reminder'|'deposit_reminder'
  -- 'price_alert'|'waitlist_available'|'abandoned_booking'|'agent_reply'
  channel               TEXT NOT NULL,                  -- 'email'|'sms'|'whatsapp'|'push_android'|'push_ios'|'push_web'|'messenger'
  recipient             TEXT NOT NULL,
  template_name         TEXT,
  status                TEXT DEFAULT 'pending',
  gateway_message_id    TEXT,
  error_code            TEXT,
  error_message         TEXT,
  opened_at             TIMESTAMPTZ,
  clicked_at            TIMESTAMPTZ,
  delivery_confirmed_at TIMESTAMPTZ,
  retry_count           SMALLINT DEFAULT 0,
  created_at            TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- 13. PRIVACY & COMPLIANCE
-- ============================================================

CREATE TABLE consent_records (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  consent_type      TEXT NOT NULL,
  consented         BOOLEAN NOT NULL,
  ip_address        INET,
  user_agent        TEXT,
  consent_version   TEXT,
  recorded_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE privacy_requests (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id          UUID NOT NULL REFERENCES users(id) ON DELETE SET NULL,
  request_type     TEXT NOT NULL,                       -- 'erasure'|'export'|'access'|'portability'
  status           TEXT DEFAULT 'pending',
  requester_ip     INET,
  data_categories  TEXT[] DEFAULT '{}',
  export_url       TEXT,
  completed_at     TIMESTAMPTZ,
  must_complete_by TIMESTAMPTZ,
  created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE compliance_audit_log (
  id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_type            TEXT NOT NULL,
  user_id               UUID,
  performed_by_user_id  UUID,
  ip_address            INET,
  data_categories       TEXT[] DEFAULT '{}',
  details               JSONB DEFAULT '{}',
  created_at            TIMESTAMPTZ DEFAULT NOW()
  -- Append-only: NEVER grant UPDATE or DELETE on this table
);


-- ============================================================
-- 14. ANALYTICS EVENTS
-- ============================================================

CREATE TABLE analytics_events (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_type   TEXT NOT NULL,
  -- 'session_started'|'intent_classified'|'booking_completed'|'booking_modified'
  -- 'booking_cancelled'|'payment_attempted'|'review_submitted'|'loyalty_points_awarded'
  -- 'voucher_applied'|'waitlist_joined'|'price_alert_triggered'|'abandoned_booking'
  -- 'overbooking_incident'|'agent_handoff'|'service_booking_made'
  session_id   UUID REFERENCES chat_sessions(id) ON DELETE SET NULL,
  user_id      UUID REFERENCES users(id) ON DELETE SET NULL,
  booking_id   UUID REFERENCES bookings(id) ON DELETE SET NULL,
  hotel_id     UUID REFERENCES hotel_partners(id) ON DELETE SET NULL,
  properties   JSONB DEFAULT '{}',
  revenue_usd  NUMERIC(10,2),
  language     TEXT,
  channel      TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- 15. SUPPORTING TABLES
-- ============================================================

CREATE TABLE system_config (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  config_key    TEXT NOT NULL UNIQUE,
  config_value  JSONB NOT NULL,
  description   TEXT,
  updated_by    UUID REFERENCES users(id) ON DELETE SET NULL,
  updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE airport_codes (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  iata_code     TEXT NOT NULL UNIQUE,
  airport_name  TEXT,
  city          TEXT,
  country       TEXT,
  lat           NUMERIC(10,7),
  lng           NUMERIC(10,7),
  timezone      TEXT
);

CREATE TABLE geocoding_cache (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  query         TEXT NOT NULL UNIQUE,
  lat           NUMERIC(10,7) NOT NULL,
  lng           NUMERIC(10,7) NOT NULL,
  display_name  TEXT,
  confidence    NUMERIC(4,3),
  cached_at     TIMESTAMPTZ DEFAULT NOW(),
  expires_at    TIMESTAMPTZ DEFAULT NOW() + INTERVAL '24 hours'
);

CREATE TABLE fx_rates_cache (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  from_currency   TEXT NOT NULL,
  to_currency     TEXT NOT NULL,
  rate            NUMERIC(14,8) NOT NULL,
  source          TEXT DEFAULT 'open_exchange_rates',
  fetched_at      TIMESTAMPTZ DEFAULT NOW(),
  expires_at      TIMESTAMPTZ DEFAULT NOW() + INTERVAL '1 hour',
  UNIQUE (from_currency, to_currency)
);


-- ============================================================
-- 16. INDEXES — Complete & Optimized
-- ============================================================

-- ── Users ──────────────────────────────────────────────────
CREATE INDEX idx_users_email         ON users(email)                  WHERE email IS NOT NULL;
CREATE INDEX idx_users_phone         ON users(phone)                  WHERE phone IS NOT NULL;
CREATE INDEX idx_users_active        ON users(is_active)              WHERE is_active = TRUE;
-- Trigram for fuzzy name search
CREATE INDEX idx_users_first_trgm    ON users USING gin(first_name gin_trgm_ops);
CREATE INDEX idx_users_last_trgm     ON users USING gin(last_name gin_trgm_ops);

-- ── Platform identities ────────────────────────────────────
CREATE INDEX idx_platform_id_user     ON platform_identities(user_id);
CREATE INDEX idx_platform_id_platform ON platform_identities(platform, platform_user_id);

-- ── OAuth ──────────────────────────────────────────────────
CREATE INDEX idx_oauth_user           ON oauth_accounts(user_id);
CREATE INDEX idx_oauth_provider       ON oauth_accounts(provider, provider_user_id);

-- ── Auth blacklist ─────────────────────────────────────────
CREATE INDEX idx_auth_blacklist_hash   ON auth_token_blacklist(token_hash);
CREATE INDEX idx_auth_blacklist_expiry ON auth_token_blacklist(expires_at);

-- ── Chat sessions ──────────────────────────────────────────
CREATE INDEX idx_chat_sessions_user       ON chat_sessions(user_id);
CREATE INDEX idx_chat_sessions_session_id ON chat_sessions(session_id);
CREATE INDEX idx_chat_sessions_channel    ON chat_sessions(channel);
CREATE INDEX idx_chat_sessions_platform   ON chat_sessions(platform_identity_id);
CREATE INDEX idx_chat_sessions_started    ON chat_sessions(session_started_at);

-- ── Messages ───────────────────────────────────────────────
CREATE INDEX idx_conv_messages_session ON conversation_messages(session_id, turn_number);
CREATE INDEX idx_conv_messages_created ON conversation_messages(created_at);

-- ── Booking sessions log ───────────────────────────────────
CREATE INDEX idx_bsl_user            ON booking_sessions_log(user_id);
CREATE INDEX idx_bsl_abandoned       ON booking_sessions_log(was_abandoned, recovery_sent_at)
                                     WHERE was_abandoned = TRUE;
CREATE INDEX idx_bsl_hotel_checkin   ON booking_sessions_log(hotel_id, checkin);

-- ── NLU ────────────────────────────────────────────────────
CREATE INDEX idx_nlu_session         ON nlu_confidence_log(session_id);
CREATE INDEX idx_nlu_intent          ON nlu_confidence_log(intent);
CREATE INDEX idx_nlu_fallback        ON nlu_confidence_log(was_fallback) WHERE was_fallback = TRUE;

-- ── Hotels ─────────────────────────────────────────────────
CREATE INDEX idx_hotel_city          ON hotel_partners(city);
CREATE INDEX idx_hotel_country       ON hotel_partners(country);
CREATE INDEX idx_hotel_stars         ON hotel_partners(star_rating);
CREATE INDEX idx_hotel_active        ON hotel_partners(is_active)   WHERE is_active = TRUE;
CREATE INDEX idx_hotel_lat_lng       ON hotel_partners(lat, lng);
CREATE INDEX idx_hotel_pet           ON hotel_partners(is_pet_friendly) WHERE is_pet_friendly = TRUE;
CREATE INDEX idx_hotel_halal         ON hotel_partners(is_halal_certified) WHERE is_halal_certified = TRUE;
-- Trigram for fuzzy hotel name search (e.g. "Oberoi" -> "Grand Oberoi")
CREATE INDEX idx_hotel_name_trgm     ON hotel_partners USING gin(name gin_trgm_ops);
CREATE INDEX idx_hotel_city_trgm     ON hotel_partners USING gin(city gin_trgm_ops);

-- ── Room availability (hottest table — most reads) ─────────
CREATE INDEX idx_avail_type_date     ON room_availability(room_type_id, date);
CREATE INDEX idx_avail_hotel_date    ON room_availability(hotel_id, date);
CREATE INDEX idx_avail_date          ON room_availability(date);
CREATE INDEX idx_avail_available     ON room_availability(hotel_id, date, available_count)
                                     WHERE available_count > 0 AND is_blackout = FALSE;
-- Partial index for non-blackout dates (most queries filter this)
CREATE INDEX idx_avail_open          ON room_availability(room_type_id, date, available_count)
                                     WHERE is_blackout = FALSE AND available_count > 0;

-- ── Room rates ─────────────────────────────────────────────
CREATE INDEX idx_rates_type          ON room_rates(room_type_id);
CREATE INDEX idx_rates_hotel         ON room_rates(hotel_id);
CREATE INDEX idx_rates_active_plan   ON room_rates(hotel_id, rate_plan) WHERE is_active = TRUE;
CREATE INDEX idx_rates_validity      ON room_rates(valid_from, valid_until);

-- ── Seasonal pricing ───────────────────────────────────────
CREATE INDEX idx_seasonal_dates      ON seasonal_pricing_rules(date_from, date_to);
CREATE INDEX idx_seasonal_city       ON seasonal_pricing_rules(city);
CREATE INDEX idx_seasonal_hotel      ON seasonal_pricing_rules(hotel_id);

-- ── Bookings (most critical table) ────────────────────────
CREATE INDEX idx_bookings_user       ON bookings(user_id);
CREATE INDEX idx_bookings_hotel      ON bookings(hotel_id);
CREATE INDEX idx_bookings_ref        ON bookings(booking_reference);
CREATE INDEX idx_bookings_status     ON bookings(status);
CREATE INDEX idx_bookings_checkin    ON bookings(check_in);
CREATE INDEX idx_bookings_checkout   ON bookings(check_out);
CREATE INDEX idx_bookings_created    ON bookings(created_at);
-- Compound: hotel + checkin (PMS dashboard view)
CREATE INDEX idx_bookings_hotel_ci   ON bookings(hotel_id, check_in, status);
-- Compound: user + status (guest's booking list)
CREATE INDEX idx_bookings_user_stat  ON bookings(user_id, status, check_in);
-- Partial: pending review
CREATE INDEX idx_bookings_review     ON bookings(review_requested, check_out)
                                     WHERE review_requested = FALSE;
-- Partial: active stays
CREATE INDEX idx_bookings_active     ON bookings(hotel_id, check_in, check_out)
                                     WHERE status IN ('confirmed','checked_in');
-- Partial: abandoned (soft lock expired)
CREATE INDEX idx_bookings_softlock   ON bookings(soft_lock_expires_at)
                                     WHERE soft_lock_id IS NOT NULL;
-- Corporate bookings
CREATE INDEX idx_bookings_corp       ON bookings(corporate_account_id)
                                     WHERE is_corporate = TRUE;

-- ── Guests ─────────────────────────────────────────────────
CREATE INDEX idx_guests_booking      ON guests(booking_id);
CREATE INDEX idx_guests_email        ON guests(email)        WHERE email IS NOT NULL;
CREATE INDEX idx_guests_primary      ON guests(booking_id)   WHERE is_primary = TRUE;

-- ── Service bookings (in-stay) ─────────────────────────────
CREATE INDEX idx_svc_booking         ON service_bookings(booking_id);
CREATE INDEX idx_svc_type            ON service_bookings(service_type);
CREATE INDEX idx_svc_status          ON service_bookings(status);
CREATE INDEX idx_svc_scheduled       ON service_bookings(scheduled_at) WHERE scheduled_at IS NOT NULL;

-- ── In-stay requests ───────────────────────────────────────
CREATE INDEX idx_instay_booking      ON in_stay_requests(booking_id);
CREATE INDEX idx_instay_open         ON in_stay_requests(status) WHERE status IN ('open','in_progress');

-- ── Payments ───────────────────────────────────────────────
CREATE INDEX idx_payments_booking    ON payments(booking_id);
CREATE INDEX idx_payments_user       ON payments(user_id);
CREATE INDEX idx_payments_status     ON payments(status);
CREATE INDEX idx_payments_txn        ON payments(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX idx_payments_gateway    ON payments(gateway);
CREATE INDEX idx_payments_intent     ON payments(payment_intent_id) WHERE payment_intent_id IS NOT NULL;
-- Partial: failed payments (for fraud analysis)
CREATE INDEX idx_payments_failed     ON payments(user_id, created_at)
                                     WHERE status = 'failed';

-- ── Payment attempts ───────────────────────────────────────
CREATE INDEX idx_pmt_attempts_booking ON payment_attempts(booking_id);
CREATE INDEX idx_pmt_attempts_user    ON payment_attempts(user_id, attempted_at);

-- ── Vouchers ───────────────────────────────────────────────
CREATE INDEX idx_vouchers_code        ON vouchers(code);
CREATE INDEX idx_vouchers_active      ON vouchers(is_active, valid_until) WHERE is_active = TRUE;

-- ── Waitlist & price alerts ────────────────────────────────
CREATE INDEX idx_waitlist_hotel_dates ON room_waitlist(hotel_id, checkin, checkout);
CREATE INDEX idx_waitlist_user        ON room_waitlist(user_id);
CREATE INDEX idx_waitlist_status      ON room_waitlist(status) WHERE status = 'waiting';
CREATE INDEX idx_alerts_hotel_dates   ON price_alerts(hotel_id, checkin, checkout);
CREATE INDEX idx_alerts_user          ON price_alerts(user_id);
CREATE INDEX idx_alerts_active        ON price_alerts(status) WHERE status = 'active';

-- ── Support tickets ────────────────────────────────────────
CREATE INDEX idx_tickets_user         ON support_tickets(user_id);
CREATE INDEX idx_tickets_booking      ON support_tickets(booking_id);
CREATE INDEX idx_tickets_status       ON support_tickets(status) WHERE status IN ('open','in_progress');
CREATE INDEX idx_tickets_priority     ON support_tickets(priority, created_at);

-- ── Addons ─────────────────────────────────────────────────
CREATE INDEX idx_addons_hotel         ON addons(hotel_id);
CREATE INDEX idx_addons_category      ON addons(category);
CREATE INDEX idx_addons_available     ON addons(hotel_id, category) WHERE is_available = TRUE;

-- ── Loyalty ────────────────────────────────────────────────
CREATE INDEX idx_loyalty_user         ON loyalty_accounts(user_id);
CREATE INDEX idx_loyalty_tier         ON loyalty_accounts(tier);
CREATE INDEX idx_loyalty_leaderboard  ON loyalty_accounts(leaderboard_points_this_month DESC);
CREATE INDEX idx_loyalty_txn_user     ON loyalty_transactions(user_id, created_at);
CREATE INDEX idx_loyalty_txn_booking  ON loyalty_transactions(booking_id);

-- ── Reviews ────────────────────────────────────────────────
CREATE INDEX idx_reviews_hotel        ON reviews(hotel_id, is_published);
CREATE INDEX idx_reviews_user         ON reviews(user_id);
CREATE INDEX idx_reviews_rating       ON reviews(hotel_id, overall_rating);
CREATE INDEX idx_reviews_negative     ON reviews(is_negative_alert)
                                      WHERE is_negative_alert = TRUE;
CREATE INDEX idx_reviews_unpublished  ON reviews(is_published) WHERE is_published = FALSE;

-- ── Notifications ──────────────────────────────────────────
CREATE INDEX idx_notif_logs_user      ON notification_logs(user_id);
CREATE INDEX idx_notif_logs_booking   ON notification_logs(booking_id);
CREATE INDEX idx_notif_logs_type      ON notification_logs(notification_type);
CREATE INDEX idx_notif_logs_status    ON notification_logs(status);
-- Partial: failed — for retry worker
CREATE INDEX idx_notif_logs_failed    ON notification_logs(created_at)
                                      WHERE status = 'failed';

-- ── Analytics ──────────────────────────────────────────────
CREATE INDEX idx_analytics_type       ON analytics_events(event_type);
CREATE INDEX idx_analytics_session    ON analytics_events(session_id);
CREATE INDEX idx_analytics_created    ON analytics_events(created_at);
CREATE INDEX idx_analytics_hotel      ON analytics_events(hotel_id);
CREATE INDEX idx_analytics_user       ON analytics_events(user_id);
-- Compound for revenue reporting
CREATE INDEX idx_analytics_revenue    ON analytics_events(hotel_id, event_type, created_at)
                                      WHERE revenue_usd IS NOT NULL;

-- ── Corporate ──────────────────────────────────────────────
CREATE INDEX idx_corp_employees_acct  ON corporate_employees(corporate_account_id);
CREATE INDEX idx_corp_employees_user  ON corporate_employees(user_id);
CREATE INDEX idx_corp_rates_acct      ON corporate_rates(corporate_account_id, hotel_id);
CREATE INDEX idx_approval_booking     ON approval_requests(booking_id);
CREATE INDEX idx_approval_status      ON approval_requests(status) WHERE status = 'pending';

-- ── Force majeure ──────────────────────────────────────────
CREATE INDEX idx_fm_country           ON force_majeure_events(country_code);
CREATE INDEX idx_fm_status            ON force_majeure_events(status);
CREATE INDEX idx_fm_dates             ON force_majeure_events(affected_date_from, affected_date_until);

-- ── Nearby places ──────────────────────────────────────────
CREATE INDEX idx_nearby_hotel         ON hotel_nearby_places(hotel_id);
CREATE INDEX idx_nearby_type          ON hotel_nearby_places(hotel_id, place_type);
CREATE INDEX idx_nearby_emergency     ON hotel_nearby_places(hotel_id) WHERE is_emergency = TRUE;

-- ── Geocoding / FX ─────────────────────────────────────────
CREATE INDEX idx_geocoding_expires    ON geocoding_cache(expires_at);
CREATE INDEX idx_geocoding_query_trgm ON geocoding_cache USING gin(query gin_trgm_ops);
CREATE INDEX idx_fx_expiry            ON fx_rates_cache(expires_at);
CREATE INDEX idx_fx_pair              ON fx_rates_cache(from_currency, to_currency);


-- ============================================================
-- 17. ROW LEVEL SECURITY
-- ============================================================

ALTER TABLE users                    ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions            ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_messages    ENABLE ROW LEVEL SECURITY;
ALTER TABLE bookings                 ENABLE ROW LEVEL SECURITY;
ALTER TABLE guests                   ENABLE ROW LEVEL SECURITY;
ALTER TABLE payments                 ENABLE ROW LEVEL SECURITY;
ALTER TABLE loyalty_accounts         ENABLE ROW LEVEL SECURITY;
ALTER TABLE loyalty_transactions     ENABLE ROW LEVEL SECURITY;
ALTER TABLE reviews                  ENABLE ROW LEVEL SECURITY;
ALTER TABLE notification_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE device_push_tokens       ENABLE ROW LEVEL SECURITY;
ALTER TABLE consent_records          ENABLE ROW LEVEL SECURITY;
ALTER TABLE privacy_requests         ENABLE ROW LEVEL SECURITY;
ALTER TABLE platform_identities      ENABLE ROW LEVEL SECURITY;
ALTER TABLE service_bookings         ENABLE ROW LEVEL SECURITY;
ALTER TABLE in_stay_requests         ENABLE ROW LEVEL SECURITY;
ALTER TABLE lost_found_reports       ENABLE ROW LEVEL SECURITY;
ALTER TABLE price_alerts             ENABLE ROW LEVEL SECURITY;
ALTER TABLE room_waitlist            ENABLE ROW LEVEL SECURITY;
ALTER TABLE support_tickets          ENABLE ROW LEVEL SECURITY;

-- Users: own row only
CREATE POLICY users_self ON users
  FOR ALL USING (auth.uid() = id);

-- Bookings: own bookings
CREATE POLICY bookings_user ON bookings
  FOR SELECT USING (auth.uid() = user_id);

-- Guests: via own bookings
CREATE POLICY guests_user ON guests
  FOR SELECT USING (
    booking_id IN (SELECT id FROM bookings WHERE user_id = auth.uid())
  );

-- Payments: own
CREATE POLICY payments_user ON payments
  FOR SELECT USING (auth.uid() = user_id);

-- Loyalty
CREATE POLICY loyalty_accounts_user ON loyalty_accounts
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY loyalty_txn_user ON loyalty_transactions
  FOR SELECT USING (auth.uid() = user_id);

-- Reviews: own
CREATE POLICY reviews_user ON reviews
  FOR ALL USING (auth.uid() = user_id);

-- Notification prefs: own
CREATE POLICY notif_prefs_user ON notification_preferences
  FOR ALL USING (auth.uid() = user_id);

-- Device tokens: own
CREATE POLICY device_tokens_user ON device_push_tokens
  FOR ALL USING (auth.uid() = user_id);

-- Consent: own
CREATE POLICY consent_user ON consent_records
  FOR ALL USING (auth.uid() = user_id);

-- Privacy: own
CREATE POLICY privacy_req_user ON privacy_requests
  FOR ALL USING (auth.uid() = user_id);

-- Platform identities: own
CREATE POLICY platform_id_user ON platform_identities
  FOR ALL USING (auth.uid() = user_id);

-- Service bookings: via own bookings
CREATE POLICY svc_bookings_user ON service_bookings
  FOR SELECT USING (
    booking_id IN (SELECT id FROM bookings WHERE user_id = auth.uid())
  );

-- In-stay requests: via own bookings
CREATE POLICY instay_user ON in_stay_requests
  FOR SELECT USING (
    booking_id IN (SELECT id FROM bookings WHERE user_id = auth.uid())
  );

-- Lost & found: via own bookings
CREATE POLICY lost_found_user ON lost_found_reports
  FOR SELECT USING (
    booking_id IN (SELECT id FROM bookings WHERE user_id = auth.uid())
  );

-- Price alerts: own
CREATE POLICY alerts_user ON price_alerts
  FOR ALL USING (auth.uid() = user_id);

-- Waitlist: own
CREATE POLICY waitlist_user ON room_waitlist
  FOR ALL USING (auth.uid() = user_id);

-- Support tickets: own
CREATE POLICY tickets_user ON support_tickets
  FOR SELECT USING (auth.uid() = user_id);


-- ============================================================
-- 18. UPDATED_AT TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to every table with updated_at
DO $$
DECLARE
  tbl TEXT;
BEGIN
  FOR tbl IN SELECT unnest(ARRAY[
    'users','hotel_partners','room_types','addons','bookings','guests',
    'payments','loyalty_accounts','room_blocks','group_bookings',
    'corporate_accounts','notification_preferences','hotel_faqs',
    'hotel_accessibility_features','hotel_services','service_bookings',
    'in_stay_requests','lost_found_reports','overbooking_incidents',
    'support_tickets','booking_sessions_log','corporate_travel_policies'
  ]) LOOP
    EXECUTE format(
      'CREATE TRIGGER trg_%s_updated_at
       BEFORE UPDATE ON %I
       FOR EACH ROW EXECUTE FUNCTION set_updated_at();',
      tbl, tbl
    );
  END LOOP;
END;
$$;


-- ============================================================
-- 19. SEED: SYSTEM CONFIG DEFAULTS
-- ============================================================

INSERT INTO system_config (config_key, config_value, description) VALUES
  ('points_per_dollar',               '10',      'Earn rate: 10 pts per $1 / per ₹100'),
  ('points_redemption_rate',          '2',       'Redemption: 2 pts = ₹1 / $0.01'),
  ('tier_silver_threshold',           '10000',   'Points YTD for Silver'),
  ('tier_gold_threshold',             '50000',   'Points YTD for Gold'),
  ('tier_platinum_threshold',         '100000',  'Points YTD for Platinum'),
  ('tier_black_threshold',            '500000',  'Points YTD for Black tier review'),
  ('tier_multiplier_silver',          '1.10',    'Silver 10% bonus'),
  ('tier_multiplier_gold',            '1.15',    'Gold 15% bonus'),
  ('tier_multiplier_platinum',        '1.20',    'Platinum 20% bonus'),
  ('tier_multiplier_black',           '3.0',     'Black 3× multiplier'),
  ('bonus_points_review',             '100',     'Points for review'),
  ('bonus_points_referral',           '500',     'Points for converted referral'),
  ('bonus_points_first_booking',      '500',     'Welcome bonus'),
  ('bonus_points_overbooking',        '5000',    'Compensation for overbooking incident'),
  ('soft_lock_duration_minutes',      '15',      'Room hold during booking flow'),
  ('group_booking_min_rooms',         '10',      'Rooms threshold for group flow'),
  ('fraud_block_threshold',           '71',      'Risk score ≥ this = block'),
  ('fraud_3ds_threshold',             '31',      'Risk score ≥ this = require 3DS'),
  ('velocity_check_max_attempts',     '5',       'Max payment attempts per user/hour'),
  ('review_send_hours_after_checkout','24',      'Hours after checkout for review request'),
  ('nlu_fallback_threshold',          '0.60',    'Confidence below = fallback'),
  ('language_detect_confidence_min',  '0.85',    'Min confidence for langdetect'),
  ('checkin_reminder_hours',          '24',      'Pre-arrival reminder lead time (hours)'),
  ('abandoned_booking_recovery_hours','2',       'Hours idle before recovery message'),
  ('price_alert_check_interval_mins', '60',      'How often to check price alerts'),
  ('max_split_payment_parts',         '5',       'Maximum people in a split payment'),
  ('late_checkout_half_rate_hours',   '4',       'Hours beyond checkout = 50% charge'),
  ('early_checkin_guaranteed_fee_pct','15',      'Guaranteed early CI as % of nightly'),
  ('waitlist_expiry_days',            '30',      'Auto-expire waitlist entries'),
  ('sms_fallback_enabled',            'true',    'Fall back to SMS if messenger fails');


-- ============================================================
-- 20. SEED: BADGE DEFINITIONS
-- ============================================================

INSERT INTO badge_definitions (badge_key, name, description, trigger_event, condition_type, condition_value, points_bonus) VALUES
  ('first_booking',      'First Steps',           'Completed your very first booking',                'booking_completed',    'count',             1,  500),
  ('globetrotter_5',     'Globetrotter',          'Stayed in 5 different countries',                  'booking_completed',    'unique_countries',  5,  300),
  ('globetrotter_10',    'World Explorer',        'Stayed in 10 different countries',                 'booking_completed',    'unique_countries',  10, 1000),
  ('streak_3',           'Loyal Traveler',        '3 consecutive months with a booking',              'streak_checkin',       'consecutive_months',3,  200),
  ('streak_6',           'Frequent Flyer',        '6 consecutive months with a booking',              'streak_checkin',       'consecutive_months',6,  500),
  ('reviewer',           'Honest Voice',          'Posted your first review',                         'review_posted',        'count',             1,  100),
  ('super_reviewer',     'Top Reviewer',          'Posted 10 reviews',                                'review_posted',        'count',             10, 500),
  ('honeymoon',          'Love Traveler',         'Booked a honeymoon stay',                          'booking_completed',    'specific_purpose',  0,  200),
  ('referral_hero',      'Referral Champion',     'Successfully referred 5 friends',                  'referral_converted',   'count',             5,  1000),
  ('night_owl_50',       '50 Nights',             'Accumulated 50 hotel nights',                      'booking_completed',    'total_nights',      50, 500),
  ('night_owl_200',      '200 Night Club',        'Accumulated 200 hotel nights',                     'booking_completed',    'total_nights',      200,2000),
  ('voice_user',         'Voice Pioneer',         'Made a booking via voice message',                 'booking_completed',    'specific_purpose',  0,  100);


-- ============================================================
-- END OF SCHEMA
-- ============================================================
-- BookBot v2.0 — Complete Normalized Schema
-- Tables: 55 | Indexes: 90+ | RLS Policies: 20
-- New in v2.0: platform_identities, vouchers, price_alerts,
--   room_waitlist, service_bookings, in_stay_requests,
--   lost_found_reports, support_tickets, agent_sessions,
--   seasonal_pricing_rules, overbooking_incidents,
--   hotel_accessibility_features, corporate_travel_policies,
--   split_payment_requests, booking_sessions_log, and more.
-- ============================================================