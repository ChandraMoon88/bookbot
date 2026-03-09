-- ============================================================
-- HOTEL BOOKING AI CHATBOT SYSTEM
-- Intelligent Normalized PostgreSQL Schema — v2.0
-- Design: 10 tables, smart JSONB, composite + partial indexes
-- ============================================================
--
-- TABLE MAP (what each table absorbs from the 47-table model):
-- ┌─────────────────┬──────────────────────────────────────────────────────────────────┐
-- │ users           │ identity + auth + oauth + loyalty account + preferences          │
-- │ hotels          │ hotel profile + policies + FAQs + amenity config                 │
-- │ inventory       │ room types + rate plans + daily availability                     │
-- │ catalog         │ addons + badge defs + system config + FX + airports              │
-- │ sessions        │ chat sessions + NLU log + handoff context                        │
-- │ bookings        │ booking + guests (JSONB) + addons cart + modification log        │
-- │ payments        │ payment records + attempt log + fraud metadata                   │
-- │ organizations   │ corporate accounts + employees + rates + group bookings           │
-- │ engagements     │ reviews + loyalty txns + notifications + consent                 │
-- │ system_events   │ analytics + audit log + force majeure (append-only)              │
-- └─────────────────┴──────────────────────────────────────────────────────────────────┘
--
-- NORMALIZATION DECISIONS:
--   1NF/2NF/3NF on all scalar columns.
--   JSONB only where: data is variable-length arrays always fetched as a unit,
--   translated text (i18n), snapshot/audit data, or sparse config.
--   Every JSONB column queried with @> or ->> has a GIN index.
--   No JSONB on any column used in WHERE with equality, ORDER BY, or GROUP BY.
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";


-- ============================================================
-- TABLE 1: users
-- Absorbs: users, oauth_accounts, auth_token_blacklist,
--          loyalty_accounts, notification_preferences, device_push_tokens
-- Why embedded: loyalty + prefs are 1:1 with user so a JOIN
--   would be pure overhead on every read. OAuth providers are
--   max 3 per user — fits cleanly in a JSONB array.
-- ============================================================

CREATE TABLE users (
  id                        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  email                     TEXT UNIQUE,
  phone                     TEXT UNIQUE,
  password_hash             TEXT,
  first_name                TEXT,
  last_name                 TEXT,
  nationality               TEXT,
  country_code              TEXT,
  timezone                  TEXT,
  profile_photo_url         TEXT,

  -- Language & voice
  preferred_language        TEXT NOT NULL DEFAULT 'en',
  is_rtl                    BOOLEAN NOT NULL DEFAULT FALSE,
  language_tier             SMALLINT NOT NULL DEFAULT 1,
  voice_gender_preference   TEXT NOT NULL DEFAULT 'female',
  voice_speed               NUMERIC(3,2) NOT NULL DEFAULT 1.0,

  -- Auth state
  is_email_verified         BOOLEAN NOT NULL DEFAULT FALSE,
  is_phone_verified         BOOLEAN NOT NULL DEFAULT FALSE,
  is_active                 BOOLEAN NOT NULL DEFAULT TRUE,
  failed_login_attempts     SMALLINT NOT NULL DEFAULT 0,
  locked_until              TIMESTAMPTZ,
  last_login_at             TIMESTAMPTZ,

  -- OAuth [{provider, provider_user_id, access_token, refresh_token, token_expires_at}]
  oauth_accounts            JSONB NOT NULL DEFAULT '[]',

  -- Blacklisted refresh tokens [{hash, expires_at}] — Celery purges expired entries nightly
  blacklisted_tokens        JSONB NOT NULL DEFAULT '[]',

  -- Loyalty (1:1 — zero joins needed for loyalty reads)
  loyalty_tier              TEXT NOT NULL DEFAULT 'bronze',
  loyalty_points_total      INTEGER NOT NULL DEFAULT 0,
  loyalty_points_available  INTEGER NOT NULL DEFAULT 0,
  loyalty_points_ytd        INTEGER NOT NULL DEFAULT 0,
  loyalty_tier_expiry       DATE,
  loyalty_total_bookings    INTEGER NOT NULL DEFAULT 0,
  loyalty_total_nights      INTEGER NOT NULL DEFAULT 0,
  loyalty_unique_countries  TEXT[] NOT NULL DEFAULT '{}',
  loyalty_streak_months     SMALLINT NOT NULL DEFAULT 0,
  loyalty_streak_updated    DATE,
  loyalty_leaderboard_rank  INTEGER,
  loyalty_badges            TEXT[] NOT NULL DEFAULT '{}',
  loyalty_points_this_month INTEGER NOT NULL DEFAULT 0,

  -- Notification prefs (1:1 — avoid join on every notification send)
  notif_preferred_channel   TEXT NOT NULL DEFAULT 'email',
  notif_email_enabled       BOOLEAN NOT NULL DEFAULT TRUE,
  notif_sms_enabled         BOOLEAN NOT NULL DEFAULT FALSE,
  notif_whatsapp_enabled    BOOLEAN NOT NULL DEFAULT FALSE,
  notif_push_enabled        BOOLEAN NOT NULL DEFAULT TRUE,
  notif_marketing           BOOLEAN NOT NULL DEFAULT FALSE,
  notif_review_requests     BOOLEAN NOT NULL DEFAULT TRUE,

  -- Device push tokens [{token, platform, device_model, app_version, is_active, last_used_at}]
  device_push_tokens        JSONB NOT NULL DEFAULT '[]',

  -- Privacy
  gdpr_consent              BOOLEAN NOT NULL DEFAULT FALSE,
  gdpr_consent_at           TIMESTAMPTZ,
  marketing_consent         BOOLEAN NOT NULL DEFAULT FALSE,
  ccpa_opt_out              BOOLEAN NOT NULL DEFAULT FALSE,
  data_deletion_requested   BOOLEAN NOT NULL DEFAULT FALSE,

  -- Referral
  referral_code             TEXT UNIQUE,
  referred_by_user_id       UUID REFERENCES users(id) ON DELETE SET NULL,

  created_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at                TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email         ON users(email) WHERE email IS NOT NULL;
CREATE INDEX idx_users_phone         ON users(phone) WHERE phone IS NOT NULL;
CREATE INDEX idx_users_loyalty_tier  ON users(loyalty_tier);
CREATE INDEX idx_users_referral_code ON users(referral_code) WHERE referral_code IS NOT NULL;
CREATE INDEX idx_users_oauth         ON users USING GIN(oauth_accounts jsonb_path_ops);
CREATE INDEX idx_users_push_tokens   ON users USING GIN(device_push_tokens jsonb_path_ops);
CREATE INDEX idx_users_name_trgm     ON users USING GIN((first_name || ' ' || last_name) gin_trgm_ops);


-- ============================================================
-- TABLE 2: hotels
-- Absorbs: hotel_partners, hotel_policies, hotel_faqs
-- Why embedded: policies are a small fixed-shape config block
--   never queried independently. FAQs are seeded to Qdrant at
--   onboarding — SQL never searches FAQ text.
-- ============================================================

CREATE TABLE hotels (
  id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  partner_user_id        UUID REFERENCES users(id) ON DELETE SET NULL,
  name                   TEXT NOT NULL,
  legal_name             TEXT,
  description            TEXT,
  description_i18n       JSONB NOT NULL DEFAULT '{}',

  -- Location
  city                   TEXT NOT NULL,
  country                TEXT NOT NULL,
  state_province         TEXT,
  address                TEXT,
  postal_code            TEXT,
  lat                    NUMERIC(10,7),
  lng                    NUMERIC(10,7),

  -- Contact
  email                  TEXT NOT NULL,
  phone                  TEXT,

  -- Classification
  star_rating            SMALLINT CHECK (star_rating BETWEEN 1 AND 5),
  currency               TEXT NOT NULL DEFAULT 'USD',
  check_in_time          TIME NOT NULL DEFAULT '15:00',
  check_out_time         TIME NOT NULL DEFAULT '11:00',
  amenities              TEXT[] NOT NULL DEFAULT '{}',
  photos                 TEXT[] NOT NULL DEFAULT '{}',
  logo_url               TEXT,
  thumbnail_url          TEXT,

  -- Cancellation + modification policy (small fixed shape, no independent queries)
  -- {free_cancel_days, partial_refund_days, partial_refund_pct, no_refund_days,
  --  mod_free_days, mod_fee_days, mod_fee_amount, mod_fee_currency, mod_fee_type}
  policy                 JSONB NOT NULL DEFAULT '{}',

  -- FAQs for Qdrant seeding [{question, answer, category, language, qdrant_vector_id}]
  faqs                   JSONB NOT NULL DEFAULT '[]',

  is_active              BOOLEAN NOT NULL DEFAULT TRUE,
  elasticsearch_id       TEXT,
  qdrant_indexed         BOOLEAN NOT NULL DEFAULT FALSE,
  partner_contract_type  TEXT NOT NULL DEFAULT 'standard',
  onboarded_at           TIMESTAMPTZ,

  created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hotels_city       ON hotels(city);
CREATE INDEX idx_hotels_country    ON hotels(country);
CREATE INDEX idx_hotels_stars      ON hotels(star_rating);
CREATE INDEX idx_hotels_search     ON hotels(city, star_rating) WHERE is_active = TRUE;
CREATE INDEX idx_hotels_geo        ON hotels(lat, lng) WHERE lat IS NOT NULL;
CREATE INDEX idx_hotels_amenities  ON hotels USING GIN(amenities);
CREATE INDEX idx_hotels_name_trgm  ON hotels USING GIN(name gin_trgm_ops);


-- ============================================================
-- TABLE 3: inventory
-- Absorbs: room_types, room_rates, room_availability
-- Why one row per room_type+date: a single SELECT returns
--   type definition + all rate plans + availability for a stay
--   without any JOIN. rate_plans is JSONB keyed by plan name —
--   plan names are stable and the full map is always fetched.
-- ============================================================

CREATE TABLE inventory (
  id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  hotel_id            UUID NOT NULL REFERENCES hotels(id) ON DELETE CASCADE,

  -- Room type (stable definition columns — denormalized for single-scan reads)
  room_type_code      TEXT NOT NULL,
  room_type_name      TEXT NOT NULL,
  room_name_i18n      JSONB NOT NULL DEFAULT '{}',
  room_description    TEXT,
  size_sqm            SMALLINT,
  max_adults          SMALLINT NOT NULL DEFAULT 2,
  max_children        SMALLINT NOT NULL DEFAULT 0,
  bed_type            TEXT,
  bed_count           SMALLINT NOT NULL DEFAULT 1,
  has_balcony         BOOLEAN NOT NULL DEFAULT FALSE,
  has_sea_view        BOOLEAN NOT NULL DEFAULT FALSE,
  has_city_view       BOOLEAN NOT NULL DEFAULT FALSE,
  room_amenities      TEXT[] NOT NULL DEFAULT '{}',
  room_photos         TEXT[] NOT NULL DEFAULT '{}',
  room_thumbnail_url  TEXT,

  -- Daily availability
  date                DATE NOT NULL,
  total_rooms         SMALLINT NOT NULL,
  available_count     SMALLINT NOT NULL,
  is_blackout         BOOLEAN NOT NULL DEFAULT FALSE,

  -- Rate plans for this room on this date
  -- {room_only:{price,currency,min_stay,is_refundable,advance_purchase_days},
  --  breakfast:{...}, half_board:{...}, full_board:{...}}
  rate_plans          JSONB NOT NULL DEFAULT '{}',

  -- Soft lock (persisted for audit; Redis is the authoritative lock)
  soft_lock_session   TEXT,
  soft_lock_expires   TIMESTAMPTZ,

  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  UNIQUE (hotel_id, room_type_code, date)
);

CREATE INDEX idx_inv_hotel_date       ON inventory(hotel_id, date);
CREATE INDEX idx_inv_type_date        ON inventory(hotel_id, room_type_code, date);
CREATE INDEX idx_inv_available        ON inventory(hotel_id, date, available_count)
  WHERE available_count > 0 AND is_blackout = FALSE;
CREATE INDEX idx_inv_soft_lock_expiry ON inventory(soft_lock_expires)
  WHERE soft_lock_session IS NOT NULL;
CREATE INDEX idx_inv_rate_plans       ON inventory USING GIN(rate_plans jsonb_path_ops);
CREATE INDEX idx_inv_room_amenities   ON inventory USING GIN(room_amenities);


-- ============================================================
-- TABLE 4: catalog
-- Absorbs: addons, badge_definitions, system_config,
--          airport_codes, geocoding_cache, fx_rates_cache
-- Why unified: all are lookup/reference data with the same
--   access pattern: fetch by (catalog_type, catalog_key).
--   Eliminates 6 small independent tables with no inter-FK needs.
-- ============================================================

CREATE TABLE catalog (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  catalog_type  TEXT NOT NULL,
    -- 'addon' | 'badge' | 'config' | 'airport' | 'geo_cache' | 'fx_rate'
  catalog_key   TEXT NOT NULL,
  hotel_id      UUID REFERENCES hotels(id) ON DELETE CASCADE,
  name          TEXT,
  description   TEXT,
  is_active     BOOLEAN NOT NULL DEFAULT TRUE,
  -- addon:     {category,price,currency,unit,trip_purpose_tags,recommended_rank,min_advance_hours,image_url,name_i18n}
  -- badge:     {trigger_event,condition_type,condition_value,points_bonus,icon_url}
  -- config:    {value}
  -- airport:   {airport_name,city,country,lat,lng,timezone}
  -- geo_cache: {lat,lng,display_name,confidence,expires_at}
  -- fx_rate:   {rate,source,expires_at}
  data          JSONB NOT NULL DEFAULT '{}',
  created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (catalog_type, catalog_key)
);

CREATE INDEX idx_catalog_type        ON catalog(catalog_type);
CREATE INDEX idx_catalog_hotel_addon ON catalog(hotel_id, catalog_type) WHERE catalog_type = 'addon';
CREATE INDEX idx_catalog_config_key  ON catalog(catalog_key) WHERE catalog_type = 'config';
CREATE INDEX idx_catalog_data        ON catalog USING GIN(data jsonb_path_ops);
CREATE INDEX idx_catalog_expiry      ON catalog((data->>'expires_at'))
  WHERE catalog_type IN ('geo_cache','fx_rate');

INSERT INTO catalog (catalog_type, catalog_key, name, data) VALUES
  ('config','points_per_dollar',         'Base earn rate',              '{"value":"1"}'),
  ('config','tier_silver_threshold',     'Silver threshold',            '{"value":"2000"}'),
  ('config','tier_gold_threshold',       'Gold threshold',              '{"value":"5000"}'),
  ('config','tier_platinum_threshold',   'Platinum threshold',          '{"value":"15000"}'),
  ('config','tier_black_threshold',      'Black tier flag',             '{"value":"50000"}'),
  ('config','tier_multiplier_silver',    'Silver multiplier',           '{"value":"1.25"}'),
  ('config','tier_multiplier_gold',      'Gold multiplier',             '{"value":"1.5"}'),
  ('config','tier_multiplier_platinum',  'Platinum multiplier',         '{"value":"2.0"}'),
  ('config','tier_multiplier_black',     'Black multiplier',            '{"value":"3.0"}'),
  ('config','bonus_pts_review',          'Review bonus points',         '{"value":"50"}'),
  ('config','bonus_pts_referral',        'Referral bonus',              '{"value":"200"}'),
  ('config','bonus_pts_first_booking',   'First booking bonus',         '{"value":"500"}'),
  ('config','soft_lock_minutes',         'Room soft-lock duration',     '{"value":"15"}'),
  ('config','group_booking_min_rooms',   'Group booking trigger',       '{"value":"10"}'),
  ('config','fraud_block_threshold',     'Fraud block score',           '{"value":"71"}'),
  ('config','fraud_3ds_threshold',       '3DS required score',          '{"value":"31"}'),
  ('config','velocity_max_attempts',     'Max payments/hour',           '{"value":"5"}'),
  ('config','review_send_hours',         'Hours after checkout',        '{"value":"24"}'),
  ('config','nlu_fallback_threshold',    'NLU fallback confidence',     '{"value":"0.60"}'),
  ('config','lang_detect_min_conf',      'LangDetect min confidence',   '{"value":"0.85"}');

INSERT INTO catalog (catalog_type, catalog_key, name, data) VALUES
  ('badge','first_booking', 'First Booking',    '{"trigger_event":"booking_completed","condition_type":"count","condition_value":1,"points_bonus":0}'),
  ('badge','globetrotter',  'Globetrotter',     '{"trigger_event":"booking_completed","condition_type":"unique_countries","condition_value":5,"points_bonus":100}'),
  ('badge','streak_3',      'Loyal 3 Months',   '{"trigger_event":"streak_checkin","condition_type":"consecutive_months","condition_value":3,"points_bonus":50}'),
  ('badge','first_review',  'First Review',     '{"trigger_event":"review_posted","condition_type":"count","condition_value":1,"points_bonus":50}');


-- ============================================================
-- TABLE 5: sessions
-- Absorbs: chat_sessions, conversation_messages,
--          nlu_confidence_log, handoff_requests
-- Why JSONB for messages: always fetched as a full conversation
--   unit. SQL never filters individual messages. Rasa/Chatwoot
--   own message-level queries. NLU stats kept scalar for GROUP BY.
-- ============================================================

CREATE TABLE sessions (
  id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_key              TEXT UNIQUE NOT NULL,
  user_id                  UUID REFERENCES users(id) ON DELETE SET NULL,
  hotel_id                 UUID REFERENCES hotels(id) ON DELETE SET NULL,

  channel                  TEXT NOT NULL DEFAULT 'web',
  is_voice_session         BOOLEAN NOT NULL DEFAULT FALSE,
  device_type              TEXT,
  ip_address               INET,
  country_from_ip          TEXT,
  user_agent               TEXT,

  -- Language (scalar for analytics GROUP BY)
  detected_language        TEXT NOT NULL DEFAULT 'en',
  language_confidence      NUMERIC(4,3),
  is_rtl                   BOOLEAN NOT NULL DEFAULT FALSE,
  language_tier            SMALLINT NOT NULL DEFAULT 1,

  -- NLU stats (scalar for aggregation)
  total_turns              SMALLINT NOT NULL DEFAULT 0,
  fallback_count           SMALLINT NOT NULL DEFAULT 0,
  last_intent              TEXT,
  last_confidence          NUMERIC(4,3),
  booking_completed        BOOLEAN NOT NULL DEFAULT FALSE,
  drop_off_intent          TEXT,

  -- Full conversation [{role,text,intent,confidence,entities,language,is_voice,audio_url,turn_number,created_at}]
  messages                 JSONB NOT NULL DEFAULT '[]',

  -- NLU confidence log [{intent,confidence,language,entities,was_fallback,timestamp}]
  -- Written by Rasa, read only by ClickHouse consumer — never SQL-queried
  nlu_log                  JSONB NOT NULL DEFAULT '[]',

  -- Handoff state
  handoff_active           BOOLEAN NOT NULL DEFAULT FALSE,
  handoff_reason           TEXT,
  handoff_slots_filled     JSONB NOT NULL DEFAULT '{}',
  chatwoot_conversation_id TEXT,
  assigned_agent_name      TEXT,
  estimated_wait_minutes   SMALLINT,
  handoff_status           TEXT,
  handoff_resolved_at      TIMESTAMPTZ,

  started_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ended_at                 TIMESTAMPTZ,
  created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_user        ON sessions(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_sessions_hotel       ON sessions(hotel_id) WHERE hotel_id IS NOT NULL;
CREATE INDEX idx_sessions_key         ON sessions(session_key);
CREATE INDEX idx_sessions_language    ON sessions(detected_language);
CREATE INDEX idx_sessions_channel     ON sessions(channel);
CREATE INDEX idx_sessions_started     ON sessions(started_at);
CREATE INDEX idx_sessions_handoff     ON sessions(chatwoot_conversation_id)
  WHERE handoff_active = TRUE;
CREATE INDEX idx_sessions_converted   ON sessions(started_at, channel)
  WHERE booking_completed = TRUE;


-- ============================================================
-- TABLE 6: bookings
-- Absorbs: bookings, guests, booking_addons, booking_modifications,
--          cancellations
-- Why JSONB for guests: always fetched as a unit (max 8 guests).
--   Primary guest email/name kept scalar for indexing + notifications.
--   Modification log is an immutable audit trail — append-only JSONB.
--   Cancellation data is inline (1:1 with booking).
-- ============================================================

CREATE TABLE bookings (
  id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_reference       TEXT UNIQUE NOT NULL,
  user_id                 UUID REFERENCES users(id) ON DELETE SET NULL,
  hotel_id                UUID NOT NULL REFERENCES hotels(id),
  session_id              UUID REFERENCES sessions(id) ON DELETE SET NULL,
  organization_id         UUID,

  room_type_code          TEXT NOT NULL,
  rate_plan               TEXT NOT NULL,
  meal_plan               TEXT,

  -- Stay
  check_in                DATE NOT NULL,
  check_out               DATE NOT NULL,
  nights                  SMALLINT GENERATED ALWAYS AS (CAST(check_out - check_in AS SMALLINT)) STORED,
  num_adults              SMALLINT NOT NULL DEFAULT 1,
  num_children            SMALLINT NOT NULL DEFAULT 0,
  num_rooms               SMALLINT NOT NULL DEFAULT 1,
  trip_purpose            TEXT,
  special_requests        TEXT,

  -- Primary guest (scalar — indexed for notifications + agent search)
  primary_guest_name      TEXT NOT NULL,
  primary_guest_email     TEXT NOT NULL,
  primary_guest_phone     TEXT,

  -- All guests [{is_primary,first_name,last_name,email,phone,dob,nationality,
  --              passport_number,passport_expiry,passport_country,passport_scan_url,
  --              ocr_confidence,gender,dietary_needs,accessibility_needs,is_child,child_age}]
  guests                  JSONB NOT NULL DEFAULT '[]',

  -- Pricing (scalar for analytics)
  base_amount             NUMERIC(10,2) NOT NULL,
  addons_amount           NUMERIC(10,2) NOT NULL DEFAULT 0,
  tax_amount              NUMERIC(10,2) NOT NULL DEFAULT 0,
  total_amount            NUMERIC(10,2) NOT NULL,
  currency                TEXT NOT NULL,
  total_amount_usd        NUMERIC(10,2),
  fx_rate_used            NUMERIC(12,6),

  -- Add-ons cart snapshot [{addon_id,name,category,quantity,price_at_time,currency,status}]
  addons_cart             JSONB NOT NULL DEFAULT '[]',

  -- Status
  status                  TEXT NOT NULL DEFAULT 'pending',
  payment_status          TEXT NOT NULL DEFAULT 'unpaid',

  -- Soft lock
  soft_lock_id            TEXT,
  soft_lock_expires_at    TIMESTAMPTZ,

  -- Modification log [{change_type,old_values,new_values,policy_tier,days_until_checkin,
  --                     mod_fee,price_diff,status,confirmed_at,created_at}]
  modification_log        JSONB NOT NULL DEFAULT '[]',

  -- Cancellation (1:1 inline)
  cancellation_reason     TEXT,
  cancellation_policy     TEXT,
  refund_percent          NUMERIC(5,2),
  refund_amount           NUMERIC(10,2),
  refund_currency         TEXT,
  cancelled_at            TIMESTAMPTZ,
  cancelled_by            TEXT,

  -- Type flags
  is_group_booking        BOOLEAN NOT NULL DEFAULT FALSE,
  is_corporate            BOOLEAN NOT NULL DEFAULT FALSE,
  force_majeure_flag      BOOLEAN NOT NULL DEFAULT FALSE,
  force_majeure_event_id  UUID,

  -- Artefacts
  pdf_voucher_url         TEXT,
  qr_code_url             TEXT,
  ics_calendar_url        TEXT,
  digital_key_token       TEXT,
  checkin_completed       BOOLEAN NOT NULL DEFAULT FALSE,
  checkin_completed_at    TIMESTAMPTZ,

  review_requested        BOOLEAN NOT NULL DEFAULT FALSE,
  review_requested_at     TIMESTAMPTZ,

  confirmed_at            TIMESTAMPTZ,
  completed_at            TIMESTAMPTZ,
  created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_bookings_user          ON bookings(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_bookings_hotel         ON bookings(hotel_id);
CREATE INDEX idx_bookings_reference     ON bookings(booking_reference);
CREATE INDEX idx_bookings_status        ON bookings(status);
CREATE INDEX idx_bookings_check_in      ON bookings(check_in);
CREATE INDEX idx_bookings_user_status   ON bookings(user_id, status) WHERE user_id IS NOT NULL;
CREATE INDEX idx_bookings_review_due    ON bookings(check_out, primary_guest_email)
  WHERE review_requested = FALSE AND status = 'completed';
CREATE INDEX idx_bookings_org           ON bookings(organization_id) WHERE organization_id IS NOT NULL;
CREATE INDEX idx_bookings_force_maj     ON bookings(check_in, hotel_id)
  WHERE force_majeure_flag = FALSE AND status = 'confirmed';
CREATE INDEX idx_bookings_guests        ON bookings USING GIN(guests jsonb_path_ops);
CREATE INDEX idx_bookings_addons_cart   ON bookings USING GIN(addons_cart jsonb_path_ops);


-- ============================================================
-- TABLE 7: payments
-- Absorbs: payments, payment_attempts
-- Why JSONB for attempts: 1-3 attempts per booking, always
--   fetched together for fraud review, never individually filtered.
-- ============================================================

CREATE TABLE payments (
  id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  booking_id           UUID NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
  user_id              UUID REFERENCES users(id) ON DELETE SET NULL,

  amount               NUMERIC(10,2) NOT NULL,
  currency             TEXT NOT NULL,
  amount_usd           NUMERIC(10,2),
  gateway              TEXT NOT NULL,
  transaction_id       TEXT,
  payment_method_type  TEXT,
  card_last4           TEXT,
  card_brand           TEXT,
  card_country         TEXT,

  status               TEXT NOT NULL DEFAULT 'pending',
  idempotency_key      TEXT UNIQUE,

  risk_score           SMALLINT,
  fraud_action         TEXT,
  stripe_radar_score   SMALLINT,
  requires_3ds         BOOLEAN NOT NULL DEFAULT FALSE,
  three_ds_completed   BOOLEAN NOT NULL DEFAULT FALSE,
  blocked_reason       TEXT,

  refund_id            TEXT,
  refund_amount        NUMERIC(10,2),
  refund_status        TEXT,
  refund_initiated_at  TIMESTAMPTZ,
  refund_reason        TEXT,
  receipt_url          TEXT,
  webhook_verified     BOOLEAN NOT NULL DEFAULT FALSE,

  -- Attempt log [{gateway,amount,currency,error_code,error_message,risk_score,
  --               blocked_by,ip_address,device_fingerprint,card_country,attempted_at}]
  attempt_log          JSONB NOT NULL DEFAULT '[]',
  metadata             JSONB NOT NULL DEFAULT '{}',

  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_payments_booking      ON payments(booking_id);
CREATE INDEX idx_payments_user         ON payments(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX idx_payments_status       ON payments(status);
CREATE INDEX idx_payments_gateway      ON payments(gateway);
CREATE INDEX idx_payments_transaction  ON payments(transaction_id) WHERE transaction_id IS NOT NULL;
CREATE INDEX idx_payments_analytics    ON payments(gateway, currency, created_at)
  WHERE status = 'succeeded';
CREATE INDEX idx_payments_fraud        ON payments(risk_score, created_at)
  WHERE risk_score >= 31;


-- ============================================================
-- TABLE 8: organizations
-- Absorbs: corporate_accounts, corporate_employees, corporate_rates,
--          rate_blackout_dates, approval_requests, group_bookings,
--          group_deposit_payments, room_blocks, referrals
-- Why unified: all are "entity groupings a user belongs to."
--   org_type discriminator routes to type-specific JSONB.
--   Members, rates, deposits are bounded arrays never individually
--   filtered in SQL — always fetched as a whole unit.
-- ============================================================

CREATE TABLE organizations (
  id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  org_type       TEXT NOT NULL,         -- 'corporate' | 'group' | 'referral'
  owner_user_id  UUID REFERENCES users(id) ON DELETE SET NULL,
  name           TEXT,
  email          TEXT,
  is_active      BOOLEAN NOT NULL DEFAULT TRUE,

  -- Corporate: {billing_address,tax_id,account_manager_email,
  --             auto_approve_threshold,currency,company_domain}
  config         JSONB NOT NULL DEFAULT '{}',

  -- Corporate employees [{user_id,employee_id,cost_center,department,role,spending_limit,is_active}]
  employees      JSONB NOT NULL DEFAULT '[]',

  -- Corporate rates [{hotel_id,room_type_code,rate_per_night,currency,includes,
  --                   valid_from,valid_until,blackout_dates,is_active}]
  rates          JSONB NOT NULL DEFAULT '[]',

  -- Approval requests [{booking_id,employee_user_id,approver_email,booking_total,
  --                     currency,cost_center,status,approval_token,decided_at,expires_at,notes,created_at}]
  approval_requests JSONB NOT NULL DEFAULT '[]',

  -- Group: {event_type,num_rooms,total_amount,currency,contract_type,
  --         room_block_status,group_booking_url,room_block_expires_at,
  --         rooms_held,hotel_id,room_type_code,check_in,check_out}
  group_meta     JSONB NOT NULL DEFAULT '{}',

  -- Deposit schedule [{installment_number,due_date,amount,percent_of_total,
  --                    currency,status,payment_id,reminder_sent_at,paid_at}]
  deposit_schedule JSONB NOT NULL DEFAULT '[]',

  -- Referral: {referral_code,referred_user_id,status,converted_booking_id,
  --            referrer_points_awarded,referred_discount_pct,converted_at}
  referral_meta  JSONB NOT NULL DEFAULT '{}',

  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE bookings ADD CONSTRAINT fk_bookings_org
  FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL;

CREATE INDEX idx_orgs_type       ON organizations(org_type);
CREATE INDEX idx_orgs_owner      ON organizations(owner_user_id);
CREATE INDEX idx_orgs_active     ON organizations(org_type, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_orgs_employees  ON organizations USING GIN(employees jsonb_path_ops);
CREATE INDEX idx_orgs_approvals  ON organizations USING GIN(approval_requests jsonb_path_ops);


-- ============================================================
-- TABLE 9: engagements
-- Absorbs: reviews, loyalty_transactions, notification_logs,
--          consent_records, privacy_requests
-- Why unified: all are "time-series events attached to a user."
--   Same access pattern: list by (user_id, engagement_type, date).
--   Scalar columns cover all GROUP BY / ORDER BY / WHERE cases.
--   Type-specific detail lives in JSONB data column.
--   Append-only — past engagements are immutable facts.
-- ============================================================

CREATE TABLE engagements (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  engagement_type  TEXT NOT NULL,
    -- 'review' | 'loyalty_txn' | 'notification' | 'consent' | 'privacy_request'
  user_id          UUID REFERENCES users(id) ON DELETE CASCADE,
  booking_id       UUID REFERENCES bookings(id) ON DELETE SET NULL,
  hotel_id         UUID REFERENCES hotels(id) ON DELETE SET NULL,

  -- Scalar fields — used in WHERE / GROUP BY across all types
  status           TEXT,
  channel          TEXT,
  rating           SMALLINT CHECK (rating BETWEEN 1 AND 5),
  sentiment        TEXT,
  sentiment_score  NUMERIC(4,3),
  is_negative_alert BOOLEAN NOT NULL DEFAULT FALSE,
  points_delta     INTEGER DEFAULT 0,
  points_balance   INTEGER,

  -- Type-specific detail:
  -- review:       {text,language,aspects:{room,food,staff,location},hotel_response,hotel_responded_at,is_published,follow_up_sent}
  -- loyalty_txn:  {action_type,multiplier,base_points,bonus_points,description,tier_at_time}
  -- notification: {notification_type,template_name,recipient,gateway_message_id,error_code,error_message,opened_at,clicked_at,delivery_confirmed_at,retry_count}
  -- consent:      {consent_type,consent_version,ip_address,user_agent}
  -- privacy_req:  {request_type,data_categories,export_url,must_complete_by,requester_ip}
  data             JSONB NOT NULL DEFAULT '{}',

  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
  -- No updated_at — append-only by design
);

CREATE INDEX idx_eng_user          ON engagements(user_id, engagement_type);
CREATE INDEX idx_eng_booking       ON engagements(booking_id) WHERE booking_id IS NOT NULL;
CREATE INDEX idx_eng_hotel         ON engagements(hotel_id, engagement_type) WHERE hotel_id IS NOT NULL;
CREATE INDEX idx_eng_type_date     ON engagements(engagement_type, created_at);
CREATE INDEX idx_eng_status        ON engagements(status, engagement_type);
CREATE INDEX idx_eng_neg_alert     ON engagements(hotel_id, created_at)
  WHERE engagement_type = 'review' AND is_negative_alert = TRUE AND status = 'pending';
CREATE INDEX idx_eng_notif_due     ON engagements(created_at, channel)
  WHERE engagement_type = 'notification' AND status = 'pending';
CREATE INDEX idx_eng_privacy_due   ON engagements((data->>'must_complete_by'))
  WHERE engagement_type = 'privacy_request' AND status IN ('pending','processing');
CREATE INDEX idx_eng_data          ON engagements USING GIN(data jsonb_path_ops);


-- ============================================================
-- TABLE 10: system_events
-- Absorbs: analytics_events, compliance_audit_log,
--          force_majeure_events
-- Why unified: all are append-only event log records read by
--   Kafka consumers and the ClickHouse pipeline.
--   compliance_audit entries are legally immutable — GRANT INSERT
--   only on this table; never UPDATE/DELETE.
-- ============================================================

CREATE TABLE system_events (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  event_type      TEXT NOT NULL,
    -- analytics:  'session_started'|'intent_classified'|'booking_completed'|
    --             'booking_modified'|'booking_cancelled'|'payment_attempted'|
    --             'review_submitted'|'loyalty_points_awarded'
    -- audit:      'data_erasure'|'data_export'|'login'|'payment_processed'|'data_access'
    -- force_maj:  'travel_ban_detected'|'natural_disaster'|'border_closed'
  event_category  TEXT NOT NULL,   -- 'analytics' | 'audit' | 'force_majeure'
  user_id         UUID,            -- intentionally no FK — audit logs survive user deletion
  session_id      UUID,
  booking_id      UUID,
  hotel_id        UUID,
  performed_by    UUID,
  ip_address      INET,

  -- Scalar metrics extracted for ClickHouse aggregations (no JSONB parsing needed)
  revenue_usd     NUMERIC(10,2),
  language        TEXT,
  channel         TEXT,
  confidence      NUMERIC(4,3),

  -- Full event payload
  -- analytics: {intent,entities,hotel_id,room_type,gateway,amount,currency,...}
  -- audit:     {data_categories,action_detail}
  -- force_maj: {country_code,event_description,confidence,news_source_urls,
  --             affected_bookings_count,affected_date_from,affected_date_until,
  --             status,reviewed_by,reviewed_at,auto_refund_triggered}
  properties      JSONB NOT NULL DEFAULT '{}',

  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
  -- APPEND-ONLY: GRANT INSERT ONLY. Never UPDATE or DELETE.
);

CREATE INDEX idx_sysev_type         ON system_events(event_type, created_at);
CREATE INDEX idx_sysev_category     ON system_events(event_category, created_at);
CREATE INDEX idx_sysev_user         ON system_events(user_id, created_at) WHERE user_id IS NOT NULL;
CREATE INDEX idx_sysev_booking      ON system_events(booking_id) WHERE booking_id IS NOT NULL;
CREATE INDEX idx_sysev_hotel        ON system_events(hotel_id, event_type) WHERE hotel_id IS NOT NULL;
CREATE INDEX idx_sysev_force_maj    ON system_events((properties->>'status'), created_at)
  WHERE event_category = 'force_majeure';
CREATE INDEX idx_sysev_kafka_range  ON system_events(event_category, created_at DESC);
CREATE INDEX idx_sysev_properties   ON system_events USING GIN(properties jsonb_path_ops);

ALTER TABLE bookings ADD CONSTRAINT fk_bookings_force_maj
  FOREIGN KEY (force_majeure_event_id) REFERENCES system_events(id) ON DELETE SET NULL;


-- ============================================================
-- AUTO updated_at TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

CREATE TRIGGER trg_users_upd        BEFORE UPDATE ON users         FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_hotels_upd       BEFORE UPDATE ON hotels        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_inventory_upd    BEFORE UPDATE ON inventory     FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_catalog_upd      BEFORE UPDATE ON catalog       FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_sessions_upd     BEFORE UPDATE ON sessions      FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_bookings_upd     BEFORE UPDATE ON bookings      FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_payments_upd     BEFORE UPDATE ON payments      FOR EACH ROW EXECUTE FUNCTION set_updated_at();
CREATE TRIGGER trg_orgs_upd         BEFORE UPDATE ON organizations FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ============================================================
-- ROW LEVEL SECURITY
-- ============================================================

ALTER TABLE users        ENABLE ROW LEVEL SECURITY;
ALTER TABLE bookings     ENABLE ROW LEVEL SECURITY;
ALTER TABLE payments     ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions     ENABLE ROW LEVEL SECURITY;
ALTER TABLE engagements  ENABLE ROW LEVEL SECURITY;

CREATE POLICY users_self        ON users        FOR ALL    USING (auth.uid() = id);
CREATE POLICY bookings_owner    ON bookings     FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY payments_owner    ON payments     FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY sessions_owner    ON sessions     FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY engagements_owner ON engagements  FOR ALL    USING (auth.uid() = user_id);


-- ============================================================
-- VIEWS (query sugar — no extra tables)
-- ============================================================

CREATE VIEW v_available_rooms AS
SELECT i.hotel_id, h.name AS hotel_name,
       i.room_type_code, i.room_type_name,
       i.date, i.available_count,
       i.max_adults, i.max_children, i.rate_plans
FROM inventory i
JOIN hotels h ON h.id = i.hotel_id
WHERE i.available_count > 0
  AND i.is_blackout = FALSE
  AND h.is_active = TRUE;

CREATE VIEW v_booking_summary AS
SELECT b.booking_reference, b.user_id, b.hotel_id,
       h.name AS hotel_name, h.city, h.country,
       b.room_type_code, b.rate_plan,
       b.check_in, b.check_out, b.nights,
       b.num_adults, b.total_amount, b.currency,
       b.status, b.payment_status,
       b.primary_guest_name, b.primary_guest_email,
       b.trip_purpose, b.created_at
FROM bookings b
JOIN hotels h ON h.id = b.hotel_id;

CREATE VIEW v_user_loyalty AS
SELECT id, first_name, last_name,
       loyalty_tier, loyalty_points_available,
       loyalty_points_ytd, loyalty_total_bookings,
       loyalty_total_nights, loyalty_badges,
       loyalty_streak_months, loyalty_leaderboard_rank
FROM users;

-- ============================================================
-- PARTITIONING (enable when row counts cross thresholds)
-- ============================================================
-- system_events: partition by month (Celery creates monthly partitions)
-- ALTER TABLE system_events PARTITION BY RANGE (created_at);
-- CREATE TABLE system_events_2026_03 PARTITION OF system_events
--   FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');

-- inventory: partition by HASH(hotel_id) when hotels > 500
-- ALTER TABLE inventory PARTITION BY HASH (hotel_id);
-- CREATE TABLE inventory_p0 PARTITION OF inventory
--   FOR VALUES WITH (MODULUS 4, REMAINDER 0);

-- ============================================================
-- END — 10 tables, 50+ indexes, production-ready PostgreSQL
-- ============================================================
