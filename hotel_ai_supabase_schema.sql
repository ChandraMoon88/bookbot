-- ============================================================
-- BookBot — Supabase PostgreSQL Schema
-- ============================================================
-- Run this in Supabase SQL Editor to create all tables/views.
-- Matches db_client.py exactly (sync urllib REST client).
-- ============================================================

-- ── Enable UUID extension ────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ── USERS ────────────────────────────────────────────────────────────────────
-- phone field stores "messenger:{psid}" as a unique identifier for FB users.
CREATE TABLE IF NOT EXISTS users (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    first_name   TEXT,
    last_name    TEXT,
    email        TEXT,
    phone        TEXT UNIQUE,           -- "messenger:{sender_id}" for Messenger users
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);


-- ── HOTELS ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hotels (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT NOT NULL,
    description     TEXT,
    city            TEXT NOT NULL,
    country         TEXT NOT NULL DEFAULT 'India',
    star_rating     INTEGER CHECK (star_rating BETWEEN 1 AND 5),
    amenities       JSONB DEFAULT '[]',      -- array of strings, e.g. ["WiFi","Pool"]
    currency        TEXT NOT NULL DEFAULT 'INR',
    check_in_time   TEXT DEFAULT '14:00',
    check_out_time  TEXT DEFAULT '12:00',
    policy          TEXT,                    -- plain-text cancellation/house rules
    photos          JSONB DEFAULT '[]',      -- array of image URLs
    thumbnail_url   TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_hotels_city       ON hotels(city);
CREATE INDEX IF NOT EXISTS idx_hotels_is_active  ON hotels(is_active);


-- ── INVENTORY ─────────────────────────────────────────────────────────────────
-- One row per hotel × room_type × date.
-- rate_plans stores pricing per meal plan, e.g.:
--   {"room_only": {"price_per_night": 2500}, "breakfast": {"price_per_night": 3200}}
CREATE TABLE IF NOT EXISTS inventory (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hotel_id         UUID NOT NULL REFERENCES hotels(id) ON DELETE CASCADE,
    date             DATE NOT NULL,
    room_type_code   TEXT NOT NULL,           -- e.g. "STD", "DLX", "STE"
    room_type_name   TEXT NOT NULL,           -- e.g. "Standard Room", "Deluxe King"
    available_count  INTEGER NOT NULL DEFAULT 0,
    is_blackout      BOOLEAN NOT NULL DEFAULT FALSE,
    max_adults       INTEGER NOT NULL DEFAULT 2,
    max_children     INTEGER NOT NULL DEFAULT 0,
    rate_plans       JSONB DEFAULT '{}',      -- per-plan pricing (see above)
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (hotel_id, date, room_type_code)
);

CREATE INDEX IF NOT EXISTS idx_inventory_hotel_date
    ON inventory(hotel_id, date);
CREATE INDEX IF NOT EXISTS idx_inventory_date_avail
    ON inventory(date, available_count) WHERE NOT is_blackout;


-- ── BOOKINGS ─────────────────────────────────────────────────────────────────
-- nights is computed automatically — do NOT pass it in INSERT (GENERATED ALWAYS).
CREATE TABLE IF NOT EXISTS bookings (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    booking_reference    TEXT UNIQUE NOT NULL,     -- "BB" + 8 [A-Z0-9] chars
    user_id              UUID REFERENCES users(id) ON DELETE SET NULL,
    hotel_id             UUID REFERENCES hotels(id) ON DELETE SET NULL,
    room_type_code       TEXT NOT NULL DEFAULT 'STD',
    rate_plan            TEXT NOT NULL DEFAULT 'room_only',
    meal_plan            TEXT NOT NULL DEFAULT 'room_only',
    check_in             DATE NOT NULL,
    check_out            DATE NOT NULL,
    nights               INTEGER GENERATED ALWAYS AS
                             ((check_out - check_in)) STORED,
    num_adults           INTEGER NOT NULL DEFAULT 1,
    num_children         INTEGER NOT NULL DEFAULT 0,
    num_rooms            INTEGER NOT NULL DEFAULT 1,
    primary_guest_name   TEXT NOT NULL,
    primary_guest_email  TEXT,
    primary_guest_phone  TEXT DEFAULT '',
    total_amount         NUMERIC(12, 2),
    base_amount          NUMERIC(12, 2),
    tax_amount           NUMERIC(12, 2) DEFAULT 0,
    currency             TEXT NOT NULL DEFAULT 'INR',
    special_requests     TEXT DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'confirmed'
                             CHECK (status IN ('pending','confirmed','cancelled','completed')),
    payment_status       TEXT NOT NULL DEFAULT 'pending'
                             CHECK (payment_status IN ('pending','paid','refunded','failed')),
    cancellation_reason  TEXT,
    cancelled_at         TIMESTAMPTZ,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bookings_reference ON bookings(booking_reference);
CREATE INDEX IF NOT EXISTS idx_bookings_user_id   ON bookings(user_id);
CREATE INDEX IF NOT EXISTS idx_bookings_hotel_id  ON bookings(hotel_id);
CREATE INDEX IF NOT EXISTS idx_bookings_status    ON bookings(status);


-- ── SESSIONS ─────────────────────────────────────────────────────────────────
-- Conversation state persistence for analytics / handoff.
CREATE TABLE IF NOT EXISTS sessions (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_key       TEXT UNIQUE NOT NULL,   -- e.g. "messenger:{sender_id}"
    user_id           UUID REFERENCES users(id) ON DELETE SET NULL,
    channel           TEXT DEFAULT 'messenger',
    detected_language TEXT DEFAULT 'en',
    total_turns       INTEGER DEFAULT 0,
    last_intent       TEXT,
    messages          JSONB DEFAULT '[]',     -- last 20 turns snapshot
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_key ON sessions(session_key);


-- ── ENGAGEMENTS ──────────────────────────────────────────────────────────────
-- Post-stay ratings & feedback.
CREATE TABLE IF NOT EXISTS engagements (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      UUID REFERENCES users(id) ON DELETE SET NULL,
    booking_id   UUID REFERENCES bookings(id) ON DELETE CASCADE,
    rating       INTEGER CHECK (rating BETWEEN 1 AND 5),
    feedback     TEXT,
    sentiment    TEXT,                      -- "positive" / "neutral" / "negative"
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_engagements_booking ON engagements(booking_id);


-- ═══════════════════════════════════════════════════════════════════════════
-- VIEWS
-- ═══════════════════════════════════════════════════════════════════════════

-- ── v_available_rooms ────────────────────────────────────────────────────────
-- Used by search_hotels() in db_client.py as a quick lookup (optional).
CREATE OR REPLACE VIEW v_available_rooms AS
SELECT
    i.hotel_id,
    h.name            AS hotel_name,
    h.city,
    h.country,
    h.star_rating,
    h.currency,
    h.amenities,
    h.thumbnail_url,
    i.date,
    i.room_type_code,
    i.room_type_name,
    i.available_count,
    i.max_adults,
    i.max_children,
    i.rate_plans
FROM inventory i
JOIN hotels     h ON h.id = i.hotel_id
WHERE i.available_count > 0
  AND i.is_blackout = FALSE
  AND h.is_active = TRUE;


-- ── v_booking_summary ────────────────────────────────────────────────────────
-- Used by get_user_bookings() and get_booking_by_ref() in db_client.py.
CREATE OR REPLACE VIEW v_booking_summary AS
SELECT
    b.id,
    b.booking_reference,
    b.user_id,
    b.hotel_id,
    h.name          AS hotel_name,
    h.city,
    h.country,
    b.room_type_code,
    b.rate_plan,
    b.meal_plan,
    b.check_in,
    b.check_out,
    b.nights,
    b.num_adults,
    b.num_children,
    b.num_rooms,
    b.primary_guest_name,
    b.primary_guest_email,
    b.primary_guest_phone,
    b.total_amount,
    b.currency,
    b.special_requests,
    b.status,
    b.payment_status,
    b.cancellation_reason,
    b.cancelled_at,
    b.created_at
FROM bookings   b
LEFT JOIN hotels h ON h.id = b.hotel_id;


-- ═══════════════════════════════════════════════════════════════════════════
-- ROW-LEVEL SECURITY  (enable after testing)
-- ═══════════════════════════════════════════════════════════════════════════
-- The db_client.py uses the service_role key which bypasses RLS.
-- These policies are set up for future anon/user-scoped access.

ALTER TABLE users      ENABLE ROW LEVEL SECURITY;
ALTER TABLE hotels     ENABLE ROW LEVEL SECURITY;
ALTER TABLE inventory  ENABLE ROW LEVEL SECURITY;
ALTER TABLE bookings   ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE engagements ENABLE ROW LEVEL SECURITY;

-- Service role (used by db_client.py) can see everything
CREATE POLICY "service_role_all_users"      ON users       FOR ALL TO service_role USING (true);
CREATE POLICY "service_role_all_hotels"     ON hotels      FOR ALL TO service_role USING (true);
CREATE POLICY "service_role_all_inventory"  ON inventory   FOR ALL TO service_role USING (true);
CREATE POLICY "service_role_all_bookings"   ON bookings    FOR ALL TO service_role USING (true);
CREATE POLICY "service_role_all_sessions"   ON sessions    FOR ALL TO service_role USING (true);
CREATE POLICY "service_role_all_engagements" ON engagements FOR ALL TO service_role USING (true);

-- Public read for hotels (browse without login)
CREATE POLICY "public_read_hotels"    ON hotels    FOR SELECT TO anon USING (is_active = TRUE);
CREATE POLICY "public_read_inventory" ON inventory FOR SELECT TO anon USING (available_count > 0);


-- ═══════════════════════════════════════════════════════════════════════════
-- SAMPLE DATA  (optional — useful for local dev / smoke testing)
-- ═══════════════════════════════════════════════════════════════════════════

-- INSERT INTO hotels (name, description, city, country, star_rating, currency, amenities, thumbnail_url, is_active)
-- VALUES
--   ('The Grand Palace', 'Luxury hotel in the heart of the city', 'Mumbai', 'India', 5, 'INR',
--    '["WiFi","Pool","Spa","Restaurant","Bar","Gym"]',
--    'https://example.com/grand_palace.jpg', true),
--   ('City Inn', 'Affordable and central', 'Delhi', 'India', 3, 'INR',
--    '["WiFi","Restaurant","Parking"]',
--    'https://example.com/city_inn.jpg', true);
