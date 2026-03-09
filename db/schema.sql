-- ============================================================
-- db/schema.sql  — BookHotel AI complete Supabase schema
-- Run in Supabase SQL Editor
-- Enable Row Level Security (RLS) on all tables after creation
-- ============================================================

-- ─── EXTENSIONS ──────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS postgis;    -- geo queries (optional)

-- ─── HOTEL PARTNERS ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hotel_partners (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name              TEXT NOT NULL,
    contact_email     TEXT,
    commission_percent DECIMAL(4,2) DEFAULT 15.0,
    active            BOOLEAN DEFAULT true,
    created_at        TIMESTAMPTZ DEFAULT now()
);

-- ─── HOTELS ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hotels (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                 TEXT NOT NULL,
    city                 TEXT NOT NULL,
    country_code         CHAR(2) NOT NULL,
    address              TEXT,
    lat                  DECIMAL(9,6),
    lng                  DECIMAL(9,6),
    star_rating          SMALLINT CHECK (star_rating BETWEEN 1 AND 5),
    amenities            JSONB DEFAULT '[]',
    cancellation_policy  JSONB,
    partner_id           UUID REFERENCES hotel_partners(id),
    active               BOOLEAN DEFAULT true,
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- ─── ROOM TYPES ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS room_types (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotel_id       UUID REFERENCES hotels(id) ON DELETE CASCADE,
    name           TEXT NOT NULL,
    description    TEXT,
    size_sqm       SMALLINT,
    max_adults     SMALLINT DEFAULT 2,
    max_children   SMALLINT DEFAULT 0,
    amenities      JSONB DEFAULT '[]',
    images         JSONB DEFAULT '[]',
    base_price_usd DECIMAL(10,2),
    active         BOOLEAN DEFAULT true
);

-- ─── ROOM RATES ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS room_rates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    room_type_id    UUID REFERENCES room_types(id) ON DELETE CASCADE,
    rate_plan       TEXT NOT NULL,
    price_per_night DECIMAL(10,2) NOT NULL,
    currency        CHAR(3) NOT NULL DEFAULT 'USD',
    meal_includes   JSONB DEFAULT '[]',
    minimum_stay    SMALLINT DEFAULT 1,
    valid_from      DATE,
    valid_until     DATE,
    non_refundable  BOOLEAN DEFAULT false
);

-- ─── ROOM AVAILABILITY ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS room_availability (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    room_type_id   UUID REFERENCES room_types(id) ON DELETE CASCADE,
    date           DATE NOT NULL,
    available_count SMALLINT NOT NULL DEFAULT 0,
    UNIQUE(room_type_id, date)
);

-- ─── GUESTS ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS guests (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    messenger_psid       TEXT UNIQUE,
    email                TEXT UNIQUE,
    phone                TEXT,
    first_name           TEXT,
    last_name            TEXT,
    nationality          CHAR(2),
    passport_number      TEXT,
    passport_expiry      DATE,
    preferred_language   CHAR(5) DEFAULT 'en',
    preferred_channel    TEXT DEFAULT 'messenger',
    gdpr_consent         BOOLEAN DEFAULT false,
    gdpr_consent_at      TIMESTAMPTZ,
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- ─── BOOKINGS ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bookings (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    booking_reference    TEXT UNIQUE NOT NULL,
    guest_id             UUID REFERENCES guests(id),
    hotel_id             UUID REFERENCES hotels(id),
    room_type_id         UUID REFERENCES room_types(id),
    rate_plan            TEXT NOT NULL,
    check_in             DATE NOT NULL,
    check_out            DATE NOT NULL,
    num_adults           SMALLINT NOT NULL,
    num_children         SMALLINT DEFAULT 0,
    trip_purpose         TEXT,
    dietary_needs        TEXT,
    accessibility_needs  TEXT,
    addons               JSONB DEFAULT '[]',
    subtotal             DECIMAL(10,2),
    taxes                DECIMAL(10,2),
    total_amount         DECIMAL(10,2) NOT NULL,
    currency             CHAR(3) NOT NULL,
    status               TEXT DEFAULT 'pending',
    payment_status       TEXT DEFAULT 'unpaid',
    force_majeure_flag   BOOLEAN DEFAULT false,
    review_requested     BOOLEAN DEFAULT false,
    review_requested_at  TIMESTAMPTZ,
    modification_history JSONB DEFAULT '[]',
    cancellation_reason  TEXT,
    cancelled_at         TIMESTAMPTZ,
    created_at           TIMESTAMPTZ DEFAULT now(),
    updated_at           TIMESTAMPTZ DEFAULT now()
);

-- ─── PAYMENTS ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS payments (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    booking_id           UUID REFERENCES bookings(id),
    gateway              TEXT NOT NULL,
    transaction_id       TEXT UNIQUE,
    amount               DECIMAL(10,2) NOT NULL,
    currency             CHAR(3) NOT NULL,
    status               TEXT NOT NULL,
    risk_score           SMALLINT,
    stripe_radar_score   SMALLINT,
    refund_id            TEXT,
    refund_amount        DECIMAL(10,2),
    refund_status        TEXT,
    refund_initiated_at  TIMESTAMPTZ,
    error_code           TEXT,
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- ─── ADDONS ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS addons (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotel_id             UUID REFERENCES hotels(id) ON DELETE CASCADE,
    name                 TEXT NOT NULL,
    description          TEXT,
    category             TEXT,
    price                DECIMAL(10,2) NOT NULL,
    currency             CHAR(3) NOT NULL,
    trip_purpose_tags    JSONB DEFAULT '[]',
    recommended_rank     SMALLINT DEFAULT 50,
    available            BOOLEAN DEFAULT true
);

-- ─── LOYALTY ACCOUNTS ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS loyalty_accounts (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    guest_id       UUID UNIQUE REFERENCES guests(id),
    total_points   INT DEFAULT 0,
    points_ytd     INT DEFAULT 0,
    tier           TEXT DEFAULT 'bronze',
    badges         JSONB DEFAULT '[]',
    streak_months  INT DEFAULT 0,
    created_at     TIMESTAMPTZ DEFAULT now(),
    updated_at     TIMESTAMPTZ DEFAULT now()
);

-- ─── LOYALTY TRANSACTIONS ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS loyalty_transactions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    guest_id       UUID REFERENCES guests(id),
    booking_id     UUID REFERENCES bookings(id),
    action_type    TEXT NOT NULL,
    points_delta   INT NOT NULL,
    multiplier     DECIMAL(4,2) DEFAULT 1.0,
    balance_after  INT NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- ─── REVIEWS ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reviews (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    booking_id      UUID UNIQUE REFERENCES bookings(id),
    guest_id        UUID REFERENCES guests(id),
    hotel_id        UUID REFERENCES hotels(id),
    overall_score   SMALLINT CHECK (overall_score BETWEEN 1 AND 5),
    room_score      SMALLINT,
    food_score      SMALLINT,
    staff_score     SMALLINT,
    location_score  SMALLINT,
    review_text     TEXT,
    sentiment       TEXT,
    sentiment_score DECIMAL(4,3),
    language        TEXT DEFAULT 'en',
    alert_sent      BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ─── CORPORATE ACCOUNTS ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS corporate_accounts (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name             TEXT NOT NULL,
    auto_approve_threshold   DECIMAL(10,2) DEFAULT 500.00,
    currency                 CHAR(3) DEFAULT 'USD',
    billing_email            TEXT,
    created_at               TIMESTAMPTZ DEFAULT now()
);

-- ─── CORPORATE RATES ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS corporate_rates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id      UUID REFERENCES corporate_accounts(id),
    hotel_id        UUID REFERENCES hotels(id),
    room_type_id    UUID REFERENCES room_types(id),
    rate_per_night  DECIMAL(10,2) NOT NULL,
    currency        CHAR(3) NOT NULL,
    includes_json   JSONB DEFAULT '[]',
    valid_from      DATE,
    valid_until     DATE
);

-- ─── GROUP (ROOM BLOCKS) ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS room_blocks (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotel_id             UUID REFERENCES hotels(id),
    room_type_id         UUID REFERENCES room_types(id),
    count                SMALLINT NOT NULL,
    check_in             DATE NOT NULL,
    check_out            DATE NOT NULL,
    coordinator_guest_id UUID REFERENCES guests(id),
    status               TEXT DEFAULT 'held',
    deposit_schedule     JSONB DEFAULT '[]',
    expires_at           TIMESTAMPTZ NOT NULL,
    group_booking_url    TEXT,
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- ─── APPROVAL REQUESTS ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS approval_requests (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    booking_id      UUID REFERENCES bookings(id),
    company_id      UUID REFERENCES corporate_accounts(id),
    employee_id     UUID REFERENCES guests(id),
    approver_email  TEXT NOT NULL,
    status          TEXT DEFAULT 'pending',
    total_amount    DECIMAL(10,2),
    currency        CHAR(3),
    created_at      TIMESTAMPTZ DEFAULT now(),
    resolved_at     TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ
);

-- ─── CHAT SESSIONS ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_sessions (
    id                        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id                TEXT UNIQUE NOT NULL,
    messenger_psid            TEXT,
    language                  CHAR(5),
    guest_id                  UUID REFERENCES guests(id),
    booking_id                UUID REFERENCES bookings(id),
    handoff_active            BOOLEAN DEFAULT false,
    chatwoot_conversation_id  TEXT,
    started_at                TIMESTAMPTZ DEFAULT now(),
    last_active_at            TIMESTAMPTZ DEFAULT now()
);

-- ─── COMPLIANCE REQUESTS (GDPR) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS compliance_requests (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    guest_id               UUID REFERENCES guests(id),
    request_type           TEXT NOT NULL,
    status                 TEXT DEFAULT 'pending',
    request_date           TIMESTAMPTZ DEFAULT now(),
    completed_date         TIMESTAMPTZ,
    requester_ip           TEXT,
    data_categories_erased JSONB
);

-- ─── INDEXES ─────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_bookings_guest    ON bookings(guest_id);
CREATE INDEX IF NOT EXISTS idx_bookings_hotel    ON bookings(hotel_id);
CREATE INDEX IF NOT EXISTS idx_bookings_status   ON bookings(status);
CREATE INDEX IF NOT EXISTS idx_bookings_checkin  ON bookings(check_in);
CREATE INDEX IF NOT EXISTS idx_room_avail_date   ON room_availability(room_type_id, date);
CREATE INDEX IF NOT EXISTS idx_guests_psid       ON guests(messenger_psid);
CREATE INDEX IF NOT EXISTS idx_payments_booking  ON payments(booking_id);
CREATE INDEX IF NOT EXISTS idx_reviews_hotel     ON reviews(hotel_id);

-- ─── ROW LEVEL SECURITY ───────────────────────────────────────────────────────
ALTER TABLE guests             ENABLE ROW LEVEL SECURITY;
ALTER TABLE bookings           ENABLE ROW LEVEL SECURITY;
ALTER TABLE payments           ENABLE ROW LEVEL SECURITY;
ALTER TABLE loyalty_accounts   ENABLE ROW LEVEL SECURITY;
ALTER TABLE compliance_requests ENABLE ROW LEVEL SECURITY;

-- Service role key bypasses RLS — use for all server-side operations
