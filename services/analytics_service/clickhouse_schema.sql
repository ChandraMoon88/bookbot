-- ============================================================
-- ClickHouse schema for BookHotel analytics
-- ============================================================

CREATE DATABASE IF NOT EXISTS bookhotel;

-- ---------------------------------------------------------------
-- Booking Events
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bookhotel.booking_events
(
    event_type  LowCardinality(String),  -- created | succeeded | cancelled | modified
    booking_id  String,
    hotel_id    String,
    guest_id    String,
    amount_usd  Float64,
    check_in    String,
    check_out   String,
    created_at  DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (event_type, created_at);

-- ---------------------------------------------------------------
-- NLU Logs
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bookhotel.nlu_logs
(
    sender_id   String,
    intent      LowCardinality(String),
    confidence  Float32,
    language    LowCardinality(String),
    created_at  DateTime
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(created_at)
ORDER BY (intent, created_at);

-- ---------------------------------------------------------------
-- Revenue Summary (materialized view refreshed hourly)
-- ---------------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS bookhotel.revenue_daily
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(day)
ORDER BY (hotel_id, day)
AS
SELECT
    hotel_id,
    toDate(created_at) AS day,
    count()            AS bookings,
    sum(amount_usd)    AS revenue_usd
FROM bookhotel.booking_events
WHERE event_type = 'succeeded'
GROUP BY hotel_id, day;
