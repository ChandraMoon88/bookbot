"""
services/analytics_service/kafka_consumer.py
----------------------------------------------
Consumes booking events from Kafka and inserts them into ClickHouse.

Topics consumed:
  booking.created
  booking.payment.succeeded
  booking.cancelled
  booking.modified
  nlu.intent.logged

Runs as a standalone worker process (not a FastAPI route).
"""

import os
import json
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

KAFKA_BOOTSTRAP  = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_GROUP_ID   = os.environ.get("KAFKA_CONSUMER_GROUP",    "analytics-consumer")
KAFKA_TOPICS     = ["booking.created", "booking.payment.succeeded",
                    "booking.cancelled", "booking.modified", "nlu.intent.logged"]

CLICKHOUSE_HOST  = os.environ.get("CLICKHOUSE_HOST",     "localhost")
CLICKHOUSE_PORT  = int(os.environ.get("CLICKHOUSE_PORT", "9000"))
CLICKHOUSE_DB    = os.environ.get("CLICKHOUSE_DB",       "bookhotel")
CLICKHOUSE_USER  = os.environ.get("CLICKHOUSE_USER",     "default")
CLICKHOUSE_PASS  = os.environ.get("CLICKHOUSE_PASSWORD", "")


def _get_ch_client():
    try:
        from clickhouse_driver import Client
        return Client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            database=CLICKHOUSE_DB,
            user=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASS,
        )
    except ImportError:
        raise RuntimeError("clickhouse-driver is required: pip install clickhouse-driver")


def _insert_booking_event(ch, event: dict) -> None:
    ch.execute(
        "INSERT INTO booking_events (event_type, booking_id, hotel_id, guest_id, "
        "amount_usd, check_in, check_out, created_at) VALUES",
        [{
            "event_type": event.get("type", "unknown"),
            "booking_id": event.get("booking_id", ""),
            "hotel_id":   event.get("hotel_id", ""),
            "guest_id":   event.get("guest_id", ""),
            "amount_usd": float(event.get("amount_usd", 0)),
            "check_in":   event.get("check_in", ""),
            "check_out":  event.get("check_out", ""),
            "created_at": datetime.now(timezone.utc),
        }],
    )


def _insert_nlu_event(ch, event: dict) -> None:
    ch.execute(
        "INSERT INTO nlu_logs (sender_id, intent, confidence, language, created_at) VALUES",
        [{
            "sender_id":  event.get("sender_id", ""),
            "intent":     event.get("intent", ""),
            "confidence": float(event.get("confidence", 0)),
            "language":   event.get("language", "en"),
            "created_at": datetime.now(timezone.utc),
        }],
    )


def run() -> None:
    """
    Blocking Kafka consumer loop.
    Call from a Celery worker or a dedicated Render background worker.
    """
    try:
        from kafka import KafkaConsumer
    except ImportError:
        raise RuntimeError("kafka-python is required: pip install kafka-python")

    consumer = KafkaConsumer(
        *KAFKA_TOPICS,
        bootstrap_servers=KAFKA_BOOTSTRAP.split(","),
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    ch = _get_ch_client()
    log.info("Kafka consumer started; topics=%s", KAFKA_TOPICS)

    for message in consumer:
        try:
            event = message.value
            topic = message.topic

            if topic == "nlu.intent.logged":
                _insert_nlu_event(ch, event)
            else:
                event["type"] = topic.split(".")[-1]
                _insert_booking_event(ch, event)
        except Exception as exc:
            log.error("Error processing Kafka message: %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
