"""
services/group_service/room_block.py
--------------------------------------
Manages group / rooming-list room blocks.
A room block reserves N rooms of a type from date_from to date_to,
with a pickup deadline and optional deposit schedule.
"""

import os
import logging
from datetime import datetime, timezone

log = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _get_conn():
    import psycopg2
    import psycopg2.extras
    dsn = DATABASE_URL
    if dsn and "sslmode" not in dsn:
        dsn += ("&" if "?" in dsn else "?") + "sslmode=require"
    return psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)


def create_block(
    hotel_id:       str,
    room_type_id:   str,
    rooms_reserved: int,
    date_from:      str,
    date_to:        str,
    group_name:     str,
    pickup_deadline: str,
    rate_per_night:  float,
    organiser_email: str,
) -> dict:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO room_blocks "
                "(hotel_id, room_type_id, rooms_reserved, rooms_picked_up, date_from, date_to, "
                " group_name, pickup_deadline, rate_per_night, organiser_email, status, created_at) "
                "VALUES (%s,%s,%s,0,%s,%s,%s,%s,%s,%s,'active',%s) RETURNING *",
                (hotel_id, room_type_id, rooms_reserved, date_from, date_to,
                 group_name, pickup_deadline, rate_per_night, organiser_email,
                 datetime.now(timezone.utc)),
            )
            block = dict(cur.fetchone())
            conn.commit()
    log.info("Room block created: %s (%d rooms)", group_name, rooms_reserved)
    return block


def pickup_room(block_id: str, booking_id: str) -> dict:
    """Marks one room from the block as picked up."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM room_blocks WHERE id = %s FOR UPDATE", (block_id,))
            block = cur.fetchone()
            if not block:
                raise ValueError(f"Block {block_id} not found")
            block = dict(block)
            if block["rooms_picked_up"] >= block["rooms_reserved"]:
                raise ValueError("All rooms already picked up")
            cur.execute(
                "UPDATE room_blocks SET rooms_picked_up = rooms_picked_up + 1, "
                "updated_at = %s WHERE id = %s RETURNING *",
                (datetime.now(timezone.utc), block_id),
            )
            result = dict(cur.fetchone())
            conn.commit()
    return result


def get_block(block_id: str) -> dict:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM room_blocks WHERE id = %s", (block_id,))
            row = cur.fetchone()
    if not row:
        raise ValueError(f"Block {block_id} not found")
    return dict(row)
