"""
services/confirmation_service/calendar.py
-------------------------------------------
Generates an iCalendar (.ics) file for the booking.
Standard: RFC 5545
Check-in  : 15:00 local time → stored as UTC
Check-out : 11:00 local time → stored as UTC
VALARM    : 24 h before check-in
"""

from datetime import datetime, timedelta, timezone


def _parse_date(date_str: str) -> datetime:
    """Accepts 'YYYY-MM-DD', returns naive date at midnight UTC."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def generate(booking: dict) -> str:
    """
    Returns ICS text string.

    booking keys used: ref, guest_name, hotel_name, check_in (YYYY-MM-DD),
                        check_out (YYYY-MM-DD), room_type
    """
    try:
        from icalendar import Calendar, Event, Alarm, vText, vDatetime, vDuration
    except ImportError:
        raise RuntimeError("icalendar is required: pip install icalendar")

    ref        = booking.get("ref", "BOOK000")
    guest_name = booking.get("guest_name", "Guest")
    hotel_name = booking.get("hotel_name", "Hotel")
    room_type  = booking.get("room_type", "")

    check_in_date  = _parse_date(booking["check_in"])
    check_out_date = _parse_date(booking["check_out"])

    # 15:00 check-in, 11:00 check-out (UTC)
    checkin_dt  = check_in_date.replace(hour=15)
    checkout_dt = check_out_date.replace(hour=11)

    cal = Calendar()
    cal.add("prodid", "-//BookHotel Bot//bookhotel.ai//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("method", "PUBLISH")

    event = Event()
    event.add("uid",     f"{ref}@bookhotel.ai")
    event.add("summary", f"Hotel Stay – {hotel_name}")
    event.add("dtstart", checkin_dt)
    event.add("dtend",   checkout_dt)
    event.add("dtstamp", datetime.now(timezone.utc))
    event.add("location", vText(hotel_name))
    event.add("description", vText(
        f"Booking Ref: {ref}\nGuest: {guest_name}\nRoom: {room_type}\n"
        f"Check-in: {booking['check_in']} from 15:00\n"
        f"Check-out: {booking['check_out']} by 11:00"
    ))

    # VALARM: reminder 24 h before check-in
    alarm = Alarm()
    alarm.add("action",  "DISPLAY")
    alarm.add("trigger", timedelta(hours=-24))
    alarm.add("description", f"Reminder: Check-in tomorrow at {hotel_name}")
    event.add_component(alarm)

    cal.add_component(event)
    return cal.to_ical().decode("utf-8")
