"""
services/force_majeure_service/news_monitor.py
------------------------------------------------
Monitors news feeds for force-majeure events (severe weather, natural disasters,
political unrest, pandemics) near hotel locations.

Sources:
  1. GNews API  (free tier, configurable)
  2. ReliefWeb API (UN OCHA humanitarian events, free / no key)

Returns a severity level: "none" | "watch" | "warning" | "critical"
"""

import os
import json
import logging
import urllib.request
import urllib.parse
from typing import Optional

log = logging.getLogger(__name__)

GNEWS_KEY = os.environ.get("GNEWS_API_KEY", "")
FM_KEYWORDS = [
    "earthquake", "hurricane", "typhoon", "cyclone", "tsunami", "volcano",
    "flood", "wildfire", "war", "conflict", "martial law", "airport closed",
    "travel ban", "pandemic", "lockdown", "curfew",
]


def _gnews_search(city: str, country: str) -> list[dict]:
    if not GNEWS_KEY:
        return []
    query = urllib.parse.quote(f"{city} {country} disaster OR emergency OR warning")
    url   = (
        f"https://gnews.io/api/v4/search?q={query}"
        f"&lang=en&max=5&apikey={GNEWS_KEY}"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read()).get("articles", [])
    except Exception as exc:
        log.debug("GNews unavailable: %s", exc)
        return []


def _reliefweb_search(country_iso2: str) -> list[dict]:
    url = (
        "https://api.reliefweb.int/v1/disasters?appname=bookhotel"
        f"&filter[field]=country.iso3&filter[value]={country_iso2}"
        "&filter[field]=status&filter[value]=current&limit=5"
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("data", [])
    except Exception as exc:
        log.debug("ReliefWeb unavailable: %s", exc)
        return []


def assess(
    city:          str,
    country:       str,
    country_iso2:  Optional[str] = "",
) -> dict:
    """
    Returns {severity, events, message}.
    severity: "none" | "watch" | "warning" | "critical"
    """
    articles    = _gnews_search(city, country)
    rw_events   = _reliefweb_search(country_iso2 or country)

    triggered = []
    for art in articles:
        title = (art.get("title") or "").lower()
        desc  = (art.get("description") or "").lower()
        for kw in FM_KEYWORDS:
            if kw in title or kw in desc:
                triggered.append({"source": "gnews", "keyword": kw, "title": art["title"]})
                break

    for evt in rw_events:
        fields = evt.get("fields", {})
        triggered.append({"source": "reliefweb", "name": fields.get("name"), "type": fields.get("type")})

    if not triggered:
        severity = "none"
        message  = "No active force-majeure events detected."
    elif len(triggered) >= 3:
        severity = "critical"
        message  = f"Multiple force-majeure events near {city}. Full waiver recommended."
    elif len(triggered) >= 1:
        severity = "watch"
        message  = f"Potential disruption near {city}. Monitor situation."
    else:
        severity = "none"
        message  = "No active force-majeure events."

    return {"severity": severity, "events": triggered, "message": message}
