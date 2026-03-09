"""
services/maps_service/nominatim.py
------------------------------------
Geocoding + reverse geocoding via OpenStreetMap Nominatim.
No API key required.  Rate limited to 1 req/s per OSM policy.
"""

import json
import logging
import time
import urllib.request
import urllib.parse
from typing import Optional

log = logging.getLogger(__name__)

NOMINATIM_BASE   = "https://nominatim.openstreetmap.org"
USER_AGENT       = "BookHotelBot/1.0 (contact@bookhotel.ai)"
_last_call       = 0.0   # timestamp of last request (rate limit)


def _request(url: str) -> dict:
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        _last_call = time.time()
        return json.loads(resp.read())


def geocode(query: str) -> Optional[dict]:
    """
    Forward geocoding.
    Returns {lat, lon, display_name, country_code} or None.
    """
    params = urllib.parse.urlencode({
        "q":      query,
        "format": "json",
        "limit":  1,
        "addressdetails": 1,
    })
    url    = f"{NOMINATIM_BASE}/search?{params}"
    try:
        results = _request(url)
        if results:
            r = results[0]
            addr = r.get("address", {})
            return {
                "lat":          float(r["lat"]),
                "lon":          float(r["lon"]),
                "display_name": r["display_name"],
                "country_code": addr.get("country_code", "").upper(),
                "city":         addr.get("city") or addr.get("town") or addr.get("village", ""),
                "country":      addr.get("country", ""),
            }
    except Exception as exc:
        log.error("Nominatim geocode error: %s", exc)
    return None


def reverse_geocode(lat: float, lon: float) -> Optional[dict]:
    """
    Reverse geocoding.
    Returns {display_name, city, country, country_code} or None.
    """
    params = urllib.parse.urlencode({
        "lat":    lat,
        "lon":    lon,
        "format": "json",
        "addressdetails": 1,
    })
    url = f"{NOMINATIM_BASE}/reverse?{params}"
    try:
        r    = _request(url)
        addr = r.get("address", {})
        return {
            "display_name": r.get("display_name", ""),
            "city":         addr.get("city") or addr.get("town") or addr.get("village", ""),
            "country":      addr.get("country", ""),
            "country_code": addr.get("country_code", "").upper(),
        }
    except Exception as exc:
        log.error("Nominatim reverse geocode error: %s", exc)
    return None


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km between two coordinates."""
    from math import radians, sin, cos, sqrt, atan2
    R    = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a    = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))
