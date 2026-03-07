"""
BookBot End-to-End Test Suite
=============================
Run: python test_all.py

Tests every component before going live on Messenger.
Set these env vars before running:
  REDIS_URL        - Upstash Redis URL (just the URL, e.g. rediss://default:PASSWORD@host:port)
  HF_PROCESSOR_URL - e.g. https://your-space.hf.space
"""

import os
import sys
import json
import base64
import asyncio
import httpx

# ─────────────────────────────────────────────
# CONFIG — edit these or set as env vars
# ─────────────────────────────────────────────
HF_PROCESSOR_URL = os.getenv("HF_PROCESSOR_URL", "https://your-space.hf.space")
REDIS_URL        = os.getenv("REDIS_URL", "")
TEST_SENDER_ID   = "test_user_001"

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []


def log(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed))
    print(f"{status}  {name}")
    if detail:
        print(f"       {detail}")


# ─────────────────────────────────────────────
# 1. HEALTH CHECK
# ─────────────────────────────────────────────
def test_health():
    print("\n── 1. Health Check ──────────────────────────")
    try:
        r = httpx.get(f"{HF_PROCESSOR_URL}/health", timeout=10)
        ok = r.status_code == 200 and r.json().get("status") == "ok"
        log("Processor /health", ok, r.text)
    except Exception as e:
        log("Processor /health", False, str(e))


# ─────────────────────────────────────────────
# 2. TEXT MESSAGE — English greeting
# ─────────────────────────────────────────────
def test_text_english():
    print("\n── 2. Text Message (English Greeting) ───────")
    payload = {"sender_id": TEST_SENDER_ID, "type": "text", "message": "Hello"}
    try:
        r = httpx.post(f"{HF_PROCESSOR_URL}/process", json=payload, timeout=30)
        data = r.json()
        has_text  = bool(data.get("text"))
        has_audio = bool(data.get("audio_b64"))
        lang_ok   = data.get("lang") == "en"
        log("Response has text",  has_text,  data.get("text", "")[:80])
        log("Response has audio", has_audio)
        log("Language detected as 'en'", lang_ok, f"got: {data.get('lang')}")
    except Exception as e:
        log("Text English", False, str(e))


# ─────────────────────────────────────────────
# 3. TEXT MESSAGE — Telugu (non-Latin script)
# ─────────────────────────────────────────────
def test_text_telugu():
    print("\n── 3. Text Message (Telugu) ─────────────────")
    payload = {"sender_id": "test_user_te", "type": "text", "message": "నమస్కారం"}
    try:
        r = httpx.post(f"{HF_PROCESSOR_URL}/process", json=payload, timeout=30)
        data = r.json()
        has_text  = bool(data.get("text"))
        has_audio = bool(data.get("audio_b64"))
        lang_ok   = data.get("lang") == "te"
        log("Response has text",  has_text,  data.get("text", "")[:80])
        log("Response has audio", has_audio)
        log("Language detected as 'te'", lang_ok, f"got: {data.get('lang')}")
    except Exception as e:
        log("Text Telugu", False, str(e))


# ─────────────────────────────────────────────
# 4. TEXT MESSAGE — Hindi
# ─────────────────────────────────────────────
def test_text_hindi():
    print("\n── 4. Text Message (Hindi) ──────────────────")
    payload = {"sender_id": "test_user_hi", "type": "text", "message": "नमस्ते"}
    try:
        r = httpx.post(f"{HF_PROCESSOR_URL}/process", json=payload, timeout=30)
        data = r.json()
        has_text  = bool(data.get("text"))
        lang_ok   = data.get("lang") == "hi"
        log("Response has text",  has_text,  data.get("text", "")[:80])
        log("Language detected as 'hi'", lang_ok, f"got: {data.get('lang')}")
    except Exception as e:
        log("Text Hindi", False, str(e))


# ─────────────────────────────────────────────
# 5. VOICE MESSAGE — Synthetic silence WAV
#    (tests the voice pipeline end-to-end without a mic)
# ─────────────────────────────────────────────
def test_voice_pipeline():
    print("\n── 5. Voice Pipeline (synthetic audio) ──────")
    # Minimal valid WAV: 1 second of silence at 16kHz mono 16-bit
    import struct, math
    sample_rate = 16000
    duration    = 1
    num_samples = sample_rate * duration
    wav_header  = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + num_samples * 2, b"WAVE",
        b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16,
        b"data", num_samples * 2,
    )
    audio_bytes = wav_header + b"\x00" * (num_samples * 2)
    audio_b64   = base64.b64encode(audio_bytes).decode()

    payload = {"sender_id": "test_user_voice", "type": "voice", "audio_b64": audio_b64}
    try:
        r = httpx.post(f"{HF_PROCESSOR_URL}/process", json=payload, timeout=60)
        data = r.json()
        has_text  = bool(data.get("text"))
        no_crash  = "Something went wrong" not in data.get("text", "")
        log("Voice endpoint responds", r.status_code == 200, f"status={r.status_code}")
        log("Voice response has text", has_text, data.get("text", "")[:80])
        log("No crash in voice path",  no_crash)
    except Exception as e:
        log("Voice pipeline", False, str(e))


# ─────────────────────────────────────────────
# 6. TTS — Audio is valid MP3/audio bytes
# ─────────────────────────────────────────────
def test_audio_validity():
    print("\n── 6. TTS Audio Validity ────────────────────")
    payload = {"sender_id": "test_audio", "type": "text", "message": "Hello"}
    try:
        r    = httpx.post(f"{HF_PROCESSOR_URL}/process", json=payload, timeout=30)
        data = r.json()
        b64  = data.get("audio_b64", "")
        if not b64:
            log("Audio b64 present", False, "No audio in response")
            return
        raw = base64.b64decode(b64)
        # MP3 starts with ID3 or 0xFF 0xFB/0xFA/0xF3
        is_mp3 = raw[:3] == b"ID3" or (raw[0] == 0xFF and raw[1] in (0xFA, 0xFB, 0xF3))
        log("Audio b64 present",     True,   f"{len(raw):,} bytes")
        log("Audio is valid MP3",    is_mp3, f"header bytes: {raw[:4].hex()}")
    except Exception as e:
        log("Audio validity", False, str(e))


# ─────────────────────────────────────────────
# 7. REDIS — session save/get
# ─────────────────────────────────────────────
def test_redis():
    print("\n── 7. Redis Session Store ───────────────────")
    if not REDIS_URL:
        log("Redis URL configured", False, "REDIS_URL env var not set — skipping")
        return
    try:
        import redis as redis_lib
        # redis-py v4+ uses Redis.from_url; fallback for older versions
        if hasattr(redis_lib, "from_url"):
            client = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)
        else:
            client = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True, socket_timeout=5)
        client.setex("test:bookbot", 60, "ping")
        val = client.get("test:bookbot")
        log("Redis write",        True)
        log("Redis read back",    val == "ping", f"got: {val}")
        client.delete("test:bookbot")
        log("Redis delete",       True)
    except Exception as e:
        log("Redis connection",   False, str(e))


# ─────────────────────────────────────────────
# 8. LANGUAGE PERSISTENCE — second message uses stored lang
# ─────────────────────────────────────────────
def test_language_persistence():
    print("\n── 8. Language Persistence ──────────────────")
    uid = "test_persist_001"
    # First message sets language
    r1 = httpx.post(f"{HF_PROCESSOR_URL}/process",
                    json={"sender_id": uid, "type": "text", "message": "నమస్కారం"},
                    timeout=30)
    lang1 = r1.json().get("lang")
    # Second message should still be Telugu
    r2 = httpx.post(f"{HF_PROCESSOR_URL}/process",
                    json={"sender_id": uid, "type": "text", "message": "book"},
                    timeout=30)
    lang2 = r2.json().get("lang")
    try:
        log("First message lang set to 'te'",       lang1 == "te", f"got: {lang1}")
        log("Second message lang persists as 'te'", lang2 == "te", f"got: {lang2}")
    except Exception as e:
        log("Language persistence", False, str(e))


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def summary():
    print("\n" + "═" * 50)
    total  = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  |  {failed} failed:")
        for name, ok in results:
            if not ok:
                print(f"    ❌  {name}")
    else:
        print("  🎉  All tests passed — safe to test on Messenger!")
    print("═" * 50)


if __name__ == "__main__":
    print("═" * 50)
    print("  BookBot Test Suite")
    print(f"  Target: {HF_PROCESSOR_URL}")
    print("═" * 50)

    if "your-space" in HF_PROCESSOR_URL:
        print("\n⚠️  Set HF_PROCESSOR_URL before running:")
        print("   export HF_PROCESSOR_URL=https://your-username-your-space.hf.space")
        sys.exit(1)

    test_health()
    test_text_english()
    test_text_telugu()
    test_text_hindi()
    test_voice_pipeline()
    test_audio_validity()
    test_redis()
    test_language_persistence()
    summary()