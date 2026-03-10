"""
hf_space/app.py
----------------
HuggingFace Spaces entry point — forwards to the root processor.py app.

The Dockerfile CMD is:
    uvicorn processor:app --host 0.0.0.0 --port 7860

This file re-exports the same FastAPI app from processor.py so that
both `uvicorn processor:app` and `uvicorn hf_space.app:app` work.

Architecture:
  Render (main.py) -> HF Spaces (processor.py) -> Supabase DB (db_client.py)

Endpoints (from processor.py):
  GET  /health   -- readiness check, per-model status
  POST /process  -- {sender_id, type, message, audio_b64} -> {text, buttons, audio_b64, lang}
"""

# Re-export processor app so both uvicorn paths work.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processor import app  # noqa: F401

__all__ = ["app"]
