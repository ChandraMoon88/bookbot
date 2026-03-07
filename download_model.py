"""
download_model.py
-----------------
Pre-downloads all AI models at Docker BUILD time so the running container
starts instantly with no cold-download delays.

Models downloaded:
  1. Whisper small      — STT (speech-to-text)
  2. NLLB-200 600M      — Translation (replaces deep-translator)
  3. SpeechT5 + HiFiGAN + CMU Arctic xvectors — English TTS
  4. MMS-TTS            — Multilingual TTS for the 3 most-used languages
                          (eng / hin / tel — extend the list as needed)

Run in Dockerfile as:   RUN python download_model.py
"""

import os
from pathlib import Path

WHISPER_CACHE = "/app/.cache/whisper"
HF_CACHE      = "/app/.cache/huggingface"

os.makedirs(WHISPER_CACHE, exist_ok=True)
os.makedirs(HF_CACHE, exist_ok=True)

# ── Shared kwargs for all HF model downloads ─────────────────────────────────
HF_KWARGS = dict(cache_dir=HF_CACHE)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Whisper STT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Downloading Whisper 'small' (STT)…")
from faster_whisper import WhisperModel
WhisperModel("small", device="cpu", compute_type="int8", download_root=WHISPER_CACHE)
print("      ✅ Whisper done.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. NLLB-200 Translation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Downloading NLLB-200-distilled-600M (translation)…")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NLLB_ID = "facebook/nllb-200-distilled-600M"
AutoTokenizer.from_pretrained(NLLB_ID, **HF_KWARGS)
AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID, **HF_KWARGS)
print("      ✅ NLLB-200 done.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SpeechT5 — English TTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Downloading SpeechT5 TTS + HiFiGAN vocoder + speaker embeddings…")
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

SpeechT5Processor.from_pretrained("microsoft/speecht5_tts",    **HF_KWARGS)
SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", **HF_KWARGS)
SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",  **HF_KWARGS)

# Pre-fetch CMU Arctic speaker embeddings dataset (tiny — ~2 MB)
load_dataset(
    "Matthijs/cmu-arctic-xvectors",
    split="validation",
    cache_dir=HF_CACHE,
    trust_remote_code=True,
)
print("      ✅ SpeechT5 done.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MMS-TTS — Multilingual TTS
#    Pre-cache the languages most likely to be used first.
#    Add more codes from MMS_LANG_MAP in autotranslator.py as needed.
# ─────────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, VitsModel

# Top languages for this bot — adjust to match your user base
MMS_PRELOAD = [
    "eng",   # English   (backup path)
    "hin",   # Hindi
    "tel",   # Telugu
    "tam",   # Tamil
    "fra",   # French
    "spa",   # Spanish
    "arb",   # Arabic
]

print(f"\n[4/4] Downloading MMS-TTS models for: {MMS_PRELOAD}")
for code in MMS_PRELOAD:
    model_id = f"facebook/mms-tts-{code}"
    print(f"      Fetching {model_id}…")
    try:
        AutoTokenizer.from_pretrained(model_id, **HF_KWARGS)
        VitsModel.from_pretrained(model_id, **HF_KWARGS)
        print(f"      ✅ {code} done.")
    except Exception as e:
        print(f"      ⚠️  {code} skipped ({e})")

print("\n✅  All models cached — container is ready for cold-start-free deployment.")