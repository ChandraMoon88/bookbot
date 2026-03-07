import os
from faster_whisper import WhisperModel

os.makedirs("/app/.cache/whisper", exist_ok=True)
print("Downloading Whisper 'small' model...")
WhisperModel("small", device="cpu", compute_type="int8", download_root="/app/.cache/whisper")
print("Whisper model cached at /app/.cache/whisper")