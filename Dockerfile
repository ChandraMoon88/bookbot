FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Whisper model to a fixed, explicit path — not home-relative
# so it works regardless of which user HF Spaces runs the container as
ENV WHISPER_CACHE=/app/.cache/whisper
RUN python -c "
from faster_whisper import WhisperModel
import os
os.makedirs('/app/.cache/whisper', exist_ok=True)
WhisperModel('small', device='cpu', compute_type='int8', download_root='/app/.cache/whisper')
print('Whisper model cached at /app/.cache/whisper')
"

COPY . .

EXPOSE 7860

RUN python -c "import processor; print('Import OK')"
# Shell form so stderr is captured; --log-level debug surfaces import crashes
CMD ["uvicorn", "processor:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "debug"]