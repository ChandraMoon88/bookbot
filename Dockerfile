FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download Whisper model at build time into a fixed path
RUN python download_model.py

# This will FAIL THE BUILD (showing the exact error) if any import is broken
# Remove this line once confirmed working
RUN python -c "import processor; print('=== Import check OK ===')"

EXPOSE 7860

CMD ["uvicorn", "processor:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "debug"]