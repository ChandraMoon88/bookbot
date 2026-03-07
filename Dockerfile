FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY . .

EXPOSE 7860

# Run processor.py on HF Space
CMD ["uvicorn", "processor:app", "--host", "0.0.0.0", "--port", "7860"]