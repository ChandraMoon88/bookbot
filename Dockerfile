FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg gcc g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python download_model.py

EXPOSE 7860

CMD ["uvicorn", "processor:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "debug"]