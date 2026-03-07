FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg gcc g++ libsndfile1 patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HF Spaces blocks shared libs that request executable stack.
# ctranslate2 ships with this flag set — clear it with patchelf.
RUN find /usr/local/lib/python3.10/site-packages/ctranslate2 -name "*.so*" \
    | xargs -I{} patchelf --clear-execstack {}

COPY . .

RUN python download_model.py

RUN python -c "import processor; print('=== Import check OK ===')"

EXPOSE 7860

CMD ["uvicorn", "processor:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "debug"]