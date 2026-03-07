FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg gcc g++ libsndfile1 patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clear executable stack flag from ctranslate2 bundled libs
RUN find /usr/local/lib/python3.10/site-packages -name "*.so*" \
    -exec patchelf --clear-execstack {} \; 2>/dev/null || true

RUN python -c "import ctranslate2; print('ctranslate2 OK:', ctranslate2.__version__)"

COPY . .

RUN python download_model.py

# Catch any import-time crash — shows exact error in build logs
RUN python -c "import processor; print('=== Import check OK ===')"

EXPOSE 7860

CMD ["uvicorn", "processor:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "debug"]