FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg gcc g++ libsndfile1 patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ctranslate2 bundles its own libctranslate2*.so in a .libs/ subdirectory
# inside the wheel — patchelf must search the full site-packages tree.
RUN find /usr/local/lib/python3.10/site-packages -name "*.so*" -exec patchelf --clear-execstack {} \; 2>/dev/null || true

# Verify the flag is gone before proceeding
RUN python -c "import ctranslate2; print('ctranslate2 OK:', ctranslate2.__version__)"

COPY . .

RUN python download_model.py

# Simulate runtime environment — expose the crash with verbose output
RUN python -c "
import traceback, sys
try:
    import processor
    print('OK')
except Exception as e:
    traceback.print_exc()
    sys.exit(1)
"

EXPOSE 7860

CMD ["uvicorn", "processor:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "debug"]