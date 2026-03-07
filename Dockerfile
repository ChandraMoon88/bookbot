# Python 3.10 slim base
FROM python:3.10-slim

# Install system dependencies
# ffmpeg needed for pydub audio processing
# gcc needed for noisereduce compilation
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    g++ \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (faster rebuilds)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code
COPY . .

# Hugging Face uses port 7860
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]