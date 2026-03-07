# Use Python 3.10
FROM python:3.10-slim

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Start your bot
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]