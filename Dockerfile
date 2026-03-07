# Use Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for faster builds)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code
COPY . .

# Expose port 7860 (Hugging Face uses this port)
EXPOSE 7860

# Start your FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]