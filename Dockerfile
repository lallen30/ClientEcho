# Build stage
FROM python:3.9-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg \
  build-essential \
  curl \
  sqlite3 \
  tesseract-ocr \
  tesseract-ocr-eng \
  libtesseract-dev \
  git \
  cmake \
  libatlas-base-dev \
  libffi-dev \
  python3-dev \
  sox \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory for the builder
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg \
  sqlite3 \
  tesseract-ocr \
  tesseract-ocr-eng \
  libtesseract-dev \
  libatlas-base-dev \
  python3-dev \
  sox \
  nano \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy the application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/instance /app/secure_uploads && \
  touch /app/instance/clientecho.db && \
  useradd -m appuser && \
  chown -R appuser:appuser /app && \
  chmod 755 /app/instance && \
  chmod 664 /app/instance/clientecho.db && \
  chmod 777 /app/secure_uploads

# Initialize the database as root
RUN python init_db.py

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "1", "--threads", "2", "app:app"]
