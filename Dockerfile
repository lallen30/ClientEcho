# Build stage
FROM python:3.9-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg \
  build-essential \
  curl \
  sqlite3 \
  && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory for the builder
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim

# Install ffmpeg and sqlite3
RUN apt-get update && apt-get install -y \
  ffmpeg \
  sqlite3 \
  && rm -rf /var/lib/apt/lists/*

# Install a text editor
RUN apt-get update && apt-get install -y nano

# Set the working directory
WORKDIR /app

# Copy the installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY . .

# Create a non-root user
RUN useradd -m appuser

# Create necessary directories and set permissions
RUN mkdir -p /app/instance && \
  touch /app/instance/clientecho.db && \
  chown -R appuser:appuser /app && \
  chmod 755 /app/instance && \
  chmod 664 /app/instance/clientecho.db

# Initialize the database as root
RUN python init_db.py

# Switch to non-root user
USER appuser

# Expose the port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the application with increased timeout to allow for longer video processing times
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "1", "--threads", "2", "app:app"]
