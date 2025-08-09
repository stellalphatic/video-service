FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies for video processing and AI models
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV MODELS_DIR=/app/models
ENV TEMP_DIR=/app/temp
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Create necessary directories
RUN mkdir -p ${MODELS_DIR} && \
    mkdir -p ${TEMP_DIR} && \
    mkdir -p ${HF_HOME} && \
    mkdir -p ${TORCH_HOME}

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download models script
COPY download_models.py .
RUN python download_models.py

# Copy the FastAPI application
COPY app.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
