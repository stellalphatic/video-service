FROM python:3.8-slim

# Set production environment
ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV MODELS_DIR=/app/models
ENV TEMP_DIR=/app/temp
ENV PYTHONPATH="/app:/app/models/SadTalker:${PYTHONPATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    gcc \
    g++ \
    wget \
    make \
    cmake \
    curl \
    ffmpeg \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libx11-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-thread-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies with version pinning
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install dlib
RUN git clone --depth 1 https://github.com/davisking/dlib.git && \
    cd dlib && \
    python3 setup.py install && \
    cd .. && \
    rm -rf dlib

# Copy only necessary files
COPY download_models.py .
COPY app.py .

# Download models and set up directories
RUN python download_models.py && \
    mkdir -p temp/videos \
             temp/errors \
             temp/avatars \
             temp/streams \
             temp/sadtalker_results \
             temp/wav2lip_results \
             src/config && \
    chmod -R 777 temp && \
    chmod -R 777 models

# Set up SadTalker paths (use copying instead of symlinks for Cloud Run)
RUN cp -r /app/models/SadTalker/src/config/* /app/src/config/ && \
    cp -r /app/models/SadTalker/checkpoints /app/checkpoints

# Non-root user for security
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Configure uvicorn for production
ENV WORKERS=2
ENV TIMEOUT=300
ENV MAX_REQUESTS=1000
ENV MAX_REQUESTS_JITTER=50

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production-ready uvicorn configuration
CMD ["uvicorn", \
     "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "75", \
     "--limit-max-requests", "1000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]