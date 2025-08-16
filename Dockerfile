# Start with a Python slim base image
FROM python:3.8-slim

# Set production environment variables
ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV MODELS_DIR=/app/models
ENV TEMP_DIR=/app/temp
ENV PYTHONPATH="/app:/app/models/SadTalker:${PYTHONPATH}"

# Install all necessary system dependencies in one go
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
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

# Copy and install all Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Clone and install dlib, which will take a long time to compile
RUN git clone --depth 1 https://github.com/davisking/dlib.git && \
    cd dlib && \
    python3 setup.py install && \
    cd .. && \
    rm -rf dlib

# Copy your application files and download models
COPY download_models.py .
COPY app.py .
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

# Set up SadTalker paths
RUN cp -r /app/models/SadTalker/src/config/* /app/src/config/ && \
    cp -r /app/models/SadTalker/checkpoints /app/checkpoints

# Create a non-root user for security
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port and set up uvicorn
EXPOSE 8000
ENV WORKERS=2
ENV TIMEOUT=300
ENV MAX_REQUESTS=1000
ENV MAX_REQUESTS_JITTER=50

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", \
     "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--timeout-keep-alive", "75", \
     "--limit-max-requests", "1000", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]