# Stage 1: The Builder (for heavy lifting)
FROM python:3.8-slim AS builder

# Set up environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build-time system dependencies. This is where dlib is compiled.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libboost-python-dev \
    libboost-thread-dev \
    libopenblas-dev \
    liblapack-dev \
    wget \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install all Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone and compile dlib. This is the most time-consuming step.
RUN git clone --depth 1 https://github.com/davisking/dlib.git && \
    cd dlib && \
    python3 setup.py install && \
    cd .. && \
    rm -rf dlib

# Download all models to a dedicated directory
COPY download_models.py .
RUN python download_models.py

# ---
# Stage 2: The Final (lightweight) Image
# ---
FROM python:3.8-slim

# Set up runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV MODELS_DIR=/app/models
ENV TEMP_DIR=/app/temp
ENV PYTHONPATH="/app:/app/models/SadTalker:${PYTHONPATH}"

# Install only the system dependencies required *at runtime*
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libx11-dev \
    libgomp1 \
    libtcmalloc-minimal4 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the necessary files from the builder stage. This is the key.
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /app/models /app/models

# Copy the application source code
COPY app.py .
COPY requirements.txt .
COPY download_models.py .

# Create directories and set permissions
RUN mkdir -p temp/videos temp/errors temp/avatars temp/streams temp/sadtalker_results temp/wav2lip_results src/config && \
    chmod -R 777 temp && \
    chmod -R 777 models && \
    cp -r /app/models/SadTalker/src/config/* /app/src/config/ && \
    cp -r /app/models/SadTalker/checkpoints /app/checkpoints

# Create and switch to a non-root user for security
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port and start the application
EXPOSE 8000
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