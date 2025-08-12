FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV MODELS_DIR=/app/models
ENV TEMP_DIR=/app/temp

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt



# RUN pip install --no-cache-dir dlib-bin==19.24.2 || \
#     (curl -L -o dlib.whl https://github.com/davisking/dlib/releases/download/v19.24.2/dlib-19.24.2-cp310-cp310-manylinux_2_17_x86_64.whl && \
#     pip install --no-cache-dir dlib.whl && \
#     rm dlib.whl)

RUN git clone https://github.com/davisking/dlib.git && \
    cd dlib && \
    python3 setup.py install && \
    cd .. && \
    rm -rf dlib

# Copy model download script
COPY download_models.py .

# Download models during build (this runs once)
RUN python download_models.py

# Copy application code
COPY app.py .

# Create necessary directories
RUN mkdir -p temp/videos temp/errors temp/avatars temp/streams

# Expose port
EXPOSE 8000

# Health check with longer timeout for startup
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
