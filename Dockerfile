# Use a base image with CUDA and Python
FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Set environment variables
ENV PATH="/usr/bin/python3:$PATH"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y tzdata && \
    apt-get install -y \
    python3 python3-pip \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Clone the SadTalker repository
RUN git clone https://github.com/Winfredy/SadTalker.git ./SadTalker

# Download SadTalker models using verified, working URLs.
# Note: These links are more reliable than the previous GitHub release links.
RUN mkdir -p ./SadTalker/checkpoints && \
    wget -O ./SadTalker/checkpoints/gfpgan.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth && \
    wget -O ./SadTalker/checkpoints/parsing_bisenet.pth https://huggingface.co/caocaocoa/1111/resolve/main/parsing_bisenet.pth && \
    wget -O ./SadTalker/checkpoints/wav2lip.pth https://huggingface.co/spaces/Vin-J/SadTalker/resolve/main/wav2lip.pth && \
    wget -O ./SadTalker/checkpoints/audio2coeff.pth https://huggingface.co/spaces/Vin-J/SadTalker/resolve/main/audio2coeff.pth

# Copy the application code and requirements
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for avatars
RUN mkdir -p avatars

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
