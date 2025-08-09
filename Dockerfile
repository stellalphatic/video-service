# Use a base image with CUDA 11.7. This is necessary for GPU support.
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive.
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.9, pip for 3.9, and other necessary dependencies.
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /app

# Copy the requirements.txt file.
COPY requirements.txt .

# This command will now succeed because python3.9-pip is installed.
# We are explicitly using the Python 3.9 pip.
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Clone the SadTalker repository.
RUN git clone https://github.com/OpenTalker/SadTalker.git

# Change to the cloned directory to pull large model files.
WORKDIR /app/SadTalker

# Install git-lfs and pull the large model files.
RUN git lfs install && git lfs pull

# Copy your application file to the parent directory.
WORKDIR /app
COPY app.py .

# Define the command to run the application using Python 3.9.
CMD ["python3.9", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
