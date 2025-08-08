# Use a base image with CUDA 11.7, which is compatible with the L4 GPU.
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive to prevent prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, git-lfs, and other necessary dependencies.
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as the default python.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file into the container.
COPY requirements.txt .

# Install all the Python dependencies. The key here is the `--extra-index-url`
# which tells pip where to find the CUDA-specific PyTorch binaries.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Clone the SadTalker repository.
RUN git clone https://github.com/OpenTalker/SadTalker.git

# Change to the cloned directory to pull large model files.
WORKDIR /app/SadTalker

# Install git-lfs and pull the large model files.
RUN git lfs install && git lfs pull

# The app.py file expects the SadTalker directory to be next to it.
WORKDIR /app

# Copy your application file into the container.
COPY app.py .

# Define the command to run when the container starts.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
