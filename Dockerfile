# Use a base image with CUDA 11.7, which is compatible with the PyTorch version you need and the L4 GPU.
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive.
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and other necessary dependencies.
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

# Set Python 3.8 as the default.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory.
WORKDIR /app

# Copy the requirements.txt file.
COPY requirements.txt .

# Install Python dependencies, including the specific PyTorch version.
RUN pip install --no-cache-dir -r requirements.txt

# Clone the SadTalker repository.
RUN git clone https://github.com/OpenTalker/SadTalker.git

# Change to the cloned directory to pull large model files.
WORKDIR /app/SadTalker

# Install git-lfs and pull the large model files.
RUN git lfs install && git lfs pull

# Copy your application file to the parent directory.
WORKDIR /app
COPY app.py .

# Define the command to run the application.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
