# Start with a base image that has Python and CUDA, as this is a deep learning model.
# Using a specific version ensures reproducibility.
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Set DEBIAN_FRONTEND to noninteractive to prevent prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, git-lfs, and other necessary dependencies.
# We include git-lfs here to handle the large model files.
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

# Clone the SadTalker repository. This will pull the source code.
RUN git clone https://github.com/OpenTalker/SadTalker.git /app/SadTalker

# Change to the cloned directory to run subsequent commands.
WORKDIR /app/SadTalker

# Install git-lfs and pull the large model files.
RUN git lfs install && git lfs pull

# Install all the Python dependencies from the repository's requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt

# This is the new line to create the 'models' directory.
RUN mkdir -p models/

# Move the model files into the newly created 'models' folder.
RUN mv audio2exp300-model.pth auido2pose_00140-model.pth wav2lip.pth models/

# Define the command to run when the container starts.
# You will likely need to adjust this command based on what you want to do.
# For example, to run the gradio web UI, you would use a command like:
# CMD ["python3", "app_sadtalker.py"]
# I'm providing a placeholder command for now.
CMD ["python3", "-c", "print('Dockerfile setup complete. Please add your command to run the SadTalker application.')"]
