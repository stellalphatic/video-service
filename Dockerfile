# Start with a base image that has Python and CUDA, as this is a deep learning model.
# Using a specific version ensures reproducibility.
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install Python and other necessary dependencies.
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    git \
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

# Copy the model files from the host to the container.
# These files are crucial for the model to work.
# Note: The user mentioned `audio2coeff.pth` but it's not in the screenshot.
# I'm including `auido2pose_00140-model.pth` and `audio2exp300-model.pth` as they seem related to the same function.
# Please ensure these files are in the same directory as the Dockerfile on your local machine.
COPY ./auido2pose_00140-model.pth /app/
COPY ./audio2exp300-model.pth /app/
COPY ./epoch_20.pth /app/
COPY ./facevid2vid_00189-model.pth.tar /app/
COPY ./mapping_00109-model.pth.tar /app/
COPY ./mapping_00229-model.pth.tar /app/
COPY ./shape_predictor_68_face_landmarks.dat /app/
COPY ./wav2lip.pth /app/

# Install the Python dependencies. Since we don't have a requirements.txt,
# we'll install common libraries for this type of model.
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html \
    && pip install opencv-python==4.5.5.64 \
    && pip install scipy==1.7.3 \
    && pip install numpy==1.22.4 \
    && pip install pillow==9.1.1 \
    && pip install dlib==19.24.0 \
    && pip install face-alignment==1.3.5

# Install the SadTalker repository itself.
RUN git clone https://github.com/OpenTalker/SadTalker.git /app/SadTalker \
    && mv /app/SadTalker/* /app/ \
    && rm -rf /app/SadTalker

# Define the command to run when the container starts.
# This is a placeholder and will need to be replaced with the actual command
# to run the SadTalker application, e.g., 'python app.py'.
# You may need to create a python file to wrap the SadTalker main functionality.
CMD ["python", "-c", "print('Dockerfile setup complete. Please add your command here to run the application.')"]
