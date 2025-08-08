# Start with a base image that has Python and the correct CUDA version.
FROM pytorch/pytorch:1.13.1-cuda11.7-cudnn8-runtime

# Set the working directory inside the container.
WORKDIR /app

# Copy the requirements file and install dependencies.
COPY requirements.txt .

# Install all the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

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
