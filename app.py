# app.py

import asyncio
import base64
import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Union

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

# Import SadTalker dependencies
# NOTE: This assumes the SadTalker repository is cloned locally and its path
# is correctly added to the system path.
import sys
sys.path.append('./SadTalker')
from src.utils.cropper import Cropper
from src.facerender.animate import AnimateFromCoeff
from src.test_audio2coeff import Audio2Coeff
from src.utils.sadtalker_utils import sad_talker_initialization, get_video_from_frames, get_facerender_model
from src.utils.face_parsing.face_parsing import BiSeNet
from src.test_audio2coeff import get_audio_feature_from_audio

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State and Configuration ---
app = FastAPI()

# Executor for background tasks to avoid blocking the main event loop
executor = ThreadPoolExecutor(max_workers=4)

# In-memory storage for video generation tasks
# NOTE: In a production environment, this should be a persistent store like Redis or a database.
video_tasks: Dict[str, dict] = {}

# Paths to SadTalker checkpoints
# NOTE: You MUST configure these paths to your SadTalker model files.
CHECKPOINTS_DIR = os.path.join('./SadTalker/checkpoints')
CONFIG_PATH = os.path.join(CHECKPOINTS_DIR, 'config/facerender.yaml')
INFERENCE_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'facerender.pth')
FACE_PARSER_PATH = os.path.join(CHECKPOINTS_DIR, 'face_parsing.pth')
AUDIO2COEFF_PATH = os.path.join(CHECKPOINTS_DIR, 'audio2coeff.pth')

# --- Model Singleton for GPU loading ---
class SadTalkerModel:
    """
    A singleton class to ensure SadTalker models are loaded only once and
    can be accessed by all endpoints. This is crucial for GPU memory management.
    """
    _instance = None
    _is_loading = False
    _lock = asyncio.Lock()

    @staticmethod
    async def get_instance():
        """
        Retrieves the singleton instance, loading the models if necessary.
        Uses a lock to prevent multiple threads from trying to load at the same time.
        """
        async with SadTalkerModel._lock:
            if SadTalkerModel._instance is None:
                if SadTalkerModel._is_loading:
                    raise RuntimeError("SadTalker is already loading. Please wait.")
                SadTalkerModel._is_loading = True
                try:
                    logger.info("Initializing SadTalker models...")
                    # Initialize the main SadTalker components
                    audio_to_coeff_model = Audio2Coeff(AUDIO2COEFF_PATH)
                    animate_model = get_facerender_model(CONFIG_PATH, INFERENCE_MODEL_PATH)
                    face_parser = BiSeNet(FACE_PARSER_PATH)
                    cropper = Cropper()
                    
                    SadTalkerModel._instance = {
                        "audio_to_coeff_model": audio_to_coeff_model,
                        "animate_model": animate_model,
                        "face_parser": face_parser,
                        "cropper": cropper
                    }
                    logger.info("SadTalker models initialized successfully.")
                except Exception as e:
                    logger.error(f"Failed to initialize SadTalker models: {e}", exc_info=True)
                    SadTalkerModel._instance = None
                    raise HTTPException(status_code=500, detail=f"Failed to load SadTalker models: {e}")
                finally:
                    SadTalkerModel._is_loading = False
            return SadTalkerModel._instance

# --- Endpoints ---

class VideoRequest(BaseModel):
    """
    Pydantic model for the HTTP video generation request.
    """
    image_url: str
    audio_url: str

@app.post("/generate-video")
async def generate_video(
    request: VideoRequest,
    background_tasks: BackgroundTasks
):
    """
    HTTP endpoint for generating a full talking-head video from a single image and audio URL.
    This runs SadTalker in a background task to prevent timeouts. Instead of returning
    the file directly, it returns a task ID which the client can use to poll for the result.
    """
    logger.info("Received request for full video generation.")

    # Create a unique task ID
    task_id = os.urandom(16).hex()
    output_dir = tempfile.mkdtemp()
    
    video_tasks[task_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
        "output_dir": output_dir,
    }

    try:
        # Run SadTalker in a background task and return the path
        background_tasks.add_task(
            _run_sadtalker_and_cleanup,
            task_id,
            request.image_url,
            request.audio_url,
            output_dir
        )

        return JSONResponse({"task_id": task_id, "status": "processing"})

    except Exception as e:
        logger.error(f"Error during video generation request: {e}", exc_info=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Failed to initiate video generation: {e}")

@app.get("/video-status/{task_id}")
async def get_video_status(task_id: str):
    """
    Endpoint for a client to check the status of a video generation task.
    """
    task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found.")

    status = task.get("status")
    if status == "completed":
        # Return the video file
        return FileResponse(task["output_path"], media_type="video/mp4", filename="generated_video.mp4")
    elif status == "failed":
        # Return an error message
        raise HTTPException(status_code=500, detail=task["error"])
    else:
        # Return the current status
        return JSONResponse({"task_id": task_id, "status": status})

@app.websocket("/real-time-stream/{avatar_id}")
async def real_time_stream(websocket: WebSocket, avatar_id: str):
    """
    WebSocket endpoint for real-time video streaming.
    Receives audio chunks from the voice service and sends back video frames.
    """
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for avatar {avatar_id}.")

    # Per-connection state
    connection_state = {
        "audio_buffer": [],
        "preprocessed_avatar_data": None,
    }
    
    try:
        models = await SadTalkerModel.get_instance()
        
        # NOTE: In a real-world scenario, the avatar data would be pre-cached
        # and loaded from a permanent store. For this example, we'll
        # simulate the preprocessing here.
        # This part should be triggered by an 'init' message from the client.
        static_image_path = f'./avatars/{avatar_id}/base_image.png'
        if not os.path.exists(static_image_path):
            await websocket.close(code=1008, reason=f"Avatar image not found for {avatar_id}")
            return

        # Preprocess the avatar image once
        connection_state["preprocessed_avatar_data"] = _preprocess_avatar(
            models, static_image_path
        )

        while True:
            # Receive audio data from the voice service
            audio_chunk = await websocket.receive_bytes()
            
            if not audio_chunk:
                continue
                
            # Append audio chunk to the buffer
            connection_state["audio_buffer"].append(audio_chunk)
            
            # Simple buffering logic: wait for a certain amount of audio before processing
            # This is a critical parameter to tune for latency vs. smoothness.
            if len(connection_state["audio_buffer"]) * len(audio_chunk) > 4000: # Example: 4KB of audio
                full_audio_chunk = b''.join(connection_state["audio_buffer"])
                connection_state["audio_buffer"] = []
                
                # Get SadTalker coefficients from the audio chunk
                coeff = _get_audio_features(models, full_audio_chunk)
                
                # Generate video frames from the coefficients
                frames = _generate_frames_from_features(models, connection_state["preprocessed_avatar_data"], coeff)
                
                # Stream the frames back to the client
                for frame in frames:
                    await websocket.send_bytes(frame)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for avatar {avatar_id}.")
    except Exception as e:
        logger.error(f"WebSocket connection error for avatar {avatar_id}: {e}", exc_info=True)
        # Clean up any temporary files if needed
    finally:
        logger.info(f"WebSocket connection closed for avatar {avatar_id}.")


def _run_sadtalker_and_cleanup(task_id, image_url, audio_url, output_dir):
    """
    Runs the full SadTalker pipeline in a background thread and cleans up.
    This replaces the user's placeholder function with a real workflow.
    """
    temp_dir = tempfile.mkdtemp()
    
    try:
        models = SadTalkerModel.get_instance()
        
        # Download image and audio files
        image_path = os.path.join(temp_dir, "input_image.png")
        audio_path = os.path.join(temp_dir, "input_audio.wav")
        output_video_path = os.path.join(output_dir, "output.mp4")

        # Use requests for blocking downloads in this background task
        image_response = requests.get(image_url)
        audio_response = requests.get(audio_url)
        
        image_response.raise_for_status()
        audio_response.raise_for_status()

        with open(image_path, "wb") as f: f.write(image_response.content)
        with open(audio_path, "wb") as f: f.write(audio_response.content)
        
        # --- SadTalker Full Pipeline ---
        # 1. Preprocess the image (crop and resize)
        image_data = models["cropper"].crop_image(image_path)
        
        # 2. Extract audio features and coefficients
        audio_coeff, audio_mel_features = get_audio_feature_from_audio(
            audio_path, models["audio_to_coeff_model"]
        )

        # 3. Animate the face from the coefficients
        video_frames = models["animate_model"].animate(
            image_data, audio_coeff, audio_mel_features
        )
        
        # 4. Save the final video
        get_video_from_frames(video_frames, audio_path, output_video_path)
        
        logger.info(f"Video generated successfully at {output_video_path}")
        
        # Update the task state
        video_tasks[task_id]["status"] = "completed"
        video_tasks[task_id]["output_path"] = output_video_path
        
    except Exception as e:
        logger.error(f"Error running SadTalker in background task: {e}", exc_info=True)
        # Update the task state to failed
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = f"SadTalker processing failed: {e}"
        
    finally:
        # Clean up temporary download directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def _preprocess_avatar(models: dict, image_path: str) -> dict:
    """
    Preprocesses the avatar image once to be used in the real-time stream.
    Returns the pre-computed data required by the animation model.
    """
    # This is a conceptual implementation. Real-time SadTalker may require
    # more or different pre-computed data.
    cropped_image_data = models["cropper"].crop_image(image_path)
    # The actual animation model might need more pre-processing here
    # For now, we'll just return the cropped image data.
    return {"cropped_image_data": cropped_image_data}

def _get_audio_features(models: dict, audio_chunk: bytes) -> Any:
    """
    Processes an audio chunk and returns features for animation.
    This would be a custom implementation for streaming, but we can
    simulate it by treating the chunk as a small file.
    """
    # Write the chunk to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio_chunk)
        temp_audio_path = temp_audio.name

    try:
        # Use the SadTalker function to process the audio.
        # This is a simplification; a true streaming solution would process
        # the raw bytes in memory.
        coeff, _ = get_audio_feature_from_audio(
            temp_audio_path, models["audio_to_coeff_model"]
        )
        return coeff
    finally:
        os.remove(temp_audio_path)

def _generate_frames_from_features(models: dict, preprocessed_data: dict, audio_features: Any) -> list[bytes]:
    """
    Generates video frames from audio features using the pre-processed avatar data.
    """
    # This is highly conceptual, as SadTalker's `animate` function is not
    # designed for a frame-by-frame loop. This represents where you would
    # hook into the model's core logic to get individual frames.
    
    # We will simulate a small animation run here for a single audio chunk.
    # In a real-time system, this would be a tight loop that generates one frame at a time.
    generated_frames = models["animate_model"].animate(
        preprocessed_data["cropped_image_data"], audio_features, None
    )
    
    # In a real system, you would get a list of image bytes (or a stream) from this.
    # We'll just return a mock list of bytes for now.
    return [b'fake_speaking_frame'] * len(generated_frames)

if __name__ == "__main__":
    # Load SadTalker model at startup
    asyncio.run(SadTalkerModel.get_instance())
    uvicorn.run(app, host="0.0.0.0", port=8000)

