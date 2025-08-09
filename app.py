import asyncio
import base64
import json
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, Tuple
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import requests
import torch
import cv2
import numpy as np
from PIL import Image
import io
import subprocess
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration ---
app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)
video_tasks: Dict[str, dict] = {}

# Model paths and configuration
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
TEMP_DIR = os.environ.get("TEMP_DIR", "/app/temp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Model Management Classes ---
class SadTalkerModel:
    """SadTalker model for high-quality offline video generation"""
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.device = DEVICE
        self.models_loaded = False
        self.model_components = {}

    @staticmethod
    async def get_instance():
        async with SadTalkerModel._lock:
            if SadTalkerModel._instance is None:
                logger.info("Loading SadTalker models for offline generation...")
                instance = SadTalkerModel()
                await instance.load_models()
                SadTalkerModel._instance = instance
            return SadTalkerModel._instance

    async def load_models(self):
        """Load SadTalker models"""
        try:
            logger.info(f"Loading SadTalker models on {self.device}")
            
            # Simulate SadTalker model loading
            # In production, replace with actual SadTalker model loading
            await asyncio.sleep(3)  # Simulate loading time
            
            self.models_loaded = True
            logger.info("SadTalker models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SadTalker models: {e}")
            raise

    def generate_video(self, image_path: str, audio_path: str, output_path: str) -> str:
        """Generate high-quality video using SadTalker"""
        try:
            logger.info(f"SadTalker generation: {image_path} + {audio_path} -> {output_path}")
            
            # Placeholder implementation - replace with actual SadTalker
            self._create_high_quality_video(image_path, audio_path, output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"SadTalker generation error: {e}")
            raise

    def _create_high_quality_video(self, image_path: str, audio_path: str, output_path: str):
        """Placeholder for SadTalker - replace with actual implementation"""
        try:
            # Get audio duration
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ], capture_output=True, text=True)
            
            duration = float(result.stdout.strip()) if result.stdout.strip() else 5.0
            
            # Create high-quality video with the image
            subprocess.run([
                'ffmpeg', '-loop', '1', '-i', image_path,
                '-i', audio_path,
                '-c:v', 'libx264', '-tune', 'stillimage',
                '-c:a', 'aac', '-b:a', '192k',
                '-pix_fmt', 'yuv420p', '-shortest',
                '-t', str(duration),
                output_path, '-y'
            ], check=True)
            
            logger.info(f"High-quality video created: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating high-quality video: {e}")
            raise


class RealtimeVideoModel:
    """Wav2Lip + FOMM model for real-time video generation"""
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.device = DEVICE
        self.models_loaded = False
        self.wav2lip_model = None
        self.fomm_model = None
        self.preprocessed_avatars = {}  # Cache for preprocessed avatar data

    @staticmethod
    async def get_instance():
        async with RealtimeVideoModel._lock:
            if RealtimeVideoModel._instance is None:
                logger.info("Loading real-time video models (Wav2Lip + FOMM)...")
                instance = RealtimeVideoModel()
                await instance.load_models()
                RealtimeVideoModel._instance = instance
            return RealtimeVideoModel._instance

    async def load_models(self):
        """Load Wav2Lip and FOMM models"""
        try:
            logger.info(f"Loading real-time models on {self.device}")
            
            # Simulate model loading
            await asyncio.sleep(2)
            
            self.models_loaded = True
            logger.info("Real-time video models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading real-time models: {e}")
            raise

    async def preprocess_avatar(self, avatar_id: str, image_path: str) -> dict:
        """Preprocess avatar image for real-time generation"""
        try:
            if avatar_id in self.preprocessed_avatars:
                return self.preprocessed_avatars[avatar_id]

            logger.info(f"Preprocessing avatar {avatar_id}")
            
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize to standard size for real-time processing
            img_resized = cv2.resize(img, (256, 256))
            
            # Store preprocessed data
            preprocessed_data = {
                "original_image": img,
                "resized_image": img_resized,
                "image_path": image_path,
                "processed_at": time.time()
            }
            
            self.preprocessed_avatars[avatar_id] = preprocessed_data
            logger.info(f"Avatar {avatar_id} preprocessed successfully")
            
            return preprocessed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing avatar {avatar_id}: {e}")
            raise

    def generate_frame_from_audio(self, avatar_data: dict, audio_chunk: bytes) -> Optional[bytes]:
        """Generate a single video frame from audio chunk"""
        try:
            # This is a simplified implementation
            # In production, implement actual Wav2Lip + FOMM pipeline
            
            # Get the preprocessed image
            img = avatar_data["resized_image"].copy()
            
            # Simulate lip movement based on audio intensity
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            if len(audio_array) > 0:
                audio_intensity = np.abs(audio_array).mean() / 32767.0
                
                # Simple mouth animation (replace with Wav2Lip)
                mouth_opening = int(audio_intensity * 10)
                cv2.ellipse(img, (128, 180), (15, mouth_opening), 0, 0, 180, (0, 0, 0), -1)
            
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            return None


# --- Request Models ---
class VideoRequest(BaseModel):
    image_url: str
    audio_url: str
    quality: str = "high"  # "high" for SadTalker, "realtime" for Wav2Lip+FOMM

class RealtimeInitRequest(BaseModel):
    avatar_id: str
    image_url: str

# --- HTTP Endpoints ---
@app.post("/generate-video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Generate video - uses SadTalker for high quality or Wav2Lip for fast generation"""
    logger.info(f"Video generation request - Quality: {request.quality}")
    
    task_id = os.urandom(16).hex()
    output_dir = tempfile.mkdtemp()
    
    video_tasks[task_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
        "output_dir": output_dir,
        "quality": request.quality
    }
    
    try:
        if request.quality == "high":
            # Use SadTalker for high-quality generation
            background_tasks.add_task(
                _run_sadtalker_generation,
                task_id, request.image_url, request.audio_url, output_dir
            )
        else:
            # Use Wav2Lip for faster generation
            background_tasks.add_task(
                _run_realtime_generation,
                task_id, request.image_url, request.audio_url, output_dir
            )
        
        return JSONResponse({
            "task_id": task_id, 
            "status": "processing",
            "quality": request.quality
        })
        
    except Exception as e:
        logger.error(f"Error initiating video generation: {e}")
        shutil.rmtree(output_dir, ignore_errors=True)
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video-status/{task_id}")
async def get_video_status(task_id: str):
    """Check video generation status"""
    task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")
    
    status = task.get("status")
    if status == "completed":
        return FileResponse(
            task["output_path"],
            media_type="video/mp4",
            filename=f"generated_video_{task['quality']}.mp4"
        )
    elif status == "failed":
        raise HTTPException(status_code=500, detail=task["error"])
    else:
        return JSONResponse({
            "task_id": task_id,
            "status": status,
            "quality": task.get("quality", "unknown")
        })

@app.post("/preprocess-avatar")
async def preprocess_avatar_endpoint(request: RealtimeInitRequest):
    """Preprocess avatar for real-time streaming"""
    try:
        model = await RealtimeVideoModel.get_instance()
        
        # Download image temporarily
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "avatar.jpg")
        
        await _download_file(request.image_url, image_path)
        
        # Preprocess avatar
        await model.preprocess_avatar(request.avatar_id, image_path)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return JSONResponse({
            "status": "success",
            "message": f"Avatar {request.avatar_id} preprocessed for real-time streaming"
        })
        
    except Exception as e:
        logger.error(f"Error preprocessing avatar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/real-time-stream/{avatar_id}")
async def real_time_stream(websocket: WebSocket, avatar_id: str):
    """WebSocket for real-time video streaming"""
    await websocket.accept()
    logger.info(f"Real-time stream started for avatar {avatar_id}")
    
    try:
        model = await RealtimeVideoModel.get_instance()
        
        # Check if avatar is preprocessed
        if avatar_id not in model.preprocessed_avatars:
            await websocket.send_json({
                "type": "error",
                "message": f"Avatar {avatar_id} not preprocessed. Call /preprocess-avatar first."
            })
            return
        
        avatar_data = model.preprocessed_avatars[avatar_id]
        
        # Send ready signal
        await websocket.send_json({
            "type": "ready",
            "message": "Real-time video service ready"
        })
        
        frame_count = 0
        audio_buffer = b""
        
        while True:
            try:
                data = await websocket.receive()
                
                if "bytes" in data:
                    # Received audio chunk
                    audio_chunk = data["bytes"]
                    audio_buffer += audio_chunk
                    
                    # Process when we have enough audio (adjust threshold)
                    if len(audio_buffer) >= 4096:  # 4KB threshold
                        # Generate video frame
                        frame_data = model.generate_frame_from_audio(avatar_data, audio_buffer)
                        
                        if frame_data:
                            await websocket.send_bytes(frame_data)
                            frame_count += 1
                        
                        # Clear buffer
                        audio_buffer = b""
                
                elif "text" in data:
                    # Handle control messages
                    try:
                        message = json.loads(data["text"])
                        if message.get("type") == "stop_speaking":
                            await websocket.send_json({"type": "speech_end"})
                        elif message.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
                        
            except Exception as e:
                logger.error(f"Error in real-time stream: {e}")
                break
                
    except Exception as e:
        logger.error(f"Real-time stream error for avatar {avatar_id}: {e}")
    finally:
        logger.info(f"Real-time stream ended for avatar {avatar_id}")

# --- Background Task Functions ---
async def _run_sadtalker_generation(task_id: str, image_url: str, audio_url: str, output_dir: str):
    """Run SadTalker generation in background"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        model = await SadTalkerModel.get_instance()
        
        # Download files
        image_path = os.path.join(temp_dir, "input.jpg")
        audio_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(output_dir, "output.mp4")
        
        await _download_file(image_url, image_path)
        await _download_file(audio_url, audio_path)
        
        # Generate high-quality video
        model.generate_video(image_path, audio_path, output_path)
        
        video_tasks[task_id]["status"] = "completed"
        video_tasks[task_id]["output_path"] = output_path
        
        logger.info(f"SadTalker generation completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"SadTalker generation failed for task {task_id}: {e}")
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def _run_realtime_generation(task_id: str, image_url: str, audio_url: str, output_dir: str):
    """Run Wav2Lip generation in background (faster)"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        model = await RealtimeVideoModel.get_instance()
        
        # Download files
        image_path = os.path.join(temp_dir, "input.jpg")
        audio_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(output_dir, "output.mp4")
        
        await _download_file(image_url, image_path)
        await _download_file(audio_url, audio_path)
        
        # Generate video using real-time model (faster but lower quality)
        await _generate_realtime_video(image_path, audio_path, output_path)
        
        video_tasks[task_id]["status"] = "completed"
        video_tasks[task_id]["output_path"] = output_path
        
        logger.info(f"Real-time generation completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Real-time generation failed for task {task_id}: {e}")
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def _generate_realtime_video(image_path: str, audio_path: str, output_path: str):
    """Generate video using real-time model"""
    try:
        # Simplified implementation - replace with actual Wav2Lip
        subprocess.run([
            'ffmpeg', '-loop', '1', '-i', image_path,
            '-i', audio_path,
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac', '-b:a', '128k',
            '-pix_fmt', 'yuv420p', '-shortest',
            output_path, '-y'
        ], check=True)
        
        logger.info(f"Real-time video generated: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating real-time video: {e}")
        raise

# --- Helper Functions ---
async def _download_file(url: str, local_path: str):
    """Download file from URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Downloaded {url} to {local_path}")
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "hybrid-video-generation",
        "device": DEVICE,
        "models": {
            "sadtalker_loaded": SadTalkerModel._instance is not None,
            "realtime_loaded": RealtimeVideoModel._instance is not None
        }
    }

@app.get("/models/status")
async def models_status():
    """Check model loading status"""
    return {
        "sadtalker": {
            "loaded": SadTalkerModel._instance is not None,
            "loading": SadTalkerModel._instance is not None and not SadTalkerModel._instance.models_loaded
        },
        "realtime": {
            "loaded": RealtimeVideoModel._instance is not None,
            "loading": RealtimeVideoModel._instance is not None and not RealtimeVideoModel._instance.models_loaded
        },
        "device": DEVICE
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
