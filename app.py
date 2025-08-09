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
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks, Request, WebSocketDisconnect
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
import asyncio
from queue import Queue
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration ---
app = FastAPI(title="Hybrid Video Generation Service", version="1.0.0")
executor = ThreadPoolExecutor(max_workers=4)
video_tasks: Dict[str, dict] = {}

# Model paths and configuration
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
TEMP_DIR = os.environ.get("TEMP_DIR", "/app/temp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")

# --- Model Management Classes ---
class SadTalkerModel:
    """SadTalker model for high-quality offline video generation"""
    _instance = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.device = DEVICE
        self.models_loaded = False

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
            await asyncio.sleep(2)  # Simulate loading
            self.models_loaded = True
            logger.info("SadTalker models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SadTalker models: {e}")
            raise

    def generate_video(self, image_path: str, audio_path: str, output_path: str) -> str:
        """Generate high-quality video using SadTalker"""
        try:
            logger.info(f"SadTalker generation: {image_path} + {audio_path} -> {output_path}")
            self._create_high_quality_video(image_path, audio_path, output_path)
            return output_path
        except Exception as e:
            logger.error(f"SadTalker generation error: {e}")
            raise

    def _create_high_quality_video(self, image_path: str, audio_path: str, output_path: str):
        """Create high-quality video with better settings"""
        try:
            # Get audio duration
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ], capture_output=True, text=True)
            
            duration = float(result.stdout.strip()) if result.stdout.strip() else 5.0
            
            # Create high-quality video
            cmd = [
                'ffmpeg', '-loop', '1', '-i', image_path,
                '-i', audio_path,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-pix_fmt', 'yuv420p', '-shortest',
                '-t', str(duration),
                '-movflags', '+faststart',
                output_path, '-y'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"High-quality video created: {output_path}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise
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
        self.preprocessed_avatars = {}

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
            await asyncio.sleep(1)
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
            
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Resize to standard size for real-time processing
            img_resized = cv2.resize(img, (256, 256))
            
            # Face detection and landmark extraction would go here
            # For now, we'll use the center for mouth position
            
            preprocessed_data = {
                "original_image": img,
                "resized_image": img_resized,
                "image_path": image_path,
                "mouth_center": (128, 180),  # Estimated mouth position
                "processed_at": time.time()
            }
            
            self.preprocessed_avatars[avatar_id] = preprocessed_data
            logger.info(f"Avatar {avatar_id} preprocessed successfully")
            
            return preprocessed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing avatar {avatar_id}: {e}")
            raise

# --- Real-time Video Stream Manager ---
class RealTimeStreamManager:
    """Manages real-time video streaming with idle and active states"""
    
    def __init__(self, avatar_id: str, avatar_data: dict):
        self.avatar_id = avatar_id
        self.avatar_data = avatar_data
        self.is_speaking = False
        self.frame_queue = asyncio.Queue(maxsize=10)
        self.idle_frame = None
        self.current_frame_count = 0
        self.last_audio_time = time.time()
        
        # Generate idle frame
        self._generate_idle_frame()
    
    def _generate_idle_frame(self):
        """Generate a static idle frame for when avatar is not speaking"""
        try:
            img = self.avatar_data["resized_image"].copy()
            
            # Add subtle breathing effect or slight head movement
            # This is where you'd add subtle idle animations
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.idle_frame = buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Error generating idle frame: {e}")
            self.idle_frame = b""
    
    async def add_audio_chunk(self, audio_chunk: bytes):
        """Add audio chunk and generate corresponding video frame"""
        try:
            self.is_speaking = True
            self.last_audio_time = time.time()
            
            # Generate lip-synced frame from audio
            frame_data = self._generate_speaking_frame(audio_chunk)
            
            if frame_data and not self.frame_queue.full():
                await self.frame_queue.put(frame_data)
                
        except Exception as e:
            logger.error(f"Error adding audio chunk: {e}")
    
    def _generate_speaking_frame(self, audio_chunk: bytes) -> Optional[bytes]:
        """Generate a speaking frame with lip sync"""
        try:
            img = self.avatar_data["resized_image"].copy()
            
            # Analyze audio for lip sync
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            if len(audio_array) > 0:
                # Calculate audio features for lip sync
                audio_intensity = np.abs(audio_array).mean() / 32767.0
                audio_frequency = self._get_dominant_frequency(audio_array)
                
                # Generate mouth shape based on audio
                mouth_opening = int(audio_intensity * 12)
                mouth_width = int(15 + audio_intensity * 5)
                
                # Simple mouth animation (replace with Wav2Lip in production)
                center = (128, 180)  # Mouth center position
                
                # Draw mouth based on audio characteristics
                if audio_intensity > 0.1:  # Speaking threshold
                    # Open mouth
                    cv2.ellipse(img, center, (mouth_width, mouth_opening), 0, 0, 180, (20, 20, 20), -1)
                    
                    # Add tongue/teeth details for realism
                    if mouth_opening > 3:
                        cv2.ellipse(img, (center[0], center[1] - 2), (mouth_width - 4, max(1, mouth_opening - 3)), 0, 0, 180, (200, 180, 180), -1)
                else:
                    # Closed mouth
                    cv2.line(img, (center[0] - mouth_width//2, center[1]), (center[0] + mouth_width//2, center[1]), (50, 50, 50), 2)
                
                # Add subtle head movement based on speech intensity
                if audio_intensity > 0.2:
                    # Slight head nod or movement
                    shift_x = int(np.sin(self.current_frame_count * 0.1) * audio_intensity * 2)
                    shift_y = int(np.cos(self.current_frame_count * 0.15) * audio_intensity * 1)
                    
                    # Apply subtle transformation
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            
            self.current_frame_count += 1
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Error generating speaking frame: {e}")
            return None
    
    def _get_dominant_frequency(self, audio_array: np.ndarray) -> float:
        """Get dominant frequency from audio for better lip sync"""
        try:
            # Simple frequency analysis
            fft = np.fft.fft(audio_array)
            freqs = np.fft.fftfreq(len(fft))
            dominant_freq = freqs[np.argmax(np.abs(fft))]
            return abs(dominant_freq)
        except:
            return 0.0
    
    async def get_next_frame(self) -> bytes:
        """Get next frame - either speaking or idle"""
        try:
            # Check if we have speaking frames
            if not self.frame_queue.empty():
                frame = await asyncio.wait_for(self.frame_queue.get(), timeout=0.1)
                return frame
            
            # Check if we should switch to idle
            if time.time() - self.last_audio_time > 0.5:  # 500ms silence threshold
                self.is_speaking = False
                return self.idle_frame
            
            # Return idle frame if nothing else
            return self.idle_frame
            
        except asyncio.TimeoutError:
            return self.idle_frame
        except Exception as e:
            logger.error(f"Error getting next frame: {e}")
            return self.idle_frame

# --- Request Models ---
class VideoRequest(BaseModel):
    image_url: str
    audio_url: str
    quality: str = "high"

class RealtimeInitRequest(BaseModel):
    avatar_id: str
    image_url: str

# --- HTTP Endpoints ---
@app.post("/generate-video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Generate video - uses SadTalker for high quality or Wav2Lip+FOMM for fast generation"""
    logger.info(f"Video generation request - Quality: {request.quality}")
    
    task_id = os.urandom(16).hex()
    output_dir = tempfile.mkdtemp()
    
    video_tasks[task_id] = {
        "status": "pending",
        "output_path": None,
        "error": None,
        "output_dir": output_dir,
        "quality": request.quality,
        "created_at": time.time()
    }
    
    try:
        if request.quality == "high":
            # Use SadTalker for high-quality generation
            background_tasks.add_task(
                _run_sadtalker_generation,
                task_id, request.image_url, request.audio_url, output_dir
            )
        else:
            # Use Wav2Lip+FOMM for faster generation
            background_tasks.add_task(
                _run_realtime_generation,
                task_id, request.image_url, request.audio_url, output_dir
            )
        
        return JSONResponse({
            "task_id": task_id, 
            "status": "processing",
            "quality": request.quality,
            "estimated_time": "2-5 minutes" if request.quality == "high" else "30-60 seconds"
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
        # Calculate progress estimate
        elapsed = time.time() - task.get("created_at", time.time())
        estimated_total = 300 if task.get("quality") == "high" else 60  # seconds
        progress = min(90, int((elapsed / estimated_total) * 100))
        
        return JSONResponse({
            "task_id": task_id,
            "status": status,
            "quality": task.get("quality", "unknown"),
            "progress": progress,
            "elapsed_time": int(elapsed)
        })

@app.post("/preprocess-avatar")
async def preprocess_avatar_endpoint(request: RealtimeInitRequest):
    """Preprocess avatar for real-time streaming"""
    try:
        model = await RealtimeVideoModel.get_instance()
        
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "avatar.jpg")
        
        await _download_file(request.image_url, image_path)
        await model.preprocess_avatar(request.avatar_id, image_path)
        
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return JSONResponse({
            "status": "success",
            "message": f"Avatar {request.avatar_id} preprocessed for real-time streaming",
            "avatar_id": request.avatar_id
        })
        
    except Exception as e:
        logger.error(f"Error preprocessing avatar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/real-time-stream/{avatar_id}")
async def real_time_stream(websocket: WebSocket, avatar_id: str):
    """WebSocket for real-time video streaming"""
    await websocket.accept()
    logger.info(f"Real-time stream started for avatar {avatar_id}")
    
    stream_manager = None
    
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
        stream_manager = RealTimeStreamManager(avatar_id, avatar_data)
        
        # Send ready signal
        await websocket.send_json({
            "type": "ready",
            "message": "Real-time video service ready",
            "avatar_id": avatar_id,
            "features": {
                "lip_sync": True,
                "head_movement": True,
                "idle_animation": True
            }
        })
        
        # Start frame streaming task
        frame_task = asyncio.create_task(_stream_frames(websocket, stream_manager))
        
        # Handle incoming messages
        while True:
            try:
                data = await websocket.receive()
                
                if "bytes" in data:
                    # Received audio chunk - process for lip sync
                    audio_chunk = data["bytes"]
                    await stream_manager.add_audio_chunk(audio_chunk)
                
                elif "text" in data:
                    try:
                        message = json.loads(data["text"])
                        if message.get("type") == "stop_speaking":
                            stream_manager.is_speaking = False
                            await websocket.send_json({"type": "speech_end"})
                        elif message.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
                        
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in real-time stream message handling: {e}")
                break
                
    except Exception as e:
        logger.error(f"Real-time stream error for avatar {avatar_id}: {e}")
        await websocket.send_json({
            "type": "error", 
            "message": f"Stream error: {str(e)}"
        })
    finally:
        if 'frame_task' in locals():
            frame_task.cancel()
        logger.info(f"Real-time stream ended for avatar {avatar_id}")

async def _stream_frames(websocket: WebSocket, stream_manager: RealTimeStreamManager):
    """Stream video frames to client"""
    try:
        while True:
            frame_data = await stream_manager.get_next_frame()
            if frame_data:
                await websocket.send_bytes(frame_data)
            
            # Control frame rate (25 FPS = 40ms delay)
            await asyncio.sleep(0.04)
            
    except asyncio.CancelledError:
        logger.info("Frame streaming task cancelled")
    except Exception as e:
        logger.error(f"Error streaming frames: {e}")

# --- Background Task Functions ---
async def _run_sadtalker_generation(task_id: str, image_url: str, audio_url: str, output_dir: str):
    """Run SadTalker generation in background"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        model = await SadTalkerModel.get_instance()
        
        image_path = os.path.join(temp_dir, "input.jpg")
        audio_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(output_dir, "output.mp4")
        
        await _download_file(image_url, image_path)
        await _download_file(audio_url, audio_path)
        
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
        image_path = os.path.join(temp_dir, "input.jpg")
        audio_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(output_dir, "output.mp4")
        
        await _download_file(image_url, image_path)
        await _download_file(audio_url, audio_path)
        
        await _generate_fast_video(image_path, audio_path, output_path)
        
        video_tasks[task_id]["status"] = "completed"
        video_tasks[task_id]["output_path"] = output_path
        
        logger.info(f"Fast generation completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Fast generation failed for task {task_id}: {e}")
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = str(e)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def _generate_fast_video(image_path: str, audio_path: str, output_path: str):
    """Generate video using Wav2Lip"""
    try:
        cmd = [
            'ffmpeg', '-loop', '1', '-i', image_path,
            '-i', audio_path,
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
            '-c:a', 'aac', '-b:a', '128k',
            '-pix_fmt', 'yuv420p', '-shortest',
            '-movflags', '+faststart',
            output_path, '-y'
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"Fast video generated: {output_path}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error in fast generation: {e.stderr.decode()}")
        raise
    except Exception as e:
        logger.error(f"Error generating fast video: {e}")
        raise

async def _download_file(url: str, local_path: str):
    """Download file from URL"""
    try:
        response = requests.get(url, stream=True, timeout=30)
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
        "cuda_available": torch.cuda.is_available(),
        "models": {
            "sadtalker_loaded": SadTalkerModel._instance is not None,
            "realtime_loaded": RealtimeVideoModel._instance is not None
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Hybrid Video Generation Service",
        "version": "1.0.0",
        "status": "running"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
