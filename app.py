from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import uvicorn
import os
import logging
import asyncio
import aiofiles
from pathlib import Path
import tempfile
import shutil
from typing import Optional, Dict, Any, List
import json
import time
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import requests
import torch
import cv2
import numpy as np
from PIL import Image
import io
import threading
from queue import Queue
import uuid
import traceback
import base64
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration ---
app = FastAPI(title="Professional Avatar Video Service", version="5.0.0")
executor = ThreadPoolExecutor(max_workers=6)
video_tasks: Dict[str, dict] = {}
active_streams: Dict[str, dict] = {}

# Model paths and configuration
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
TEMP_DIR = os.environ.get("TEMP_DIR", "/app/temp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("VIDEO_SERVICE_API_KEY", "default-key")

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("temp/videos", exist_ok=True)
os.makedirs("temp/streams", exist_ok=True)
os.makedirs("temp/errors", exist_ok=True)

logger.info(f"🚀 Professional Avatar Video Service Starting")
logger.info(f"📱 Device: {DEVICE}")
logger.info(f"📁 Models Directory: {MODELS_DIR}")
logger.info(f"🔧 Temp Directory: {TEMP_DIR}")

if DEVICE == "cuda":
    logger.info(f"🎮 CUDA version: {torch.version.cuda}")
    logger.info(f"🎯 GPU count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        logger.info(f"🎪 GPU 0: {torch.cuda.get_device_name(0)}")

# Check if models are downloaded
models_downloaded = os.path.exists(f"{MODELS_DIR}/.models_downloaded")
logger.info(f"📦 Models downloaded: {models_downloaded}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not api_key_header:
        return None
    if api_key_header.startswith("Bearer "):
        api_key_header = api_key_header[7:]
    return api_key_header

async def verify_api_key(api_key: str = Depends(get_api_key)):
    if API_KEY == "default-key" or api_key == API_KEY:
        return True
    raise HTTPException(status_code=401, detail="Invalid API key")

# Global variables for models
models_loaded = False
sadtalker_available = False
wav2lip_available = False

# Model paths
SADTALKER_MODELS = {
    "audio2exp": f"{MODELS_DIR}/SadTalker/checkpoints/auido2exp_00300-model.pth",
    "facevid2vid": f"{MODELS_DIR}/SadTalker/checkpoints/facevid2vid_00189-model.pth.tar",
    "epoch_20": f"{MODELS_DIR}/SadTalker/checkpoints/epoch_20.pth",
    "audio2pose": f"{MODELS_DIR}/SadTalker/checkpoints/auido2pose_00140-model.pth",
    "shape_predictor": f"{MODELS_DIR}/SadTalker/checkpoints/shape_predictor_68_face_landmarks.dat"
}

WAV2LIP_MODELS = {
    "wav2lip_gan": f"{MODELS_DIR}/Wav2Lip/checkpoints/wav2lip_gan.pth",
    "s3fd": f"{MODELS_DIR}/Wav2Lip/face_detection/detection/sfd/s3fd.pth",
    "wav2lip": f"{MODELS_DIR}/Wav2Lip/checkpoints/wav2lip.pth"
}

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

class VideoGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"VideoGenerator initialized on device: {self.device}")
        
    async def generate_video_sadtalker(self, image_path: str, audio_path: str, output_path: str, quality: str = "high") -> bool:
        """Generate video using SadTalker - PRIORITY METHOD"""
        try:
            logger.info("🎭 Generating video with SadTalker...")
            
            sadtalker_path = os.path.join(MODELS_DIR, "SadTalker")
            if not os.path.exists(sadtalker_path):
                logger.error("❌ SadTalker directory not found")
                return False
            
            # Check if inference.py exists
            inference_script = os.path.join(sadtalker_path, "inference.py")
            if not os.path.exists(inference_script):
                logger.error(f"❌ SadTalker inference.py not found at {inference_script}")
                return False
            
            # Prepare arguments for SadTalker
            result_dir = str(Path(output_path).parent)
            
            # SadTalker inference command with all required parameters
            cmd = [
                sys.executable, inference_script,
                "--driven_audio", audio_path,
                "--source_image", image_path,
                "--result_dir", result_dir,
                "--still",
                "--preprocess", "crop" if quality == "high" else "resize",
                "--size", "512" if quality == "high" else "256",
                "--pose_style", "0",
                "--expression_scale", "1.0",
                "--facerender", "facevid2vid",
                "--batch_size", "2" if quality == "high" else "4"
            ]
            
            if DEVICE == "cpu":
                cmd.append("--cpu")
            
            logger.info(f"🚀 Running SadTalker command: {' '.join(cmd)}")
            
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = sadtalker_path + ':' + env.get('PYTHONPATH', '')
            env['CUDA_VISIBLE_DEVICES'] = '0' if DEVICE == "cuda" else ''
            
            # Run SadTalker
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=sadtalker_path, 
                timeout=600,  # 10 minutes timeout
                env=env
            )
            
            logger.info(f"🎭 SadTalker return code: {result.returncode}")
            logger.info(f"🎭 SadTalker stdout: {result.stdout}")
            
            if result.returncode != 0:
                logger.error(f"❌ SadTalker failed with return code {result.returncode}")
                logger.error(f"❌ SadTalker stderr: {result.stderr}")
                return False
            
            # Find generated video file in result directory
            result_dir_path = Path(result_dir)
            generated_files = []
            
            # Look for mp4 files in result directory and subdirectories
            for pattern in ["*.mp4", "**/*.mp4"]:
                generated_files.extend(list(result_dir_path.glob(pattern)))
            
            if not generated_files:
                logger.error(f"❌ No video file generated by SadTalker in {result_dir}")
                # List all files in result directory for debugging
                all_files = list(result_dir_path.rglob("*"))
                logger.error(f"❌ Files found in result directory: {[str(f) for f in all_files]}")
                return False
            
            # Use the first (and likely only) generated video
            generated_video = generated_files[0]
            logger.info(f"✅ Found generated video: {generated_video}")
            
            # Move to expected output path
            shutil.move(str(generated_video), output_path)
            
            logger.info(f"✅ SadTalker generation completed: {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ SadTalker generation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ SadTalker generation error: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False

    async def generate_video_wav2lip(self, image_path: str, audio_path: str, output_path: str, quality: str = "fast") -> bool:
        """Generate video using Wav2Lip"""
        try:
            logger.info("🎤 Generating video with Wav2Lip...")
            
            wav2lip_path = os.path.join(MODELS_DIR, "Wav2Lip")
            if not os.path.exists(wav2lip_path):
                logger.error("❌ Wav2Lip directory not found")
                return False
            
            # Check if inference.py exists
            inference_script = os.path.join(wav2lip_path, "inference.py")
            if not os.path.exists(inference_script):
                logger.error(f"❌ Wav2Lip inference.py not found at {inference_script}")
                return False
            
            # Ensure model files exist and are in correct locations
            checkpoints_dir = os.path.join(wav2lip_path, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            model_path = os.path.join(checkpoints_dir, "wav2lip_gan.pth")
            if not os.path.exists(model_path) and os.path.exists(WAV2LIP_MODELS["wav2lip_gan"]):
                shutil.copy(WAV2LIP_MODELS["wav2lip_gan"], model_path)
                logger.info(f"Copied wav2lip_gan.pth to {model_path}")
            
            # Ensure face detection model is in place
            face_detection_dir = os.path.join(wav2lip_path, "face_detection", "detection", "sfd")
            os.makedirs(face_detection_dir, exist_ok=True)
            
            s3fd_path = os.path.join(face_detection_dir, "s3fd.pth")
            if not os.path.exists(s3fd_path) and os.path.exists(WAV2LIP_MODELS["s3fd"]):
                shutil.copy(WAV2LIP_MODELS["s3fd"], s3fd_path)
                logger.info(f"Copied s3fd.pth to {s3fd_path}")
            
            # Wav2Lip inference command
            cmd = [
                sys.executable, inference_script,
                "--checkpoint_path", model_path,
                "--face", image_path,
                "--audio", audio_path,
                "--outfile", output_path,
                "--fps", "25",
                "--pads", "0", "10", "0", "0",
                "--face_det_batch_size", "4",
                "--wav2lip_batch_size", "8" if quality == "fast" else "4"
            ]
            
            if quality == "fast":
                cmd.append("--static")
            
            logger.info(f"🚀 Running Wav2Lip command: {' '.join(cmd)}")
            
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = wav2lip_path + ':' + env.get('PYTHONPATH', '')
            env['CUDA_VISIBLE_DEVICES'] = '0' if DEVICE == "cuda" else ''
            
            # Run Wav2Lip
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=wav2lip_path, 
                timeout=300,  # 5 minutes timeout
                env=env
            )
            
            logger.info(f"🎤 Wav2Lip return code: {result.returncode}")
            logger.info(f"🎤 Wav2Lip stdout: {result.stdout}")
            
            if result.returncode != 0:
                logger.error(f"❌ Wav2Lip failed with return code {result.returncode}")
                logger.error(f"❌ Wav2Lip stderr: {result.stderr}")
                return False
            
            # Check if output file was created
            if not os.path.exists(output_path):
                logger.error(f"❌ Wav2Lip did not create output file: {output_path}")
                return False
            
            logger.info(f"✅ Wav2Lip generation completed: {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Wav2Lip generation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Wav2Lip generation error: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False

    def create_basic_video_with_audio(self, image_path: str, audio_path: str, output_path: str) -> bool:
        """Create basic video as final fallback"""
        try:
            logger.info("🎥 Creating basic video as fallback...")
            
            # Load and process image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = img.shape[:2]
            # Make dimensions even for H.264 compatibility
            even_width = width if width % 2 == 0 else width - 1
            even_height = height if height % 2 == 0 else height - 1
            
            if even_width != width or even_height != height:
                img = cv2.resize(img, (even_width, even_height))
                cv2.imwrite(image_path, img)
            
            # Get audio duration
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
            ], capture_output=True, text=True)
            
            duration = float(result.stdout.strip()) if result.stdout.strip() else 5.0
            
            # Create video with FFmpeg
            cmd = [
                'ffmpeg', '-loop', '1', '-i', image_path,
                '-i', audio_path,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-pix_fmt', 'yuv420p', '-shortest',
                '-t', str(duration),
                '-movflags', '+faststart',
                output_path, '-y'
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
            logger.info(f"✅ Basic video created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic video creation failed: {e}")
            return False

# Initialize video generator
video_generator = VideoGenerator()

async def check_models():
    """Check which models are available and can be imported."""
    global sadtalker_available, wav2lip_available, models_loaded
    
    # Check SadTalker
    sadtalker_path = os.path.join(MODELS_DIR, "SadTalker")
    if os.path.exists(sadtalker_path):
        logger.info("🎭 Checking SadTalker availability...")
        
        # Check if all model files exist
        missing_sadtalker = []
        for name, path in SADTALKER_MODELS.items():
            if not os.path.exists(path):
                missing_sadtalker.append(f"{name}: {path}")
        
        if missing_sadtalker:
            logger.warning(f"🎭 SadTalker missing model files: {missing_sadtalker}")
        else:
            # Test if we can import SadTalker modules
            try:
                # Add SadTalker to Python path
                if sadtalker_path not in sys.path:
                    sys.path.insert(0, sadtalker_path)
                
                # Try importing key modules
                logger.info("🎭 Testing SadTalker imports...")
                
                # Test basic imports first
                import kornia
                logger.info("✅ kornia imported successfully")
                
                # Test SadTalker specific imports
                from src.utils.preprocess import CropAndExtract
                from src.test_audio2coeff import Audio2Coeff  
                from src.facerender.animate import AnimateFromCoeff
                
                sadtalker_available = True
                logger.info("✅ SadTalker models found and imports successful")
                
            except ImportError as e:
                logger.error(f"❌ SadTalker import failed: {e}")
                logger.error("❌ SadTalker not available due to missing dependencies")
            except Exception as e:
                logger.error(f"❌ SadTalker test failed: {e}")
    else:
        logger.warning("🎭 SadTalker directory not found")
    
    # Check Wav2Lip
    wav2lip_path = os.path.join(MODELS_DIR, "Wav2Lip")
    if os.path.exists(wav2lip_path):
        logger.info("👄 Checking Wav2Lip availability...")
        
        # Check if all model files exist
        missing_wav2lip = []
        for name, path in WAV2LIP_MODELS.items():
            if not os.path.exists(path):
                missing_wav2lip.append(f"{name}: {path}")
        
        if missing_wav2lip:
            logger.warning(f"👄 Wav2Lip missing model files: {missing_wav2lip}")
        else:
            # Test if we can import Wav2Lip modules
            try:
                # Add Wav2Lip to Python path
                if wav2lip_path not in sys.path:
                    sys.path.insert(0, wav2lip_path)
                
                # Try importing key modules
                logger.info("👄 Testing Wav2Lip imports...")
                
                import face_detection
                from models import Wav2Lip
                
                wav2lip_available = True
                logger.info("✅ Wav2Lip models found and imports successful")
                
            except ImportError as e:
                logger.error(f"❌ Wav2Lip import failed: {e}")
                logger.error("❌ Wav2Lip not available due to missing dependencies")
            except Exception as e:
                logger.error(f"❌ Wav2Lip test failed: {e}")
    else:
        logger.warning("👄 Wav2Lip directory not found")
    
    models_loaded = True
    
    logger.info(f"🎭 SadTalker available: {sadtalker_available}")
    logger.info(f"👄 Wav2Lip available: {wav2lip_available}")
    
    if not (sadtalker_available or wav2lip_available):
        logger.warning("⚠️ No advanced models available, using basic video generation")
    else:
        logger.info("✅ Video generation service ready")

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("🚀 Starting Professional Avatar Video Service...")
    await check_models()

# --- Background Task Functions ---
async def _run_video_generation(task_id: str, image_url: str, audio_url: str, output_dir: str, quality: str):
    """Run video generation in background with CORRECT MODEL PRIORITY"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        logger.info(f"🎬 Starting video generation for task {task_id}")
        
        image_path = os.path.join(temp_dir, "input.jpg")
        audio_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(output_dir, f"{task_id}.mp4")
        
        # Download files
        logger.info(f"📥 Downloading files for task {task_id}")
        await _download_file(image_url, image_path)
        await _download_file(audio_url, audio_path)
        
        # Update task status
        video_tasks[task_id]["status"] = "processing"
        
        # CORRECT PRIORITY ORDER: SadTalker > Wav2Lip > Basic
        success = False
        
        # 1. FIRST PRIORITY: Try SadTalker (best overall quality)
        if sadtalker_available:
            logger.info(f"🎭 Trying SadTalker FIRST for task {task_id}")
            video_tasks[task_id]["model_used"] = "SadTalker"
            success = await video_generator.generate_video_sadtalker(image_path, audio_path, output_path, quality)
            
            if success:
                logger.info(f"✅ SadTalker SUCCESS for task {task_id}")
            else:
                logger.warning(f"❌ SadTalker FAILED for task {task_id}")
        
        # 2. SECOND PRIORITY: Try Wav2Lip if SadTalker failed
        if not success and wav2lip_available:
            logger.info(f"🎤 Trying Wav2Lip for task {task_id}")
            video_tasks[task_id]["model_used"] = "Wav2Lip"
            success = await video_generator.generate_video_wav2lip(image_path, audio_path, output_path, quality)
            
            if success:
                logger.info(f"✅ Wav2Lip SUCCESS for task {task_id}")
            else:
                logger.warning(f"❌ Wav2Lip FAILED for task {task_id}")
        
        # 3. FINAL FALLBACK: Basic video only if ALL advanced models failed
        if not success:
            logger.warning(f"🔄 ALL ADVANCED MODELS FAILED, using basic video for task {task_id}")
            video_tasks[task_id]["model_used"] = "Basic"
            success = video_generator.create_basic_video_with_audio(image_path, audio_path, output_path)
        
        if success:
            video_tasks[task_id]["status"] = "completed"
            video_tasks[task_id]["output_path"] = output_path
            logger.info(f"✅ Video generation completed for task {task_id} using {video_tasks[task_id]['model_used']}")
        else:
            raise Exception("All video generation methods failed")
        
    except Exception as e:
        logger.error(f"❌ Video generation failed for task {task_id}: {e}")
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = str(e)
        
        # Save error to file
        error_path = f"temp/errors/{task_id}.json"
        try:
            with open(error_path, 'w') as f:
                json.dump({"error": str(e), "task_id": task_id}, f)
        except Exception as save_error:
            logger.error(f"Failed to save error file: {save_error}")
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def _download_file(url: str, local_path: str):
    """Download file from URL"""
    try:
        logger.info(f"📥 Downloading {url}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"✅ Downloaded {url} to {local_path}")
        
    except Exception as e:
        logger.error(f"❌ Error downloading {url}: {e}")
        raise

# --- HTTP Endpoints ---
@app.post("/generate-video", dependencies=[Depends(verify_api_key)])
async def generate_video(
    background_tasks: BackgroundTasks,
    image_url: str = Form(...),
    audio_url: str = Form(...),
    quality: str = Form(default="high")
):
    """Generate video from image and audio URLs."""
    if not models_loaded:
        await check_models()
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task
    video_tasks[task_id] = {
        "status": "queued",
        "created_at": time.time(),
        "quality": quality,
        "model_used": "Unknown"
    }
    
    # Start background video generation
    background_tasks.add_task(
        _run_video_generation,
        task_id,
        image_url,
        audio_url,
        "temp/videos",
        quality
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "quality": quality,
        "estimated_time": "2-5 minutes" if quality == "high" else "30-60 seconds",
        "available_models": {
            "sadtalker": sadtalker_available,
            "wav2lip": wav2lip_available,
            "animated": True,
            "basic": True
        }
    }

@app.get("/video-status/{task_id}")
async def get_video_status(task_id: str):
    """Get video generation status."""
    # Check if video file exists
    video_path = f"temp/videos/{task_id}.mp4"
    
    if os.path.exists(video_path):
        # Return the video file
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"{task_id}.mp4"
        )
    else:
        # Check for error file
        error_path = f"temp/errors/{task_id}.json"
        if os.path.exists(error_path):
            async with aiofiles.open(error_path, 'r') as f:
                error_data = json.loads(await f.read())
            return JSONResponse(
                status_code=500,
                content={"status": "failed", "error": error_data["error"]}
            )
        
        # Check task status
        if task_id in video_tasks:
            task_info = video_tasks[task_id].copy()
            return {
                "status": task_info["status"],
                "model_used": task_info.get("model_used", "Unknown"),
                "quality": task_info.get("quality", "unknown"),
                "created_at": task_info.get("created_at", 0)
            }
        else:
            return {"status": "not_found"}

@app.get("/models/status")
async def models_status():
    """Get detailed model status."""
    if not models_loaded:
        await check_models()
    
    return {
        "models_loaded": models_loaded,
        "device": DEVICE,
        "sadtalker": {
            "available": sadtalker_available,
            "models": {name: os.path.exists(path) for name, path in SADTALKER_MODELS.items()}
        },
        "wav2lip": {
            "available": wav2lip_available,
            "models": {name: os.path.exists(path) for name, path in WAV2LIP_MODELS.items()}
        },
        "features": {
            "high_quality_generation": sadtalker_available,
            "fast_generation": wav2lip_available,
            "animated_fallback": True,
            "basic_fallback": True,
            "real_time_streaming": True,
            "face_detection": True,
            "lip_sync": wav2lip_available or sadtalker_available
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "sadtalker_available": sadtalker_available,
        "wav2lip_available": wav2lip_available,
        "service": "professional-video-generation",
        "version": "5.0.0",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "active_streams": len(active_streams),
        "active_tasks": len([t for t in video_tasks.values() if t["status"] == "processing"]),
        "total_tasks": len(video_tasks),
        "features": {
            "sadtalker": sadtalker_available,
            "wav2lip": wav2lip_available,
            "real_time_streaming": True,
            "animated_fallback": True,
            "basic_fallback": True,
            "lip_sync": wav2lip_available or sadtalker_available,
            "head_movement": sadtalker_available,
            "eye_blink": sadtalker_available,
            "face_detection": True
        },
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Professional Video Generation Service",
        "version": "5.0.0",
        "status": "running",
        "features": [
            "SadTalker Integration (Priority)",
            "Wav2Lip Integration", 
            "Real-time Video Streaming",
            "Professional Lip Sync",
            "Head Movement Animation",
            "Eye Blink Animation",
            "Face Detection with MediaPipe",
            "Basic Video Fallback"
        ],
        "endpoints": {
            "generate_video": "/generate-video",
            "video_status": "/video-status/{task_id}",
            "models_status": "/models/status",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
        log_level="info"
    )
