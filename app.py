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
import glob


sys.path.insert(0, "/app/models/SadTalker")
from src.gradio_demo import SadTalker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration ---
app = FastAPI(title="Professional Avatar Video Service", version="4.0.0")
executor = ThreadPoolExecutor(max_workers=6)
video_tasks: Dict[str, dict] = {}
active_streams: Dict[str, dict] = {}
preprocessed_avatars: Dict[str, dict] = {}

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
os.makedirs("temp/avatars", exist_ok=True)

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
sadtalker_models = {}
wav2lip_models = {}
realtime_models = {}

# Global variables for model loading
models_loaded = False
sadtalker_available = False
wav2lip_available = False

# Model paths
SADTALKER_MODELS = {
    "auido2exp": f"{MODELS_DIR}/SadTalker/checkpoints/auido2exp_00300-model.pth",
    "facevid2vid": f"{MODELS_DIR}/SadTalker/checkpoints/facevid2vid_00189-model.pth.tar",
    "epoch_20": f"{MODELS_DIR}/SadTalker/checkpoints/epoch_20.pth",
    "auido2pose": f"{MODELS_DIR}/SadTalker/checkpoints/auido2pose_00140-model.pth",
    "shape_predictor": f"{MODELS_DIR}/SadTalker/checkpoints/shape_predictor_68_face_landmarks.dat",
    "mapping": f"{MODELS_DIR}/SadTalker/checkpoints/mapping_00229-model.pth.tar",
    "wav2lip": f"{MODELS_DIR}/SadTalker/checkpoints/wav2lip.pth"
}

WAV2LIP_MODELS = {
    "wav2lip_gan": f"{MODELS_DIR}/Wav2Lip/checkpoints/wav2lip_gan.pth",
    "s3fd": f"{MODELS_DIR}/Wav2Lip/face_detection/detection/sfd/s3fd.pth",
    "wav2lip": f"{MODELS_DIR}/Wav2Lip/checkpoints/wav2lip.pth"
}

# Initialize MediaPipe Face Detection (lighter alternative to dlib)
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

class VideoGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sadtalker_model = None
        self.wav2lip_model = None
        logger.info(f"VideoGenerator initialized on device: {self.device}")

    async def load_sadtalker_models(self):
        ...
        try:
            from src.gradio_demo import SadTalker
            self.sadtalker = SadTalker(lazy_load=True)
            logger.info("✅ SadTalker class loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load SadTalker: {e}")
            return False

    async def load_wav2lip_models(self):
        """Load Wav2Lip models"""
        try:
            logger.info("🎤 Loading Wav2Lip models...")
            wav2lip_path = os.path.join(MODELS_DIR, "Wav2Lip")
            if not os.path.exists(wav2lip_path):
                logger.error("❌ Wav2Lip repository not found")
                return False

            # Add Wav2Lip to Python path
            if wav2lip_path not in sys.path:
                sys.path.insert(0, wav2lip_path)

            # Check if all model files exist
            missing_models = []
            for name, path in WAV2LIP_MODELS.items():
                if not os.path.exists(path):
                    missing_models.append(f"{name}: {path}")

            if missing_models:
                logger.error(f"❌ Missing Wav2Lip model files: {missing_models}")
                return False

            # Try to import and load Wav2Lip model
            try:
                # Fix path issues with Wav2Lip
                wav2lip_dir = os.path.dirname(os.path.dirname(WAV2LIP_MODELS["wav2lip_gan"]))
                if wav2lip_dir not in sys.path:
                    sys.path.insert(0, wav2lip_dir)
                
                # Create symbolic links if needed for compatibility
                checkpoints_dir = os.path.join(wav2lip_path, "checkpoints")
                os.makedirs(checkpoints_dir, exist_ok=True)
                
                # Create symbolic links for model files if they don't exist in expected locations
                for model_name, model_path in WAV2LIP_MODELS.items():
                    if "wav2lip" in model_name and not os.path.exists(os.path.join(checkpoints_dir, os.path.basename(model_path))):
                        try:
                            os.symlink(model_path, os.path.join(checkpoints_dir, os.path.basename(model_path)))
                            logger.info(f"Created symlink for {model_name}")
                        except Exception as e:
                            logger.error(f"Failed to create symlink for {model_name}: {e}")
                
                # Create face_detection directory structure if needed
                face_detection_dir = os.path.join(wav2lip_path, "face_detection", "detection", "sfd")
                os.makedirs(face_detection_dir, exist_ok=True)
                
                if not os.path.exists(os.path.join(face_detection_dir, "s3fd.pth")) and os.path.exists(WAV2LIP_MODELS["s3fd"]):
                    try:
                        os.symlink(WAV2LIP_MODELS["s3fd"], os.path.join(face_detection_dir, "s3fd.pth"))
                        logger.info("Created symlink for s3fd.pth")
                    except Exception as e:
                        logger.error(f"Failed to create symlink for s3fd.pth: {e}")
                
                # Try importing Wav2Lip modules
                sys.path.insert(0, wav2lip_path)
                
                # Check if we can import the modules
                try:
                    import face_detection
                    from models import Wav2Lip
                    logger.info("✅ Wav2Lip modules imported successfully")
                    return True
                except ImportError as e:
                    logger.error(f"❌ Failed to import Wav2Lip modules: {e}")
                    return False
                
            except Exception as e:
                logger.error(f"❌ Failed to load Wav2Lip model: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to load Wav2Lip models: {e}")
            return False

  

    async def generate_video_sadtalker(self, image_path, audio_path, output_path, quality="high"):
      try:
        logger.info("🎭 Generating video with SadTalker (Python API)...")
        if not self.sadtalker:
            logger.error("SadTalker not loaded")
            return False

        # preprocess_type = "crop" if quality == "high" else "resize"
        preprocess_type = "full"
        is_still_mode = True
        enhancer = "gfpgan"
        batch_size = 2 if quality == "high" else 4
        size_of_image = 512 if quality == "high" else 256
        pose_style = 0
        facerender = "facevid2vid"
        exp_weight = 1.0
        use_ref_video = False
        ref_video = None
        ref_info = "pose"
        use_idle_mode = False
        length_of_audio = None
        blink_every = True

        # Use a unique result directory for this task
        result_dir = os.path.abspath(os.path.join("temp", "sadtalker_results", str(uuid.uuid4())))
        os.makedirs(result_dir, exist_ok=True)

        logger.info(f"SadTalker.test args: {image_path}, {audio_path}, {preprocess_type}, {is_still_mode}, {enhancer}, {batch_size}, {size_of_image}, {pose_style}, {facerender}, {exp_weight}, {use_ref_video}, {ref_video}, {ref_info}, {use_idle_mode}, {length_of_audio}, {blink_every}, result_dir={result_dir}")

        # Try API usage
        try:
            result = self.sadtalker.test(
                image_path,
                audio_path,
                preprocess_type,
                is_still_mode,
                enhancer,
                batch_size,
                size_of_image,
                pose_style,
                facerender,
                exp_weight,
                use_ref_video,
                ref_video,
                ref_info,
                use_idle_mode,
                length_of_audio,
                blink_every,
                result_dir
            )
            # Find the generated mp4 in result_dir
            mp4_files = glob.glob(os.path.join(result_dir, "*.mp4"))
            if result and isinstance(result, str) and os.path.exists(result):
                shutil.move(result, output_path)
                logger.info(f"✅ SadTalker generation completed: {output_path}")
                return True
            elif mp4_files:
                shutil.move(mp4_files[0], output_path)
                logger.info(f"✅ SadTalker generation completed (from result_dir): {output_path}")
                return True
            else:
                logger.error(f"SadTalker did not return a valid video path: {result}")
        except Exception as api_e:
            logger.error(f"❌ SadTalker API error: {api_e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")

        # --- CLI fallback ---
        logger.info("⚠️ SadTalker API failed, falling back to CLI usage...")
        sadtalker_dir = os.path.join(MODELS_DIR, "SadTalker")
        inference_py = os.path.join(sadtalker_dir, "inference.py")
        cli_result_dir = os.path.abspath(os.path.join("temp", "sadtalker_cli_results", str(uuid.uuid4())))
        os.makedirs(cli_result_dir, exist_ok=True)

        cmd = [
            sys.executable, inference_py,
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", cli_result_dir,
            "--still",
            "--preprocess", preprocess_type,
            "--enhancer", "gfpgan"
        ]
        logger.info(f"Running SadTalker CLI: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=sadtalker_dir, timeout=600)
        logger.info(f"SadTalker CLI stdout: {result.stdout}")
        logger.info(f"SadTalker CLI stderr: {result.stderr}")

        # Find the generated mp4 in cli_result_dir
        mp4_files = glob.glob(os.path.join(cli_result_dir, "*.mp4"))
        if mp4_files:
            shutil.move(mp4_files[0], output_path)
            logger.info(f"✅ SadTalker CLI generation completed: {output_path}")
            return True
        else:
            logger.error("❌ SadTalker CLI did not produce a video.")
            logger.error(f"❌ SadTalker CLI stdout: {result.stdout}")
            logger.error(f"❌ SadTalker CLI stderr: {result.stderr}")
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
                logger.error("❌ Wav2Lip not found")
                return False
            
            # Ensure the model files are in the expected locations
            checkpoints_dir = os.path.join(wav2lip_path, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            model_path = os.path.join(checkpoints_dir, "wav2lip_gan.pth")
            if not os.path.exists(model_path) and os.path.exists(WAV2LIP_MODELS["wav2lip_gan"]):
                try:
                    shutil.copy(WAV2LIP_MODELS["wav2lip_gan"], model_path)
                    logger.info(f"Copied wav2lip_gan.pth to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to copy wav2lip_gan.pth: {e}")
                    return False
            
            # Ensure face detection model is in place
            face_detection_dir = os.path.join(wav2lip_path, "face_detection", "detection", "sfd")
            os.makedirs(face_detection_dir, exist_ok=True)
            
            s3fd_path = os.path.join(face_detection_dir, "s3fd.pth")
            if not os.path.exists(s3fd_path) and os.path.exists(WAV2LIP_MODELS["s3fd"]):
                try:
                    shutil.copy(WAV2LIP_MODELS["s3fd"], s3fd_path)
                    logger.info(f"Copied s3fd.pth to {s3fd_path}")
                except Exception as e:
                    logger.error(f"Failed to copy s3fd.pth: {e}")
                    return False
            
            # Wav2Lip inference command
            cmd = [
                sys.executable, os.path.join(wav2lip_path, "inference.py"),
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
            
            logger.info(f"🚀 Running Wav2Lip: {' '.join(cmd)}")
            
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = wav2lip_path + ':' + env.get('PYTHONPATH', '')
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=wav2lip_path, 
                timeout=300,  # 5 minutes timeout
                env=env
            )
            logger.info(f"Wav2Lip stdout: {result.stdout}")
            logger.info(f"Wav2Lip stderr: {result.stderr}")
            logger.info(f"Checking for output file: {output_path} -> {os.path.exists(output_path)}")
            
            if result.returncode != 0:
                logger.error(f"❌ Wav2Lip failed: {result.stderr}")
                logger.error(f"❌ Wav2Lip stdout: {result.stdout}")
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
    """Check which models are available."""
    global sadtalker_available, wav2lip_available, models_loaded

    # Only check if models were downloaded during build
    if not models_downloaded:
        logger.warning("⚠️ Models not downloaded during build, using basic generation only")
        sadtalker_available = False
        wav2lip_available = False
        models_loaded = True
        return

    # Try to load models
    sadtalker_available = await video_generator.load_sadtalker_models()
    wav2lip_available = await video_generator.load_wav2lip_models()
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

def detect_face_landmarks(image):
    """Detect face landmarks using MediaPipe"""
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            logger.warning("No faces detected in image")
            return None
        
        # Get the first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        height, width = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.append([x, y])
        
        landmarks = np.array(landmarks)
        
        # MediaPipe face mesh landmark indices for different facial features
        # Mouth landmarks (lips)
        mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        # Eye landmarks
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        mouth_points = landmarks[mouth_indices] if len(landmarks) > max(mouth_indices) else landmarks[:12]
        left_eye = landmarks[left_eye_indices] if len(landmarks) > max(left_eye_indices) else landmarks[33:49]
        right_eye = landmarks[right_eye_indices] if len(landmarks) > max(right_eye_indices) else landmarks[42:48]
        
        return {
            'landmarks': landmarks,
            'mouth_points': mouth_points,
            'left_eye': left_eye,
            'right_eye': right_eye,
            'face_center': np.mean(landmarks, axis=0).astype(np.int32)
        }
        
    except Exception as e:
        logger.error(f"Error detecting face landmarks: {e}")
        return None

def make_dimensions_even(width: int, height: int) -> tuple:
    """Make dimensions even (divisible by 2) for H.264 compatibility while preserving aspect ratio"""
    even_width = width if width % 2 == 0 else width - 1
    even_height = height if height % 2 == 0 else height - 1
    return even_width, even_height

def create_animated_talking_video(image_path: str, audio_path: str, output_path: str, quality: str = "high") -> str:
    """Create animated talking video with MediaPipe face detection and lip sync"""
    try:
        logger.info(f"Creating animated talking video: {image_path} + {audio_path} -> {output_path}")
        
        # Load and analyze the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_height, original_width = img.shape[:2]
        logger.info(f"Original image dimensions: {original_width}x{original_height}")
        
        # Detect face landmarks
        face_data = detect_face_landmarks(img)
        if face_data is None:
            logger.warning("No face detected - using basic animation")
            return create_basic_animated_video(image_path, audio_path, output_path, quality)
        
        logger.info("Face detected successfully with MediaPipe")
        
        # Make dimensions even while preserving aspect ratio
        even_width, even_height = make_dimensions_even(original_width, original_height)
        
        if even_width != original_width or even_height != original_height:
            logger.info(f"Adjusting dimensions from {original_width}x{original_height} to {even_width}x{even_height}")
            img = cv2.resize(img, (even_width, even_height))
            # Recalculate face landmarks for resized image
            scale_x = even_width / original_width
            scale_y = even_height / original_height
            face_data['landmarks'] = face_data['landmarks'] * [scale_x, scale_y]
            face_data['mouth_points'] = face_data['mouth_points'] * [scale_x, scale_y]
            face_data['left_eye'] = face_data['left_eye'] * [scale_x, scale_y]
            face_data['right_eye'] = face_data['right_eye'] * [scale_x, scale_y]
            face_data['face_center'] = face_data['face_center'] * [scale_x, scale_y]
            cv2.imwrite(image_path, img)
        
        # Get audio duration
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ], capture_output=True, text=True)
        
        duration = float(result.stdout.strip()) if result.stdout.strip() else 5.0
        logger.info(f"Audio duration: {duration} seconds")
        
        # Create animated frames
        temp_frames_dir = tempfile.mkdtemp()
        frame_rate = 25
        total_frames = int(duration * frame_rate)
        
        logger.info(f"Generating {total_frames} animated frames at {frame_rate} FPS with MediaPipe face animation")
        
        # Analyze audio for animation data
        audio_analysis = analyze_audio_for_animation(audio_path, frame_rate)
        
        # Generate animated frames with face landmarks
        for frame_idx in range(total_frames):
            animated_frame = create_face_animated_frame(
                img.copy(), 
                face_data,
                frame_idx, 
                frame_rate, 
                audio_analysis.get(frame_idx, {'intensity': 0.0, 'is_speaking': False})
            )
            
            frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, animated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Create video from animated frames
        frames_pattern = os.path.join(temp_frames_dir, "frame_%06d.jpg")
        
        # FFmpeg command for high-quality animated video
        if quality == "high":
            cmd = [
                'ffmpeg', '-r', str(frame_rate), '-i', frames_pattern,
                '-i', audio_path,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
                '-c:a', 'aac', '-b:a', '192k',
                '-pix_fmt', 'yuv420p',
                '-shortest', '-movflags', '+faststart',
                output_path, '-y'
            ]
        else:
            cmd = [
                'ffmpeg', '-r', str(frame_rate), '-i', frames_pattern,
                '-i', audio_path,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-pix_fmt', 'yuv420p',
                '-shortest', '-movflags', '+faststart',
                output_path, '-y'
            ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        shutil.rmtree(temp_frames_dir, ignore_errors=True)
        
        logger.info(f"Animated talking video created successfully: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise Exception(f"Video generation failed: {e.stderr}")
    except Exception as e:
        logger.error(f"Error creating animated video: {e}")
        raise

def create_face_animated_frame(base_image: np.ndarray, face_data: dict, frame_idx: int, frame_rate: int, audio_data: dict) -> np.ndarray:
    """Create an animated frame with realistic face movements based on MediaPipe landmarks"""
    try:
        img = base_image.copy()
        height, width = img.shape[:2]
        
        # Extract audio features
        intensity = audio_data.get('intensity', 0.0)
        is_speaking = audio_data.get('is_speaking', False)
        spectral_centroid = audio_data.get('spectral_centroid', 0.0)
        
        # Time-based animation
        time_sec = frame_idx / frame_rate
        
        # Get face landmarks
        mouth_points = face_data['mouth_points'].astype(np.int32)
        left_eye = face_data['left_eye'].astype(np.int32)
        right_eye = face_data['right_eye'].astype(np.int32)
        face_center = face_data['face_center'].astype(np.int32)
        
        # 1. Mouth animation based on audio
        if is_speaking and intensity > 0.1:
            # Calculate mouth opening based on audio intensity
            mouth_opening_factor = min(intensity * 3, 1.0)
            
            # Get mouth center from landmarks
            mouth_center = np.mean(mouth_points, axis=0).astype(np.int32)
            
            # Create mouth opening effect
            mouth_width = int(20 + intensity * 15)
            mouth_height = int(5 + intensity * 12)
            
            # Different mouth shapes based on spectral content
            if spectral_centroid > 1500:  # Higher frequencies - more open mouth
                mouth_height = int(mouth_height * 1.3)
            elif spectral_centroid < 800:  # Lower frequencies - wider mouth
                mouth_width = int(mouth_width * 1.2)
            
            # Draw mouth opening
            cv2.ellipse(img, tuple(mouth_center), (mouth_width, mouth_height), 0, 0, 180, (20, 20, 20), -1)
            
            # Add teeth for realism
            if mouth_height > 8:
                cv2.ellipse(img, tuple(mouth_center), (mouth_width - 4, max(2, mouth_height - 4)), 0, 0, 180, (220, 220, 220), -1)
        
        # 2. Eye blinking animation
        if frame_idx % 150 == 0 or (frame_idx % 75 == 0 and np.random.random() < 0.3):
            # Create blink effect by darkening eye regions
            for eye_points in [left_eye, right_eye]:
                if len(eye_points) > 0:
                    # Create eye mask
                    eye_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(eye_mask, [eye_points], 255)
                    
                    # Darken eye area for blink
                    img[eye_mask > 0] = img[eye_mask > 0] * 0.3
        
        # 3. Subtle head movement
        if is_speaking:
            # More pronounced movement when speaking
            head_sway_x = int(3 * np.sin(time_sec * 1.2) * intensity)
            head_bob_y = int(2 * np.sin(time_sec * 0.8) * intensity)
        else:
            # Subtle idle movement
            head_sway_x = int(1 * np.sin(time_sec * 0.3))
            head_bob_y = int(1 * np.sin(time_sec * 0.2))
        
        # Apply head movement
        if abs(head_sway_x) > 0 or abs(head_bob_y) > 0:
            M = np.float32([[1, 0, head_sway_x], [0, 1, head_bob_y]])
            img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
        
        return img
        
    except Exception as e:
        logger.error(f"Error creating face animated frame {frame_idx}: {e}")
        return base_image

def create_basic_animated_video(image_path: str, audio_path: str, output_path: str, quality: str = "high") -> str:
    """Fallback basic animation when face detection fails"""
    logger.info("Using basic animation fallback")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    height, width = img.shape[:2]
    even_width, even_height = make_dimensions_even(width, height)
    
    if even_width != width or even_height != height:
        img = cv2.resize(img, (even_width, even_height))
        cv2.imwrite(image_path, img)
    
    # Get audio duration
    result = subprocess.run([
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
    ], capture_output=True, text=True)
    
    duration = float(result.stdout.strip()) if result.stdout.strip() else 5.0
    
    # Create simple animated frames
    temp_frames_dir = tempfile.mkdtemp()
    frame_rate = 25
    total_frames = int(duration * frame_rate)
    
    audio_analysis = analyze_audio_for_animation(audio_path, frame_rate)
    
    for frame_idx in range(total_frames):
        # Basic animation without face detection
        animated_frame = create_basic_animated_frame(
            img.copy(), 
            frame_idx, 
            frame_rate, 
            audio_analysis.get(frame_idx, {'intensity': 0.0, 'is_speaking': False})
        )
        
        frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_path, animated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Create video
    frames_pattern = os.path.join(temp_frames_dir, "frame_%06d.jpg")
    
    cmd = [
        'ffmpeg', '-r', str(frame_rate), '-i', frames_pattern,
        '-i', audio_path,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-c:a', 'aac', '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-shortest', '-movflags', '+faststart',
        output_path, '-y'
    ]
    
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    shutil.rmtree(temp_frames_dir, ignore_errors=True)
    
    return output_path

def create_basic_animated_frame(base_image: np.ndarray, frame_idx: int, frame_rate: int, audio_data: dict) -> np.ndarray:
    """Create basic animated frame without face detection"""
    img = base_image.copy()
    height, width = img.shape[:2]
    
    intensity = audio_data.get('intensity', 0.0)
    is_speaking = audio_data.get('is_speaking', False)
    time_sec = frame_idx / frame_rate
    
    # Basic mouth animation in center-bottom area
    if is_speaking and intensity > 0.1:
        mouth_center_x = width // 2
        mouth_center_y = int(height * 0.75)
        
        mouth_width = int(15 + intensity * 20)
        mouth_height = int(5 + intensity * 10)
        
        # Draw simple mouth
        cv2.ellipse(img, (mouth_center_x, mouth_center_y), (mouth_width, mouth_height), 0, 0, 180, (20, 20, 20), -1)
        
        if mouth_height > 6:
            cv2.ellipse(img, (mouth_center_x, mouth_center_y - 2), (mouth_width - 3, max(1, mouth_height - 4)), 0, 0, 180, (200, 200, 200), -1)
    
    # Basic head movement
    head_sway_x = int(2 * np.sin(time_sec * 0.5) * intensity) if is_speaking else int(1 * np.sin(time_sec * 0.3))
    head_bob_y = int(1 * np.sin(time_sec * 0.3) * intensity) if is_speaking else int(1 * np.sin(time_sec * 0.2))
    
    if abs(head_sway_x) > 0 or abs(head_bob_y) > 0:
        M = np.float32([[1, 0, head_sway_x], [0, 1, head_bob_y]])
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    
    return img

def analyze_audio_for_animation(audio_path: str, frame_rate: int) -> dict:
    """Analyze audio to extract animation data for each frame"""
    try:
        import librosa
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Calculate frame duration
        frame_duration = 1.0 / frame_rate
        
        # Extract features for animation
        animation_data = {}
        
        for frame_idx in range(int(len(y) / sr * frame_rate)):
            start_sample = int(frame_idx * frame_duration * sr)
            end_sample = int((frame_idx + 1) * frame_duration * sr)
            
            if end_sample > len(y):
                end_sample = len(y)
            
            frame_audio = y[start_sample:end_sample]
            
            if len(frame_audio) > 0:
                # Calculate audio intensity (for mouth opening)
                intensity = np.sqrt(np.mean(frame_audio ** 2))
                
                # Calculate spectral features (for mouth shape)
                if len(frame_audio) > 512:
                    stft = librosa.stft(frame_audio, n_fft=512)
                    spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft))[0]
                    avg_centroid = np.mean(spectral_centroid) if len(spectral_centroid) > 0 else 0
                else:
                    avg_centroid = 0
                
                animation_data[frame_idx] = {
                    'intensity': intensity,
                    'spectral_centroid': avg_centroid,
                    'is_speaking': intensity > 0.01
                }
            else:
                animation_data[frame_idx] = {
                    'intensity': 0.0,
                    'spectral_centroid': 0.0,
                    'is_speaking': False
                }
        
        logger.info(f"Analyzed audio for {len(animation_data)} frames")
        return animation_data
        
    except ImportError:
        logger.warning("librosa not available, using simple audio analysis")
        return simple_audio_analysis(audio_path, frame_rate)
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        return simple_audio_analysis(audio_path, frame_rate)

def simple_audio_analysis(audio_path: str, frame_rate: int) -> dict:
    """Simple audio analysis without librosa"""
    try:
        # Get audio duration
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ], capture_output=True, text=True)
        
        duration = float(result.stdout.strip()) if result.stdout.strip() else 5.0
        total_frames = int(duration * frame_rate)
        
        # Create simple animation pattern
        animation_data = {}
        for frame_idx in range(total_frames):
            # Create a speaking pattern (simulate speech)
            time_sec = frame_idx / frame_rate
            
            # Simple sine wave pattern for mouth movement
            base_intensity = 0.3 + 0.7 * abs(np.sin(time_sec * 8))  # 8 Hz speech-like pattern
            
            # Add some randomness
            noise = np.random.normal(0, 0.1)
            intensity = max(0, min(1, base_intensity + noise))
            
            animation_data[frame_idx] = {
                'intensity': intensity,
                'spectral_centroid': 1000 + 500 * intensity,
                'is_speaking': intensity > 0.2
            }
        
        logger.info(f"Generated simple animation data for {len(animation_data)} frames")
        return animation_data
        
    except Exception as e:
        logger.error(f"Error in simple audio analysis: {e}")
        return {}

# --- Background Task Functions ---
async def _run_video_generation(task_id: str, image_url: str, audio_url: str, output_dir: str, quality: str):
    temp_dir = tempfile.mkdtemp()
    try:
        logger.info(f"🎬 Starting video generation for task {task_id}")
        image_path = os.path.join(temp_dir, "input.jpg")
        audio_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.abspath(os.path.join(output_dir, f"{task_id}.mp4"))

        await _download_file(image_url, image_path)
        await _download_file(audio_url, audio_path)

        video_tasks[task_id]["status"] = "processing"
        success = False

        # 1. Try SadTalker
        if sadtalker_available:
            logger.info(f"🎭 Trying SadTalker for task {task_id}")
            video_tasks[task_id]["model_used"] = "SadTalker"
            try:
               result = await video_generator.generate_video_sadtalker(image_path, audio_path, output_path, quality)
               if result and os.path.exists(output_path):
                 success = True
            except Exception as e:
               logger.error(f"SadTalker failed: {e}")

        # 2. Try Wav2Lip if SadTalker failed
        if not success and wav2lip_available:
             logger.info(f"👄 Trying Wav2Lip for task {task_id}")
             video_tasks[task_id]["model_used"] = "Wav2Lip"
             try:
               result = await video_generator.generate_video_wav2lip(image_path, audio_path, output_path, quality)
               for _ in range(10):
                   if os.path.exists(output_path):
                     logger.info(f"✅ Wav2Lip output file detected: {output_path}")
                     success = True
                     break
                   time.sleep(0.2)
               if not success:
                 logger.error(f"❌ Wav2Lip reported success but output file not found: {output_path}")
             except Exception as e:
               logger.error(f"Wav2Lip failed: {e}")

        # 3. Fallback to animated
        if not success:
           logger.info(f"🎬 Falling back to animated video for task {task_id}")
           video_tasks[task_id]["model_used"] = "Animated"
           try:
             create_animated_talking_video(image_path, audio_path, output_path, quality)
             if os.path.exists(output_path):
               success = True
           except Exception as e:
              logger.error(f"❌ Animated video failed: {e}")

        # 4. Fallback to basic
        if not success:
           logger.info(f"🎬 Falling back to basic video for task {task_id}")
           video_tasks[task_id]["model_used"] = "Basic"
           try:
               success = video_generator.create_basic_video_with_audio(image_path, audio_path, output_path)
           except Exception as e:
               logger.error(f"❌ Basic video creation failed: {e}")

        if success and os.path.exists(output_path):
            video_tasks[task_id]["status"] = "completed"
            video_tasks[task_id]["output_path"] = output_path
            logger.info(f"✅ Video generation completed for task {task_id} using {video_tasks[task_id]['model_used']}")
        else:
            raise Exception("All video generation methods failed or output file missing")

    except Exception as e:
        logger.error(f"❌ Video generation failed for task {task_id}: {e}")
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = str(e)
        # Write error file so /video-status can return failure
        error_path = f"temp/errors/{task_id}.json"
        with open(error_path, "w") as f:
            json.dump({"status": "failed", "error": str(e)}, f)
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
    quality: str = Form(default="fast")):
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
           return {
              "status": "failed",
              "error": error_data.get("error", "Unknown error")
             }

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

@app.post("/init-stream", dependencies=[Depends(verify_api_key)])
async def init_stream(request: Request):
    """Initialize a video streaming session"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        avatar_id = data.get("avatar_id")
        image_url = data.get("image_url")
        
        if not session_id or not avatar_id or not image_url:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Missing required parameters"}
            )
        
        # Create session directory
        session_dir = os.path.join("temp/streams", session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Download avatar image
        avatar_path = os.path.join(session_dir, "avatar.jpg")
        response = requests.get(image_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(avatar_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Initialize session data
        active_streams[session_id] = {
            "avatar_id": avatar_id,
            "avatar_path": avatar_path,
            "created_at": time.time(),
            "last_activity": time.time(),
            "is_speaking": False
        }
        
        # Pre-process avatar for faster streaming
        img = cv2.imread(avatar_path)
        if img is not None:
            # Detect face landmarks
            face_data = detect_face_landmarks(img)
            if face_data:
                active_streams[session_id]["face_data"] = face_data
                
                # Generate idle animation frames
                idle_frames_dir = os.path.join(session_dir, "idle_frames")
                os.makedirs(idle_frames_dir, exist_ok=True)
                
                # Generate 50 idle animation frames (2 seconds at 25fps)
                for i in range(50):
                    idle_frame = create_face_animated_frame(
                        img.copy(),
                        face_data,
                        i,
                        25,
                        {"intensity": 0.0, "is_speaking": False}
                    )
                    cv2.imwrite(os.path.join(idle_frames_dir, f"idle_{i:03d}.jpg"), idle_frame)
                
                active_streams[session_id]["idle_frames_dir"] = idle_frames_dir
        
        return {"status": "success", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Error initializing stream: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/end-stream", dependencies=[Depends(verify_api_key)])
async def end_stream(request: Request):
    """End a video streaming session"""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        
        if not session_id:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Missing session_id"}
            )
        
        if session_id in active_streams:
            # Clean up session data
            session_dir = os.path.join("temp/streams", session_id)
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir, ignore_errors=True)
            
            del active_streams[session_id]
            
            return {"status": "success", "message": "Stream ended"}
        else:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Session not found"}
            )
            
    except Exception as e:
        logger.error(f"Error ending stream: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.websocket("/stream/{session_id}")
async def stream_video(websocket: WebSocket, session_id: str):
    """Stream video frames for real-time avatar animation"""
    await websocket.accept()
    
    if session_id not in active_streams:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return
    
    session_data = active_streams[session_id]
    avatar_path = session_data["avatar_path"]
    face_data = session_data.get("face_data")
    idle_frames_dir = session_data.get("idle_frames_dir")
    
    # Load avatar image
    img = cv2.imread(avatar_path)
    if img is None:
        await websocket.send_json({"type": "error", "message": "Failed to load avatar image"})
        await websocket.close()
        return
    
    # If no face data, try to detect face
    if not face_data:
        face_data = detect_face_landmarks(img)
        if face_data:
            session_data["face_data"] = face_data

    # Initialize variables
    is_speaking = False
    audio_buffer = []
    frame_rate = 25
    frame_interval = 1.0 / frame_rate
    idle_frame_index = 0
    last_frame_time = time.time()
    
    try:
        # Send ready message
        await websocket.send_json({"type": "ready"})
        
        # Main streaming loop
        while True:
            # Update last activity time
            session_data["last_activity"] = time.time()
            
            # Check for incoming messages
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=0.01)
                
                if isinstance(data, dict) and data.get("type") == "text":
                    message = json.loads(data["text"])
                    
                    if message.get("type") == "speech_start":
                        is_speaking = True
                        session_data["is_speaking"] = True
                    elif message.get("type") == "speech_end":
                        is_speaking = False
                        session_data["is_speaking"] = False
                    elif message.get("type") == "stop_speaking":
                        is_speaking = False
                        session_data["is_speaking"] = False
                        audio_buffer = []
                        
                elif isinstance(data, bytes) or isinstance(data, bytearray):
                    # Audio data for lip sync
                    audio_buffer.append(data)
            except asyncio.TimeoutError:
                pass
            
            # Check if it's time to send a new frame
            current_time = time.time()
            if current_time - last_frame_time >= frame_interval:
                last_frame_time = current_time
                
                # Generate frame based on state
                if is_speaking and face_data:
                    # Generate speaking frame with lip sync
                    # Use audio buffer to determine mouth shape
                    audio_intensity = 0.5  # Default intensity
                    if audio_buffer:
                        # Simple analysis of latest audio chunk
                        latest_audio = audio_buffer[-1]
                        if isinstance(latest_audio, bytes) or isinstance(latest_audio, bytearray):
                            try:
                                # Convert to numpy array and calculate intensity
                                audio_np = np.frombuffer(latest_audio, dtype=np.int16).astype(np.float32) / 32768.0
                                audio_intensity = min(1.0, np.sqrt(np.mean(audio_np ** 2)) * 5.0)
                            except Exception as e:
                                logger.error(f"Error processing audio: {e}")
                    
                    # Create animated frame
                    frame = create_face_animated_frame(
                        img.copy(),
                        face_data,
                        int(time.time() * frame_rate) % 1000,  # Frame index based on time
                        frame_rate,
                        {
                            "intensity": audio_intensity,
                            "is_speaking": True,
                            "spectral_centroid": 1000 + 500 * audio_intensity
                        }
                    )
                    
                    # Limit audio buffer size
                    if len(audio_buffer) > 10:
                        audio_buffer = audio_buffer[-10:]
                        
                elif idle_frames_dir and os.path.exists(idle_frames_dir):
                    # Use pre-generated idle animation frames
                    idle_frame_path = os.path.join(idle_frames_dir, f"idle_{idle_frame_index:03d}.jpg")
                    if os.path.exists(idle_frame_path):
                        frame = cv2.imread(idle_frame_path)
                        idle_frame_index = (idle_frame_index + 1) % 50  # Loop through 50 frames
                    else:
                        # Fallback to basic animation
                        frame = create_face_animated_frame(
                            img.copy(),
                            face_data,
                            int(time.time() * frame_rate) % 1000,
                            frame_rate,
                            {"intensity": 0.0, "is_speaking": False}
                        )
                else:
                    # Fallback to basic animation
                    frame = create_face_animated_frame(
                        img.copy(),
                        face_data,
                        int(time.time() * frame_rate) % 1000,
                        frame_rate,
                        {"intensity": 0.0, "is_speaking": False}
                    )
                
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                # Send frame to client
                await websocket.send_bytes(buffer.tobytes())
                
                # Send frame ready message
                await websocket.send_json({"type": "frame_ready"})
            
            # Small delay to prevent CPU overload
            await asyncio.sleep(0.01)
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from stream {session_id}")
    except Exception as e:
        logger.error(f"Error in video stream: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        # Update session data
        if session_id in active_streams:
            active_streams[session_id]["is_speaking"] = False

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
            "lip_sync": True
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
        "version": "3.0.0",
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
            "lip_sync": True,
            "head_movement": True,
            "eye_blink": True,
            "face_detection": True
        },
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Professional Video Generation Service",
        "version": "3.0.0",
        "status": "running",
        "features": [
            "SadTalker Integration",
            "Wav2Lip Integration", 
            "Real-time Video Streaming",
            "Professional Lip Sync",
            "Head Movement Animation",
            "Eye Blink Animation",
            "Face Detection with MediaPipe",
            "Animated Video Fallback",
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
