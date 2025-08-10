from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
import asyncio
import aiofiles
from pathlib import Path
import tempfile
import shutil
from typing import Optional, Dict, Any, Tuple, List
import json
import base64
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration ---
app = FastAPI(title="Professional Video Generation Service", version="3.0.0")
executor = ThreadPoolExecutor(max_workers=4)
video_tasks: Dict[str, dict] = {}

# Model paths and configuration
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
TEMP_DIR = os.environ.get("TEMP_DIR", "/app/temp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
API_KEY = os.getenv("VIDEO_SERVICE_API_KEY", "default-key")

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs("temp/videos", exist_ok=True)
os.makedirs("temp/errors", exist_ok=True)
os.makedirs("temp/avatars", exist_ok=True)

logger.info(f"🚀 Professional Video Service Starting")
logger.info(f"📱 Device: {DEVICE}")
logger.info(f"📁 Models Directory: {MODELS_DIR}")
logger.info(f"🔧 Temp Directory: {TEMP_DIR}")

if DEVICE == "cuda":
    logger.info(f"🎮 CUDA version: {torch.version.cuda}")
    logger.info(f"🎯 GPU count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        logger.info(f"🎪 GPU 0: {torch.cuda.get_device_name(0)}")

# Global model instances
sadtalker_model = None
wav2lip_model = None
face_detection_model = None

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
sadtalker_models = {}
wav2lip_models = {}
realtime_models = {}
preprocessed_avatars = {}
active_streams = {}

# Global variables for model loading
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
    "s3fd": f"{MODELS_DIR}/Wav2Lip/checkpoints/s3fd.pth",
    "wav2lip": f"{MODELS_DIR}/Wav2Lip/checkpoints/wav2lip.pth"
}

class VideoGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    async def load_sadtalker_models(self):
        """Load SadTalker models"""
        try:
            logger.info("🎭 Loading SadTalker models...")
            
            # Clone SadTalker repository if not exists
            sadtalker_path = os.path.join(MODELS_DIR, "SadTalker")
            if not os.path.exists(sadtalker_path):
                logger.info("📥 Cloning SadTalker repository...")
                subprocess.run([
                    "git", "clone", "https://github.com/OpenTalker/SadTalker.git", sadtalker_path
                ], check=True)
                
            # Add SadTalker to Python path
            sys.path.insert(0, sadtalker_path)
            
            # Download models if they don't exist
            await self._download_sadtalker_models()
            
            logger.info("✅ SadTalker models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load SadTalker models: {e}")
            return False
    
    async def _download_sadtalker_models(self):
        """Download SadTalker model files"""
        model_urls = {
            "auido2exp_00300-model.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/auido2exp_00300-model.pth",
            "facevid2vid_00189-model.pth.tar": "https://huggingface.co/vinthony/SadTalker/resolve/main/facevid2vid_00189-model.pth.tar",
            "epoch_20.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/epoch_20.pth",
            "auido2pose_00140-model.pth": "https://huggingface.co/vinthony/SadTalker/resolve/main/auido2pose_00140-model.pth",
            "shape_predictor_68_face_landmarks.dat": "https://huggingface.co/vinthony/SadTalker/resolve/main/shape_predictor_68_face_landmarks.dat"
        }
        
        checkpoints_dir = os.path.join(MODELS_DIR, "SadTalker", "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        for filename, url in model_urls.items():
            file_path = os.path.join(checkpoints_dir, filename)
            if not os.path.exists(file_path):
                logger.info(f"📥 Downloading {filename}...")
                await self._download_file(url, file_path)
    
    async def load_wav2lip_models(self):
        """Load Wav2Lip models"""
        try:
            logger.info("🎤 Loading Wav2Lip models...")
            
            # Clone Wav2Lip repository if not exists
            wav2lip_path = os.path.join(MODELS_DIR, "Wav2Lip")
            if not os.path.exists(wav2lip_path):
                logger.info("📥 Cloning Wav2Lip repository...")
                subprocess.run([
                    "git", "clone", "https://github.com/Rudrabha/Wav2Lip.git", wav2lip_path
                ], check=True)
            
            # Download models if they don't exist
            await self._download_wav2lip_models()
            
            logger.info("✅ Wav2Lip models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Wav2Lip models: {e}")
            return False
    
    async def _download_wav2lip_models(self):
        """Download Wav2Lip model files"""
        model_urls = {
            "wav2lip_gan.pth": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
            "s3fd.pth": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/s3fd-619a316812.pth",
            "wav2lip.pth": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip.pth"
        }
        
        checkpoints_dir = os.path.join(MODELS_DIR, "Wav2Lip", "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        for filename, url in model_urls.items():
            file_path = os.path.join(checkpoints_dir, filename)
            if not os.path.exists(file_path):
                logger.info(f"📥 Downloading {filename}...")
                await self._download_file(url, file_path)
    
    async def _download_file(self, url: str, file_path: str):
        """Download a file from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Downloaded {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"❌ Failed to download {url}: {e}")
            raise
    
    async def generate_video_sadtalker(self, image_path: str, audio_path: str, output_path: str, quality: str = "high") -> bool:
        """Generate video using SadTalker"""
        try:
            logger.info("🎭 Generating video with SadTalker...")
            
            sadtalker_path = os.path.join(MODELS_DIR, "SadTalker")
            if not os.path.exists(sadtalker_path):
                logger.error("❌ SadTalker not found")
                return False
            
            # SadTalker inference command
            cmd = [
                "python", os.path.join(sadtalker_path, "inference.py"),
                "--driven_audio", audio_path,
                "--source_image", image_path,
                "--result_dir", str(Path(output_path).parent),
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
            
            logger.info(f"🚀 Running SadTalker: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=sadtalker_path, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"❌ SadTalker failed: {result.stderr}")
                return False
            
            # Find generated video file
            result_dir = Path(output_path).parent
            generated_files = list(result_dir.glob("*.mp4"))
            
            if not generated_files:
                logger.error("❌ No video file generated by SadTalker")
                return False
            
            # Move to expected output path
            shutil.move(str(generated_files[0]), output_path)
            
            logger.info(f"✅ SadTalker generation completed: {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ SadTalker generation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ SadTalker generation error: {e}")
            return False
    
    async def generate_video_wav2lip(self, image_path: str, audio_path: str, output_path: str, quality: str = "fast") -> bool:
        """Generate video using Wav2Lip"""
        try:
            logger.info("🎤 Generating video with Wav2Lip...")
            
            wav2lip_path = os.path.join(MODELS_DIR, "Wav2Lip")
            if not os.path.exists(wav2lip_path):
                logger.error("❌ Wav2Lip not found")
                return False
            
            # Wav2Lip inference
            model_path = os.path.join(wav2lip_path, "checkpoints", "wav2lip_gan.pth")
            cmd = [
                "python", os.path.join(wav2lip_path, "inference.py"),
                "--checkpoint_path", model_path,
                "--face", image_path,
                "--audio", audio_path,
                "--outfile", output_path,
                "--fps", "25",
                "--pads", "0", "10", "0", "0"
            ]
            
            if quality == "fast":
                cmd.append("--static")
            
            logger.info(f"🚀 Running Wav2Lip: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=wav2lip_path, timeout=180)
            
            if result.returncode != 0:
                logger.error(f"❌ Wav2Lip failed: {result.stderr}")
                return False
            
            logger.info(f"✅ Wav2Lip generation completed: {output_path}")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Wav2Lip generation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Wav2Lip generation error: {e}")
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
    logger.info("🚀 Starting Avatar Video Generation Service...")
    await check_models()

# Initialize MediaPipe Face Detection (lighter alternative to dlib)
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

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
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Cleanup temp frames
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

# --- Real-time Video Streaming Classes ---
class RealtimeVideoStreamer:
    """Real-time video streaming with proper lip sync"""
    
    def __init__(self, avatar_id: str, image_path: str):
        self.avatar_id = avatar_id
        self.image_path = image_path
        self.base_image = None
        self.is_speaking = False
        self.frame_queue = asyncio.Queue(maxsize=5)
        self.audio_queue = Queue(maxsize=10)
        self.current_frame = None
        self.frame_count = 0
        
        # Load and preprocess image
        self.load_image()
        
        # Start frame generation thread
        self.frame_thread = threading.Thread(target=self._frame_generation_loop, daemon=True)
        self.frame_thread.start()
        
        logger.info(f"🎥 Real-time streamer initialized for avatar {avatar_id}")
    
    def load_image(self):
        """Load and preprocess the avatar image"""
        try:
            self.base_image = cv2.imread(self.image_path)
            if self.base_image is None:
                raise ValueError(f"Could not load image: {self.image_path}")
            
            # Resize to standard size
            self.base_image = cv2.resize(self.base_image, (512, 512))
            
            # Generate idle frame
            _, buffer = cv2.imencode('.jpg', self.base_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.current_frame = buffer.tobytes()
            
            logger.info(f"✅ Avatar image loaded and preprocessed: {self.image_path}")
            
        except Exception as e:
            logger.error(f"❌ Error loading avatar image: {e}")
            raise
    
    def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk for lip sync"""
        try:
            if not self.audio_queue.full():
                self.audio_queue.put(audio_data)
                self.is_speaking = True
        except Exception as e:
            logger.error(f"❌ Error adding audio chunk: {e}")
    
    def _frame_generation_loop(self):
        """Background thread for frame generation"""
        while True:
            try:
                # Get audio data if available
                audio_data = None
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                
                # Generate frame
                if audio_data:
                    frame = self._generate_speaking_frame(audio_data)
                else:
                    frame = self._generate_idle_frame()
                    self.is_speaking = False
                
                # Update current frame
                if frame:
                    self.current_frame = frame
                
                # Control frame rate (25 FPS)
                time.sleep(0.04)
                
            except Exception as e:
                logger.error(f"❌ Error in frame generation loop: {e}")
                time.sleep(0.1)
    
    def _generate_speaking_frame(self, audio_data: bytes) -> bytes:
        """Generate frame with lip sync animation"""
        try:
            img = self.base_image.copy()
            
            # Analyze audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) > 0:
                # Calculate audio intensity
                intensity = np.abs(audio_array).mean() / 32767.0
                
                # Simple lip sync animation
                if intensity > 0.1:
                    # Mouth center (approximate)
                    mouth_center = (256, 380)
                    
                    # Mouth opening based on audio intensity
                    mouth_width = int(20 + intensity * 25)
                    mouth_height = int(5 + intensity * 15)
                    
                    # Draw mouth opening
                    cv2.ellipse(img, mouth_center, (mouth_width, mouth_height), 0, 0, 180, (20, 20, 20), -1)
                    
                    # Add subtle head movement
                    if intensity > 0.3:
                        shift_x = int(np.sin(self.frame_count * 0.1) * intensity * 3)
                        shift_y = int(np.cos(self.frame_count * 0.15) * intensity * 2)
                        
                        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            
            self.frame_count += 1
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"❌ Error generating speaking frame: {e}")
            return self.current_frame
    
    def _generate_idle_frame(self) -> bytes:
        """Generate idle frame with subtle animation"""
        try:
            img = self.base_image.copy()
            
            # Subtle breathing effect
            scale = 1.0 + 0.01 * np.sin(self.frame_count * 0.05)
            height, width = img.shape[:2]
            
            # Slight scale change for breathing
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if new_width != width or new_height != height:
                img = cv2.resize(img, (new_width, new_height))
                
                # Center the image
                if new_width > width or new_height > height:
                    # Crop to original size
                    start_x = (new_width - width) // 2
                    start_y = (new_height - height) // 2
                    img = img[start_y:start_y+height, start_x:start_x+width]
                else:
                    # Pad to original size
                    pad_x = (width - new_width) // 2
                    pad_y = (height - new_height) // 2
                    img = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
            
            self.frame_count += 1
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"❌ Error generating idle frame: {e}")
            return self.current_frame
    
    async def get_frame(self) -> bytes:
        """Get current frame"""
        return self.current_frame if self.current_frame else b""

# Global streamers
active_streamers: Dict[str, RealtimeVideoStreamer] = {}

# --- Background Task Functions ---
async def _run_video_generation(task_id: str, image_url: str, audio_url: str, output_dir: str, quality: str):
    """Run video generation in background"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        logger.info(f"🎬 Starting video generation for task {task_id}")
        
        image_path = os.path.join(temp_dir, "input.jpg")
        audio_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(output_dir, "output.mp4")
        
        # Download files
        logger.info(f"📥 Downloading files for task {task_id}")
        await _download_file(image_url, image_path)
        await _download_file(audio_url, audio_path)
        
        # Update task status
        video_tasks[task_id]["status"] = "processing"
        
        # Try different generation methods based on quality and availability
        success = False
        
        if quality == "high" and sadtalker_available:
            video_tasks[task_id]["model_used"] = "SadTalker"
            success = await video_generator.generate_video_sadtalker(image_path, audio_path, output_path, quality)
            
            if not success and wav2lip_available:
                logger.info(f"🔄 SadTalker failed, falling back to Wav2Lip for task {task_id}")
                video_tasks[task_id]["model_used"] = "Wav2Lip"
                success = await video_generator.generate_video_wav2lip(image_path, audio_path, output_path, "fast")
        
        elif quality == "fast" and wav2lip_available:
            video_tasks[task_id]["model_used"] = "Wav2Lip"
            success = await video_generator.generate_video_wav2lip(image_path, audio_path, output_path, quality)
            
            if not success and sadtalker_available:
                logger.info(f"🔄 Wav2Lip failed, falling back to SadTalker for task {task_id}")
                video_tasks[task_id]["model_used"] = "SadTalker"
                success = await video_generator.generate_video_sadtalker(image_path, audio_path, output_path, "high")
        
        # Final fallback to animated video
        if not success:
            logger.info(f"🔄 Advanced models failed, using animated video for task {task_id}")
            video_tasks[task_id]["model_used"] = "Animated"
            try:
                create_animated_talking_video(image_path, audio_path, output_path, quality)
                success = True
            except Exception as e:
                logger.error(f"❌ Animated video failed: {e}")
                # Ultimate fallback to basic video
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
@app.post("/generate-video")
async def generate_video(
    background_tasks: BackgroundTasks,
    image_url: str = Form(...),
    audio_url: str = Form(...),
    quality: str = Form(default="fast")
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
        async def generate():
            async with aiofiles.open(video_path, 'rb') as f:
                while chunk := await f.read(8192):
                    yield chunk
        
        return StreamingResponse(
            generate(),
            media_type="video/mp4",
            headers={"Content-Disposition": f"attachment; filename={task_id}.mp4"}
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

@app.post("/preprocess-avatar")
async def preprocess_avatar(
    avatar_id: str = Form(...),
    image_url: str = Form(...)
):
    """Preprocess avatar for real-time video generation."""
    try:
        # Create avatar directory
        avatar_dir = f"temp/avatars/{avatar_id}"
        os.makedirs(avatar_dir, exist_ok=True)
        
        # Download and preprocess image
        response = requests.get(image_url)
        response.raise_for_status()
        
        image_path = f"{avatar_dir}/image.jpg"
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        # Initialize real-time streamer
        if avatar_id not in active_streamers:
            active_streamers[avatar_id] = RealtimeVideoStreamer(avatar_id, image_path)
        
        return {
            "status": "success",
            "avatar_id": avatar_id,
            "preprocessed": True,
            "streaming_ready": True
        }
    
    except Exception as e:
        logger.error(f"Error preprocessing avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

@app.get("/stream-frame/{avatar_id}")
async def stream_frame(avatar_id: str):
    """Get current frame for real-time streaming"""
    try:
        if avatar_id not in active_streamers:
            raise HTTPException(status_code=404, detail="Avatar not found or not preprocessed")
        
        streamer = active_streamers[avatar_id]
        frame_data = await streamer.get_frame()
        
        return StreamingResponse(
            io.BytesIO(frame_data),
            media_type="image/jpeg"
        )
    
    except Exception as e:
        logger.error(f"Error streaming frame for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@app.post("/add-audio-chunk/{avatar_id}")
async def add_audio_chunk(
    avatar_id: str,
    audio_file: UploadFile = File(...)
):
    """Add audio chunk for real-time lip sync"""
    try:
        if avatar_id not in active_streamers:
            raise HTTPException(status_code=404, detail="Avatar not found or not preprocessed")
        
        streamer = active_streamers[avatar_id]
        audio_data = await audio_file.read()
        streamer.add_audio_chunk(audio_data)
        
        return {
            "status": "success",
            "message": "Audio chunk added for lip sync"
        }
    
    except Exception as e:
        logger.error(f"Error adding audio chunk for avatar {avatar_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

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
        "active_streams": len(active_streamers),
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
            "preprocess_avatar": "/preprocess-avatar",
            "stream_frame": "/stream-frame/{avatar_id}",
            "add_audio_chunk": "/add-audio-chunk/{avatar_id}",
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
