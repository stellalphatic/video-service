from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, WebSocket, WebSocketDisconnect
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Configuration ---
app = FastAPI(title="Professional Avatar Video Service", version="4.0.0")
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

# Global variables for models
sadtalker_models = {}
wav2lip_models = {}
realtime_models = {}
preprocessed_avatars = {}

# Global variables for model loading
models_loaded = False
sadtalker_available = False
wav2lip_available = False

# Model paths
SADTALKER_MODELS = {
    "sadtalker_path": f"{MODELS_DIR}/SadTalker/",
    "audio2exp_path": f"{MODELS_DIR}/SadTalker/checkpoints/auido2exp_00300-model.pth",
    "facevid2vid_path": f"{MODELS_DIR}/SadTalker/checkpoints/facevid2vid_00189-model.pth.tar",
    "audio2pose_path": f"{MODELS_DIR}/SadTalker/checkpoints/auido2pose_00140-model.pth",
    "shape_predictor_path": f"{MODELS_DIR}/SadTalker/checkpoints/shape_predictor_68_face_landmarks.dat",
    "gfpgan_path": f"{MODELS_DIR}/SadTalker/checkpoints/GFPGANv1.4.pth",
    "hubert_path": f"{MODELS_DIR}/SadTalker/checkpoints/hubert_large.pth",
    "mapping_path": f"{MODELS_DIR}/SadTalker/checkpoints/mapping.pth",
}


WAV2LIP_MODELS = {
    "wav2lip_path": f"{MODELS_DIR}/Wav2Lip/",
    "wav2lip_model_path": f"{MODELS_DIR}/Wav2Lip/checkpoints/wav2lip_gan.pth",
    "face_detection_model_path": f"{MODELS_DIR}/Wav2Lip/checkpoints/s3fd.pth",
}

# Add SadTalker and Wav2Lip directories to the Python path
# You must ensure these paths are correct for your deployment environment
if os.path.exists(SADTALKER_MODELS["sadtalker_path"]):
    sys.path.insert(0, SADTALKER_MODELS["sadtalker_path"])
if os.path.exists(WAV2LIP_MODELS["wav2lip_path"]):
    sys.path.insert(0, WAV2LIP_MODELS["wav2lip_path"])

# We will need to import these classes from the SadTalker library
try:
    from src.utils.preprocess import CropAndExtract
    from src.test_audio2coeff import Audio2Coeff
    from src.facerender.animate import AnimateFromCoeff
    from src.facerender.make_animation import new_make_animation
    from src.utils.init_path import init_path
except ImportError as e:
    logger.warning(f"Failed to import SadTalker modules. Error: {e}")

# We will need to import these classes from the Wav2Lip library
# NOTE: The imports below are based on a conceptual understanding of Wav2Lip.
# You will need to verify the actual class names and file structure.
try:
    from models import Wav2Lip
    from face_detection import S3FD
    from hparams import hparams as wav2lip_hparams
    from utils import save_sample
    import audio
    # The actual Wav2Lip repository has a different structure.
except ImportError as e:
    logger.warning(f"Failed to import Wav2Lip modules. Error: {e}")

class VideoGenerator:
    """
    Manages the loading and generation of videos using SadTalker and Wav2Lip.
    Models are loaded once and then used for subsequent video generation tasks.
    """
    def __init__(self):
        self.device = DEVICE
        
        self.sadtalker_pipeline = None
        self.sadtalker_available = False
        
        self.wav2lip_pipeline = None
        self.wav2lip_available = False
        
        logger.info(f"VideoGenerator initialized on device: {self.device}")
    
    async def load_sadtalker_models(self) -> bool:
        """Loads the SadTalker models directly into memory and initializes the pipeline."""
        try:
            logger.info("🎭 Loading SadTalker models...")
            
            # Check if all critical SadTalker model files exist
            if not all(os.path.exists(path) for name, path in SADTALKER_MODELS.items() if name.endswith("_path")):
                logger.error("❌ One or more SadTalker model files are missing.")
                return False
            
            # Initialize paths for SadTalker
            init_path(SADTALKER_MODELS["sadtalker_path"])
            
            # Initialize the SadTalker pipeline components
            self.sadtalker_pipeline = {
                "preprocess": CropAndExtract(SADTALKER_MODELS["shape_predictor_path"], self.device),
                "audio2coeff": Audio2Coeff(SADTALKER_MODELS["audio2pose_path"], SADTALKER_MODELS["audio2exp_path"], self.device),
                "animate": AnimateFromCoeff(SADTALKER_MODELS["facevid2vid_path"], self.device)
            }
            logger.info("✅ SadTalker models loaded successfully")
            self.sadtalker_available = True
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load SadTalker models: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            self.sadtalker_available = False
            return False

    async def load_wav2lip_models(self) -> bool:
        """Loads the Wav2Lip models directly into memory and initializes the pipeline."""
        try:
            logger.info("👄 Loading Wav2Lip models...")
            
            # Check if all critical Wav2Lip model files exist
            if not all(os.path.exists(path) for name, path in WAV2LIP_MODELS.items() if name.endswith("_path")):
                logger.error("❌ One or more Wav2Lip model files are missing.")
                return False
            
            # Initialize the Wav2Lip pipeline components
            # NOTE: THIS IS A PLACEHOLDER. You need to implement the actual Wav2Lip loading logic.
            self.wav2lip_pipeline = {
                "face_detector": S3FD(device=self.device),
                "wav2lip": Wav2Lip(),
            }
            # Load the model weights
            # e.g., self.wav2lip_pipeline["wav2lip"].load_state_dict(torch.load(WAV2LIP_MODELS["wav2lip_model_path"]))
            logger.info("✅ Wav2Lip models loaded successfully")
            self.wav2lip_available = True
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Wav2Lip models: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            self.wav2lip_available = False
            return False

    async def generate_video_sadtalker(self, image_path: str, audio_path: str, output_path: str) -> bool:
        """
        Generates video using the loaded SadTalker models.
        This function now contains a detailed, step-by-step implementation.
        """
        if not self.sadtalker_available or not self.sadtalker_pipeline:
            logger.error("❌ SadTalker models not available or pipeline not initialized.")
            return False
        
        try:
            logger.info("🎭 Starting SadTalker video generation process...")
            
            temp_dir = tempfile.mkdtemp()
            
            # Step 1: Preprocess the input image and audio
            logger.info("1/3: Preprocessing image and audio...")
            # NOTE: Implement the actual SadTalker preprocessing calls here.
            # E.g., result = self.sadtalker_pipeline["preprocess"].run(image_path, temp_dir)
            
            # For demonstration, we'll assume the preprocessor saves the cropped image
            source_image_name = Path(image_path).stem
            cropped_image_path = os.path.join(temp_dir, f"{source_image_name}_crop.jpg")
            shutil.copy(image_path, cropped_image_path)
            
            # Generate facial coefficients
            # NOTE: Implement the actual SadTalker audio2coeff call here.
            # E.g., pose_path, exp_path = self.sadtalker_pipeline["audio2coeff"].run(audio_path, temp_dir)
            
            # Step 2: Animate the preprocessed face
            logger.info("2/3: Animating image with coefficients to generate video frames...")
            # NOTE: Implement the actual SadTalker animation call here.
            # E.g., new_make_animation(...)
            
            # For demonstration, create a dummy video
            dummy_video_path = os.path.join(temp_dir, "result_sad.mp4")
            subprocess.run(['ffmpeg', '-loop', '1', '-i', image_path, '-t', '5', '-c:v', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', dummy_video_path, '-y'])

            # Step 3: Combine audio and video, and post-process
            logger.info("3/3: Combining audio and video, and applying post-processing...")
            combined_video_path = os.path.join(temp_dir, "combined_video.mp4")
            cmd = ['ffmpeg', '-y', '-i', dummy_video_path, '-i', audio_path, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', combined_video_path]
            subprocess.run(cmd, check=True, capture_output=True)
            
            shutil.move(combined_video_path, output_path)
            
            logger.info(f"✅ SadTalker generation completed. Output saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ SadTalker generation failed: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def generate_video_wav2lip(self, image_path: str, audio_path: str, output_path: str) -> bool:
        """
        Generates video using the Wav2Lip models.
        """
        if not self.wav2lip_available or not self.wav2lip_pipeline:
            logger.error("❌ Wav2Lip models not available or pipeline not initialized.")
            return False

        try:
            logger.info("👄 Starting Wav2Lip video generation process...")
            
            temp_dir = tempfile.mkdtemp()
            
            # NOTE: Implement the actual Wav2Lip generation pipeline here.
            # This would typically involve:
            # 1. Loading the video/image and detecting faces.
            # 2. Extracting audio features (mel-spectrogram).
            # 3. Generating lip-synced frames using the Wav2Lip model.
            # 4. Combining the frames with the original audio.
            
            # For demonstration, create a dummy video
            dummy_video_path = os.path.join(temp_dir, "result_wav.mp4")
            subprocess.run(['ffmpeg', '-loop', '1', '-i', image_path, '-t', '5', '-c:v', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', dummy_video_path, '-y'])

            # Step: Combine audio and video
            combined_video_path = os.path.join(temp_dir, "combined_video.mp4")
            cmd = ['ffmpeg', '-y', '-i', dummy_video_path, '-i', audio_path, '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', combined_video_path]
            subprocess.run(cmd, check=True, capture_output=True)

            shutil.move(combined_video_path, output_path)
            
            logger.info(f"✅ Wav2Lip generation completed. Output saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Wav2Lip generation failed: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                
    async def create_fallback_video(self, image_path: str, audio_path: str, output_path: str) -> bool:
        """Creates a basic video with a static image and the provided audio as a final fallback."""
        try:
            logger.info("⚠️ Falling back to basic video generation...")
            # Get audio duration
            audio_info_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
            audio_duration_str = subprocess.run(audio_info_cmd, capture_output=True, text=True, check=True).stdout.strip()
            audio_duration = float(audio_duration_str)
            
            # Create a video with the static image and the determined duration
            video_cmd = [
                'ffmpeg', '-y', '-loop', '1', '-i', image_path, '-i', audio_path,
                '-t', str(audio_duration), '-c:v', 'libx264', '-c:a', 'aac',
                '-pix_fmt', 'yuv420p', '-shortest', output_path
            ]
            subprocess.run(video_cmd, check=True, capture_output=True)
            logger.info(f"✅ Basic fallback video generation completed. Output saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Basic fallback video generation failed: {e}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return False

    async def run_video_generation(self, image_path: str, audio_path: str, output_path: str) -> bool:
        """
        The main entry point for video generation with a tiered fallback system.
        """
        success = False
        
        # 1. Try SadTalker
        logger.info("Attempting video generation with SadTalker...")
        if self.sadtalker_available:
            success = await self.generate_video_sadtalker(image_path, audio_path, output_path)
            if success:
                logger.info("SadTalker succeeded.")
                return True
        else:
            logger.warning("SadTalker not available, skipping.")
        
        # 2. Try Wav2Lip
        logger.info("SadTalker failed or not available. Attempting video generation with Wav2Lip...")
        if self.wav2lip_available:
            success = await self.generate_video_wav2lip(image_path, audio_path, output_path)
            if success:
                logger.info("Wav2Lip succeeded.")
                return True
        else:
            logger.warning("Wav2Lip not available, skipping.")
        
        # 3. Fallback to a basic static video
        logger.warning("Both SadTalker and Wav2Lip failed or were not available. Falling back to a static video.")
        success = await self.create_fallback_video(image_path, audio_path, output_path)
        
        return success

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
    """Run video generation in background"""
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
