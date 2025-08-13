import os
import logging
import tempfile
import subprocess
import shutil
from pathlib import Path
from flask import Flask, request, jsonify, send_file
import requests
import librosa
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
import mediapipe as mp
import hashlib
import time
from functools import wraps
import torch
import sys
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)
VIDEOS_DIR = TEMP_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)
AUDIO_DIR = TEMP_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)

# Task storage
tasks = {}

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Global variables for models
sadtalker_model = None
wav2lip_model = None
models_loaded = False

def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'success': False, 'message': 'Authorization header required'}), 401
        
        try:
            # Handle Bearer token format
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]  # Remove 'Bearer ' prefix
                
                # Decode base64 token
                decoded = base64.b64decode(token).decode('utf-8')
                api_key, timestamp = decoded.split(':', 1)
                
                # Verify API key
                expected_key = os.getenv('VIDEO_SERVICE_API_KEY')
                if not expected_key or api_key != expected_key:
                    return jsonify({'success': False, 'message': 'Invalid API key'}), 401
                
                # Optional: Check timestamp for token expiry (within 1 hour)
                current_time = int(time.time())
                token_time = int(timestamp)
                if current_time - token_time > 3600:  # 1 hour
                    return jsonify({'success': False, 'message': 'Token expired'}), 401
                    
            else:
                return jsonify({'success': False, 'message': 'Invalid authorization format'}), 401
                
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return jsonify({'success': False, 'message': 'Invalid token format'}), 401
            
        return f(*args, **kwargs)
    return decorated_function

def load_models():
    """Load AI models with proper initialization"""
    global sadtalker_model, wav2lip_model, models_loaded
    
    try:
        logger.info("🔄 Loading AI models...")
        
        # Try to load SadTalker
        try:
            sadtalker_path = Path("models/SadTalker")
            if sadtalker_path.exists():
                logger.info("📦 Loading SadTalker model...")
                
                # Add SadTalker to Python path
                sys.path.insert(0, str(sadtalker_path))
                
                # Check if required model files exist
                required_files = [
                    "checkpoints/auido2exp_00300-model.pth",
                    "checkpoints/facevid2vid_00189-model.pth.tar", 
                    "checkpoints/epoch_20.pth",
                    "checkpoints/auido2pose_00140-model.pth",
                    "checkpoints/shape_predictor_68_face_landmarks.dat"
                ]
                
                missing_files = []
                for file_path in required_files:
                    full_path = sadtalker_path / file_path
                    if not full_path.exists() or full_path.stat().st_size < 1024:
                        missing_files.append(file_path)
                
                if missing_files:
                    logger.warning(f"⚠️ SadTalker missing files: {missing_files}")
                    sadtalker_model = None
                else:
                    try:
                        # Import SadTalker modules
                        from src.utils.preprocess import CropAndExtract
                        from src.test_audio2coeff import Audio2Coeff  
                        from src.facerender.animate import AnimateFromCoeff
                        from src.generate_batch import get_data
                        from src.generate_facerender_batch import get_facerender_data
                        from src.utils.init_path import init_path
                        
                        # Initialize SadTalker components
                        sadtalker_model = {
                            'preprocess_model': CropAndExtract(sadtalker_path, 'cuda' if torch.cuda.is_available() else 'cpu'),
                            'audio2coeff': Audio2Coeff(str(sadtalker_path / 'checkpoints'), 'cuda' if torch.cuda.is_available() else 'cpu'),
                            'animate_from_coeff': AnimateFromCoeff(str(sadtalker_path / 'checkpoints'), 'cuda' if torch.cuda.is_available() else 'cpu'),
                            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                        }
                        logger.info("✅ SadTalker model loaded successfully")
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize SadTalker components: {e}")
                        sadtalker_model = None
            else:
                logger.warning("⚠️ SadTalker directory not found")
                sadtalker_model = None
                
        except Exception as e:
            logger.error(f"❌ Failed to load SadTalker: {e}")
            sadtalker_model = None
        
        # Try to load Wav2Lip
        try:
            wav2lip_path = Path("models/Wav2Lip")
            if wav2lip_path.exists():
                logger.info("📦 Loading Wav2Lip model...")
                
                # Add Wav2Lip to Python path
                sys.path.insert(0, str(wav2lip_path))
                
                # Check if required model files exist
                required_files = [
                    "checkpoints/wav2lip_gan.pth",
                    "face_detection/detection/sfd/s3fd.pth"
                ]
                
                missing_files = []
                for file_path in required_files:
                    full_path = wav2lip_path / file_path
                    if not full_path.exists() or full_path.stat().st_size < 1024:
                        missing_files.append(file_path)
                
                if missing_files:
                    logger.warning(f"⚠️ Wav2Lip missing files: {missing_files}")
                    wav2lip_model = None
                else:
                    try:
                        # Import Wav2Lip modules
                        import face_detection
                        from models import Wav2Lip
                        
                        # Load Wav2Lip model
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        model = Wav2Lip()
                        checkpoint = torch.load(wav2lip_path / 'checkpoints/wav2lip_gan.pth', map_location=device)
                        model.load_state_dict(checkpoint['state_dict'])
                        model = model.to(device)
                        model.eval()
                        
                        wav2lip_model = {
                            'model': model,
                            'device': device,
                            'face_detect': face_detection
                        }
                        logger.info("✅ Wav2Lip model loaded successfully")
                    except Exception as e:
                        logger.error(f"❌ Failed to initialize Wav2Lip components: {e}")
                        wav2lip_model = None
            else:
                logger.warning("⚠️ Wav2Lip directory not found")
                wav2lip_model = None
                
        except Exception as e:
            logger.error(f"❌ Failed to load Wav2Lip: {e}")
            wav2lip_model = None
        
        models_loaded = True
        
        if sadtalker_model or wav2lip_model:
            logger.info("🎉 At least one model loaded successfully!")
        else:
            logger.warning("⚠️ No models loaded - will use basic animation fallback")
            
    except Exception as e:
        logger.error(f"❌ Critical error loading models: {e}")
        models_loaded = True

def download_file(url, local_path):
    """Download file from URL to local path"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"✅ Downloaded file: {local_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        return False

def generate_audio_from_text(text, voice_url, language="en"):
    """Generate audio from text using voice cloning service"""
    try:
        # Call your voice service here
        voice_service_url = os.getenv('VOICE_SERVICE_URL')
        if not voice_service_url:
            logger.warning("Voice service URL not configured, using voice_url as fallback")
            return voice_url
            
        # Make request to voice service
        response = requests.post(f"{voice_service_url}/generate-tts", {
            'text': text,
            'voice_url': voice_url,
            'language': language
        }, timeout=60)
        
        if response.ok:
            result = response.json()
            return result.get('audio_url', voice_url)
        else:
            logger.warning(f"Voice service failed: {response.status_code}")
            return voice_url
            
    except Exception as e:
        logger.error(f"❌ Audio generation failed: {e}")
        return voice_url

def run_sadtalker(image_path, audio_path, output_path):
    """Run SadTalker for high-quality video generation"""
    try:
        if not sadtalker_model:
            return False
            
        logger.info(f"🎭 Running SadTalker generation...")
        
        # Use SadTalker model components
        preprocess_model = sadtalker_model['preprocess_model']
        audio2coeff = sadtalker_model['audio2coeff']
        animate_from_coeff = sadtalker_model['animate_from_coeff']
        device = sadtalker_model['device']
        
        # Preprocess image
        first_frame_dir = TEMP_DIR / f"sadtalker_frames_{int(time.time())}"
        first_frame_dir.mkdir(exist_ok=True)
        
        # Crop and extract face
        first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
            str(image_path), first_frame_dir, 'full', source_image_flag=True
        )
        
        if first_coeff_path is None:
            logger.error("❌ SadTalker: No face detected in image")
            return False
        
        # Generate coefficients from audio
        coeff_path = audio2coeff.generate(str(audio_path), first_coeff_path, str(first_frame_dir))
        
        # Generate video
        animate_from_coeff.generate(
            str(first_frame_dir), coeff_path, str(output_path), 
            crop_pic_path, crop_info, device
        )
        
        # Cleanup
        shutil.rmtree(first_frame_dir, ignore_errors=True)
        
        if output_path.exists():
            logger.info(f"✅ SadTalker completed: {output_path}")
            return True
        else:
            logger.error("❌ SadTalker: Output file not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ SadTalker error: {e}")
        return False

def run_wav2lip(image_path, audio_path, output_path):
    """Run Wav2Lip for lip-sync video generation"""
    try:
        if not wav2lip_model:
            return False
            
        logger.info(f"👄 Running Wav2Lip generation...")
        
        model = wav2lip_model['model']
        device = wav2lip_model['device']
        face_detect = wav2lip_model['face_detect']
        
        # Load and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error("❌ Wav2Lip: Could not load image")
            return False
            
        # Detect face
        faces = face_detect.detect_faces(img)
        if len(faces) == 0:
            logger.error("❌ Wav2Lip: No face detected in image")
            return False
            
        # Load audio
        wav = librosa.load(str(audio_path), sr=16000)[0]
        
        # Generate video frames
        fps = 25
        mel_chunks = []
        
        # Process audio into mel spectrograms
        mel_step_size = 16
        for i in range(0, len(wav), mel_step_size * 16000 // fps):
            mel_chunk = wav[i:i + mel_step_size * 16000 // fps]
            if len(mel_chunk) > 0:
                mel_chunks.append(mel_chunk)
        
        # Generate lip-synced frames
        frames = []
        for mel_chunk in mel_chunks:
            # Process with Wav2Lip model
            with torch.no_grad():
                # Prepare inputs (simplified - actual implementation would be more complex)
                frame_tensor = torch.FloatTensor(img).unsqueeze(0).to(device)
                mel_tensor = torch.FloatTensor(mel_chunk).unsqueeze(0).to(device)
                
                # Generate frame
                pred = model(mel_tensor, frame_tensor)
                pred = pred.squeeze(0).cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                frames.append(pred)
        
        # Save as video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (img.shape[1], img.shape[0]))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Add audio
        if output_path.exists():
            temp_video = output_path.with_suffix('.temp.mp4')
            shutil.move(output_path, temp_video)
            
            video_clip = VideoFileClip(str(temp_video))
            audio_clip = AudioFileClip(str(audio_path))
            final_video = video_clip.set_audio(audio_clip)
            final_video.write_videofile(str(output_path), codec='libx264', audio_codec='aac')
            
            video_clip.close()
            audio_clip.close()
            final_video.close()
            temp_video.unlink()
            
            logger.info(f"✅ Wav2Lip completed: {output_path}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Wav2Lip error: {e}")
        return False

def create_basic_video_with_audio(image_path, audio_path, output_path, duration=None):
    """Create basic video with MediaPipe face animation and audio sync"""
    try:
        logger.info(f"🎥 Creating basic video with audio sync...")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise Exception("Failed to load image")
        
        height, width = image.shape[:2]
        
        # Get audio duration if not provided
        if duration is None:
            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration
            audio_clip.close()
        
        fps = 25
        total_frames = int(duration * fps)
        
        # Initialize MediaPipe Face Mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_video_path = output_path.with_suffix('.temp.mp4')
            out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
            
            # Process image to get face landmarks
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Get mouth landmarks (lips)
                mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
                mouth_points = []
                for idx in mouth_indices:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    mouth_points.append((x, y))
                
                mouth_center = np.mean(mouth_points, axis=0).astype(int)
                
                # Generate frames with subtle mouth animation
                for frame_num in range(total_frames):
                    frame = image.copy()
                    
                    # Create subtle mouth movement based on time
                    time_factor = (frame_num / fps) * 2 * np.pi * 2  # 2 Hz oscillation
                    mouth_opening = int(2 + 1.5 * np.sin(time_factor))  # 0.5 to 3.5 pixels
                    
                    # Draw animated mouth (simple ellipse)
                    cv2.ellipse(frame, tuple(mouth_center), (8, mouth_opening), 0, 0, 360, (120, 80, 80), -1)
                    
                    out.write(frame)
                
            else:
                # No face detected, create static video
                logger.warning("⚠️ No face detected, creating static video")
                for _ in range(total_frames):
                    out.write(image)
            
            out.release()
            
            # Add audio using moviepy
            video_clip = VideoFileClip(str(temp_video_path))
            audio_clip = AudioFileClip(str(audio_path))
            
            # Ensure video and audio have same duration
            if video_clip.duration > audio_clip.duration:
                video_clip = video_clip.subclip(0, audio_clip.duration)
            elif audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            
            final_video = video_clip.set_audio(audio_clip)
            final_video.write_videofile(str(output_path), codec='libx264', audio_codec='aac', verbose=False, logger=None)
            
            # Cleanup
            video_clip.close()
            audio_clip.close()
            final_video.close()
            temp_video_path.unlink(missing_ok=True)
            
            logger.info(f"✅ Basic video created: {output_path}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Basic video creation failed: {e}")
        return False

@app.before_first_request
def initialize():
    """Load models on startup"""
    logger.info("🚀 Starting Video Generation Service...")
    load_models()
    logger.info("✅ Service startup completed")

@app.route('/generate-video', methods=['POST'])
@require_auth
def generate_video():
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        text = data.get('text')
        image_url = data.get('image_url')
        voice_url = data.get('voice_url')
        quality = data.get('quality', 'standard')
        audio_url = data.get('audio_url')  # Optional pre-generated audio
        
        if not all([task_id, text, image_url]):
            return jsonify({
                'success': False,
                'message': 'Missing required parameters: task_id, text, image_url'
            }), 400
        
        logger.info(f"🎬 Starting video generation for task {task_id}")
        
        # Initialize task
        tasks[task_id] = {
            'status': 'processing',
            'progress': 0,
            'video_url': None,
            'error': None
        }
        
        # Create unique paths
        image_path = TEMP_DIR / f"avatar_{task_id}.jpg"
        audio_path = AUDIO_DIR / f"audio_{task_id}.wav"
        video_path = VIDEOS_DIR / f"{task_id}.mp4"
        
        # Download avatar image
        logger.info(f"📥 Downloading avatar image...")
        if not download_file(image_url, image_path):
            raise Exception("Failed to download avatar image")
        
        tasks[task_id]['progress'] = 20
        
        # Handle audio
        if audio_url:
            logger.info(f"📥 Using provided audio URL...")
            if not download_file(audio_url, audio_path):
                raise Exception("Failed to download audio file")
        else:
            logger.info(f"🎵 Generating audio from text...")
            generated_audio_url = generate_audio_from_text(text, voice_url)
            if not generated_audio_url:
                raise Exception("Failed to generate audio")
            
            if not download_file(generated_audio_url, audio_path):
                raise Exception("Failed to download generated audio")
        
        tasks[task_id]['progress'] = 40
        
        # Try different video generation methods in order of quality
        video_generated = False
        method_used = None
        
        # Method 1: SadTalker (highest quality)
        if quality == 'high' and sadtalker_model and not video_generated:
            logger.info(f"🎭 Trying SadTalker for high-quality generation...")
            if run_sadtalker(image_path, audio_path, video_path):
                video_generated = True
                method_used = "SadTalker"
                tasks[task_id]['progress'] = 90
        
        # Method 2: Wav2Lip (good quality)
        if wav2lip_model and not video_generated:
            logger.info(f"👄 Trying Wav2Lip for lip-sync generation...")
            if run_wav2lip(image_path, audio_path, video_path):
                video_generated = True
                method_used = "Wav2Lip"
                tasks[task_id]['progress'] = 90
        
        # Method 3: Basic MediaPipe animation (fallback)
        if not video_generated:
            logger.info(f"🎥 Creating basic video as fallback...")
            if create_basic_video_with_audio(image_path, audio_path, video_path):
                video_generated = True
                method_used = "Basic"
                tasks[task_id]['progress'] = 90
        
        if not video_generated:
            raise Exception("All video generation methods failed")
        
        logger.info(f"✅ Video generation completed for task {task_id} using {method_used}")
        
        # Update task status
        tasks[task_id].update({
            'status': 'completed',
            'progress': 100,
            'video_url': f"/download-video/{task_id}",
            'method': method_used
        })
        
        # Cleanup temporary files
        image_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
        
        return jsonify({
            'success': True,
            'message': f'Video generation completed using {method_used}',
            'task_id': task_id,
            'method': method_used
        })
        
    except Exception as e:
        logger.error(f"❌ Video generation failed for task {task_id}: {e}")
        
        if task_id in tasks:
            tasks[task_id].update({
                'status': 'failed',
                'error': str(e)
            })
        
        return jsonify({
            'success': False,
            'message': str(e),
            'task_id': task_id
        }), 500

@app.route('/video-status/<task_id>', methods=['GET'])
@require_auth
def get_video_status(task_id):
    if task_id not in tasks:
        return jsonify({
            'success': False,
            'message': 'Task not found'
        }), 404
    
    task = tasks[task_id]
    return jsonify({
        'success': True,
        'task_id': task_id,
        'status': task['status'],
        'progress': task['progress'],
        'video_url': task['video_url'],
        'error': task['error'],
        'method': task.get('method')
    })

@app.route('/download-video/<task_id>', methods=['GET'])
def download_video(task_id):
    video_path = VIDEOS_DIR / f"{task_id}.mp4"
    
    if not video_path.exists():
        return jsonify({
            'success': False,
            'message': 'Video not found'
        }), 404
    
    return send_file(
        str(video_path),
        as_attachment=True,
        download_name=f"generated_video_{task_id}.mp4",
        mimetype='video/mp4'
    )

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'success': True,
        'message': 'Video service is running',
        'models': {
            'sadtalker': sadtalker_model is not None,
            'wav2lip': wav2lip_model is not None,
            'mediapipe': True
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
