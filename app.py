import os
import sys
import tempfile
import shutil
import logging
import traceback
import subprocess
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify, send_file
import requests
from pathlib import Path
import json
import time
import mediapipe as mp
from scipy.io import wavfile
import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models
SADTALKER_AVAILABLE = False
WAV2LIP_AVAILABLE = False
MODELS_LOADED = False

# Store completed videos in memory (in production, use Redis or database)
completed_videos = {}

def fix_librosa_compatibility():
    """Fix librosa compatibility issues for Wav2Lip"""
    try:
        # Add compatibility layer for older librosa versions
        import librosa.filters
        if not hasattr(librosa.filters, 'mel'):
            # Create compatibility wrapper
            def mel_wrapper(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney', dtype=np.float32):
                return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk, norm=norm, dtype=dtype)
            librosa.filters.mel = mel_wrapper
        logger.info("✅ Librosa compatibility fixed")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to fix librosa compatibility: {e}")
        return False

def test_model_availability():
    """Test which models are actually available and can be imported"""
    global SADTALKER_AVAILABLE, WAV2LIP_AVAILABLE
    
    logger.info("🔍 Testing model availability...")
    
    # Fix librosa compatibility first
    fix_librosa_compatibility()
    
    # Test SadTalker
    try:
        logger.info("Testing SadTalker availability...")
        sadtalker_path = "/app/SadTalker"
        if os.path.exists(sadtalker_path):
            # Check for required files
            required_files = [
                "src/utils/preprocess.py",
                "src/test_audio2coeff.py",
                "src/facerender/animate.py",
                "inference.py"
            ]
            
            missing_files = []
            for file_path in required_files:
                full_path = os.path.join(sadtalker_path, file_path)
                if not os.path.exists(full_path):
                    missing_files.append(file_path)
            
            if missing_files:
                logger.warning(f"⚠️ SadTalker missing files: {missing_files}")
                SADTALKER_AVAILABLE = False
            else:
                # Test basic import
                sys.path.insert(0, sadtalker_path)
                try:
                    import kornia
                    import einops
                    from omegaconf import OmegaConf
                    SADTALKER_AVAILABLE = True
                    logger.info("✅ SadTalker is AVAILABLE and ready to use!")
                except ImportError as e:
                    logger.error(f"❌ SadTalker import failed: {e}")
                    SADTALKER_AVAILABLE = False
        else:
            logger.error("❌ SadTalker directory not found")
            SADTALKER_AVAILABLE = False
    except Exception as e:
        logger.error(f"❌ SadTalker NOT available: {str(e)}")
        SADTALKER_AVAILABLE = False
    
    # Test Wav2Lip
    try:
        logger.info("Testing Wav2Lip availability...")
        wav2lip_path = "/app/Wav2Lip"
        if os.path.exists(wav2lip_path):
            # Check for required files
            required_files = [
                "models/Wav2Lip.py",
                "inference.py",
                "audio.py"
            ]
            
            missing_files = []
            for file_path in required_files:
                full_path = os.path.join(wav2lip_path, file_path)
                if not os.path.exists(full_path):
                    missing_files.append(file_path)
            
            if missing_files:
                logger.warning(f"⚠️ Wav2Lip missing files: {missing_files}")
                WAV2LIP_AVAILABLE = False
            else:
                # Test basic import with fixed librosa
                sys.path.insert(0, wav2lip_path)
                try:
                    # Import with error handling
                    from models import Wav2Lip
                    import face_detection
                    WAV2LIP_AVAILABLE = True
                    logger.info("✅ Wav2Lip is AVAILABLE and ready to use!")
                except Exception as e:
                    logger.error(f"❌ Wav2Lip import failed: {e}")
                    WAV2LIP_AVAILABLE = False
        else:
            logger.error("❌ Wav2Lip directory not found")
            WAV2LIP_AVAILABLE = False
    except Exception as e:
        logger.error(f"❌ Wav2Lip NOT available: {str(e)}")
        WAV2LIP_AVAILABLE = False
    
    logger.info(f"📊 Model Status: SadTalker={SADTALKER_AVAILABLE}, Wav2Lip={WAV2LIP_AVAILABLE}")

def load_models():
    """Load and initialize models"""
    global MODELS_LOADED
    
    if MODELS_LOADED:
        return
    
    logger.info("🚀 Loading models...")
    
    try:
        # Test model availability first
        test_model_availability()
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        MODELS_LOADED = True
        logger.info("✅ Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        logger.error(traceback.format_exc())

def generate_video_with_sadtalker(image_path, audio_path, output_path):
    """Generate video using SadTalker"""
    try:
        logger.info("🎭 Using SadTalker for video generation")
        
        # Set up SadTalker paths
        sadtalker_dir = "/app/SadTalker"
        checkpoint_dir = "/app/checkpoints"
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare command with proper paths
        cmd = [
            "python", "inference.py",
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", os.path.dirname(output_path),
            "--checkpoint_dir", checkpoint_dir,
            "--still",
            "--preprocess", "full",
            "--enhancer", "gfpgan"
        ]
        
        logger.info(f"Running SadTalker command: {' '.join(cmd)}")
        
        # Set environment variables
        env = os.environ.copy()
        env['PYTHONPATH'] = sadtalker_dir
        
        # Run SadTalker
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=sadtalker_dir, env=env, timeout=300)
        
        logger.info(f"SadTalker stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"SadTalker stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise Exception(f"SadTalker failed with return code {result.returncode}: {result.stderr}")
        
        # Find the generated video file
        result_dir = os.path.dirname(output_path)
        generated_files = []
        for root, dirs, files in os.walk(result_dir):
            for file in files:
                if file.endswith('.mp4'):
                    generated_files.append(os.path.join(root, file))
        
        if generated_files:
            # Use the first generated video
            generated_path = generated_files[0]
            shutil.move(generated_path, output_path)
            logger.info(f"✅ SadTalker video generated: {output_path}")
            return True
        else:
            raise Exception("SadTalker did not generate output file")
        
    except subprocess.TimeoutExpired:
        logger.error("❌ SadTalker generation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ SadTalker generation failed: {str(e)}")
        return False

def generate_video_with_wav2lip(image_path, audio_path, output_path):
    """Generate video using Wav2Lip with fixed librosa compatibility"""
    try:
        logger.info("🎤 Using Wav2Lip for video generation")
        
        # Set up Wav2Lip paths
        wav2lip_dir = "/app/Wav2Lip"
        checkpoint_path = "/app/checkpoints/wav2lip_gan.pth"
        
        # Ensure checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise Exception(f"Wav2Lip checkpoint not found: {checkpoint_path}")
        
        # Copy face detection model if needed
        face_detection_path = "/app/checkpoints/s3fd.pth"
        wav2lip_face_detection = f"{wav2lip_dir}/face_detection/detection/sfd/s3fd.pth"
        if os.path.exists(face_detection_path) and not os.path.exists(wav2lip_face_detection):
            os.makedirs(os.path.dirname(wav2lip_face_detection), exist_ok=True)
            shutil.copy2(face_detection_path, wav2lip_face_detection)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepare command
        cmd = [
            "python", "inference.py",
            "--checkpoint_path", checkpoint_path,
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", output_path
        ]
        
        logger.info(f"Running Wav2Lip command: {' '.join(cmd)}")
        
        # Set environment variables with librosa fix
        env = os.environ.copy()
        env['PYTHONPATH'] = wav2lip_dir
        
        # Run Wav2Lip with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=wav2lip_dir, env=env, timeout=300)
        
        logger.info(f"Wav2Lip stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Wav2Lip stderr: {result.stderr}")
        
        if result.returncode != 0:
            raise Exception(f"Wav2Lip failed with return code {result.returncode}: {result.stderr}")
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"✅ Wav2Lip video generated: {output_path}")
            return True
        else:
            raise Exception("Wav2Lip did not generate output file or file is empty")
        
    except subprocess.TimeoutExpired:
        logger.error("❌ Wav2Lip generation timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Wav2Lip generation failed: {str(e)}")
        return False

def create_basic_talking_video(image_path, audio_path, output_path):
    """Create basic animated talking video using MediaPipe"""
    try:
        logger.info("🎨 Creating basic animated talking video")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Could not load image: {image_path}")
        
        height, width = image.shape[:2]
        logger.info(f"Original image dimensions: {width}x{height}")
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Detect face landmarks
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            raise Exception("No face detected in image")
        
        logger.info("Face detected successfully with MediaPipe")
        
        # Get audio info
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        audio_duration = len(audio_data) / sample_rate
        logger.info(f"Audio duration: {audio_duration} seconds")
        
        # Video settings
        fps = 25
        total_frames = int(audio_duration * fps)
        logger.info(f"Generating {total_frames} animated frames at {fps} FPS")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Get face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract mouth landmarks (lips)
        mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        mouth_points = []
        for idx in mouth_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            mouth_points.append([x, y])
        mouth_points = np.array(mouth_points, dtype=np.int32)
        
        # Analyze audio for mouth movement
        hop_length = max(1, len(audio_data) // total_frames)
        audio_frames = []
        for i in range(total_frames):
            start_idx = i * hop_length
            end_idx = min(start_idx + hop_length, len(audio_data))
            if start_idx < len(audio_data):
                frame_audio = audio_data[start_idx:end_idx]
                # Calculate RMS energy for this frame
                rms = np.sqrt(np.mean(frame_audio**2)) if len(frame_audio) > 0 else 0
                audio_frames.append(rms)
            else:
                audio_frames.append(0)
        
        logger.info(f"Analyzed audio for {len(audio_frames)} frames")
        
        # Normalize audio frames
        max_rms = max(audio_frames) if audio_frames else 1
        if max_rms > 0:
            audio_frames = [rms / max_rms for rms in audio_frames]
        
        # Generate frames with mouth animation
        for frame_idx in range(total_frames):
            frame = image.copy()
            
            # Get audio intensity for this frame
            audio_intensity = audio_frames[frame_idx] if frame_idx < len(audio_frames) else 0
            
            # Create mouth animation based on audio
            mouth_opening = int(audio_intensity * 8)  # Scale mouth opening
            
            if mouth_opening > 0:
                # Create animated mouth by modifying the mouth region
                mouth_center = np.mean(mouth_points, axis=0).astype(int)
                
                # Create ellipse for mouth opening
                ellipse_height = max(3, mouth_opening)
                ellipse_width = max(15, int(mouth_opening * 2))
                
                # Draw animated mouth
                cv2.ellipse(frame, tuple(mouth_center), (ellipse_width, ellipse_height), 
                           0, 0, 360, (50, 50, 50), -1)
            
            video_writer.write(frame)
        
        video_writer.release()
        
        # Add audio to video using ffmpeg
        temp_video = output_path + "_temp.mp4"
        shutil.move(output_path, temp_video)
        
        cmd = [
            'ffmpeg', '-y', '-i', temp_video, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-shortest', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.remove(temp_video)
            logger.info(f"✅ Basic animated talking video created successfully: {output_path}")
            return True
        else:
            # Fallback: use video without audio sync
            shutil.move(temp_video, output_path)
            logger.warning("⚠️ Audio sync failed, using video without audio")
            return True
        
    except Exception as e:
        logger.error(f"❌ Basic video generation failed: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': MODELS_LOADED,
        'sadtalker_available': SADTALKER_AVAILABLE,
        'wav2lip_available': WAV2LIP_AVAILABLE,
        'gpu_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })

@app.route('/generate-video', methods=['POST'])
def generate_video():
    """Generate talking head video"""
    temp_dir = None
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
        
        task_id = data.get('task_id')
        image_url = data.get('image_url')
        audio_url = data.get('audio_url')
        quality = data.get('quality', 'high')
        
        if not all([task_id, image_url, audio_url]):
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
        
        logger.info(f"🎬 Starting video generation for task {task_id}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Download files
        logger.info(f"📥 Downloading files for task {task_id}")
        
        # Download image
        logger.info(f"📥 Downloading image: {image_url}")
        image_response = requests.get(image_url, timeout=30)
        image_response.raise_for_status()
        image_path = os.path.join(temp_dir, 'input.jpg')
        with open(image_path, 'wb') as f:
            f.write(image_response.content)
        logger.info(f"✅ Downloaded image to {image_path}")
        
        # Download audio
        logger.info(f"📥 Downloading audio: {audio_url}")
        audio_response = requests.get(audio_url, timeout=30)
        audio_response.raise_for_status()
        audio_path = os.path.join(temp_dir, 'input.wav')
        with open(audio_path, 'wb') as f:
            f.write(audio_response.content)
        logger.info(f"✅ Downloaded audio to {audio_path}")
        
        # Generate video with priority: SadTalker > Wav2Lip > Basic
        output_path = f"temp/videos/{task_id}.mp4"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success = False
        method_used = "None"
        
        # Try SadTalker FIRST for high quality
        if quality == "high" and SADTALKER_AVAILABLE:
            logger.info("🎭 Trying SadTalker FIRST (highest quality)")
            if generate_video_with_sadtalker(image_path, audio_path, output_path):
                success = True
                method_used = "SadTalker"
            else:
                logger.warning("SadTalker failed, trying Wav2Lip...")
        
        # Try Wav2Lip if SadTalker failed or for fast quality
        if not success and WAV2LIP_AVAILABLE:
            logger.info("🎤 Trying Wav2Lip (good quality)")
            if generate_video_with_wav2lip(image_path, audio_path, output_path):
                success = True
                method_used = "Wav2Lip"
            else:
                logger.warning("Wav2Lip failed, falling back to basic animation...")
        
        # Fall back to basic animation
        if not success:
            logger.info("🎨 Using basic animation as fallback")
            if create_basic_talking_video(image_path, audio_path, output_path):
                success = True
                method_used = "Basic"
        
        if not success:
            raise Exception("All video generation methods failed")
        
        # Store completed video
        completed_videos[task_id] = {
            'path': output_path,
            'method': method_used,
            'timestamp': time.time()
        }
        
        logger.info(f"✅ Video generation completed for task {task_id} using {method_used}")
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'method_used': method_used,
            'message': f'Video generated successfully using {method_used}'
        })
        
    except Exception as e:
        logger.error(f"❌ Error in video generation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500
        
    finally:
        # Cleanup temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")

@app.route('/video-status/<task_id>', methods=['GET'])
def get_video_status(task_id):
    """Get video status or return completed video"""
    try:
        if task_id in completed_videos:
            video_info = completed_videos[task_id]
            video_path = video_info['path']
            
            if os.path.exists(video_path):
                logger.info(f"📤 Serving completed video for task {task_id}")
                
                # Clean up after serving
                def cleanup():
                    try:
                        if os.path.exists(video_path):
                            os.remove(video_path)
                        if task_id in completed_videos:
                            del completed_videos[task_id]
                    except Exception as e:
                        logger.warning(f"Cleanup failed: {e}")
                
                # Schedule cleanup after response
                import threading
                threading.Timer(1.0, cleanup).start()
                
                return send_file(video_path, mimetype='video/mp4', as_attachment=True, 
                               download_name=f'video_{task_id}.mp4')
            else:
                # Video file missing
                if task_id in completed_videos:
                    del completed_videos[task_id]
                return jsonify({'success': False, 'error': 'Video file not found'}), 404
        else:
            # Task not found or still processing
            return jsonify({'success': False, 'error': 'Task not found or still processing'}), 404
            
    except Exception as e:
        logger.error(f"❌ Error getting video status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'Video Generation Service',
        'version': '1.0.0',
        'status': 'running',
        'models_loaded': MODELS_LOADED,
        'sadtalker_available': SADTALKER_AVAILABLE,
        'wav2lip_available': WAV2LIP_AVAILABLE
    })

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Create necessary directories
    os.makedirs('temp/videos', exist_ok=True)
    
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
