#!/usr/bin/env python3
"""
Download required models for video generation service
"""
import os
import sys
import logging
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download, snapshot_download
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

def download_sadtalker_models():
    """Download SadTalker models from HuggingFace"""
    try:
        logger.info("Downloading SadTalker models...")
        
        # Create SadTalker directory
        sadtalker_dir = Path(MODELS_DIR) / "SadTalker"
        sadtalker_dir.mkdir(parents=True, exist_ok=True)
        
        # Download checkpoints
        checkpoints_dir = sadtalker_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        # Key SadTalker model files
        model_files = [
            "auido2exp_00300-model.pth",
            "auido2pose_00140-model.pth", 
            "epoch_20.pth",
            "facevid2vid_00189-model.pth.tar",
            "shape_predictor_68_face_landmarks.dat"
        ]
        
        # Download from a reliable source (you might need to adjust URLs)
        base_url = "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/"
        
        for model_file in model_files:
            model_path = checkpoints_dir / model_file
            if not model_path.exists():
                try:
                    logger.info(f"Downloading {model_file}...")
                    response = requests.get(f"{base_url}{model_file}", stream=True)
                    if response.status_code == 200:
                        with open(model_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        logger.info(f"Downloaded {model_file}")
                    else:
                        logger.warning(f"Could not download {model_file}, will download at runtime")
                except Exception as e:
                    logger.warning(f"Failed to download {model_file}: {e}")
        
        logger.info("SadTalker models download completed")
        
    except Exception as e:
        logger.error(f"Error downloading SadTalker models: {e}")

def download_wav2lip_models():
    """Download Wav2Lip models"""
    try:
        logger.info("Downloading Wav2Lip models...")
        
        wav2lip_dir = Path(MODELS_DIR) / "Wav2Lip"
        wav2lip_dir.mkdir(parents=True, exist_ok=True)
        
        # Download Wav2Lip checkpoint
        model_url = "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0.0/wav2lip_gan.pth"
        model_path = wav2lip_dir / "wav2lip_gan.pth"
        
        if not model_path.exists():
            try:
                logger.info("Downloading Wav2Lip model...")
                response = requests.get(model_url, stream=True)
                if response.status_code == 200:
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info("Wav2Lip model downloaded")
                else:
                    logger.warning("Could not download Wav2Lip model")
            except Exception as e:
                logger.warning(f"Failed to download Wav2Lip model: {e}")
        
        logger.info("Wav2Lip models download completed")
        
    except Exception as e:
        logger.error(f"Error downloading Wav2Lip models: {e}")

def download_face_detection_models():
    """Download face detection models"""
    try:
        logger.info("Downloading face detection models...")
        
        # Download dlib face landmarks
        landmarks_url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
        landmarks_dir = Path(MODELS_DIR) / "face_detection"
        landmarks_dir.mkdir(parents=True, exist_ok=True)
        landmarks_path = landmarks_dir / "shape_predictor_68_face_landmarks.dat"
        
        if not landmarks_path.exists():
            try:
                import bz2
                logger.info("Downloading face landmarks model...")
                response = requests.get(landmarks_url, stream=True)
                if response.status_code == 200:
                    compressed_data = response.content
                    decompressed_data = bz2.decompress(compressed_data)
                    with open(landmarks_path, 'wb') as f:
                        f.write(decompressed_data)
                    logger.info("Face landmarks model downloaded")
            except Exception as e:
                logger.warning(f"Failed to download face landmarks: {e}")
        
        logger.info("Face detection models download completed")
        
    except Exception as e:
        logger.error(f"Error downloading face detection models: {e}")

def verify_pytorch():
    """Verify PyTorch installation"""
    try:
        logger.info("Verifying PyTorch installation...")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info("PyTorch verification completed")
    except Exception as e:
        logger.error(f"PyTorch verification failed: {e}")

def main():
    """Download all required models"""
    logger.info("Starting model downloads...")
    
    # Create models directory
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Verify PyTorch
    verify_pytorch()
    
    # Download models
    download_face_detection_models()
    download_wav2lip_models()
    download_sadtalker_models()
    
    logger.info("All model downloads completed!")

if __name__ == "__main__":
    main()
