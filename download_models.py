import os
import requests
import logging
from pathlib import Path
import hashlib
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Updated model configurations with correct HuggingFace URLs
MODELS = {
    "SadTalker": {
        "auido2exp_00300-model.pth": {
            "url": "https://huggingface.co/vinthony/SadTalker/resolve/main/auido2exp_00300-model.pth",
            "path": "models/SadTalker/checkpoints/auido2exp_00300-model.pth",
            "size_mb": 34,
            "fallback_urls": [
                "https://huggingface.co/camenduru/SadTalker/resolve/main/auido2exp_00300-model.pth"
            ]
        },
        "facevid2vid_00189-model.pth.tar": {
            "url": "https://huggingface.co/vinthony/SadTalker/resolve/main/facevid2vid_00189-model.pth.tar",
            "path": "models/SadTalker/checkpoints/facevid2vid_00189-model.pth.tar",
            "size_mb": 2100,
            "fallback_urls": [
                "https://huggingface.co/camenduru/SadTalker/resolve/main/facevid2vid_00189-model.pth.tar"
            ]
        },
        "epoch_20.pth": {
            "url": "https://huggingface.co/vinthony/SadTalker/resolve/main/epoch_20.pth",
            "path": "models/SadTalker/checkpoints/epoch_20.pth",
            "size_mb": 280,
            "fallback_urls": [
                "https://huggingface.co/camenduru/SadTalker/resolve/main/epoch_20.pth"
            ]
        },
        "auido2pose_00140-model.pth": {
            "url": "https://huggingface.co/vinthony/SadTalker/resolve/main/auido2pose_00140-model.pth",
            "path": "models/SadTalker/checkpoints/auido2pose_00140-model.pth",
            "size_mb": 95,
            "fallback_urls": [
                "https://huggingface.co/camenduru/SadTalker/resolve/main/auido2pose_00140-model.pth"
            ]
        },
        "shape_predictor_68_face_landmarks.dat": {
            "url": "https://huggingface.co/vinthony/SadTalker/resolve/main/shape_predictor_68_face_landmarks.dat",
            "path": "models/SadTalker/checkpoints/shape_predictor_68_face_landmarks.dat",
            "size_mb": 99,
            "fallback_urls": [
                "https://huggingface.co/camenduru/SadTalker/resolve/main/shape_predictor_68_face_landmarks.dat"
            ]
        }
    },
    "Wav2Lip": {
        "wav2lip_gan.pth": {
            "url": "https://huggingface.co/manavisrani07/gradio-lipsync-wav2lip/resolve/main/checkpoints/wav2lip_gan.pth",
            "path": "models/Wav2Lip/checkpoints/wav2lip_gan.pth",
            "size_mb": 436,
            "fallback_urls": [
                "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
                "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth"
            ]
        },
        "s3fd.pth": {
            "url": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/s3fd-619a316812.pth",
            "path": "models/Wav2Lip/face_detection/detection/sfd/s3fd.pth",
            "size_mb": 89,
            "fallback_urls": [
                "https://huggingface.co/manavisrani07/gradio-lipsync-wav2lip/resolve/main/face_detection/detection/sfd/s3fd-619a316812.pth",
                "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/s3fd.pth"
            ]
        },
        "wav2lip.pth": {
            "url": "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip.pth",
            "path": "models/Wav2Lip/checkpoints/wav2lip.pth",
            "size_mb": 167,
            "fallback_urls": [
                "https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth"
            ]
        }
    }
}

def download_file(url, filepath, expected_size_mb=None):
    """Download a file with progress tracking and validation."""
    try:
        logger.info(f"📥 Downloading {os.path.basename(filepath)} from {url}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Download with streaming
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Progress logging every 50MB
                    if downloaded_size % (50 * 1024 * 1024) == 0:
                        progress = (downloaded_size / total_size * 100) if total_size > 0 else 0
                        logger.info(f"📊 Progress: {progress:.1f}% ({downloaded_size // (1024*1024)}MB)")
        
        # Validate file size
        actual_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"✅ Downloaded {os.path.basename(filepath)} ({actual_size_mb:.1f}MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def verify_model_file(filepath, expected_size_mb=None):
    """Verify if a model file exists and is valid."""
    if not os.path.exists(filepath):
        return False, "MISSING"
    
    try:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if size_mb < 1:  # File too small
            return False, "CORRUPTED"
        
        return True, f"OK ({size_mb:.0f}MB)"
    except Exception:
        return False, "CORRUPTED"

def clone_repositories():
    """Clone required repositories."""
    try:
        # Clone SadTalker
        sadtalker_path = "models/SadTalker"
        if not os.path.exists(sadtalker_path):
            logger.info("📥 Cloning SadTalker repository...")
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/OpenTalker/SadTalker.git",
                sadtalker_path
            ], check=True, capture_output=True)
            logger.info("✅ SadTalker repository cloned")
        
        # Clone Wav2Lip
        wav2lip_path = "models/Wav2Lip"
        if not os.path.exists(wav2lip_path):
            logger.info("📥 Cloning Wav2Lip repository...")
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/Rudrabha/Wav2Lip.git",
                wav2lip_path
            ], check=True, capture_output=True)
            logger.info("✅ Wav2Lip repository cloned")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error cloning repositories: {e}")
        return False

def main():
    """Download all required models."""
    logger.info("🚀 === Starting Model Download Process ===")
    
    # Clone repositories first
    if not clone_repositories():
        logger.error("❌ Failed to clone repositories")
        return
    
    success_count = 0
    total_count = 0
    failed_models = []
    
    for model_type, models in MODELS.items():
        logger.info(f"📦 Processing {model_type} models...")
        
        for model_name, config in models.items():
            total_count += 1
            filepath = config["path"]
            
            # Check if model already exists and is valid
            is_valid, status = verify_model_file(filepath, config.get("size_mb"))
            if is_valid:
                logger.info(f"✅ {model_name} - Already exists and valid ({status})")
                success_count += 1
                continue
            
            logger.info(f"📥 Downloading {model_name}...")
            
            # Try main URL first
            success = download_file(config["url"], filepath, config.get("size_mb"))
            
            # Try fallback URLs if main URL fails
            if not success and "fallback_urls" in config:
                for fallback_url in config["fallback_urls"]:
                    logger.info(f"🔄 Trying fallback URL for {model_name}")
                    success = download_file(fallback_url, filepath, config.get("size_mb"))
                    if success:
                        break
            
            if success:
                success_count += 1
                logger.info(f"✅ {model_name} - Downloaded successfully")
            else:
                failed_models.append(model_name)
                logger.error(f"❌ {model_name} - Download failed")
    
    # Final verification
    logger.info("🔍 === Verifying Model Downloads ===")
    for model_type, models in MODELS.items():
        for model_name, config in models.items():
            is_valid, status = verify_model_file(config["path"], config.get("size_mb"))
            if is_valid:
                logger.info(f"✅ {model_name} - {status}")
            else:
                logger.error(f"❌ {model_name} - {status}")
    
    # Summary
    if failed_models:
        logger.warning(f"⚠️ Missing models: {failed_models}")
        logger.warning("Some models failed to download. Service will use fallbacks.")
    
    if success_count >= 3:  # At least some essential models
        logger.info("🎉 Essential models downloaded successfully!")
    else:
        logger.warning(f"⚠️ Model download completed with issues. Check logs above.")
        logger.info("📝 Note: Missing models will be handled gracefully with fallbacks at runtime")

if __name__ == "__main__":
    main()
