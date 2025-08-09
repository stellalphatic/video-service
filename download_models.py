import os
import requests
import logging
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configurations with fallback URLs
MODELS = {
    "SadTalker": {
        "auido2exp_00300-model.pth": {
            "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2exp_00300-model.pth",
            "path": "models/SadTalker/checkpoints/auido2exp_00300-model.pth",
            "size_mb": 17,
            "fallback_urls": [
                "https://huggingface.co/vinthony/SadTalker/resolve/main/auido2exp_00300-model.pth"
            ]
        },
        "facevid2vid_00189-model.pth.tar": {
            "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/facevid2vid_00189-model.pth.tar",
            "path": "models/SadTalker/checkpoints/facevid2vid_00189-model.pth.tar",
            "size_mb": 367,
            "fallback_urls": [
                "https://huggingface.co/vinthony/SadTalker/resolve/main/facevid2vid_00189-model.pth.tar"
            ]
        },
        "epoch_20.pth": {
            "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/epoch_20.pth",
            "path": "models/SadTalker/checkpoints/epoch_20.pth",
            "size_mb": 383,
            "fallback_urls": [
                "https://huggingface.co/vinthony/SadTalker/resolve/main/epoch_20.pth"
            ]
        }
    },
    "Wav2Lip": {
        "wav2lip_gan.pth": {
            "url": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth",
            "path": "models/Wav2Lip/checkpoints/wav2lip_gan.pth",
            "size_mb": 338,
            "fallback_urls": [
                "https://huggingface.co/spaces/Rudrabha/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
            ]
        },
        "s3fd.pth": {
            "url": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/s3fd.pth",
            "path": "models/Wav2Lip/face_detection/detection/sfd/s3fd.pth",
            "size_mb": 85,
            "fallback_urls": [
                "https://huggingface.co/spaces/Rudrabha/Wav2Lip/resolve/main/face_detection/detection/sfd/s3fd.pth"
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
        response = requests.get(url, stream=True, timeout=300)
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
        
        if expected_size_mb and abs(actual_size_mb - expected_size_mb) > 5:
            logger.warning(f"⚠️ Size mismatch: expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB")
        
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
        
        if expected_size_mb and abs(size_mb - expected_size_mb) > 10:
            return False, "SIZE_MISMATCH"
        
        return True, f"OK ({size_mb:.0f}MB)"
    except Exception:
        return False, "CORRUPTED"

def main():
    """Download all required models."""
    logger.info("🚀 === Starting Model Download Process ===")
    
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
    
    if success_count == total_count:
        logger.info("🎉 All models downloaded successfully!")
    else:
        logger.warning(f"⚠️ Model download completed with some issues. Check logs above.")
        logger.info("📝 Note: Missing models will be handled gracefully with fallbacks at runtime")

if __name__ == "__main__":
    main()
