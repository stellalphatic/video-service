#!/usr/bin/env python3
"""
Professional model downloader for video generation service
Downloads SadTalker, Wav2Lip, and other required models
"""
import os
import sys
import logging
import requests
import zipfile
import tarfile
import gzip
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import hashlib

# Set non-interactive environment variables
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '0'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
TIMEOUT = 600  # 10 minutes timeout for large downloads

def download_file_with_progress(url: str, destination: Path, expected_hash: str = None):
    """Download file with progress and hash verification"""
    try:
        logger.info(f"📥 Downloading {url}")
        
        response = requests.get(url, stream=True, timeout=TIMEOUT)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
                        percent = (downloaded / total_size) * 100
                        logger.info(f"📊 Progress: {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB ({percent:.1f}%)")
        
        # Verify hash if provided
        if expected_hash:
            file_hash = hashlib.md5(destination.read_bytes()).hexdigest()
            if file_hash != expected_hash:
                logger.error(f"❌ Hash mismatch for {destination.name}")
                destination.unlink()
                return False
        
        logger.info(f"✅ Successfully downloaded {destination.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False

def clone_repository(repo_url: str, destination: Path):
    """Clone a git repository"""
    try:
        if destination.exists():
            logger.info(f"📁 Repository already exists: {destination}")
            return True
        
        logger.info(f"📥 Cloning repository: {repo_url}")
        subprocess.run([
            "git", "clone", "--depth", "1", repo_url, str(destination)
        ], check=True, capture_output=True)
        
        logger.info(f"✅ Repository cloned: {destination}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to clone repository {repo_url}: {e}")
        return False

def download_sadtalker_models():
    """Download SadTalker models and repository"""
    try:
        logger.info("🎭 === Downloading SadTalker Models ===")
        
        # Clone SadTalker repository
        sadtalker_dir = Path(MODELS_DIR) / "SadTalker"
        if not clone_repository("https://github.com/OpenTalker/SadTalker.git", sadtalker_dir):
            logger.error("❌ Failed to clone SadTalker repository")
            return False
        
        checkpoints_dir = sadtalker_dir / "checkpoints"
        gfpgan_dir = sadtalker_dir / "gfpgan" / "weights"
        
        # Create directories
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        gfpgan_dir.mkdir(parents=True, exist_ok=True)
        
        # SadTalker model URLs (from official releases)
        models = {
            "auido2exp_00300-model.pth": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2exp_00300-model.pth",
                "path": checkpoints_dir / "auido2exp_00300-model.pth"
            },
            "auido2pose_00140-model.pth": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2pose_00140-model.pth", 
                "path": checkpoints_dir / "auido2pose_00140-model.pth"
            },
            "epoch_20.pth": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/epoch_20.pth",
                "path": checkpoints_dir / "epoch_20.pth"
            },
            "facevid2vid_00189-model.pth.tar": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/facevid2vid_00189-model.pth.tar",
                "path": checkpoints_dir / "facevid2vid_00189-model.pth.tar"
            },
            "shape_predictor_68_face_landmarks.dat": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/shape_predictor_68_face_landmarks.dat",
                "path": checkpoints_dir / "shape_predictor_68_face_landmarks.dat"
            },
            "BFM_Fitting.zip": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/BFM_Fitting.zip",
                "path": checkpoints_dir / "BFM_Fitting.zip"
            },
            "hub.zip": {
                "url": "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/hub.zip", 
                "path": checkpoints_dir / "hub.zip"
            }
        }
        
        # Download SadTalker models
        for model_name, model_info in models.items():
            if not model_info["path"].exists():
                success = download_file_with_progress(model_info["url"], model_info["path"])
                if not success:
                    logger.warning(f"⚠️ Failed to download {model_name}")
            else:
                logger.info(f"✅ {model_name} already exists")
        
        # Extract archives
        for archive_name in ["BFM_Fitting.zip", "hub.zip"]:
            archive_path = checkpoints_dir / archive_name
            if archive_path.exists():
                logger.info(f"📦 Extracting {archive_name}")
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(checkpoints_dir)
                archive_path.unlink()  # Remove archive after extraction
        
        # Download GFPGAN weights for face enhancement
        gfpgan_model = gfpgan_dir / "GFPGANv1.4.pth"
        if not gfpgan_model.exists():
            gfpgan_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            download_file_with_progress(gfpgan_url, gfpgan_model)
        
        logger.info("✅ SadTalker models download completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading SadTalker models: {e}")
        return False

def download_wav2lip_models():
    """Download Wav2Lip models and repository"""
    try:
        logger.info("🎤 === Downloading Wav2Lip Models ===")
        
        # Clone Wav2Lip repository
        wav2lip_dir = Path(MODELS_DIR) / "Wav2Lip"
        if not clone_repository("https://github.com/Rudrabha/Wav2Lip.git", wav2lip_dir):
            logger.error("❌ Failed to clone Wav2Lip repository")
            return False
        
        # Wav2Lip models
        models = {
            "wav2lip_gan.pth": {
                "url": "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp2pgHDtDw",
                "alt_url": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0.0/wav2lip_gan.pth",
                "path": wav2lip_dir / "wav2lip_gan.pth"
            },
            "wav2lip.pth": {
                "url": "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EQRvdUzUeXxOQSVNw2kSwBwBSqkqJfMGaT4J_6TL-I4GlA",
                "alt_url": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0.0/wav2lip.pth",
                "path": wav2lip_dir / "wav2lip.pth"
            },
            "s3fd.pth": {
                "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
                "alt_url": "https://github.com/1adrianb/face-alignment/releases/download/v1.0/s3fd-619a316812.pth",
                "path": wav2lip_dir / "s3fd.pth"
            }
        }
        
        for model_name, urls in models.items():
            if not urls["path"].exists():
                # Try primary URL first, then alternative
                success = download_file_with_progress(urls["url"], urls["path"])
                if not success and "alt_url" in urls:
                    logger.info(f"🔄 Trying alternative URL for {model_name}")
                    download_file_with_progress(urls["alt_url"], urls["path"])
            else:
                logger.info(f"✅ {model_name} already exists")
        
        logger.info("✅ Wav2Lip models download completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading Wav2Lip models: {e}")
        return False

def install_dependencies():
    """Install required Python dependencies"""
    try:
        logger.info("📦 === Installing Dependencies ===")
        
        # Install SadTalker dependencies
        sadtalker_dir = Path(MODELS_DIR) / "SadTalker"
        if sadtalker_dir.exists():
            requirements_file = sadtalker_dir / "requirements.txt"
            if requirements_file.exists():
                logger.info("📦 Installing SadTalker requirements")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
        
        # Install Wav2Lip dependencies
        wav2lip_dir = Path(MODELS_DIR) / "Wav2Lip"
        if wav2lip_dir.exists():
            requirements_file = wav2lip_dir / "requirements.txt"
            if requirements_file.exists():
                logger.info("📦 Installing Wav2Lip requirements")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], check=True)
        
        # Install additional dependencies
        additional_deps = [
            "face-alignment",
            "librosa",
            "soundfile",
            "resampy",
            "ffmpeg-python",
            "imageio-ffmpeg"
        ]
        
        for dep in additional_deps:
            try:
                logger.info(f"📦 Installing {dep}")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️ Failed to install {dep}")
        
        logger.info("✅ Dependencies installation completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error installing dependencies: {e}")
        return False

def verify_models():
    """Verify that essential models are downloaded"""
    try:
        logger.info("🔍 === Verifying Model Downloads ===")
        
        essential_models = [
            Path(MODELS_DIR) / "SadTalker" / "checkpoints" / "auido2exp_00300-model.pth",
            Path(MODELS_DIR) / "SadTalker" / "checkpoints" / "epoch_20.pth", 
            Path(MODELS_DIR) / "SadTalker" / "checkpoints" / "facevid2vid_00189-model.pth.tar",
            Path(MODELS_DIR) / "Wav2Lip" / "wav2lip_gan.pth",
            Path(MODELS_DIR) / "Wav2Lip" / "s3fd.pth"
        ]
        
        missing_models = []
        for model_path in essential_models:
            if model_path.exists() and model_path.stat().st_size > 1000:  # At least 1KB
                size_mb = model_path.stat().st_size // (1024*1024)
                logger.info(f"✅ {model_path.name} - OK ({size_mb}MB)")
            else:
                logger.error(f"❌ {model_path.name} - MISSING or CORRUPTED")
                missing_models.append(model_path.name)
        
        if missing_models:
            logger.warning(f"⚠️ Missing models: {missing_models}")
            logger.warning("Some models failed to download. Service will use fallbacks.")
            return False
        else:
            logger.info("✅ All essential models verified successfully!")
            return True
        
    except Exception as e:
        logger.error(f"❌ Error verifying models: {e}")
        return False

def main():
    """Main download function"""
    logger.info("🚀 Starting professional model downloads...")
    logger.info(f"📁 Models directory: {MODELS_DIR}")
    
    # Create base directory
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Download all models
    if not download_sadtalker_models():
        success = False
    
    if not download_wav2lip_models():
        success = False
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Verify downloads
    if not verify_models():
        success = False
    
    if success:
        logger.info("🎉 Professional model download completed successfully!")
    else:
        logger.warning("⚠️ Model download completed with some issues. Check logs above.")
    
    logger.info("📝 Note: Missing models will be handled gracefully with fallbacks at runtime")

if __name__ == "__main__":
    main()
