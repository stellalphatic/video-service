#!/usr/bin/env python3
"""
Download required models for video generation service
Non-interactive model downloads with proper error handling
"""
import os
import sys
import logging
import requests
import zipfile
import tarfile
import gzip
import shutil
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
        logger.info(f"Downloading {url}")
        
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
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.info(f"Downloaded {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB ({percent:.1f}%)")
        
        # Verify hash if provided
        if expected_hash:
            file_hash = hashlib.md5(destination.read_bytes()).hexdigest()
            if file_hash != expected_hash:
                logger.error(f"Hash mismatch for {destination.name}")
                destination.unlink()
                return False
        
        logger.info(f"Successfully downloaded {destination.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False

def extract_archive(archive_path: Path, extract_to: Path):
    """Extract various archive formats"""
    try:
        logger.info(f"Extracting {archive_path.name}")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.suffix == '.gz':
            with gzip.open(archive_path, 'rb') as gz_file:
                with open(extract_to / archive_path.stem, 'wb') as out_file:
                    shutil.copyfileobj(gz_file, out_file)
        
        logger.info(f"Extracted {archive_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False

def download_sadtalker_models():
    """Download SadTalker models from official sources"""
    try:
        logger.info("=== Downloading SadTalker Models ===")
        
        sadtalker_dir = Path(MODELS_DIR) / "SadTalker"
        checkpoints_dir = sadtalker_dir / "checkpoints"
        gfpgan_dir = sadtalker_dir / "gfpgan" / "weights"
        
        # Create directories
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        gfpgan_dir.mkdir(parents=True, exist_ok=True)
        
        # SadTalker model URLs (from official repo)
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
                    logger.warning(f"Failed to download {model_name}, will try alternative source")
                    # Try alternative source
                    alt_url = f"https://huggingface.co/vinthony/SadTalker/resolve/main/checkpoints/{model_name}"
                    download_file_with_progress(alt_url, model_info["path"])
            else:
                logger.info(f"{model_name} already exists")
        
        # Extract archives
        for archive_name in ["BFM_Fitting.zip", "hub.zip"]:
            archive_path = checkpoints_dir / archive_name
            if archive_path.exists():
                extract_archive(archive_path, checkpoints_dir)
                archive_path.unlink()  # Remove archive after extraction
        
        # Download GFPGAN weights for face enhancement
        gfpgan_model = gfpgan_dir / "GFPGANv1.4.pth"
        if not gfpgan_model.exists():
            gfpgan_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
            download_file_with_progress(gfpgan_url, gfpgan_model)
        
        logger.info("✅ SadTalker models download completed")
        
    except Exception as e:
        logger.error(f"Error downloading SadTalker models: {e}")

def download_wav2lip_models():
    """Download Wav2Lip models"""
    try:
        logger.info("=== Downloading Wav2Lip Models ===")
        
        wav2lip_dir = Path(MODELS_DIR) / "Wav2Lip"
        wav2lip_dir.mkdir(parents=True, exist_ok=True)
        
        # Wav2Lip models
        models = {
            "wav2lip_gan.pth": {
                "url": "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp2pgHDtDw",
                "alt_url": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0.0/wav2lip_gan.pth"
            },
            "wav2lip.pth": {
                "url": "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EQRvdUzUeXxOQSVNw2kSwBwBSqkqJfMGaT4J_6TL-I4GlA",
                "alt_url": "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0.0/wav2lip.pth"
            },
            "s3fd.pth": {
                "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
                "alt_url": "https://github.com/1adrianb/face-alignment/releases/download/v1.0/s3fd-619a316812.pth"
            }
        }
        
        for model_name, urls in models.items():
            model_path = wav2lip_dir / model_name
            if not model_path.exists():
                # Try primary URL first, then alternative
                success = download_file_with_progress(urls["url"], model_path)
                if not success and "alt_url" in urls:
                    logger.info(f"Trying alternative URL for {model_name}")
                    download_file_with_progress(urls["alt_url"], model_path)
            else:
                logger.info(f"{model_name} already exists")
        
        logger.info("✅ Wav2Lip models download completed")
        
    except Exception as e:
        logger.error(f"Error downloading Wav2Lip models: {e}")

def download_face_detection_models():
    """Download face detection and landmark models"""
    try:
        logger.info("=== Downloading Face Detection Models ===")
        
        face_dir = Path(MODELS_DIR) / "face_detection"
        face_dir.mkdir(parents=True, exist_ok=True)
        
        # Face landmark models
        models = {
            "shape_predictor_68_face_landmarks.dat": {
                "url": "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2",
                "compressed": True
            },
            "shape_predictor_5_face_landmarks.dat": {
                "url": "https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2", 
                "compressed": True
            }
        }
        
        for model_name, info in models.items():
            model_path = face_dir / model_name
            if not model_path.exists():
                if info.get("compressed"):
                    # Download compressed version
                    compressed_path = face_dir / f"{model_name}.bz2"
                    success = download_file_with_progress(info["url"], compressed_path)
                    if success:
                        # Decompress
                        import bz2
                        with bz2.BZ2File(compressed_path, 'rb') as f_in:
                            with open(model_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        compressed_path.unlink()  # Remove compressed file
                        logger.info(f"Decompressed {model_name}")
                else:
                    download_file_with_progress(info["url"], model_path)
            else:
                logger.info(f"{model_name} already exists")
        
        logger.info("✅ Face detection models download completed")
        
    except Exception as e:
        logger.error(f"Error downloading face detection models: {e}")

def download_additional_models():
    """Download additional useful models"""
    try:
        logger.info("=== Downloading Additional Models ===")
        
        # Download face parsing model
        face_parsing_dir = Path(MODELS_DIR) / "face_parsing"
        face_parsing_dir.mkdir(parents=True, exist_ok=True)
        
        # BiSeNet for face parsing
        bisenet_model = face_parsing_dir / "79999_iter.pth"
        if not bisenet_model.exists():
            bisenet_url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
            # Note: Google Drive links need special handling, using alternative
            alt_url = "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth"
            download_file_with_progress(alt_url, bisenet_model)
        
        logger.info("✅ Additional models download completed")
        
    except Exception as e:
        logger.error(f"Error downloading additional models: {e}")

def verify_models():
    """Verify that essential models are downloaded"""
    try:
        logger.info("=== Verifying Model Downloads ===")
        
        essential_models = [
            Path(MODELS_DIR) / "SadTalker" / "checkpoints" / "auido2exp_00300-model.pth",
            Path(MODELS_DIR) / "SadTalker" / "checkpoints" / "epoch_20.pth", 
            Path(MODELS_DIR) / "Wav2Lip" / "wav2lip_gan.pth",
            Path(MODELS_DIR) / "face_detection" / "shape_predictor_68_face_landmarks.dat"
        ]
        
        missing_models = []
        for model_path in essential_models:
            if model_path.exists() and model_path.stat().st_size > 1000:  # At least 1KB
                logger.info(f"✅ {model_path.name} - OK ({model_path.stat().st_size // (1024*1024)}MB)")
            else:
                logger.error(f"❌ {model_path.name} - MISSING or CORRUPTED")
                missing_models.append(model_path.name)
        
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            logger.warning("Some models failed to download. They will be downloaded at runtime.")
        else:
            logger.info("✅ All essential models verified successfully!")
        
    except Exception as e:
        logger.error(f"Error verifying models: {e}")

def setup_huggingface_cache():
    """Setup HuggingFace cache directory"""
    try:
        hf_cache_dir = Path(MODELS_DIR) / "huggingface_cache"
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(hf_cache_dir)
        logger.info(f"HuggingFace cache set to: {hf_cache_dir}")
    except Exception as e:
        logger.error(f"Error setting up HuggingFace cache: {e}")

def main():
    """Main download function"""
    logger.info("🚀 Starting comprehensive model downloads...")
    logger.info(f"Models directory: {MODELS_DIR}")
    
    # Create base directory
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Setup caches
    setup_huggingface_cache()
    
    # Download all models
    download_face_detection_models()
    download_wav2lip_models() 
    download_sadtalker_models()
    download_additional_models()
    
    # Verify downloads
    verify_models()
    
    logger.info("🎉 Model download process completed!")
    logger.info("Note: If any models failed to download, they will be downloaded at runtime")

if __name__ == "__main__":
    main()
