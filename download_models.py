import os
import requests
import subprocess
import sys
from pathlib import Path
import zipfile
import tarfile
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model directories
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
TEMP_DIR = os.environ.get("TEMP_DIR", "/app/temp")

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

def download_file(url, local_path, description=""):
    """Download file with progress"""
    try:
        logger.info(f"📥 Downloading {description}: {url}")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # if total_size > 0:
                    #     progress = (downloaded / total_size) * 100
                    #     print(f"\r📊 Progress: {progress:.1f}%", end='', flush=True)
        
        print()  # New line after progress
        logger.info(f"✅ Downloaded {description} successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {description}: {e}")
        return False

def clone_repository(repo_url, local_path, description=""):
    """Clone git repository"""
    try:
        logger.info(f"📦 Cloning {description}: {repo_url}")
        subprocess.run(['git', 'clone', repo_url, local_path], check=True, capture_output=True)
        logger.info(f"✅ Cloned {description} successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clone {description}: {e}")
        return False

def download_sadtalker_models():
    """Download SadTalker models and repository"""
    logger.info("🎭 Setting up SadTalker...")
    
    sadtalker_dir = os.path.join(MODELS_DIR, "SadTalker")
    checkpoints_dir = os.path.join(sadtalker_dir, "checkpoints")
    
    # Clone SadTalker repository
    if not os.path.exists(sadtalker_dir):
        if not clone_repository("https://github.com/OpenTalker/SadTalker.git", sadtalker_dir, "SadTalker repository"):
            return False
    
    # Create checkpoints directory
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # SadTalker model URLs- multiple sources for reliability
    models = {
        "auido2exp_00300-model.pth": [
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2exp_00300-model.pth",
        "https://huggingface.co/vinthony/SadTalker/resolve/main/auido2exp_00300-model.pth"
       ],
       "auido2pose_00140-model.pth": [
        "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/auido2pose_00140-model.pth",
        "https://huggingface.co/vinthony/SadTalker/resolve/main/auido2pose_00140-model.pth"
        ],
        "facevid2vid_00189-model.pth.tar": [
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/facevid2vid_00189-model.pth.tar",
            "https://huggingface.co/vinthony/SadTalker/resolve/main/facevid2vid_00189-model.pth.tar"
        ],
        "epoch_20.pth": [
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/epoch_20.pth",
            "https://huggingface.co/vinthony/SadTalker/resolve/main/epoch_20.pth"
        ],
        "shape_predictor_68_face_landmarks.dat": [
            "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/shape_predictor_68_face_landmarks.dat",
            "https://huggingface.co/vinthony/SadTalker/resolve/main/shape_predictor_68_face_landmarks.dat"
        ],
        "mapping_00229-model.pth.tar": [
             "https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00229-model.pth.tar",
             "https://huggingface.co/vinthony/SadTalker/resolve/main/mapping_00229-model.pth.tar"
        ]
    }
    
    # Download each model with fallback URLs
    for model_name, urls in models.items():
        model_path = os.path.join(checkpoints_dir, model_name)
        
        if os.path.exists(model_path):
            logger.info(f"✅ {model_name} already exists")
            continue
        
        downloaded = False
        for url in urls:
            if download_file(url, model_path, f"SadTalker {model_name}"):
                downloaded = True
                break
        
        if not downloaded:
            logger.error(f"❌ Failed to download {model_name} from all sources")
            return False
    
    logger.info("✅ SadTalker setup completed")
    return True

def download_wav2lip_models():
    """Download Wav2Lip models and repository"""
    logger.info("🎤 Setting up Wav2Lip...")
    
    wav2lip_dir = os.path.join(MODELS_DIR, "Wav2Lip")
    checkpoints_dir = os.path.join(wav2lip_dir, "checkpoints")
    face_detection_dir = os.path.join(wav2lip_dir, "face_detection", "detection", "sfd")
    
    # Clone Wav2Lip repository
    if not os.path.exists(wav2lip_dir):
        if not clone_repository("https://github.com/Rudrabha/Wav2Lip.git", wav2lip_dir, "Wav2Lip repository"):
            return False
    
    # Create necessary directories
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(face_detection_dir, exist_ok=True)
    
    # Wav2Lip model URLs - multiple sources for reliability
    models = {
        "wav2lip_gan.pth": [
            "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp2pgHDc0A",
            "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth",
            "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
        ],
        "wav2lip.pth": [
            "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EQRvdUzUeiVJiRPtjc_-ioEBzqkuiEWW88dkzTxPdqsw0Q",
            "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth",
            "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip.pth"
        ]
    }
    
    # Face detection model
    face_detection_models = {
        "s3fd.pth": [
            "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
            "https://github.com/1adrianb/face-alignment/releases/download/v1.3.0/s3fd-619a316812.pth",
            "https://huggingface.co/camenduru/Wav2Lip/resolve/main/face_detection/detection/sfd/s3fd.pth"
        ]
    }
    
    # Download Wav2Lip models
    for model_name, urls in models.items():
        model_path = os.path.join(checkpoints_dir, model_name)
        
        if os.path.exists(model_path):
            logger.info(f"✅ {model_name} already exists")
            continue
        
        downloaded = False
        for url in urls:
            if download_file(url, model_path, f"Wav2Lip {model_name}"):
                downloaded = True
                break
        
        if not downloaded:
            logger.error(f"❌ Failed to download {model_name} from all sources")
            return False
    
    # Download face detection models
    for model_name, urls in face_detection_models.items():
        model_path = os.path.join(face_detection_dir, model_name)
        
        if os.path.exists(model_path):
            logger.info(f"✅ {model_name} already exists")
            continue
        
        downloaded = False
        for url in urls:
            if download_file(url, model_path, f"Face Detection {model_name}"):
                downloaded = True
                break
        
        if not downloaded:
            logger.error(f"❌ Failed to download {model_name} from all sources")
            return False
    
    logger.info("✅ Wav2Lip setup completed")
    return True

def install_additional_dependencies():
    """Install additional Python dependencies for models"""
    logger.info("📦 Installing additional dependencies...")
    
    dependencies = [
        "safetensors",
        "transformers",
        "diffusers",
        "accelerate",
        "xformers",
        "librosa",
        "soundfile",
        "resampy",
        "scikit-image",
        "dlib",
        "face-alignment",
        "yacs",
        "pydub"
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True, capture_output=True)
            logger.info(f"✅ Installed {dep}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to install {dep}: {e}")
    
    logger.info("✅ Additional dependencies installation completed")

def main():
    """Main function to download all models"""
    logger.info("🚀 Starting model download process...")
    
    try:
        # Install additional dependencies first
        install_additional_dependencies()
        
        # Download SadTalker
        sadtalker_success = download_sadtalker_models()
        
        # Download Wav2Lip
        wav2lip_success = download_wav2lip_models()
        
        # Create completion marker
        if sadtalker_success or wav2lip_success:
            marker_file = os.path.join(MODELS_DIR, ".models_downloaded")
            with open(marker_file, 'w') as f:
                f.write("Models downloaded successfully\n")
                f.write(f"SadTalker: {'✅' if sadtalker_success else '❌'}\n")
                f.write(f"Wav2Lip: {'✅' if wav2lip_success else '❌'}\n")
            
            logger.info("✅ Model download process completed")
        else:
            logger.error("❌ All model downloads failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Model download process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
