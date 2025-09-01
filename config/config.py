import os
import platform

class Config:
    """Base configuration class"""
    
    # Model Configuration
    MODEL_PATH = "facenet_africain_finetuned.pth"
    EMBEDDINGS_DIR = "embeddings"
    CAPTURE_DIR = "captures"
    
    # Recognition Settings
    THRESHOLD = 0.7
    DETECTION_DELAY = 3
    
    # Device Configuration
    DEVICE = 'cuda' if os.environ.get('FORCE_CPU') != 'true' else 'cpu'
    try:
        import torch
        DEVICE = 'cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu'
    except ImportError:
        DEVICE = 'cpu'
    
    # Platform Detection
    IS_RASPBERRY_PI = platform.machine() in ('armv7l', 'armv6l', 'aarch64')
    
    # Camera Settings
    if IS_RASPBERRY_PI:
        CAMERA_WIDTH = 640
        CAMERA_HEIGHT = 480
        CAMERA_FPS = 15
        PROCESSING_WIDTH = 320
        PROCESSING_HEIGHT = 240
    else:
        CAMERA_WIDTH = 1280
        CAMERA_HEIGHT = 720
        CAMERA_FPS = 30
        PROCESSING_WIDTH = None  # Use full resolution
        PROCESSING_HEIGHT = None
    
    # Door Control Settings
    DOOR_UNLOCK_DURATION = 15  # seconds
    DOOR_AUTO_RELOCK_DELAY = 5  # seconds before starting relock countdown
    DOOR_RELOCK_COUNTDOWN = 10  # seconds for final countdown
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    ACCESS_LOG_FILE = "access_log.txt"
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # History Settings
    MAX_DETECTION_HISTORY = 100

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    THRESHOLD = 0.8  # Stricter threshold for development

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("No SECRET_KEY set for production environment")

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    THRESHOLD = 0.5  # Looser threshold for testing

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}