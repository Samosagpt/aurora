"""
Production configuration for Samosa GPT
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ProductionConfig:
    """Production-ready configuration class with validation and security"""

    def __init__(self):
        self._validate_environment()
        self._setup_logging()

    # Base paths
    BASE_DIR = Path(__file__).parent.absolute()
    ASSETS_DIR = BASE_DIR / "_assets"
    LOGS_DIR = BASE_DIR / "logs"
    CACHE_DIR = BASE_DIR / ".cache"

    # Ensure directories exist with proper permissions
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories with proper permissions"""
        directories = [cls.ASSETS_DIR, cls.LOGS_DIR, cls.CACHE_DIR]
        for directory in directories:
            directory.mkdir(exist_ok=True, mode=0o755)

    # Environment validation
    def _validate_environment(self):
        """Validate critical environment variables"""
        # Only validate in production environment
        if self.ENV == "production":
            required_vars = ["OLLAMA_HOST"]
            missing_vars = []

            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)

            if missing_vars:
                raise EnvironmentError(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )
        else:
            # In development/testing, just log warnings
            recommended_vars = ["OLLAMA_HOST"]
            for var in recommended_vars:
                if not os.getenv(var):
                    print(f"Warning: {var} not set, using default value")

    # Logging configuration
    def _setup_logging(self):
        """Set up production logging"""
        log_level = logging.INFO
        if self.DEBUG:
            log_level = logging.DEBUG
        elif self.ENV == "production":
            log_level = logging.WARNING

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.LOGS_DIR / "samosa.log", mode="a"),
                logging.StreamHandler(),
            ],
        )

    # Environment settings
    ENV = os.getenv("ENVIRONMENT", "development")  # development, staging, production
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")

    # Security settings
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "60"))

    # API Keys (with validation)
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

    # Ollama settings
    DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "samosagpt")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")  # Optional API key for Ollama

    # Web Search settings
    WEB_SEARCH_MODEL = os.getenv("WEB_SEARCH_MODEL", DEFAULT_OLLAMA_MODEL)
    ENABLE_AI_SEARCH = os.getenv("ENABLE_AI_SEARCH", "true").lower() == "true"

    # Streamlit settings
    PAGE_TITLE = "🤖 Samosa GPT"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"

    # Performance settings
    MAX_MEMORY_USAGE = int(os.getenv("MAX_MEMORY_USAGE", "4096"))  # MB
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # seconds

    # File upload limits
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "100"))  # MB
    ALLOWED_EXTENSIONS = ["txt", "md", "pdf", "doc", "docx", "jpg", "jpeg", "png", "gif"]

    # Audio settings
    AUDIO_FILE = ASSETS_DIR / "speech.wav"
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "44100"))
    CHANNELS = int(os.getenv("CHANNELS", "1"))
    CHUNK = int(os.getenv("CHUNK", "1024"))
    SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.01"))
    SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", "2.0"))

    # Whisper settings
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

    # TTS settings
    TTS_VOICE_PRESET = os.getenv("TTS_VOICE_PRESET", "v2/en_speaker_6")

    # Image generation settings
    DEFAULT_IMAGE_MODEL = os.getenv("DEFAULT_IMAGE_MODEL", "runwayml/stable-diffusion-v1-5")
    IMAGE_MODELS = {
        "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
        "Stable Diffusion v2.1": "stabilityai/stable-diffusion-2-1",
        "Stable Diffusion XL Base": "stabilityai/stable-diffusion-xl-base-1.0",
        "SDXL Turbo": "stabilityai/sdxl-turbo",
        "Dreamlike Photoreal 2.0": "dreamlike-art/dreamlike-photoreal-2.0",
        "Realistic Vision v6.0": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "Anything v5": "stablediffusionapi/anything-v5",
        "OpenJourney": "prompthero/openjourney",
        "Deliberate v2": "XpucT/Deliberate",
    }

    # Video generation settings
    DEFAULT_VIDEO_MODEL = os.getenv("DEFAULT_VIDEO_MODEL", "damo-vilab/text-to-video-ms-1.7b")
    VIDEO_MODELS = {
        "Text-to-Video MS 1.7B": "damo-vilab/text-to-video-ms-1.7b",
        "ModelScope T2V": "damo-vilab/text-to-video-ms-1.7b",
        "Zeroscope v2 XL": "cerspense/zeroscope_v2_XL",
    }

    # App version
    VERSION = "3.5.0"

    # Health check settings
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))  # seconds

    # Log files
    CONVERSATION_LOG = LOGS_DIR / "conversation_log.json"
    ASSISTANT_LOG = LOGS_DIR / "assistant_log.md"
    ERROR_LOG = LOGS_DIR / "error.log"
    ACCESS_LOG = LOGS_DIR / "access.log"

    @classmethod
    def get_safe_config(cls) -> Dict[str, Any]:
        """Get configuration without sensitive data for logging"""
        safe_config = {}
        for attr in dir(cls):
            if not attr.startswith("_") and not callable(getattr(cls, attr)):
                value = getattr(cls, attr)
                # Hide sensitive information
                if any(
                    sensitive in attr.lower()
                    for sensitive in ["key", "token", "secret", "password"]
                ):
                    safe_config[attr] = "***HIDDEN***"
                else:
                    safe_config[attr] = value
        return safe_config

    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Validate API keys"""
        return {
            "openweather": bool(
                cls.OPENWEATHER_API_KEY and cls.OPENWEATHER_API_KEY != "YOUR_API_KEY"
            ),
            "news": bool(cls.NEWS_API_KEY and cls.NEWS_API_KEY != "YOUR_API_KEY"),
            "huggingface": bool(cls.HUGGINGFACE_TOKEN and cls.HUGGINGFACE_TOKEN != "YOUR_HF_TOKEN"),
        }


# Initialize directories
ProductionConfig.ensure_directories()

# Create global config instance
config = ProductionConfig()
