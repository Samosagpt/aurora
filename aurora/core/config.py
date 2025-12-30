"""
Configuration settings for Aurora
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration class"""

    # Base paths
    BASE_DIR = Path(__file__).parent.absolute()
    ASSETS_DIR = BASE_DIR / "_assets"
    LOGS_DIR = BASE_DIR / "logs"

    # Ensure directories exist
    ASSETS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    # Log files
    CONVERSATION_LOG = LOGS_DIR / "conversation_log.json"
    ASSISTANT_LOG = LOGS_DIR / "assistant_log.md"

    # Audio settings
    AUDIO_FILE = ASSETS_DIR / "speech.wav"
    SAMPLE_RATE = 44100
    CHANNELS = 1
    CHUNK = 1024
    SILENCE_THRESHOLD = 0.01
    SILENCE_DURATION = 2.0

    # Whisper settings
    WHISPER_MODEL = "base"

    # TTS settings
    TTS_VOICE_PRESET = "v2/en_speaker_6"

    # API Keys (from environment variables)
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_API_KEY")

    # Ollama settings
    DEFAULT_OLLAMA_MODEL = "llama3.2"
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")  # Optional API key
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

    # Web Search settings
    WEB_SEARCH_MODEL = os.getenv(
        "WEB_SEARCH_MODEL", DEFAULT_OLLAMA_MODEL
    )  # Model for AI web search
    ENABLE_AI_SEARCH = os.getenv("ENABLE_AI_SEARCH", "true").lower() == "true"

    # Streamlit settings
    PAGE_TITLE = "🌅 Aurora"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"

    # Image generation settings
    DEFAULT_IMAGE_MODEL = "runwayml/stable-diffusion-v1-5"
    IMAGE_MODELS = {
        "Stable Diffusion v1.5": "runwayml/stable-diffusion-v1-5",
        "Stable Diffusion v2.1": "stabilityai/stable-diffusion-2-1",
        "Stable Diffusion XL Base": "stabilityai/stable-diffusion-xl-base-1.0",
        "Dreamlike Photoreal 2.0": "dreamlike-art/dreamlike-photoreal-2.0",
        "Realistic Vision v6.0": "SG161222/Realistic_Vision_V6.0_B1_noVAE",
        "Anything v5": "stablediffusionapi/anything-v5",
        "OpenJourney": "prompthero/openjourney",
        "Deliberate v2": "XpucT/Deliberate",
    }

    # App version
    VERSION = "1.0.0-alpha"

    # Debug mode
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"


# Create global config instance
config = Config()
