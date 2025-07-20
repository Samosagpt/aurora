# 🤖 Samosa GPT: Advanced AI Assistant with Multi-Modal Capabilities

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0.0--pre-brightgreen.svg)](https://github.com/Samosagpt/samosa)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-green.svg)](LICENSE)
[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/80x15.png)](LICENSE)



## 🌟 Overview

Samosa GPT is a comprehensive AI assistant that combines text, voice, and image generation capabilities in a single, production-ready application. Built with modern Python technologies, it offers multiple interaction modes and can be packaged as a standalone executable.

## ✨ Key Features

### 🎯 Core Capabilities
- **Multi-Modal AI Chat**: Text conversations using Ollama models
- **Voice Interaction**: Speech-to-text and text-to-speech capabilities
- **Image Generation**: Multiple Stable Diffusion models support
- **Web Search Integration**: Wikipedia, Google, YouTube search
- **Real-time Information**: Weather and news updates
- **Smart Intent Detection**: Automatic routing of user queries

### 🖥️ Interface Options
- **Web Interface**: Modern Streamlit-based GUI
- **Command Line**: Terminal-based interaction
- **Voice Mode**: Hands-free voice commands
- **Standalone Executable**: No Python installation required

### 🎨 Image Generation
- 8+ Pre-configured Stable Diffusion models
- Custom model support via Hugging Face
- Advanced generation parameters
- Download functionality
- GPU acceleration support

## 🚀 Quick Start

### 📋 Prerequisites
- Python 3.8 or higher
- [Ollama](https://ollama.ai/) (for AI chat functionality)
- Git (for cloning the repository)

### 🔧 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Samosagpt/samosagpt.git
   cd samosagpt
   ```

2. **Run Setup Script**
   ```bash
   python setup.py
   ```
   This will automatically:
   - Check Python version compatibility
   - Install all required dependencies
   - Set up environment files
   - Create necessary directories
   - Test the installation

3. **Configure Environment (Optional)**
   ```bash
   # Copy and edit .env file for API keys
   cp .env.example .env
   # Edit .env with your favorite editor
   ```

4. **Install Ollama Models**
   ```bash
   # Install recommended models
   ollama pull llama2
   ollama pull mistral
   ollama pull codellama
   ```

5. **Install Bark TTS (Optional)**
   ```bash
   # For high-quality text-to-speech
   python install_bark.py
   # Or manually:
   pip install git+https://github.com/suno-ai/bark.git
   ```

### 🎮 Usage

#### Web Interface (Recommended)
```bash
python app.py --mode web
```
Opens a modern web interface at `http://localhost:8501`

#### Command Line Interface
```bash
python app.py --mode cli
```
Text-based interaction in terminal

#### Voice Mode
```bash
python app.py --mode voice
```
Hands-free voice commands and responses

#### Debug Mode
```bash
python app.py --mode web --debug
```
Enable detailed logging and debugging

## 🎨 Image Generation

Samosa GPT includes a powerful image generation module with multiple models:

### Available Models
- **Stable Diffusion v1.5**: General purpose, reliable quality
- **Stable Diffusion v2.1**: Enhanced coherence and detail
- **Stable Diffusion XL**: High resolution, detailed outputs
- **Dreamlike Photoreal**: Photorealistic image generation
- **Realistic Vision**: Highly realistic results
- **Anything v5**: Anime and artistic styles
- **OpenJourney**: Midjourney-style artistic images
- **Deliberate v2**: Balanced artistic approach

### Features
- Model selection and switching
- Custom Hugging Face model support
- Advanced generation parameters
- Seed control for reproducibility
- Batch generation capabilities
- Image download functionality

## 🗣️ Voice Capabilities

### Speech Recognition
- **Whisper Integration**: OpenAI's Whisper for accurate transcription
- **Multiple Model Sizes**: From tiny to large models
- **Noise Handling**: Advanced silence detection
- **Real-time Processing**: Low-latency speech processing

### Text-to-Speech
- **Bark TTS**: High-quality neural voice synthesis
- **Multiple Voices**: Various voice presets available
- **Fallback Support**: pyttsx3 backup for compatibility
- **Audio Export**: Save generated speech as audio files

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with your API keys:
```env
# API Keys
OPENWEATHER_API_KEY=your_openweather_key
NEWS_API_KEY=your_news_api_key

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Debug Mode
DEBUG=false
```

### Model Configuration
Edit `config.py` to customize:
- Default models
- File paths
- API endpoints
- Application settings

## � Project Structure

```
samosa/
├── app.py                     # Main application launcher
├── config.py                  # Configuration management
├── setup.py                   # Installation script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── 
├── Core Modules/
├── ├── Generation.py          # AI chat with Ollama
├── ├── image_gen.py          # Image generation
├── ├── offline_sr_whisper.py # Speech recognition
├── ├── offline_text2speech.py# Text-to-speech
├── ├── prompthandler.py      # Query processing
├── ├── ws.py                 # Web services
├── └── logmanagement.py      # Logging system
├── 
├── Interface/
├── ├── streamlit_app.py      # Web interface
├── ├── tertiary.css          # Custom styling
├── └── script.js             # JavaScript utilities
├── 
├── Build/
├── ├── samosa.spec           # PyInstaller configuration
├── ├── build.bat             # Windows build script
├── └── build.sh              # Linux/macOS build script
├── 
└── Assets/
    ├── _assets/              # Audio and media files
    └── logs/                 # Application logs
```

## 🔧 Advanced Usage

### Custom Models
Add your own Ollama models:
```bash
ollama pull your-custom-model
```

### Custom Image Models
Use any Hugging Face diffusion model:
1. Enable "Use custom model ID" in the web interface
2. Enter the model ID (e.g., `user/model-name`)
3. The model will be downloaded automatically

### API Integration
Configure external APIs for enhanced functionality:
- OpenWeather API for weather data
- News API for current news
- Custom web services

## � Testing

### Run Tests
```bash
python -m pytest tests/
```

### Test Components
```bash
# Test speech recognition
python -c "from offline_sr_whisper import speech_recognizer; print(speech_recognizer.test_microphone())"

# Test text-to-speech
python -c "from offline_text2speech import tts_manager; print(tts_manager.test_engines())"

# Test Ollama connection
python -c "from Generation import ollama_manager; print(ollama_manager.get_available_models())"
```

## 📊 Performance Optimization

### GPU Acceleration
- CUDA support for image generation
- Automatic GPU detection and usage
- Memory optimization for large models

### Memory Management
- Model caching and lazy loading
- Automatic cleanup of temporary files
- Configurable resource limits

## 🐛 Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Ensure Ollama is running
ollama serve
```

**Audio Issues**
```bash
# Check audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"
```

**Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)**.

### You are free to:
- **Share** — copy and redistribute the material in any medium or format

### Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes
- **NoDerivatives** — If you remix, transform, or build upon the material, you may not distribute the modified material

For the full license text, see [LICENSE](LICENSE) or visit [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/)

### Third-Party Components
This software includes third-party components with their own licenses. Please refer to individual component documentation for their terms and conditions.

## 🙏 Acknowledgments

- OpenAI for Whisper speech recognition
- Stability AI for Stable Diffusion models
- Ollama team for local LLM support
- Streamlit for the web framework
- All open-source contributors

## 📞 Support

- 🐛 Issues: [GitHub Issues](https://github.com/Samosagpt/samosagpt/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Samosagpt/samosagpt/discussions)
- 📧 Email: psntejaji@gmail.com

---

**Made with ❤️ by the P.S.N Tejaji**
