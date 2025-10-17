# 🤖 AURORA: Agentic Unified multi-model Reasoning Orchestrator for Rapid One-shot Assistance

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0-brightgreen.svg)](https://github.com/Samosagpt/aurora)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-orange.svg)](https://ollama.ai/)

---

## 🌟 Overview

**AURORA** is a next-generation AI assistant that seamlessly integrates multiple AI capabilities into one powerful platform. Built by **Karthikeyan Prasanna**, **Shyam**, and **Tejaji**, AURORA combines local LLM inference, computer vision, autonomous desktop control, RAG knowledge bases, and multimodal generation capabilities.

### What Makes AURORA Unique?

- **🧠 Local-First AI**: Powered by Ollama for complete privacy and control
- **👁️ Vision-Enabled Autonomy**: See the screen, understand context, and act intelligently
- **🤖 Desktop Control**: Execute tasks through natural language commands
- **📚 Knowledge Base (RAG)**: Enhanced responses using custom knowledge bases
- **🎨 Creative Generation**: Images and videos from text descriptions
- **🗣️ Voice Interface**: Full speech-to-text and text-to-speech support
- **📎 Multimodal Understanding**: Analyze images, PDFs, and documents

---

## ✨ Key Features

### 🎯 AI Chat & Intelligence

- **🤖 Multi-Model Support**: Switch between Llama, Mistral, CodeLlama, and 50+ Ollama models
- **💬 Streaming Responses**: Real-time text generation with think-tag support
- **📚 RAG Knowledge Base**: JSON-based vector search for context-aware responses
- **🧠 Smart Intent Detection**: Automatic routing between chat, web search, and tools
- **💭 Reasoning Visualization**: Collapsible "thinking" sections for transparency

### 👁️ Vision & Autonomy

- **� Vision Agent**: Autonomous task execution with screen understanding
- **📸 OCR Integration**: Text detection using Tesseract and EasyOCR
- **🖱️ GUI Recognition**: Computer vision-based UI element detection
- **🎯 Visual Feedback Loop**: Screenshot analysis → decision → action → verify
- **⚡ Smart Pre-checks**: Context-aware initialization (e.g., auto-open websites)

### 🤖 Desktop Control (Agentic AI)

- **🖱️ Mouse Control**: Click, move, drag, scroll with pixel precision
- **⌨️ Keyboard Automation**: Type text, press keys, execute shortcuts
- **🪟 Window Management**: List, switch, focus, minimize, restore windows
- **📂 File Operations**: Read, write, search, organize files and folders
- **🔧 System Commands**: Execute shell commands with safety checks
- **🌐 Application Control**: Open, close, manage applications and URLs

### 📎 Multimodal Understanding

- **🖼️ Image Analysis**: Vision models (LLaVA, BakLLaVA, Moondream) for image Q&A
- **📄 PDF Processing**: Extract text from PDFs with PyPDF2 and pdf2image
- **🎨 Multi-file Support**: Handle multiple images and PDFs simultaneously
- **💡 Context Enhancement**: Automatic text extraction for better AI responses

### 🎨 Creative Generation

#### Image Generation
- **8+ Stable Diffusion Models**: Pre-configured quality models
- **Custom Model Support**: Load any Hugging Face diffusion model
- **Advanced Parameters**: Guidance scale, steps, seeds, batch generation
- **GPU Acceleration**: CUDA optimization with memory management
- **Model Management**: Install, delete, and manage models from UI

#### Video Generation
- **Multiple T2V Models**: Zeroscope, ModelScope, AnimateDiff support
- **Resolution Options**: From 256x256 to 1024x576
- **Frame Control**: Adjust duration, FPS, and quality
- **Memory Optimization**: CPU offloading for limited VRAM systems

### 🗣️ Voice & Speech

- **🎤 Speech Recognition**: OpenAI Whisper for accurate transcription
- **🔊 Text-to-Speech**: Multiple engines (pyttsx3, Bark TTS, Edge TTS)
- **🎭 Voice Presets**: 10+ Bark neural voice options
- **🔇 Silence Detection**: Smart audio input handling
- **💾 Audio Export**: Save generated speech as audio files

### 🔍 Web & Information

- **🌐 AI-Powered Search**: Ollama-enhanced web search
- **📰 Smart News**: Topic-based news aggregation
- **🌤️ Weather**: Real-time weather with AI fallback
- **📚 Wikipedia**: Integrated Wikipedia search
- **🎬 YouTube**: Video search integration

### 🖥️ Interface Options
- **Web Interface**: Modern Streamlit-based GUI with drag-and-drop attachments
- **Command Line**: Terminal-based interaction
- **Voice Mode**: Hands-free voice commands
- **Standalone Executable**: No Python installation required

### 🎨 Image Generation
- 8+ Pre-configured Stable Diffusion models
- Custom model support via Hugging Face
- Advanced generation parameters
- Download functionality
- GPU acceleration support

### 🤖 Agentic AI - Desktop Control (NEW!)
Turn your AI into a desktop automation assistant! The agentic AI can:
- **Control Mouse & Keyboard**: Click, type, move, shortcuts
- **Manage Applications**: Open, close, switch between programs
- **File Operations**: Read, write, search, organize files
- **Take Screenshots**: Capture and save screen images
- **Run Commands**: Execute system commands safely
- **Window Management**: List, switch, and control windows
- **System Monitoring**: Check CPU, memory, disk usage

**Example Commands:**
```text
"Open Notepad and type Hello World"
"Take a screenshot and save it"
"List all Python files in my Documents folder"
"Close all Chrome windows"
"Create a file called todo.txt with my tasks"
```

See [AGENTIC_GUIDE.md](AGENTIC_GUIDE.md) for detailed documentation.

### 📎 Attachment Support
- **Image Analysis**: Upload JPG, PNG, GIF, BMP, WEBP images
- **PDF Processing**: Extract text and analyze PDF documents
- **Vision Models**: LLaVA, BakLLaVA, Moondream, and more
- **Multi-file Support**: Attach multiple files at once
- **Smart Context**: Automatic text extraction from PDFs
- **Use Cases**: 
  - Ask questions about images
  - Summarize documents
  - Extract information from screenshots
  - Analyze diagrams and charts

**Quick Setup:**
```bash
# Install dependencies
pip install Pillow PyPDF2 pdf2image

# Install vision model
ollama pull llava

# See ATTACHMENT_GUIDE.md for full setup
```

### 🔍 AI-Powered Web Search (NEW!)
- **Intelligent Search**: AI-enhanced web search using Ollama
- **Smart Search**: Combined Wikipedia + AI knowledge
- **News Search**: Search news by topic using AI
- **Contextual Results**: Get comprehensive, AI-analyzed results
- **Multiple Sources**: Combines Wikipedia, news, and AI insights
- **Search Commands**:
  - `web search [query]` - AI-powered search
  - `smart search [topic]` - Deep research with multiple sources
  - `news about [topic]` - Topic-specific news search
  
**Example:**
```bash
"web search latest developments in quantum computing"
"smart search climate change solutions"
"news about artificial intelligence"
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **[Ollama](https://ollama.ai/)** - For local LLM inference
- **Git** - For cloning the repository
- **Windows/Linux/macOS** - Cross-platform support

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Samosagpt/aurora.git
   cd aurora
   ```

2. **Run Setup Script**

   ```bash
   python setup.py
   ```

   This automatically:
   - Checks Python version compatibility
   - Installs all required dependencies
   - Sets up environment files
   - Creates necessary directories
   - Tests the installation

3. **Install Ollama Models**

   ```bash
   # Install recommended models
   ollama pull llama3.2
   ollama pull mistral
   ollama pull codellama
   
   # For vision support (image analysis)
   ollama pull llava
   
   # For vision agent (autonomous control)
   ollama pull qwen3-vl:235b-cloud
   ```

4. **Optional: Advanced Features**

   ```bash
   # For Tesseract OCR (desktop control)
   # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   # Linux: sudo apt-get install tesseract-ocr
   # macOS: brew install tesseract
   
   # For high-quality TTS
   pip install git+https://github.com/suno-ai/bark.git
   ```

### Launch AURORA

#### Web Interface (Recommended)

```bash
# Windows
run_web.bat

# Linux/macOS
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

#### Console Mode

```bash
# Windows
run_console.bat

# Linux/macOS
python main.py
```

---

## 💡 Usage Examples

### Basic Chat

```python
from Generation import ollama_manager

# Simple chat
response = ollama_manager.chat_with_memory("Explain quantum computing", model_name="llama3.2")
print(response)

# Streaming chat
for chunk in ollama_manager.chat_with_memory_stream("Write a poem about AI"):
    print(chunk, end="", flush=True)
```

### Using RAG Knowledge Base

```python
from rag_handler import get_rag_handler

# Initialize RAG
rag = get_rag_handler()

# Add knowledge
rag.add_knowledge("AURORA is an advanced AI assistant created by Karthik, Shyam, and Tejaji")

# Query
result = rag.query("Who created AURORA?")
print(result['answer'])
```

### Desktop Control

```python
from desktop_agent import desktop_agent

# Open an application
desktop_agent.open_application("notepad")

# Type text
desktop_agent.type_text("Hello from AURORA!")

# Take screenshot
result = desktop_agent.take_screenshot()
print(f"Screenshot saved: {result['screenshot_path']}")

# OCR screen
ocr_result = desktop_agent.ocr_screen()
print(f"Found {len(ocr_result['texts'])} text elements")
```

### Vision Agent (Autonomous)

```python
from vision_agent import execute_autonomous_task

# Execute complex task
result = execute_autonomous_task(
    "Open GitHub and show me my pull requests",
    model="qwen3-vl:235b-cloud",
    max_steps=10
)
```

### Attachment Analysis

```python
from attachment_handler import attachment_handler
from Generation import ollama_manager

# Process image
image_data = attachment_handler.process_image("photo.jpg")

# Format for Ollama
formatted = attachment_handler.format_for_ollama(
    "What's in this image?",
    [image_data]
)

# Get response
response = ollama_manager.chat_with_memory(
    formatted['prompt'],
    model_name="llava",
    images=formatted.get('images')
)
```

---

## 🏗️ Architecture

### Project Structure

```
aurora/
├── 🎯 Core AI Engine
│   ├── Generation.py              # Ollama manager & chat interface
│   ├── aurora_system.py          # Identity & system configuration
│   ├── rag_handler.py            # RAG knowledge base
│   └── prompthandler.py          # Intent detection & routing
│
├── 🤖 Agentic Capabilities
│   ├── desktop_agent.py          # Desktop control tools
│   ├── agentic_handler.py        # Agent orchestration
│   └── vision_agent.py           # Autonomous vision-guided agent
│
├── 🎨 Generation Modules
│   ├── image_gen.py              # Stable Diffusion interface
│   ├── video_gen.py              # Text-to-video generation
│   ├── image_model_manager.py    # Image model management
│   └── video_model_manager.py    # Video model management
│
├── 🗣️ Voice & Speech
│   ├── offline_sr_whisper.py     # Speech recognition (Whisper)
│   └── offline_text2speech.py    # TTS (Bark, pyttsx3, Edge)
│
├── 📎 Multimodal Processing
│   ├── attachment_handler.py     # Image & PDF processing
│   └── PreTrainedResponses.py    # Response templates
│
├── 🖥️ User Interfaces
│   ├── streamlit_app.py          # Main web interface
│   ├── main.py                   # Console interface
│   └── streamlit_navbar/         # Custom navbar component
│
├── ⚙️ Configuration & Utils
│   ├── config.py / config_prod.py # Configuration management
│   ├── hardware_optimizer.py     # Hardware detection & optimization
│   ├── user_preferences.py       # User settings management
│   ├── logmanagement.py          # Logging system
│   ├── error_handler.py          # Error handling
│   └── security.py               # Security & audit
│
├── 📦 Setup & Deployment
│   ├── setup.py                  # Installation script
│   ├── installer.py              # Executable builder
│   ├── requirements.txt          # Python dependencies
│   └── install_aurora.bat        # Windows installer
│
└── 📊 Data & Logs
    ├── rag_db.json               # RAG knowledge base
    ├── aurora_config.json        # System configuration
    └── logs/                     # Execution logs & screenshots
```

### Key Components

#### 1. **OllamaManager** (`Generation.py`)

- Manages Ollama client connections
- Handles model switching and streaming
- Processes think-tags for reasoning visualization
- Supports multimodal inputs (text + images)

#### 2. **RAGHandler** (`rag_handler.py`)

- JSON-based vector database
- Keyword-based similarity search
- Document chunking and retrieval
- Ollama integration for answer generation

#### 3. **DesktopAgent** (`desktop_agent.py`)

- 20+ desktop control tools
- OCR using Tesseract & EasyOCR
- GUI element recognition
- Window management (Win32 API)
- File and system operations

#### 4. **VisionAgent** (`vision_agent.py`)

- Autonomous task execution
- Screenshot analysis loop
- Roadmap planning with LLMs
- Context-aware decision making
- Smart pre-checks (e.g., auto-open URLs)

#### 5. **AgenticHandler** (`agentic_handler.py`)

- Tool registry and execution
- Natural language → tool calling
- Multi-step workflow orchestration
- Safety checks and validation

---

## � Configuration

### Hardware Optimization

AURORA automatically detects your hardware and optimizes settings:

```python
from hardware_optimizer import get_hardware_optimizer

hw = get_hardware_optimizer()

# Get optimized settings
chat_settings = hw.get_chat_settings()
image_settings = hw.get_image_settings()
video_settings = hw.get_video_settings()
```

### Environment Variables

Create a `.env` file for API keys (optional):

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_API_KEY=your_api_key_here  # For remote Ollama instances

# Optional External APIs
OPENWEATHER_API_KEY=your_key  # Weather data
NEWS_API_KEY=your_key          # News aggregation

# Debug Mode
DEBUG=false
```

### User Preferences

AURORA remembers your preferences:

- Last used models (chat, image, video, TTS)
- Streaming preference
- Speech enablement
- Voice presets

Preferences are stored in `logs/user_preferences.json`

---

## �🐛 Troubleshooting

### Common Issues

#### Ollama Connection Failed

```bash
# Ensure Ollama is running
ollama serve

# Check if models are available
ollama list
```

#### OCR Not Working

```bash
# Install Tesseract OCR
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract

# Verify installation
tesseract --version
```

#### GPU Not Detected

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

#### Module Import Errors

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Or run setup again
python setup.py
```

---

## 👥 Team & Attribution

### Core Team

**AURORA** is developed by:

1. **Karthikeyan Prasanna** - Lead Architect & AI Finetuning Manager
   - System design and reasoning frameworks
   - AI behavior orchestration
   - Prompt architecture and tone alignment

2. **Shyam** - Data Lead & Integration Coordinator
   - Data collection and preprocessing
   - Dataset structuring for training
   - Cross-team coordination and documentation

3. **Tejaji (P.S.N Tejaji)** - Software Lead & Infrastructure Developer
   - AURORA software framework development
   - AI component integration
   - Performance optimization and deployment

### Identity

AURORA stands for **Agentic Unified multi-model Reasoning Orchestrator for Rapid One-shot Assistance** - an AI built to deliver fast, reliable, one-shot help across diverse tasks with an Alfred-like demeanor.

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)**.

### Terms

- ✅ **Share**: Copy and redistribute in any medium or format
- ⚠️ **Attribution**: Give appropriate credit to creators (Karthik, Shyam, Tejaji)
- ❌ **NonCommercial**: No commercial use without permission
- ❌ **NoDerivatives**: No distribution of modified versions

Full license: [LICENSE](LICENSE) | [Creative Commons](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## 🙏 Acknowledgments

### Technologies

- **[Ollama](https://ollama.ai/)** - Local LLM inference engine
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Speech recognition
- **[Stability AI](https://stability.ai/)** - Stable Diffusion models
- **[Streamlit](https://streamlit.io/)** - Web framework
- **[PyAutoGUI](https://pyautogui.readthedocs.io/)** - Desktop automation
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)** - Text recognition

### Open Source Libraries

- diffusers, transformers, accelerate (Hugging Face)
- torch, torchvision (PyTorch)
- opencv-python, pytesseract, easyocr (Computer Vision)
- pywin32, psutil (System integration)
- pyttsx3, bark (Text-to-Speech)

---

## 📞 Support & Community

- 🐛 **Issues**: [GitHub Issues](https://github.com/Samosagpt/aurora/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Samosagpt/aurora/discussions)
- 📧 **Email**: psntejaji@gmail.com
- 📚 **Documentation**: Check the `/logs` folder for detailed guides

---

## 🚀 Future Roadmap

- [ ] Web-based RAG document upload interface
- [ ] Multi-user support with authentication
- [ ] Cloud deployment options (Docker, Kubernetes)
- [ ] Mobile app (React Native/Flutter)
- [ ] Browser extension for quick access
- [ ] Plugin system for custom tools
- [ ] Voice cloning for personalized TTS
- [ ] Advanced scheduling and automation
- [ ] Integration with more LLM providers

---

**Made with ❤️ by Karthikeyan Prasanna, Shyam, and Tejaji**

*AURORA - Your intelligent companion for the age of AI*
