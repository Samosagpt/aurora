"""
Production-ready Streamlit application for Aurora
"""
import streamlit as st
import Generation as gen
import subprocess
import json
import time

# Desktop automation imports - optional for cloud deployment
try:
    import keyboard as kb
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    kb = None

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    sd = None
    sf = None

import os
import threading
import numpy as np
import wave
import io
import sys
import types
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

import re
import gc
import tempfile
import logging
import base64
import json
from pathlib import Path

# Import production modules
try:
    from config_prod import config
except ImportError:
    from config import config

try:
    from error_handler import error_handler, production_error_handler, performance_monitor
    from security import security_manager, audit_logger, rate_limited
    from health_check import health_check
    PRODUCTION_MODE = True
except ImportError:
    PRODUCTION_MODE = False
    logging.warning("Production modules not available, running in basic mode")

# Import core modules - audio modules may not be available on all platforms
try:
    from offline_sr_whisper import speech_recognition
    SPEECH_RECOGNITION_AVAILABLE = True
except (ImportError, OSError) as e:
    SPEECH_RECOGNITION_AVAILABLE = False
    logging.warning(f"Speech recognition not available: {e}")
    speech_recognition = None

try:
    from offline_text2speech import speak_text
    TEXT_TO_SPEECH_AVAILABLE = True
except (ImportError, OSError) as e:
    TEXT_TO_SPEECH_AVAILABLE = False
    logging.warning(f"Text-to-speech not available: {e}")
    speak_text = None

import logmanagement as lm
from user_preferences import get_preferences_manager
from streamlit_markdown_select import markdown_select, create_option
from streamlit_navbar import navbar, create_nav_item

# Import attachment handler
try:
    from attachment_handler import attachment_handler
    ATTACHMENT_HANDLER_AVAILABLE = True
except ImportError:
    ATTACHMENT_HANDLER_AVAILABLE = False
    print("⚠️ Attachment handler not available")

# Import prompt handler for web search, weather, news
try:
    from prompthandler import prompt_handler
    PROMPT_HANDLER_AVAILABLE = True
    print("✅ Prompt handler loaded (web search, weather, news)")
except ImportError:
    PROMPT_HANDLER_AVAILABLE = False
    print("⚠️ Prompt handler not available")

# Import agentic handler for desktop control
try:
    from agentic_handler import agentic_handler, is_desktop_control_request, handle_agentic_request
    AGENTIC_HANDLER_AVAILABLE = True
    print("✅ Agentic handler loaded (desktop control)")
except ImportError:
    AGENTIC_HANDLER_AVAILABLE = False
    print("⚠️ Agentic handler not available")

# Import vision agent for autonomous task execution
try:
    from vision_agent import execute_autonomous_task, vision_agent
    VISION_AGENT_AVAILABLE = True
    print("✅ Vision agent loaded (autonomous task execution)")
except ImportError:
    VISION_AGENT_AVAILABLE = False
    print("⚠️ Vision agent not available")

# Import RAG handler for knowledge base
try:
    from rag_handler import get_rag_handler, RAGHandler
    RAG_HANDLER_AVAILABLE = True
    print("✅ RAG handler loaded (knowledge base)")
except ImportError:
    RAG_HANDLER_AVAILABLE = False
    print("⚠️ RAG handler not available")

# Import AURORA system configuration
try:
    from aurora_system import get_aurora_system, AuroraSystem
    AURORA_SYSTEM_AVAILABLE = True
    print("✅ AURORA system configuration loaded")
except ImportError:
    AURORA_SYSTEM_AVAILABLE = False
    print("⚠️ AURORA system configuration not available")

# Set up logging
logger = logging.getLogger(__name__)

# Try to import torch (optional for video generation)
try:
    import torch
except ImportError:
    torch = None

# Import video model manager
try:
    from video_model_manager import video_model_manager
    VIDEO_MODEL_MANAGER_AVAILABLE = True
except ImportError:
    VIDEO_MODEL_MANAGER_AVAILABLE = False
    print("⚠️ Video model manager not available")

# Import hardware optimizer
try:
    from hardware_optimizer import get_hardware_optimizer, get_chat_settings, get_image_settings, get_video_settings, get_hardware_info
    HARDWARE_OPTIMIZER_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZER_AVAILABLE = False
    print("⚠️ Hardware optimizer not available")

# Initialize hardware optimizer
if HARDWARE_OPTIMIZER_AVAILABLE:
    try:
        _hw_optimizer = get_hardware_optimizer()
        print("✅ Hardware optimizer initialized")
    except Exception as e:
        print(f"⚠️ Hardware optimizer initialization failed: {e}")
        HARDWARE_OPTIMIZER_AVAILABLE = False

# Initialize user preferences manager
try:
    _preferences_manager = get_preferences_manager()
    print("✅ User preferences manager initialized")
except Exception as e:
    print(f"⚠️ User preferences manager initialization failed: {e}")
    _preferences_manager = None

# Configure page (only when running in Streamlit)
def configure_streamlit():
    """Configure Streamlit page settings"""
    try:
        # Load page icon from logo file
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_assets', 'aurora_logo.jpg')
        page_icon = "🤖"  # Default emoji fallback
        
        # Try to load the image for page icon
        if os.path.exists(logo_path):
            try:
                from PIL import Image
                page_icon = Image.open(logo_path)
            except Exception:
                pass  # Use emoji fallback if PIL fails
        
        st.set_page_config(
            page_title="Aurora",
            page_icon=page_icon,
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': "# Aurora\nAI Assistant with Multi-Modal Capabilities"
            }
        )
    except Exception:
        pass  # Already configured or not in Streamlit context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_CSS = os.path.join(BASE_DIR, 'tertiary.css')

def load_logo_as_base64():
    """Load logo image and convert to base64 data URL"""
    try:
        logo_path = os.path.join(BASE_DIR, '_assets', 'aurora_logo.jpg')
        if os.path.exists(logo_path):
            with open(logo_path, 'rb') as f:
                img_data = f.read()
            b64_data = base64.b64encode(img_data).decode()
            return f"data:image/jpeg;base64,{b64_data}"
        return None
    except Exception as e:
        print(f"Error loading logo: {e}")
        return None

def load_css():
    """Load CSS file safely"""
    try:
        with open(FILE_CSS) as f:
            css = f.read()
        return css
    except Exception:
        return ""

def inject_css():
    """Inject CSS styles safely"""
    try:
        # Inject style to reposition the native Streamlit button
        st.markdown(f"""
            <style>
            {load_css()}
            
            /* Custom styling for think dropdowns */
            details {
                margin: 10px 0; 
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            
            summary {
                cursor: pointer; 
                font-weight: bold;
                padding: 5px;
                background-color: #e9e9e9;
                border-radius: 3px;
            }
            
            summary:hover {
                background-color: #d9d9d9; 
            }
            
            details[open] summary {
                margin-bottom: 10px; 
            }
            
            /* Navbar component integration */
            .stApp > header {
                background-color: transparent;
                height: 0px;
            }
            
            /* Adjust main content to account for navbar */
            .main .block-container {
                padding-top: 1rem;
            }
            
            /* Ensure navbar component has proper z-index and styling */
            iframe[title="streamlit_navbar.navbar"] {
                position: sticky !important;
                top: 0 !important;
                z-index: 999 !important;
                height: 80px !important;
                border: none !important;
                width: 100% !important;
                background: var(--background-color, white) !important;
                box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24) !important;
            }
            
            /* Smooth integration with main content */
            .main .block-container {
                margin-top: 0 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    except Exception:
        pass  # Not in Streamlit context
# Function definitions
def get_ollama_models():
    """Fetch available models using the ollama module"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        
        lines = result.stdout.splitlines()
        models = set()  # Use a set to avoid duplicates
        for line in lines:
            if line and not line.startswith("failed") and not line.startswith("NAME"):
                parts = line.split()
                if len(parts) > 0:
                    model_name = parts[0].replace(":latest", "")
                    models.add(model_name)  # Add to the set to ensure uniqueness
        
        if not models:
            st.warning("No models found. Please ensure Ollama is set up correctly.")
            return []
        
        return sorted(models)  # Return a sorted list of unique models
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

def replay_audio(temp_wav_path):
    """Replay audio file"""
    if not AUDIO_AVAILABLE:
        st.warning("Audio playback not available on this platform.")
        return
    if temp_wav_path and os.path.exists(temp_wav_path):
        data, samplerate = sf.read(temp_wav_path, dtype='float32')
        sd.play(data, samplerate)
        sd.wait()
    else:
        st.warning("No audio to replay.")

def test_gpu_acceleration():
    """Test GPU acceleration functionality"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Simple GPU test
        device = torch.device('cuda')
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        
        return True, f"GPU test successful on {torch.cuda.get_device_name(0)}"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"GPU test failed: {str(e)}"

def test_bark_tts():
    """Test Bark TTS functionality"""
    try:
        from offline_text2speech import tts_manager
        if tts_manager.bark_available:
            return True, "Bark TTS is available and initialized"
        else:
            return False, "Bark TTS is not available"
    except ImportError:
        return False, "offline_text2speech module not found"
    except Exception as e:
        return False, f"Error testing Bark TTS: {str(e)}"

def chat_page():
    """Main chat interface"""
    st.title('💬 Chat Assistant')
    
    # Agentic mode toggle
    col_title1, col_title2, col_title3 = st.columns([2, 1, 1])
    with col_title1:
        st.write("") # Spacer
    with col_title2:
        if AGENTIC_HANDLER_AVAILABLE:
            agentic_mode = st.toggle("🤖 Basic Agent", value=st.session_state.get("agentic_mode", False), 
                                    help="Enable basic desktop control", key="agentic_mode_toggle")
            st.session_state["agentic_mode"] = agentic_mode
        else:
            agentic_mode = False
            st.session_state["agentic_mode"] = False
    
    with col_title3:
        if VISION_AGENT_AVAILABLE:
            vision_mode = st.toggle("�️ Vision Agent", value=st.session_state.get("vision_mode", False),
                                   help="Enable autonomous vision-guided control", key="vision_mode_toggle")
            st.session_state["vision_mode"] = vision_mode
            if vision_mode:
                st.success("🚀 Vision Agent active - I can see and control!")
        else:
            vision_mode = False
            st.session_state["vision_mode"] = False
    
    # Get hardware-optimized settings
    if HARDWARE_OPTIMIZER_AVAILABLE:
        chat_settings = get_chat_settings()
        recommended_models = chat_settings.get('recommended_models', [])
        default_streaming = chat_settings.get('streaming_enabled', True)
        hardware_info = get_hardware_info()
        
        # Show hardware performance indicator
        if hardware_info:
            perf_rec = _hw_optimizer.get_performance_recommendation()
            st.info(f"🖥️ Hardware Status: {perf_rec}")
    else:
        recommended_models = []
        default_streaming = True
    
    # Model selection with user preferences
    available_models = get_ollama_models()
    
    # Load user preferences for chat
    chat_prefs = {}
    if _preferences_manager:
        chat_prefs = _preferences_manager.get_category_preferences('chat')
    
    # Prioritize recommended models
    if recommended_models and available_models:
        # Sort available models to put recommended ones first
        sorted_models = []
        for rec_model in recommended_models:
            if rec_model in available_models:
                sorted_models.append(rec_model)
        
        # Add remaining models
        for model in available_models:
            if model not in sorted_models:
                sorted_models.append(model)
        
        available_models = sorted_models
    
    try:
        # Determine default model based on preferences and recommendations
        default_idx = 0
        preferred_model = chat_prefs.get('preferred_model') or chat_prefs.get('last_used_model')
        
        # Handle case where preferred_model might be a dictionary (from markdown_select component)
        if isinstance(preferred_model, dict):
            preferred_model = preferred_model.get('value') or preferred_model.get('optionValue')
        
        if preferred_model and isinstance(preferred_model, str) and preferred_model in available_models:
            # Use user's preferred model
            default_idx = available_models.index(preferred_model)
        elif recommended_models and available_models:
            # Fall back to first recommended model
            for i, model in enumerate(available_models):
                if model in recommended_models:
                    default_idx = i
                    break
        
        # Create options for custom select
        model_options = []
        default_model = available_models[default_idx] if available_models else None
        
        for i, model_name in enumerate(available_models):
            is_recommended = model_name in recommended_models if HARDWARE_OPTIMIZER_AVAILABLE else False
            is_default = model_name == default_model
            
            # Create option with status indicator and buttons
            buttons = [
                {"text": "🗑️ Delete", "variant": "danger", "action": "delete"}
            ]
            
            if is_default:
                # Add green indicator for default model
                option = create_option(
                    value=model_name,
                    label=f"🤖 {model_name}",
                    code="default",
                    code_color="#4CAF50",
                    buttons=buttons
                )
            else:
                option = create_option(
                    value=model_name,
                    label=f"🤖 {model_name}",
                    code=f"Model: {model_name}",
                    language="text",
                    code_color="#4CAF50" if is_recommended else "#888888",
                    buttons=buttons
                )
            model_options.append(option)
        
        # Add option to install new model
        install_option = create_option(
            value="__install_new__",
            label="📥 Install New Model",
            code="Install from Ollama registry",
            language="info",
            code_color="#FF9800",
            buttons=[
                {"text": "🔍 Browse", "variant": "primary", "action": "browse"},
                {"text": "📥 Install", "variant": "secondary", "action": "install"}
            ]
        )
        model_options.append(install_option)
        
        # Use custom markdown select
        default_value = available_models[default_idx] if available_models else None
        model_selection = markdown_select(
            options=model_options,
            key="chat_model_select",
            placeholder="🤖 Choose a model...",
            default_value=default_value
        )
        
        # Handle button clicks and model selection
        model = None
        
        # Check if we should show install interface (from button clicks or selection)
        show_install = False
        if st.session_state.get('show_install_interface'):
            show_install = True
            # Clear the flag after using it
            st.session_state['show_install_interface'] = False
        
        # Extract the actual selection value from the component
        actual_selection = None
        if model_selection:
            if isinstance(model_selection, dict):
                # Handle dictionary response from component
                if model_selection.get('type') == 'selection':
                    actual_selection = model_selection.get('value')
                elif model_selection.get('type') == 'button_click':
                    # This is a button click, not a selection
                    actual_selection = None
                else:
                    actual_selection = model_selection.get('value', model_selection)
            else:
                actual_selection = model_selection
        
        # Check if install new model is selected
        if actual_selection == "__install_new__":
            show_install = True
        
        if show_install:
            # Show install model interface
            st.subheader("📥 Install New Model")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                new_model_name = st.text_input(
                    "Model Name",
                    placeholder="e.g., llama3.2, codellama, mistral",
                    help="Enter the model name from Ollama registry"
                )
            with col2:
                if st.button("📥 Install Model", type="primary"):
                    if new_model_name:
                        with st.spinner(f"Installing {new_model_name}..."):
                            try:
                                import Generation as gen
                                ollama_manager = gen.OllamaManager()
                                success = ollama_manager.pull_model(new_model_name)
                                
                                # Clear install interface flag
                                if 'show_install_interface' in st.session_state:
                                    del st.session_state['show_install_interface']
                                
                                if success:
                                    st.success(f"✅ {new_model_name} installed successfully!")
                                    # Force rerun to refresh the model list
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to install {new_model_name}")
                            except Exception as e:
                                st.error(f"Error installing model: {e}")
                    else:
                        st.error("Please enter a model name")
            
            # Show popular models as suggestions
            st.subheader("💡 Popular Models")
            popular_models = [
                "llama3.2", "llama3.1", "codellama", "mistral", "phi3.5",
                "gemma2", "qwen2.5", "deepseek-coder", "nomic-embed-text"
            ]
            
            cols = st.columns(3)
            for i, popular_model in enumerate(popular_models):
                with cols[i % 3]:
                    if st.button(f"📥 {popular_model}", key=f"install_{popular_model}"):
                        with st.spinner(f"Installing {popular_model}..."):
                            try:
                                import Generation as gen
                                ollama_manager = gen.OllamaManager()
                                success = ollama_manager.pull_model(popular_model)
                                
                                # Clear install interface flag
                                if 'show_install_interface' in st.session_state:
                                    del st.session_state['show_install_interface']
                                
                                if success:
                                    st.success(f"✅ {popular_model} installed successfully!")
                                    # Force rerun to refresh the model list
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to install {popular_model}")
                            except Exception as e:
                                st.error(f"Error installing model: {e}")
            
            # Set model to first available for now
            model = available_models[0] if available_models else None
        elif actual_selection:
            model = actual_selection
        else:
            # If no model selected, use first available or default
            model = available_models[0] if available_models else None
        
        # Handle button clicks from component
        component_value = st.session_state.get('chat_model_select')
        if component_value and isinstance(component_value, dict) and component_value.get('type') == 'button_click':
            button_data = component_value.get('button_data')
            if button_data:
                action = button_data.get('action')
                option_value = button_data.get('optionValue')
                
                if action == "delete" and option_value:
                    # Use a separate session state key for delete confirmation
                    delete_key = f"confirm_delete_{option_value}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False
                    
                    if not st.session_state[delete_key]:
                        st.warning(f"Are you sure you want to delete model '{option_value}'?")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("🗑️ Yes, Delete", type="primary", key=f"btn_confirm_{option_value}"):
                                st.session_state[delete_key] = True
                                st.rerun()
                        with col2:
                            if st.button("❌ Cancel", key=f"btn_cancel_{option_value}"):
                                # Reset any confirmation states
                                for key in list(st.session_state.keys()):
                                    if key.startswith('confirm_delete_'):
                                        del st.session_state[key]
                                st.rerun()
                    else:
                        # Perform the deletion
                        with st.spinner(f"Deleting {option_value}..."):
                            try:
                                import Generation as gen
                                ollama_manager = gen.OllamaManager()
                                success = ollama_manager.delete_model(option_value)
                                
                                # Clear the confirmation state first
                                if delete_key in st.session_state:
                                    del st.session_state[delete_key]
                                
                                # Clear any other model-related session state
                                keys_to_remove = [key for key in st.session_state.keys() if key.startswith('chat_model_select')]
                                for key in keys_to_remove:
                                    if key in st.session_state:
                                        del st.session_state[key]
                                
                                if success:
                                    st.success(f"✅ {option_value} deleted successfully!")
                                    # Force rerun to refresh the model list
                                    st.rerun()
                                else:
                                    st.error(f"❌ Failed to delete {option_value}")
                                    
                            except Exception as e:
                                st.error(f"Error deleting model: {e}")
                                # Clear the confirmation state even on error
                                if delete_key in st.session_state:
                                    del st.session_state[delete_key]
                
                elif action == "browse":
                    # Set flag to show install interface
                    st.session_state['show_install_interface'] = True
                    st.rerun()
                
                elif action == "install":
                    # Set flag to show install interface
                    st.session_state['show_install_interface'] = True
                    st.rerun()
        
        # Fallback to first model if nothing selected
        if not model and available_models:
            model = available_models[0]
        
        # Save model preference when changed
        current_last_used = chat_prefs.get('last_used_model')
        if isinstance(current_last_used, dict):
            current_last_used = current_last_used.get('value') or current_last_used.get('optionValue')
        
        if _preferences_manager and model != current_last_used:
            _preferences_manager.set_preference('chat', 'last_used_model', model)
            _preferences_manager.save_preferences()
        
        # Show if this is a recommended model
        if HARDWARE_OPTIMIZER_AVAILABLE and model and model in recommended_models:
            st.success(f"✅ {model} is optimized for your hardware")
        elif HARDWARE_OPTIMIZER_AVAILABLE and model:
            st.warning(f"⚠️ {model} may not be optimal for your hardware")
        
        # Check if no model is available
        if not model:
            st.error("⚠️ No Ollama models available. Please install a model first:")
            st.code("ollama pull llama3.2", language="bash")
            st.info("Or use the 'Install New Model' option above to install a model from the Ollama registry.")
            return  # Exit early if no model available
            
    except Exception as e:
        st.error(f"Error selecting model: {e}")
        # Try to get first available model or show error
        if available_models:
            model = available_models[0]
        else:
            st.error("⚠️ No Ollama models available. Please install a model first:")
            st.code("ollama pull llama3.2", language="bash")
            return
    
    # Speech and streaming settings with preferences
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.write(f"**Selected Model:** {model}")
    with col2:
        # Load speech preference
        preferred_speech = chat_prefs.get('preferred_speech', True)
        enable_speech = st.checkbox("Enable Speech", value=preferred_speech, key="enable_speech")
        
        # Save speech preference when changed
        if _preferences_manager and enable_speech != chat_prefs.get('preferred_speech'):
            _preferences_manager.set_preference('chat', 'preferred_speech', enable_speech)
            _preferences_manager.save_preferences()
    with col3:
        # Load streaming preference
        preferred_streaming = chat_prefs.get('preferred_streaming', default_streaming)
        enable_streaming = st.checkbox("Stream Response", value=preferred_streaming, key="enable_streaming")
        
        # Save streaming preference when changed
        if _preferences_manager and enable_streaming != chat_prefs.get('preferred_streaming'):
            _preferences_manager.set_preference('chat', 'preferred_streaming', enable_streaming)
            _preferences_manager.save_preferences()
    with col4:
        # RAG toggle (enabled by default if RAG handler available)
        enable_rag = st.checkbox("Use RAG", value=True, key="enable_rag", 
                                help="Use knowledge base for enhanced responses",
                                disabled=not RAG_HANDLER_AVAILABLE)
        if enable_rag and RAG_HANDLER_AVAILABLE:
            try:
                rag_handler_temp = get_rag_handler()
                stats = rag_handler_temp.get_database_stats()
                if stats['total_documents'] > 0:
                    st.caption(f"📚 {stats['total_documents']} docs")
                else:
                    st.caption("⚠️ Empty KB")
            except:
                pass
    
    # TTS Engine selection with preferences
    if st.session_state.get("enable_speech", True):
        with st.expander("🔊 TTS Settings"):
            try:
                from offline_text2speech import tts_manager
                available_engines = tts_manager.get_available_engines()
                if available_engines:
                    # Load TTS engine preference
                    preferred_engine = chat_prefs.get('preferred_tts_engine', available_engines[0] if available_engines else None)
                    engine_index = 0
                    if preferred_engine and preferred_engine in available_engines:
                        engine_index = available_engines.index(preferred_engine)
                    

                    # Create options for TTS engine selection
                    tts_engine_options = []
                    for engine in available_engines:
                        engine_descriptions = {
                            "pyttsx3": "System default TTS engine",
                            "bark": "AI-powered neural TTS with natural voices",
                            "edge": "Microsoft Edge TTS (cloud-based)",
                            "festival": "Open-source speech synthesis system"
                        }
                        if engine == "bark":
                            option = create_option(
                            value=engine,
                            label=f"🔊 {engine.upper()}",
                            code = f"default",
                            code_color="#4CAF50"
                        )
                        else:
                            option = create_option(
                                value=engine,
                                label=f"🔊 {engine.upper()}",
                        )
                        tts_engine_options.append(option)
                    
                    # Use custom markdown select
                    default_engine = available_engines[engine_index] if available_engines else None
                    selected_engine = markdown_select(
                        options=tts_engine_options,
                        key="tts_engine_select",
                        placeholder="🔊 Choose TTS Engine...",
                        default_value=default_engine
                    )
                    
                    # Fallback to first engine if nothing selected
                    if not selected_engine and available_engines:
                        selected_engine = available_engines[0]
                    
                    # Save TTS engine preference when changed
                    if _preferences_manager and selected_engine != chat_prefs.get('preferred_tts_engine'):
                        _preferences_manager.set_preference('chat', 'preferred_tts_engine', selected_engine)
                        _preferences_manager.save_preferences()
                    
                    # Update the TTS manager's current engine
                    tts_manager.current_engine = selected_engine
                    
                    if selected_engine == "bark":
                        voice_presets = [
                            "v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2",
                            "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5",
                            "v2/en_speaker_6", "v2/en_speaker_7", "v2/en_speaker_8", "v2/en_speaker_9"
                        ]
                        
                        # Load voice preset preference
                        preferred_voice = chat_prefs.get('preferred_bark_voice', voice_presets[0])
                        voice_index = 0
                        if preferred_voice in voice_presets:
                            voice_index = voice_presets.index(preferred_voice)
                        
                        # Create options for voice preset selection
                        voice_options = []
                        for i, voice in enumerate(voice_presets):
                            option = create_option(
                                value=voice,
                                label=f"🎤 Speaker {i}",
                                code=f"Voice: {voice}",
                                language="text",
                                code_color="#E91E63"
                            )
                            voice_options.append(option)
                        
                        # Use custom markdown select
                        default_voice = voice_presets[voice_index] if voice_presets else None
                        selected_voice = markdown_select(
                            options=voice_options,
                            key="bark_voice_select",
                            placeholder="🎤 Choose Voice Preset...",
                            default_value=default_voice
                        )
                        
                        # Fallback to first voice if nothing selected
                        if not selected_voice and voice_presets:
                            selected_voice = voice_presets[0]
                        
                        # Save voice preset preference when changed
                        if _preferences_manager and selected_voice != chat_prefs.get('preferred_bark_voice'):
                            _preferences_manager.set_preference('chat', 'preferred_bark_voice', selected_voice)
                            _preferences_manager.save_preferences()
                        
                        st.session_state["bark_voice_preset"] = selected_voice
                else:
                    st.warning("No TTS engines available")
            except Exception as e:
                st.error(f"Error configuring TTS: {e}")
    
    # Print the selected model
    print(f"Ollama is using the model: {model}")
    
    # Initialize message history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                'role': 'assistant',
                'content': 'How can I help you today?'
            }
        ]
        lm.append_md_log("rerun")
    
    # Clear history button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("🗑️ Clear History", type="secondary"):
            lm.clear_json_history()
            st.session_state.clear() 
            st.success("Conversation log cleared successfully!")
            lm.append_md_log("success", "Conversation log cleared successfully!")
            if "messages" not in st.session_state:
                st.session_state["messages"] = [    
                    {
                        'role': 'assistant',
                        'content': 'How can I help you today?'
                    }
                ]
            st.rerun()
    
    with col2:
        # Check if current model supports multimodal
        if ATTACHMENT_HANDLER_AVAILABLE:
            multimodal_models = attachment_handler.get_multimodal_models()
            is_multimodal = any(mm in model.lower() for mm in [m.split(':')[0] for m in multimodal_models])
            
            if not is_multimodal:
                if st.button("🔄 Switch to Vision Model", type="primary", help="Current model doesn't support images"):
                    st.info("💡 To use attachments, install a vision model like llava:")
                    st.code("ollama pull llava", language="bash")
            else:
                st.success("✅ Vision Mode Active")

    
    # Show the conversation history
    st.subheader(f"Conversation with {model}", divider="gray")
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message['role']):
            # Check if message contains HTML details/summary tags (from think processing)
            if '<details>' in message['content'] and '<summary>' in message['content']:
                st.markdown(message['content'], unsafe_allow_html=True)
            else:
                st.markdown(message['content'])
            # Add replay button for assistant messages with audio
            if message['role'] == 'assistant' and message.get('audio_path'):
                if st.button("🔊 Replay", key=f"replay_{idx}"):
                    replay_audio(message['audio_path'])
    
    # Voice input section
    with st.expander("🎤 Voice Input"):
        col1, col2 = st.columns([3, 1])
        with col1:
            mic_sensitivity = st.slider("Microphone Sensitivity", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
        with col2:
            if st.button("🎤 Start Recording"):
                st.info("Listening... Please speak into the microphone.")
                lm.append_md_log("info", "Listening... Please speak into the microphone.")
                try:
                    prompt = speech_recognition()
                    if prompt:
                        st.success(f"You said: {prompt}")
                        process_user_input(prompt, model)
                    else:
                        st.warning("No speech detected. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    # Attachment section (if supported)
    if ATTACHMENT_HANDLER_AVAILABLE:
        with st.expander("📎 Attachments (Images & PDFs)"):
            st.info("💡 Attach images or PDFs to ask questions about them. Requires a vision model like llava.")
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload files",
                type=['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'pdf'],
                accept_multiple_files=True,
                help="Upload images or PDFs to include in your prompt"
            )
            
            # Store attachments in session state
            if "attachments" not in st.session_state:
                st.session_state["attachments"] = []
            
            # Process uploaded files
            if uploaded_files:
                new_attachments = []
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    
                    # Process attachment
                    attachment_data = attachment_handler.process_attachment(tmp_path)
                    if attachment_data:
                        new_attachments.append(attachment_data)
                        
                        # Display preview
                        col1, col2 = st.columns([1, 4])
                        with col1:
                            if attachment_data['type'] == 'image':
                                st.image(f"data:image/png;base64,{attachment_data['data']}", 
                                       width=100, caption=attachment_data['file_name'])
                            elif attachment_data['type'] == 'pdf':
                                st.write(f"📄 {attachment_data['file_name']}")
                                st.caption(f"{attachment_data.get('processed_pages', 0)} pages")
                        
                        with col2:
                            if attachment_data['type'] == 'image':
                                st.success(f"✅ {attachment_data['file_name']} ({attachment_data['size'][0]}x{attachment_data['size'][1]})")
                            elif attachment_data['type'] == 'pdf':
                                st.success(f"✅ {attachment_data['file_name']} - Text extracted")
                    else:
                        st.error(f"❌ Failed to process {uploaded_file.name}")
                    
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                
                # Update session state
                st.session_state["attachments"].extend(new_attachments)
            
            # Show current attachments
            if st.session_state["attachments"]:
                st.write(f"**Current Attachments:** {len(st.session_state['attachments'])}")
                cols = st.columns(min(len(st.session_state["attachments"]), 4))
                for i, att in enumerate(st.session_state["attachments"]):
                    with cols[i % 4]:
                        if att['type'] == 'image':
                            st.image(f"data:image/png;base64,{att['data']}", 
                                   width=100, caption=att['file_name'])
                        else:
                            st.write(f"📄 {att['file_name']}")
                
                if st.button("🗑️ Clear Attachments"):
                    st.session_state["attachments"] = []
                    st.rerun()
    
    # Text input
    prompt = st.chat_input("Type your message here...")
    
    if prompt:
        # Debug logging
        print(f"\n🔍 DEBUG: New prompt received: '{prompt}'")
        print(f"🔍 DEBUG: Agentic mode in session_state: {st.session_state.get('agentic_mode', 'NOT SET')}")
        print(f"🔍 DEBUG: Vision mode in session_state: {st.session_state.get('vision_mode', 'NOT SET')}")
        print(f"🔍 DEBUG: AGENTIC_HANDLER_AVAILABLE: {AGENTIC_HANDLER_AVAILABLE}")
        print(f"🔍 DEBUG: VISION_AGENT_AVAILABLE: {VISION_AGENT_AVAILABLE}")
        
        # Get attachments if available
        attachments = st.session_state.get("attachments", [])
        
        # Check which mode to use
        if st.session_state.get("vision_mode") and VISION_AGENT_AVAILABLE:
            # Use vision agent for autonomous task execution
            print(f"🔍 DEBUG: Routing to VISION AGENT")
            process_vision_input(prompt, model)
        elif st.session_state.get("agentic_mode") and AGENTIC_HANDLER_AVAILABLE:
            # Use agentic handler for basic desktop control
            print(f"🔍 DEBUG: Routing to BASIC AGENTIC handler")
            process_agentic_input(prompt, model, attachments)
        else:
            # Use normal chat processing
            print(f"🔍 DEBUG: Routing to NORMAL chat handler")
            process_user_input(prompt, model, attachments)
        
        # Clear attachments after sending
        if attachments:
            st.session_state["attachments"] = []

def remove_think_tags(text: str) -> str:
    """Remove <think> tags from text for TTS generation"""
    # Remove think tags and their content, keeping only the visible response
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Also remove any HTML tags that might be left from think processing
    # This removes <details> and <summary> tags but keeps their content
    clean_text = re.sub(r'<details[^>]*>', '', clean_text)
    clean_text = re.sub(r'</details>', '', clean_text)
    clean_text = re.sub(r'<summary[^>]*>.*?</summary>', '', clean_text, flags=re.DOTALL)
    
    # Clean up extra whitespace
    clean_text = re.sub(r'\n\s*\n', '\n', clean_text)
    clean_text = clean_text.strip()
    
    return clean_text

def process_user_input(prompt, model, attachments=None):
    """Process user input and generate response
    
    Args:
        prompt: Text prompt from user
        model: Model to use for generation
        attachments: Optional list of attachment dictionaries
    """
    # Format attachments for display (use list for efficient concatenation)
    display_parts = [prompt]
    if attachments and len(attachments) > 0:
        display_parts.append(f"\n\n📎 **Attachments:** {len(attachments)} file(s)")
        for att in attachments:
            if att['type'] == 'image':
                display_parts.append(f"\n- 🖼️ {att['file_name']}")
            elif att['type'] == 'pdf':
                display_parts.append(f"\n- 📄 {att['file_name']}")
    
    display_content = "".join(display_parts)
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": display_content, "attachments": attachments})
    
    with st.chat_message("user"):
        st.markdown(display_content)
        # Show attachment previews
        if attachments:
            cols = st.columns(min(len(attachments), 4))
            for i, att in enumerate(attachments):
                with cols[i % 4]:
                    if att['type'] == 'image':
                        st.image(f"data:image/png;base64,{att['data']}", 
                               width=150, caption=att['file_name'])
                    elif att['type'] == 'pdf':
                        st.caption(f"📄 {att['file_name']}")
    
    lm.append_md_log("user", display_content)
    
    # Prepare images for Ollama (if attachments present and handler available)
    images = None
    if attachments and ATTACHMENT_HANDLER_AVAILABLE:
        formatted_data = attachment_handler.format_for_ollama(prompt, attachments)
        prompt = formatted_data['prompt']  # Use enhanced prompt with PDF context
        if formatted_data.get('has_images'):
            images = formatted_data.get('images')
    
    # Check if prompt should be handled by prompt_handler (weather, news, web search)
    special_response = None
    if PROMPT_HANDLER_AVAILABLE:
        try:
            special_response = prompt_handler.handle_query(prompt)
        except Exception as e:
            logger.warning(f"Prompt handler failed: {e}")
            special_response = None
    
    # PRIORITY 1: Check if RAG should be used (if enabled and knowledge base has content)
    # RAG gets priority over special responses for better accuracy
    rag_context = None
    rag_answer = None
    use_rag = st.session_state.get("enable_rag", True)  # RAG enabled by default
    
    if use_rag and RAG_HANDLER_AVAILABLE:
        try:
            rag_handler = get_rag_handler()
            # Check if there's content in the knowledge base
            stats = rag_handler.get_database_stats()
            
            if stats['total_documents'] > 0:
                # Search for relevant context with lower threshold for better recall
                search_results = rag_handler.db.search(prompt, top_k=5, min_score=0.1)
                
                if search_results:
                    # High relevance - use RAG to generate complete answer
                    if search_results[0]['score'] >= 0.3:
                        logger.info(f"✅ RAG: Found highly relevant content (score: {search_results[0]['score']:.2f})")
                        
                        # Query RAG system for complete answer
                        try:
                            rag_result = rag_handler.query(prompt, context_window=min(len(search_results), 3))
                            
                            if isinstance(rag_result, dict):
                                rag_answer = rag_result.get('answer', '')
                                sources = rag_result.get('sources', [])
                                
                                # Store sources separately for dropdown display
                                if sources:
                                    # Don't add citations to answer text, we'll show them in dropdown
                                    st.session_state['rag_sources'] = sources
                                    logger.info(f"✅ RAG: Generated answer from {len(sources)} sources")
                            else:
                                rag_answer = str(rag_result)
                                st.session_state['rag_sources'] = []
                                
                        except Exception as e:
                            logger.warning(f"⚠️ RAG query failed, falling back to context enhancement: {e}")
                    
                    # Medium/Low relevance - enhance prompt with context
                    if not rag_answer and search_results:
                        rag_context_parts = []
                        for idx, result in enumerate(search_results[:3]):
                            source = result['metadata'].get('source', 'unknown')
                            score = result['score']
                            rag_context_parts.append(
                                f"[Context {idx+1}] (from {source}, relevance: {score:.0%})\n{result['content']}"
                            )
                        
                        rag_context = "\n\n".join(rag_context_parts)
                        
                        # Enhance the prompt with RAG context for better Ollama response
                        enhanced_prompt = f"""You have been provided with relevant information from a knowledge base below. Use this information to answer the user's question naturally and conversationally.

KNOWLEDGE BASE INFORMATION:
{rag_context}

---
USER QUESTION: {prompt}

CRITICAL INSTRUCTIONS:
1. Answer naturally as if you know this information directly
2. DO NOT use phrases like "according to the context", "based on the provided information", "the context mentions", etc.
3. Be conversational and confident
4. State facts directly from the knowledge base as if they are established facts
5. Only mention if information is incomplete when truly necessary
6. Provide a complete, helpful answer

Answer the user's question now:"""
                        
                        prompt = enhanced_prompt
                        logger.info(f"✅ RAG: Enhanced prompt with {len(search_results)} context chunks")
                else:
                    logger.info("ℹ️ RAG: No relevant content found in knowledge base")
        except Exception as e:
            logger.warning(f"⚠️ RAG processing failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue without RAG if it fails
    
    # Generate assistant response
    with st.chat_message("assistant"):
        if prompt == "test":
            # Test mode
            lm.append_md_log("test", "started test mode")
            progress = st.progress(0.0, text="Generating response...")
            for percent in range(1, 51):
                time.sleep(0.01)
                progress_value = min((percent + 1) / 100.0, 1.0)
                progress.progress(progress_value, text="Generating response...")
            response = "Hi! This is a test message from Aurora."
            progress.progress(1.0, text="Response generated!")
            st.markdown(response)
            progress.empty()
        elif rag_answer:
            # PRIORITY: Use RAG-generated answer (most accurate)
            response = rag_answer
            # Add RAG indicator badge
            st.info("📚 Answer from Knowledge Base")
            st.markdown(response, unsafe_allow_html=True)
            
            # Show sources in a dropdown/expander
            if st.session_state.get('rag_sources'):
                with st.expander("📖 View Sources", expanded=False):
                    sources = st.session_state.get('rag_sources', [])
                    st.markdown("**Sources used to generate this answer:**")
                    st.markdown("---")
                    
                    for idx, source in enumerate(sources[:5]):  # Show top 5 sources
                        doc_id = source.get('doc_id', 'Unknown')
                        score = source.get('score', 0)
                        metadata = source.get('metadata', {})
                        
                        # Create a nice display for each source
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            source_name = metadata.get('source', 'Unknown Source')
                            category = metadata.get('category', '')
                            st.markdown(f"**{idx+1}. {source_name}**")
                            if category:
                                st.caption(f"Category: {category}")
                        with col2:
                            # Show relevance score
                            st.metric("Relevance", f"{score:.0%}", label_visibility="collapsed")
                        
                        # Show doc ID in small text
                        st.caption(f"Document ID: `{doc_id[:30]}...`")
                        
                        if idx < len(sources) - 1:
                            st.markdown("---")
                
                # Clear sources after displaying
                st.session_state['rag_sources'] = []
            
            logger.info("✅ Used RAG-generated answer")
        elif special_response:
            # Special query handled by prompt_handler (weather, news, web search)
            response = special_response
            st.markdown(response, unsafe_allow_html=True)
        else:
            # Use Ollama for generation (may include RAG context in prompt)
            if rag_context:
                # Indicate that knowledge base context is being used
                st.caption("📚 Using knowledge base context")
            
            # Check if streaming is enabled
            if st.session_state.get("enable_streaming", True):
                # Streaming mode
                response_placeholder = st.empty()
                status_placeholder = st.empty()
                full_response = ""
                
                # Show typing indicator
                status_placeholder.info("🤖 Thinking...")
                
                # Stream the response
                try:
                    for chunk in gen.ollama_manager.chat_with_memory_stream(prompt, model, images=images):
                        if chunk:  # Only process non-empty chunks
                            full_response += chunk
                            
                            # Update status to show streaming
                            status_placeholder.info("🤖 Generating response...")
                            
                            # Check if we have think tags to process
                            if '<think>' in full_response and '</think>' in full_response:
                                # Process think tags for display
                                processed_response = gen.ollama_manager._process_think_tags(full_response)
                                response_placeholder.markdown(processed_response, unsafe_allow_html=True)
                            else:
                                response_placeholder.markdown(full_response)
                    
                    # Clear status indicator when done
                    status_placeholder.empty()
                    
                    # Final response for logging
                    response = full_response
                    
                except Exception as e:
                    status_placeholder.error(f"Error during streaming: {e}")
                    response = f"Error: {e}"
                    response_placeholder.markdown(response)
            else:
                # Non-streaming mode (original behavior)
                with st.spinner("Generating response..."):
                    response = gen.ollama_manager.chat_with_memory(prompt, model, images=images)
                # Check if response contains HTML details/summary tags (from think processing)
                if '<details>' in response and '<summary>' in response:
                    st.markdown(response, unsafe_allow_html=True)
                else:
                    st.markdown(response)
        
        lm.append_md_log("assistant", response)
        
        # Generate speech only if enabled (with error handling for Streamlit threading issues)
        audio_path = None
        if st.session_state.get("enable_speech", True):
            try:
                # Try to generate speech, but don't let it block the UI
                try:
                    # Use selected voice preset if available
                    voice_preset = st.session_state.get("bark_voice_preset", "v2/en_speaker_0")
                    # Remove think tags from response before TTS
                    clean_response = remove_think_tags(response)
                    
                    # Only generate speech if there's actual content after removing think tags
                    if clean_response.strip():
                        audio_path = speak_text(clean_response, voice_preset=voice_preset, return_path=True)
                    else:
                        print("No speech content after removing think tags")
                except Exception as speech_error:
                    # Log the speech error but don't show it to user unless verbose
                    print(f"Speech generation failed: {speech_error}")
                    # You can uncomment the line below for debugging
                    # st.warning(f"Speech generation failed: {speech_error}")
            except Exception as e:
                print(f"Error in speech generation wrapper: {e}")
        
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'audio_path': audio_path
        })

def process_agentic_input(prompt, model, attachments=None):
    """Process user input with agentic capabilities for desktop control
    
    Args:
        prompt: Text prompt from user
        model: Model to use for generation
        attachments: Optional list of attachment dictionaries
    """
    # Debug logging
    print(f"🔍 DEBUG: Agentic mode processing request: '{prompt}'")
    print(f"🔍 DEBUG: Model: {model}")
    print(f"🔍 DEBUG: Agentic handler available: {AGENTIC_HANDLER_AVAILABLE}")
    
    # Format attachments for display (use list for efficient concatenation)
    display_parts = [prompt]
    if attachments and len(attachments) > 0:
        display_parts.append(f"\n\n📎 **Attachments:** {len(attachments)} file(s)")
        for att in attachments:
            if att['type'] == 'image':
                display_parts.append(f"\n- 🖼️ {att['file_name']}")
            elif att['type'] == 'pdf':
                display_parts.append(f"\n- 📄 {att['file_name']}")
    
    display_content = "".join(display_parts)
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": display_content, "attachments": attachments})
    
    with st.chat_message("user"):
        st.markdown(display_content)
        # Show attachment previews
        if attachments:
            cols = st.columns(min(len(attachments), 4))
            for i, att in enumerate(attachments):
                with cols[i % 4]:
                    if att['type'] == 'image':
                        st.image(f"data:image/png;base64,{att['data']}", 
                               width=150, caption=att['file_name'])
                    elif att['type'] == 'pdf':
                        st.caption(f"📄 {att['file_name']}")
    
    lm.append_md_log("user", display_content)
    
    # Generate assistant response with agentic capabilities
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        try:
            # Show thinking status
            status_placeholder.info("🤖 Analyzing your request with agentic AI...")
            
            # Call agentic handler
            print(f"🔍 DEBUG: Calling handle_agentic_request with prompt: '{prompt}', model: '{model}'")
            response = handle_agentic_request(prompt, model)
            print(f"🔍 DEBUG: Agentic response: {response[:200]}...")  # First 200 chars
            
            # Clear status and show response
            status_placeholder.empty()
            response_placeholder.markdown(response, unsafe_allow_html=True)
            
        except Exception as e:
            # Show error
            status_placeholder.empty()
            error_msg = f"❌ Error in agentic processing: {str(e)}"
            print(f"🔍 DEBUG: {error_msg}")
            response_placeholder.error(error_msg)
            response = error_msg
        
        lm.append_md_log("assistant", response)
        
        # Generate speech only if enabled
        audio_path = None
        if st.session_state.get("enable_speech", True):
            try:
                voice_preset = st.session_state.get("bark_voice_preset", "v2/en_speaker_0")
                clean_response = remove_think_tags(response)
                
                if clean_response.strip():
                    audio_path = speak_text(clean_response, voice_preset=voice_preset, return_path=True)
            except Exception as e:
                print(f"Speech generation failed: {e}")
        
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'audio_path': audio_path
        })


def process_vision_input(prompt, model):
    """Process user input with vision agent for autonomous task execution
    
    Args:
        prompt: Text prompt from user describing the task
        model: Model to use for vision analysis
    """
    import time
    import traceback
    
    print(f"🔍 DEBUG: Vision agent processing request: '{prompt}'")
    print(f"🔍 DEBUG: Original model: {model}")
    
    # Use qwen3-vl:235b-cloud for vision tasks (better vision capabilities)
    vision_model = "qwen3-vl:235b-cloud"
    print(f"🔍 DEBUG: Using vision model: {vision_model}")
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    lm.append_md_log("user", prompt)
    
    # Generate assistant response with vision agent
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        try:
            # Show initial status
            status_placeholder.info("👁️ Vision Agent activating...")
            time.sleep(0.5)
            
            status_placeholder.info("📋 Creating roadmap...")
            time.sleep(0.5)
            
            status_placeholder.success("🚀 Executing autonomous task with vision feedback...")
            
            # Show progress
            progress_bar = progress_placeholder.progress(0)
            
            # Execute task with vision agent
            print(f"🔍 DEBUG: Calling execute_autonomous_task")
            
            # Execute in a way that shows progress
            result_summary = execute_autonomous_task(prompt, vision_model)
            
            progress_bar.progress(100)
            
            # Clear status and progress
            status_placeholder.empty()
            progress_placeholder.empty()
            
            # Show result
            response_placeholder.markdown(result_summary, unsafe_allow_html=True)
            
            # Add link to screenshots
            screenshots_dir = vision_agent.screenshots_dir
            if screenshots_dir.exists():
                st.info(f"📸 Screenshots and execution log saved to: `{screenshots_dir}`")
            
            response = result_summary
            
        except Exception as e:
            # Show error
            status_placeholder.empty()
            progress_placeholder.empty()
            error_msg = f"❌ Error in vision agent execution: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
            print(f"🔍 DEBUG: {error_msg}")
            response_placeholder.error(error_msg)
            response = error_msg
        
        lm.append_md_log("assistant", response)
        
        # Generate speech only if enabled
        audio_path = None
        if st.session_state.get("enable_speech", True):
            try:
                voice_preset = st.session_state.get("bark_voice_preset", "v2/en_speaker_0")
                clean_response = remove_think_tags(response)
                
                if clean_response.strip():
                    audio_path = speak_text(clean_response, voice_preset=voice_preset, return_path=True)
            except Exception as e:
                print(f"Speech generation failed: {e}")
        
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'audio_path': audio_path
        })


def image_generation_page():
    """Image generation interface"""
    # Import and run image generation module
    try:
        import image_gen
        image_gen.main()
    except ImportError as e:
        st.error(f"Image generation module not available: {e}")
        st.info("Please ensure all image generation dependencies are installed.")
        st.markdown("""
        ### Required dependencies:
        ```bash
        pip install torch torchvision diffusers transformers accelerate
        ```
        """)
        st.markdown("You can also try running the image generation separately with:")
        st.code("streamlit run image_gen.py")
    except Exception as e:
        st.error(f"Error running image generation: {e}")
        st.error(f"Error type: {type(e).__name__}")
        
        # Show detailed error for debugging
        import traceback
        with st.expander("🔍 Show full error details"):
            st.code(traceback.format_exc())
        
        st.info("Try the following solutions:")
        st.markdown("""
        1. Restart the Streamlit app
        2. Check if all dependencies are installed
        3. Run the image generation module directly: `streamlit run image_gen.py`
        4. Check the logs for more information
        """)

def video_generation_page():
    """Video generation interface"""
    st.title("🎬 AI Video Generator")
    st.markdown("Generate videos using state-of-the-art AI models")
    
    # Check for video generation dependencies
    missing_deps = []
    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        from diffusers import TextToVideoSDPipeline
    except ImportError:
        missing_deps.append("diffusers[video]")
    
    if missing_deps:
        st.error("⚠️ Missing required dependencies for video generation:")
        for dep in missing_deps:
            st.error(f"- {dep}")
        
        st.info("Please install the required dependencies:")
        st.code(f"pip install {' '.join(missing_deps)}")
        st.stop()
    
    # Video models - use video model manager if available
    if VIDEO_MODEL_MANAGER_AVAILABLE:
        video_models = video_model_manager.get_available_models()
        installed_models = video_model_manager.get_installed_models()
    else:
        video_models = {
            "Zeroscope v2 576w": "cerspense/zeroscope_v2_576w",
            "ModelScope T2V": "damo-vilab/text-to-video-ms-1.7b",
            "Text-to-Video Zero": "text-to-video-zero"
        }
        installed_models = []
    
    # Sidebar information
    with st.sidebar:
        st.header("🎬 Video Models")
        
        # Show model management status
        if VIDEO_MODEL_MANAGER_AVAILABLE:
            st.markdown(f"**📊 Model Status:**")
            st.write(f"- **Total Models**: {len(video_models)}")
            st.write(f"- **Installed**: {len(installed_models)}")
            st.write(f"- **Not Installed**: {len(video_models) - len(installed_models)}")
            
            # Show installed models
            if installed_models:
                st.markdown("**✅ Installed Models:**")
                for model in installed_models:
                    model_size = video_model_manager.get_model_size(model)
                    size_text = f" ({model_size})" if model_size else ""
                    st.write(f"- {model}{size_text}")
            
            # Show not installed models
            not_installed = [model for model in video_models.keys() if model not in installed_models]
            if not_installed:
                st.markdown("**⬇️ Available to Install:**")
                for model in not_installed:
                    st.write(f"- {model}")
        else:
            st.markdown("""
            **Default Models:**
            - **Zeroscope v2**: High quality, 576x320 resolution
            - **ModelScope T2V**: Good balance of quality and speed
            - **Text-to-Video Zero**: Experimental, image-based
            """)
            st.warning("⚠️ Video model manager not available. Install/delete features disabled.")
        
        st.header("💡 Video Tips")
        st.markdown("""
        - Keep prompts simple and clear
        - Video generation takes much longer than images
        - Lower resolutions generate faster
        - Shorter videos (2-4 seconds) work best
        - Be patient - first generation downloads models
        """)
        
        # System requirements
        st.header("⚙️ Requirements")
        if torch and torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
            allocated = torch.cuda.memory_allocated(0) // 1024**3
            cached = torch.cuda.memory_reserved(0) // 1024**3
            free = vram - cached
            
            st.write(f"**GPU**: {torch.cuda.get_device_name(0)}")
            st.write(f"**Total VRAM**: {vram} GB")
            st.write(f"**Used**: {allocated} GB")
            st.write(f"**Cached**: {cached} GB")
            st.write(f"**Free**: {free} GB")
            
            if vram >= 12:
                st.success("✅ Excellent GPU for video generation")
            elif vram >= 8:
                st.info("ℹ️ Good GPU - use lower settings")
            elif vram >= 6:
                st.warning("⚠️ Limited VRAM - use very low settings")
            else:
                st.error("❌ Insufficient VRAM for video generation")
                
            # Memory management buttons
            st.markdown("**🧹 Memory Management:**")
            if st.button("🗑️ Clear GPU Cache", key="clear_gpu_cache"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    st.success("GPU cache cleared!")
                    st.rerun()
            
            if st.button("🔄 Reset All Models", key="reset_models"):
                # Clear all video models from session state
                keys_to_remove = [key for key in st.session_state.keys() if key.startswith('video_model_')]
                for key in keys_to_remove:
                    del st.session_state[key]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                st.success("All models reset!")
                st.rerun()
                
        # Model management buttons
        if VIDEO_MODEL_MANAGER_AVAILABLE:
            st.header("🔧 Model Management")
            
            # Bulk install popular models
            if st.button("📦 Install Popular Models", key="bulk_install"):
                popular_models = ["Zeroscope v2 576w", "ModelScope T2V"]
                installed_count = 0
                failed_count = 0
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, model in enumerate(popular_models):
                    if model not in installed_models:
                        status_text.text(f"Installing {model}...")
                        progress_bar.progress((i + 1) / len(popular_models))
                        
                        success = video_model_manager.install_model(model)
                        if success:
                            installed_count += 1
                        else:
                            failed_count += 1
                
                progress_bar.empty()
                status_text.empty()
                
                if installed_count > 0:
                    st.success(f"✅ Installed {installed_count} models successfully!")
                if failed_count > 0:
                    st.error(f"❌ Failed to install {failed_count} models")
                
                if installed_count > 0:
                    st.rerun()
            
            # Bulk delete all models
            if installed_models and st.button("🗑️ Delete All Models", key="bulk_delete"):
                if st.session_state.get('confirm_bulk_delete', False):
                    deleted_count = 0
                    failed_count = 0
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, model in enumerate(installed_models):
                        status_text.text(f"Deleting {model}...")
                        progress_bar.progress((i + 1) / len(installed_models))
                        
                        success = video_model_manager.delete_model(model)
                        if success:
                            deleted_count += 1
                        else:
                            failed_count += 1
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    if deleted_count > 0:
                        st.success(f"✅ Deleted {deleted_count} models successfully!")
                    if failed_count > 0:
                        st.error(f"❌ Failed to delete {failed_count} models")
                    
                    st.session_state['confirm_bulk_delete'] = False
                    if deleted_count > 0:
                        st.rerun()
                else:
                    st.warning("⚠️ This will delete ALL installed video models!")
                    st.session_state['confirm_bulk_delete'] = True
                    st.rerun()
        else:
            st.error("❌ GPU strongly recommended for video generation")
            if not torch:
                st.error("PyTorch not available")
            
        # Quick troubleshooting
        st.header("🛠️ Troubleshooting")
        if st.button("🔧 Run GPU Test", key="gpu_test"):
            if torch and torch.cuda.is_available():
                try:
                    # Simple GPU test
                    test_tensor = torch.randn(100, 100, device='cuda')
                    result = torch.matmul(test_tensor, test_tensor)
                    del test_tensor, result
                    torch.cuda.synchronize()
                    st.success("✅ GPU test passed!")
                except Exception as e:
                    st.error(f"❌ GPU test failed: {e}")
            else:
                st.error("❌ No CUDA GPU available")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model selection with preferences
        video_prefs = {}
        if _preferences_manager:
            video_prefs = _preferences_manager.get_category_preferences('video_generation')
        
        # Load preferred model
        preferred_model = video_prefs.get('preferred_model') or video_prefs.get('last_used_model')
        
        # Handle case where preferred_model might be a dictionary (from markdown_select component)
        if isinstance(preferred_model, dict):
            preferred_model = preferred_model.get('value') or preferred_model.get('optionValue')
        
        model_index = 0
        if preferred_model and isinstance(preferred_model, str) and preferred_model in video_models.keys():
            model_index = list(video_models.keys()).index(preferred_model)
        
        # Create options for video model selection
        video_model_options = []
        default_model = list(video_models.keys())[model_index] if video_models else None
        
        for model_name, model_id in video_models.items():
            # Add model descriptions and recommendations
            descriptions = {
                "Zeroscope v2 576w": "High quality video generation at 576x320 resolution",
                "ModelScope T2V": "Balanced quality and speed for general use",
                "Text-to-Video Zero": "Experimental model with image-based generation",
                "Stable Video Diffusion": "Stability AI's video generation model",
                "VideoCrafter2": "High-quality video generation with temporal consistency",
                "LaVie": "Text-to-video generation with natural language understanding",
                "Show-1": "ShowLab's text-to-video generation model",
                "CogVideoX": "Tsinghua's cognitive video generation model"
            }
            
            is_default = model_name == default_model
            is_installed = VIDEO_MODEL_MANAGER_AVAILABLE and model_name in installed_models
            
            # Create buttons based on installation status
            buttons = []
            if is_installed:
                buttons.append({"text": "🗑️ Delete", "variant": "danger", "action": "delete"})
            else:
                buttons.append({"text": "📥 Install", "variant": "primary", "action": "install"})
            
            if is_default:
                # Add green indicator for default model
                option = create_option(
                    value=model_name,
                    label=f"🎬 {model_name}",
                    code=f"{'✅ Installed' if is_installed else '⬇️ Not installed'} - {descriptions.get(model_name, 'Video generation model')}",
                    code_color="#00cc88" if is_installed else "#ff9800",
                    buttons=buttons
                )
            else:
                option = create_option(
                    value=model_name,
                    label=f"🎬 {model_name}",
                    code=f"{'✅ Installed' if is_installed else '⬇️ Not installed'} - {descriptions.get(model_name, 'Video generation model')}",
                    code_color="#00cc88" if is_installed else "#ff9800",
                    buttons=buttons
                )
            video_model_options.append(option)
        
        # Add option to add custom model
        if VIDEO_MODEL_MANAGER_AVAILABLE:
            custom_option = create_option(
                value="__add_custom__",
                label="➕ Add Custom Model",
                code="Add model from Hugging Face",
                language="info",
                code_color="#9c27b0",
                buttons=[
                    {"text": "🔍 Browse", "variant": "primary", "action": "browse"},
                    {"text": "➕ Add", "variant": "secondary", "action": "add"}
                ]
            )
            video_model_options.append(custom_option)
        
        # Use custom markdown select
        default_model = list(video_models.keys())[model_index] if video_models else None
        selected_model_name = markdown_select(
            options=video_model_options,
            key="video_model_select",
            placeholder="🎬 Choose Video Model...",
            default_value=default_model
        )
        
        # Handle button clicks from component
        component_value = st.session_state.get('video_model_select')
        if component_value and isinstance(component_value, dict) and component_value.get('type') == 'button_click':
            button_data = component_value.get('button_data')
            if button_data:
                action = button_data.get('action')
                option_value = button_data.get('optionValue')
                
                if action == "delete" and option_value and VIDEO_MODEL_MANAGER_AVAILABLE:
                    # Use a separate session state key for delete confirmation
                    delete_key = f"confirm_delete_video_{option_value}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False
                    
                    if not st.session_state[delete_key]:
                        st.warning(f"Are you sure you want to delete model '{option_value}'?")
                        # Show model size if available
                        model_size = video_model_manager.get_model_size(option_value)
                        if model_size:
                            st.info(f"Model size: {model_size}")
                        
                        col1_del, col2_del = st.columns([1, 1])
                        with col1_del:
                            if st.button("🗑️ Yes, Delete", type="primary", key=f"btn_confirm_video_{option_value}"):
                                st.session_state[delete_key] = True
                                st.rerun()
                        with col2_del:
                            if st.button("❌ Cancel", key=f"btn_cancel_video_{option_value}"):
                                # Reset any confirmation states
                                for key in list(st.session_state.keys()):
                                    if key.startswith('confirm_delete_video_'):
                                        del st.session_state[key]
                                st.rerun()
                    else:
                        # Perform the deletion
                        with st.spinner(f"Deleting {option_value}..."):
                            success = video_model_manager.delete_model(option_value)
                            
                            # Clear the confirmation state first
                            if delete_key in st.session_state:
                                del st.session_state[delete_key]
                            
                            # Clear any other model-related session state
                            keys_to_remove = [key for key in st.session_state.keys() if key.startswith('video_model_select')]
                            for key in keys_to_remove:
                                if key in st.session_state:
                                    del st.session_state[key]
                            
                            # Clear pipeline cache if current model was deleted
                            if option_value == selected_model_name:
                                video_model_key = f"video_model_{option_value}"
                                if video_model_key in st.session_state:
                                    del st.session_state[video_model_key]
                            
                            if success:
                                st.success(f"✅ {option_value} deleted successfully!")
                                # Force rerun to refresh the model list
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to delete {option_value}")
                
                elif action == "install" and option_value and VIDEO_MODEL_MANAGER_AVAILABLE:
                    with st.spinner(f"Installing {option_value}..."):
                        success = video_model_manager.install_model(option_value)
                        
                        # Clear any model-related session state
                        keys_to_remove = [key for key in st.session_state.keys() if key.startswith('video_model_select')]
                        for key in keys_to_remove:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        if success:
                            st.success(f"✅ {option_value} installed successfully!")
                            # Force rerun to refresh the model list
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to install {option_value}")
                
                elif action == "browse":
                    # Set flag to show custom model interface
                    st.session_state['show_custom_video_model_interface'] = True
                    st.rerun()
                
                elif action == "add":
                    # Set flag to show custom model interface
                    st.session_state['show_custom_video_model_interface'] = True
                    st.rerun()
        
        # Check if we should show custom model interface
        if st.session_state.get('show_custom_video_model_interface'):
            st.subheader("➕ Add Custom Video Model")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                custom_model_name = st.text_input(
                    "Model Name",
                    placeholder="e.g., My Custom Video Model",
                    help="Enter a friendly name for the model"
                )
                custom_model_id = st.text_input(
                    "Hugging Face Model ID",
                    placeholder="e.g., user/model-name",
                    help="Enter the model ID from Hugging Face"
                )
            with col2:
                if st.button("➕ Add Model", type="primary"):
                    if custom_model_name and custom_model_id:
                        with st.spinner(f"Adding {custom_model_name}..."):
                            success = video_model_manager.add_custom_model(custom_model_name, custom_model_id)
                            
                            # Clear custom model interface flag
                            if 'show_custom_video_model_interface' in st.session_state:
                                del st.session_state['show_custom_video_model_interface']
                            
                            if success:
                                st.success(f"✅ {custom_model_name} added successfully!")
                                # Update video_models for immediate use
                                video_models[custom_model_name] = custom_model_id
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to add {custom_model_name}")
                    else:
                        st.error("Please enter both model name and Hugging Face model ID")
                
                if st.button("❌ Cancel"):
                    if 'show_custom_video_model_interface' in st.session_state:
                        del st.session_state['show_custom_video_model_interface']
                    st.rerun()
            
            # Show popular video models as suggestions
            st.subheader("💡 Popular Video Models")
            popular_models = {
                "AnimateDiff": "guoyww/animatediff-motion-adapter-v1-5-2",
                "I2VGen-XL": "ali-vilab/i2vgen-xl",
                "VideoCrafter1": "VideoCrafter/VideoCrafter1",
                "PIA": "PIA-diffusion/PIA",
                "DynamiCrafter": "Doubiiu/DynamiCrafter_512"
            }
            
            cols = st.columns(3)
            for i, (model_name, model_id) in enumerate(popular_models.items()):
                with cols[i % 3]:
                    if st.button(f"➕ {model_name}", key=f"add_popular_{model_name}"):
                        with st.spinner(f"Adding {model_name}..."):
                            success = video_model_manager.add_custom_model(model_name, model_id)
                            
                            # Clear custom model interface flag
                            if 'show_custom_video_model_interface' in st.session_state:
                                del st.session_state['show_custom_video_model_interface']
                            
                            if success:
                                st.success(f"✅ {model_name} added successfully!")
                                # Update video_models for immediate use
                                video_models[model_name] = model_id
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to add {model_name}")
            
            # Don't show the rest of the interface when adding custom model
            st.stop()
        
        # Check if selected model is custom add option
        if selected_model_name == "__add_custom__":
            st.session_state['show_custom_video_model_interface'] = True
            st.rerun()
        
        # Fallback to first model if nothing selected
        if not selected_model_name and video_models:
            selected_model_name = list(video_models.keys())[0]
        
        # Save model preference when changed
        current_last_used = video_prefs.get('last_used_model')
        if isinstance(current_last_used, dict):
            current_last_used = current_last_used.get('value') or current_last_used.get('optionValue')
        
        if _preferences_manager and selected_model_name != current_last_used:
            _preferences_manager.set_preference('video_generation', 'last_used_model', selected_model_name)
            _preferences_manager.save_preferences()
        
        # Prompt input
        # Use session state to persist the prompt value
        if "video_prompt" not in st.session_state:
            st.session_state["video_prompt"] = ""

        prompt = st.text_area(
            "✍️ Describe your video:",
            value=st.session_state["video_prompt"],
            placeholder="A cat playing with a ball of yarn in slow motion...",
            height=100,
            help="Keep it simple and descriptive for best results",
            key="video_prompt_text_area"
        )

        # Quick prompts
        st.markdown("**🎬 Quick Prompts:**")
        prompt_col1, prompt_col2 = st.columns(2)
        with prompt_col1:
            if st.button("🌊 Ocean Waves"):
                st.session_state["video_prompt"] = "Ocean waves crashing on a sandy beach at sunset"
                st.rerun()
            if st.button("🔥 Campfire"):
                st.session_state["video_prompt"] = "A cozy campfire with dancing flames at night"
                st.rerun()
        with prompt_col2:
            if st.button("🌸 Flower Bloom"):
                st.session_state["video_prompt"] = "A flower blooming in time-lapse photography"
                st.rerun()
            if st.button("☁️ Clouds"):
                st.session_state["video_prompt"] = "Fluffy white clouds moving across a blue sky"
                st.rerun()
    
    with col2:
        st.markdown("### ⚙️ Video Settings")
        
        # Use hardware-optimized settings
        if HARDWARE_OPTIMIZER_AVAILABLE:
            video_settings = get_video_settings()
            default_frames = video_settings.get('recommended_frames', 16)
            max_frames = video_settings.get('max_frames', 32)
            default_steps = video_settings.get('inference_steps', 15)
            default_fps = video_settings.get('recommended_fps', 8)
            max_fps = video_settings.get('max_fps', 24)
            default_guidance = video_settings.get('guidance_scale', 9.0)
            video_resolution = video_settings.get('recommended_resolution', [320, 576])
            max_video_resolution = video_settings.get('max_resolution', [576, 768])
            
            # Show optimization info
            hardware_info = get_hardware_info()
            vram_gb = hardware_info.get('gpu', {}).get('total_vram_gb', 0)
            if vram_gb >= 16:
                st.success("🚀 Excellent GPU - High quality video generation")
            elif vram_gb >= 12:
                st.info("✅ Great GPU - Good quality video generation")
            elif vram_gb >= 8:
                st.info("👍 Good GPU - Standard quality video generation")
            else:
                st.warning("⚠️ Limited GPU - Basic video generation")
        else:
            # Fallback: Recommended settings based on GPU memory detection
            if torch and torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
                if vram >= 12:
                    default_frames = 24
                    max_frames = 64
                    default_steps = 20
                    default_fps = 10
                    max_fps = 30
                    default_guidance = 9.0
                    video_resolution = [576, 1024]
                    max_video_resolution = [1024, 1024]
                elif vram >= 8:
                    default_frames = 16
                    max_frames = 32
                    default_steps = 15
                    default_fps = 8
                    max_fps = 24
                    default_guidance = 9.0
                    video_resolution = [320, 576]
                    max_video_resolution = [576, 768]
                    st.info("💡 Using reduced settings for 8GB GPU")
                elif vram >= 6:
                    default_frames = 8
                    max_frames = 16
                    default_steps = 10
                    default_fps = 6
                    max_fps = 16
                    default_guidance = 9.0
                    video_resolution = [320, 576]
                    max_video_resolution = [320, 768]
                    st.warning("⚠️ Using minimal settings for 6GB GPU")
                else:
                    default_frames = 8
                    max_frames = 8
                    default_steps = 10
                    default_fps = 4
                    max_fps = 12
                    default_guidance = 9.0
                    video_resolution = [320, 576]
                    max_video_resolution = [320, 576]
                    st.error("❌ Very limited settings for <6GB GPU")
            else:
                default_frames = 8
                max_frames = 16
                default_steps = 10
                default_fps = 4
                max_fps = 8
                default_guidance = 9.0
                video_resolution = [320, 576]
                max_video_resolution = [320, 576]
        
        # Video parameters with preferences
        preferred_frames = video_prefs.get('preferred_frames', default_frames)
        preferred_fps = video_prefs.get('preferred_fps', default_fps)
        
        num_frames = st.slider(
            "🎞️ Number of Frames",
            min_value=8, max_value=max_frames, value=preferred_frames,
            help=f"Optimized for your hardware: {default_frames} frames recommended"
        )
        
        # Save frames preference when changed
        if _preferences_manager and num_frames != video_prefs.get('preferred_frames'):
            _preferences_manager.set_preference('video_generation', 'preferred_frames', num_frames)
            _preferences_manager.save_preferences()
        
        fps = st.slider(
            "📹 Frames Per Second",
            min_value=4, max_value=max_fps, value=preferred_fps,
            help=f"Optimized for your hardware: {default_fps} FPS recommended"
        )
        
        # Save FPS preference when changed
        if _preferences_manager and fps != video_prefs.get('preferred_fps'):
            _preferences_manager.set_preference('video_generation', 'preferred_fps', fps)
            _preferences_manager.save_preferences()
        
        duration = round(num_frames / fps, 1)
        st.info(f"Video duration: ~{duration} seconds")
    
    # Advanced settings
    with st.expander("🔧 Advanced Video Settings (Hardware Optimized)"):
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            preferred_steps = video_prefs.get('preferred_steps', default_steps)
            preferred_guidance = video_prefs.get('preferred_guidance', default_guidance)
            
            steps = st.slider(
                "Inference Steps",
                min_value=10, max_value=100, value=preferred_steps,
                help=f"Optimized for your hardware: {default_steps} steps recommended"
            )
            
            # Save steps preference when changed
            if _preferences_manager and steps != video_prefs.get('preferred_steps'):
                _preferences_manager.set_preference('video_generation', 'preferred_steps', steps)
                _preferences_manager.save_preferences()
            
            guidance = st.slider(
                "Guidance Scale",
                min_value=1.0, max_value=20.0, value=preferred_guidance, step=0.5,
                help="Higher values follow prompt more closely"
            )
            
            # Save guidance preference when changed
            if _preferences_manager and guidance != video_prefs.get('preferred_guidance'):
                _preferences_manager.set_preference('video_generation', 'preferred_guidance', guidance)
                _preferences_manager.save_preferences()
        
        with col_adv2:
            use_seed = st.checkbox("🎲 Use Fixed Seed")
            seed = None
            if use_seed:
                seed = st.number_input(
                    "Seed Value",
                    min_value=0, max_value=2147483647, value=42
                )
            
            # Hardware-optimized resolution settings
            if HARDWARE_OPTIMIZER_AVAILABLE:
                # Use optimized resolution options
                height_options = [video_resolution[0]]
                width_options = [video_resolution[1]]
                
                # Add additional options if hardware supports it
                if max_video_resolution[0] > video_resolution[0]:
                    height_options.append(max_video_resolution[0])
                if max_video_resolution[1] > video_resolution[1]:
                    width_options.append(max_video_resolution[1])
                
                default_h_idx = 0
                default_w_idx = 0
            else:
                # Fallback: Recommended resolutions based on GPU
                if torch and torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
                    if vram >= 12:
                        height_options = [320, 576, 768]
                        width_options = [576, 1024, 1024]
                        default_h_idx = 1
                        default_w_idx = 0
                    elif vram >= 8:
                        height_options = [320, 576]
                        width_options = [576, 768]
                        default_h_idx = 0
                        default_w_idx = 0
                    else:
                        height_options = [320]
                        width_options = [576]
                        default_h_idx = 0
                        default_w_idx = 0
                else:
                    height_options = [320]
                    width_options = [576]
                    default_h_idx = 0
                    default_w_idx = 0
            # Load resolution preferences
            preferred_resolution = video_prefs.get('preferred_resolution', [video_resolution[0], video_resolution[1]])
            preferred_height = preferred_resolution[0] if len(preferred_resolution) >= 1 else video_resolution[0]
            preferred_width = preferred_resolution[1] if len(preferred_resolution) >= 2 else video_resolution[1]
            
            # Set default indices based on preferences
            height_idx = 0
            width_idx = 0
            if preferred_height in height_options:
                height_idx = height_options.index(preferred_height)
            if preferred_width in width_options:
                width_idx = width_options.index(preferred_width)
            
            # Create height options for custom select
            height_select_options = []
            for h in height_options:
                option = create_option(
                    value=str(h),
                    label=f"📏 {h}px",
                    code=f"Height: {h}px",
                    language="text",
                    code_color="#2196F3"
                )
                height_select_options.append(option)
            
            # Create width options for custom select  
            width_select_options = []
            for w in width_options:
                option = create_option(
                    value=str(w),
                    label=f"📐 {w}px",
                    code=f"Width: {w}px",
                    language="text",
                    code_color="#FF9800"
                )
                width_select_options.append(option)
            
            # Use custom markdown select for height
            default_height = str(height_options[height_idx]) if height_options else None
            height_str = markdown_select(
                options=height_select_options,
                key="video_height_select",
                placeholder="📏 Choose Height...",
                default_value=default_height
            )
            height = int(height_str) if height_str else height_options[0]
            
            # Use custom markdown select for width
            default_width = str(width_options[width_idx]) if width_options else None
            width_str = markdown_select(
                options=width_select_options,
                key="video_width_select",
                placeholder="📐 Choose Width...",
                default_value=default_width
            )
            width = int(width_str) if width_str else width_options[0]
            
            # Save resolution preference when changed
            current_resolution = [height, width]
            if _preferences_manager and current_resolution != video_prefs.get('preferred_resolution'):
                _preferences_manager.set_preference('video_generation', 'preferred_resolution', current_resolution)
                _preferences_manager.save_preferences()
            
            # Show optimization recommendation
            if HARDWARE_OPTIMIZER_AVAILABLE:
                if height == video_resolution[0] and width == video_resolution[1]:
                    st.success(f"✅ Optimized resolution: {width}x{height}")
                else:
                    st.info(f"💡 Recommended: {video_resolution[1]}x{video_resolution[0]}")
            
            # Memory usage estimate
            memory_estimate = (height * width * num_frames * steps) / 1024**3 * 2  # Rough estimate
            st.info(f"Est. memory: ~{memory_estimate:.1f}GB")
    
    # Model loading state
    model_key = f"video_model_{selected_model_name}"
    if model_key not in st.session_state:
        st.session_state[model_key] = None
    
    # Generate button
    if st.button("🎬 Generate Video", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt to generate video.")
            return
        
        # Load model if not already loaded
        if st.session_state[model_key] is None:
            model_id = video_models[selected_model_name]
            
            with st.spinner(f"🔄 Loading {selected_model_name} (this may take several minutes)..."):
                try:
                    # Clear GPU cache before loading
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Try different loading strategies based on available memory
                    pipeline = None
                    load_strategies = []
                    
                    if torch.cuda.is_available():
                        # Check available GPU memory
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
                        free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
                        free_memory_gb = free_memory // 1024**3
                        
                        st.info(f"GPU Memory: {gpu_memory}GB total, ~{free_memory_gb}GB available")
                        
                        if gpu_memory >= 8:
                            load_strategies = [
                                ("GPU with float16", "cuda", torch.float16),
                                ("CPU offload with float16", "cpu_offload", torch.float16),
                                ("CPU with float32", "cpu", torch.float32)
                            ]
                        else:
                            load_strategies = [
                                ("CPU offload with float16", "cpu_offload", torch.float16),
                                ("CPU with float32", "cpu", torch.float32)
                            ]
                    else:
                        load_strategies = [("CPU with float32", "cpu", torch.float32)]
                    
                    # Try each loading strategy
                    for strategy_name, device_strategy, dtype in load_strategies:
                        try:
                            st.info(f"Trying {strategy_name}...")
                            
                            # Load pipeline with different approaches
                            pipeline = None
                            load_approaches = []
                            
                            # For each model, try different loading approaches
                            if "zeroscope" in model_id.lower():
                                load_approaches = [
                                    ("with safetensors", {"use_safetensors": True}),
                                    ("without safetensors", {"use_safetensors": False}),
                                    ("default loading", {})
                                ]
                            else:
                                load_approaches = [
                                    ("with safetensors", {"use_safetensors": True}),
                                    ("without safetensors", {"use_safetensors": False}),
                                    ("default loading", {})
                                ]
                            
                            # Try each loading approach for this strategy
                            for approach_name, load_kwargs in load_approaches:
                                try:
                                    base_kwargs = {
                                        "torch_dtype": dtype,
                                        "low_cpu_mem_usage": True,
                                    }
                                    base_kwargs.update(load_kwargs)
                                    
                                    if "zeroscope" in model_id.lower():
                                        from diffusers import TextToVideoSDPipeline
                                        pipeline = TextToVideoSDPipeline.from_pretrained(
                                            model_id,
                                            **base_kwargs
                                        )
                                    else:
                                        from diffusers import DiffusionPipeline
                                        pipeline = DiffusionPipeline.from_pretrained(
                                            model_id,
                                            **base_kwargs
                                        )
                                    
                                    st.info(f"✅ Loaded {approach_name}")
                                    break
                                    
                                except Exception as approach_error:
                                    st.info(f"⚠️ {approach_name} failed: {str(approach_error)[:100]}...")
                                    continue
                            
                            if pipeline is None:
                                raise Exception(f"All loading approaches failed for {strategy_name}")
                            
                            # Apply memory optimizations based on strategy
                            if device_strategy == "cpu_offload" and torch.cuda.is_available():
                                pipeline.enable_model_cpu_offload()
                                pipeline.enable_attention_slicing()
                                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                                    try:
                                        pipeline.enable_xformers_memory_efficient_attention()
                                    except:
                                        pass
                            elif device_strategy == "cuda" and torch.cuda.is_available():
                                pipeline = pipeline.to("cuda")
                                pipeline.enable_attention_slicing()
                                if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                                    try:
                                        pipeline.enable_xformers_memory_efficient_attention()
                                    except:
                                        pass
                            else:
                                pipeline = pipeline.to("cpu")
                            
                            # Test the pipeline with a simple operation
                            if torch.cuda.is_available() and device_strategy != "cpu":
                                torch.cuda.synchronize()
                            
                            st.success(f"✅ {selected_model_name} loaded successfully using {strategy_name}!")
                            break
                            
                        except Exception as strategy_error:
                            st.warning(f"⚠️ {strategy_name} failed: {str(strategy_error)}")
                            pipeline = None
                            
                            # Clear memory before trying next strategy
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                            continue
                    
                    if pipeline is None:
                        raise Exception("All loading strategies failed")
                    
                    st.session_state[model_key] = pipeline
                    
                except Exception as e:
                    st.error(f"❌ Failed to load model: {str(e)}")
                    st.error("**Troubleshooting Tips:**")
                    st.error("- Close other GPU-intensive applications")
                    st.error("- Restart the application to clear GPU memory")
                    st.error("- Try a smaller model or lower resolution")
                    st.error("- Use CPU-only mode if GPU memory is insufficient")
                    
                    # Provide specific CUDA error help
                    if "CUDA" in str(e):
                        st.error("**CUDA-specific solutions:**")
                        st.code("# Set environment variable to get better error info:\nset CUDA_LAUNCH_BLOCKING=1")
                        st.error("- Update GPU drivers")
                        st.error("- Reinstall PyTorch with correct CUDA version")
                        st.error("- Check if GPU is being used by other processes")
                    
                    return
        
        pipeline = st.session_state[model_key]
        
        # Generate video
        with st.spinner(f"🎬 Generating video ({duration}s, {num_frames} frames)..."):
            try:
                # Clear GPU cache before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Prepare generation parameters
                generation_kwargs = {
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "height": height,
                    "width": width,
                }
                
                if seed is not None:
                    generator = torch.Generator()
                    if torch.cuda.is_available() and hasattr(pipeline, 'device') and 'cuda' in str(pipeline.device):
                        generator = torch.Generator(device='cuda')
                    generator.manual_seed(seed)
                    generation_kwargs["generator"] = generator
                
                # Generate with error handling
                progress_placeholder = st.empty()
                progress_placeholder.info("🎬 Initializing generation...")
                
                with torch.no_grad():
                    try:
                        # Set environment variable for better CUDA error reporting
                        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                        
                        progress_placeholder.info("🎬 Generating frames...")
                        result = pipeline(**generation_kwargs)
                        
                        progress_placeholder.info("🎬 Processing frames...")
                        
                        # Get frames with better error handling
                        frames = None
                        if hasattr(result, 'frames') and result.frames is not None:
                            if isinstance(result.frames, list) and len(result.frames) > 0:
                                frames = result.frames[0]
                            else:
                                frames = result.frames
                        elif hasattr(result, 'images') and result.images is not None:
                            frames = result.images
                        else:
                            frames = result
                        
                        if frames is None:
                            raise Exception("No frames generated - result is None")
                        
                        # Convert frames to list if it's a tensor or other format
                        if hasattr(frames, 'shape') and len(frames.shape) == 4:
                            # It's a tensor with shape (num_frames, height, width, channels)
                            frames = [frames[i] for i in range(frames.shape[0])]
                        elif not isinstance(frames, (list, tuple)):
                            # Try to convert to list
                            try:
                                frames = list(frames)
                            except Exception:
                                raise Exception(f"Cannot convert frames to list. Type: {type(frames)}")
                        
                        if not frames or len(frames) == 0:
                            raise Exception("No frames generated - empty frames list")
                        
                        # Debug frame information with better type detection
                        st.info(f"Generated {len(frames)} frames")
                        if len(frames) > 0:
                            first_frame = frames[0]
                            if hasattr(first_frame, 'size'):
                                st.info(f"Frame format: PIL Image, size: {first_frame.size}")
                            elif hasattr(first_frame, 'shape'):
                                st.info(f"Frame format: Array/Tensor, shape: {first_frame.shape}")
                            else:
                                st.info(f"Frame format: {type(first_frame)}")
                                
                            # Additional debug info for tensor frames
                            if hasattr(first_frame, 'dtype'):
                                st.info(f"Frame dtype: {first_frame.dtype}")
                            if hasattr(first_frame, 'device'):
                                st.info(f"Frame device: {first_frame.device}")
                        
                        progress_placeholder.info(f"🎬 Converting {len(frames)} frames to video...")
                        
                    except Exception as gen_error:
                        progress_placeholder.empty()
                        
                        # Handle specific CUDA errors
                        if "CUDA" in str(gen_error) and "out of memory" in str(gen_error):
                            st.error("❌ GPU out of memory during generation!")
                            st.error("**Try these solutions:**")
                            st.error("- Reduce number of frames (try 16 or 8)")
                            st.error("- Lower resolution (try 320x576)")
                            st.error("- Reduce inference steps (try 10-15)")
                            st.error("- Close other applications using GPU")
                            st.error("- Restart the application")
                        elif "CUDA" in str(gen_error):
                            st.error(f"❌ CUDA error during generation: {str(gen_error)}")
                            st.error("**Try these solutions:**")
                            st.error("- Restart the application")
                            st.error("- Update GPU drivers")
                            st.error("- Use CPU-only mode")
                        else:
                            st.error(f"❌ Generation error: {str(gen_error)}")
                        
                        return
                
                # Convert to video
                import tempfile
                
                progress_placeholder.info("🎬 Creating video file...")
                
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    video_path = tmp.name
                
                try:
                    # Create video using OpenCV with error handling
                    import cv2  # Import cv2 here to ensure it's available
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                    
                    if not out.isOpened():
                        raise Exception("Failed to open video writer")
                    
                    frame_count = 0
                    for i, frame in enumerate(frames):
                        try:
                            # Convert frame to numpy array with better error handling
                            if hasattr(frame, 'size'):  # PIL Image
                                frame_array = np.array(frame)
                            elif hasattr(frame, 'numpy'):  # Torch tensor
                                if hasattr(frame, 'cpu'):
                                    frame_array = frame.cpu().numpy()
                                else:
                                    frame_array = frame.numpy()
                            elif isinstance(frame, np.ndarray):  # Already numpy
                                frame_array = frame
                            else:
                                frame_array = np.array(frame)
                            
                            # Handle potential tensor conversion issues
                            if len(frame_array.shape) == 0:  # Scalar
                                st.warning(f"⚠️ Frame {i} is a scalar, skipping...")
                                continue
                            
                            # Ensure frame has the right shape (handle different formats)
                            if len(frame_array.shape) == 4:  # Batch dimension
                                frame_array = frame_array[0]
                            elif len(frame_array.shape) == 2:  # Grayscale
                                frame_array = np.stack([frame_array] * 3, axis=-1)
                            elif len(frame_array.shape) == 1:  # Invalid shape
                                st.warning(f"⚠️ Frame {i} has invalid shape {frame_array.shape}, skipping...")
                                continue
                            
                            # Ensure we have a valid 3D array
                            if len(frame_array.shape) != 3:
                                st.warning(f"⚠️ Frame {i} has unexpected dimensions {frame_array.shape}, skipping...")
                                continue
                            
                            # Ensure frame is the right size
                            frame_height, frame_width = frame_array.shape[:2]
                            if frame_height != height or frame_width != width:
                                from PIL import Image
                                if not hasattr(frame, 'resize'):
                                    frame = Image.fromarray(frame_array.astype(np.uint8))
                                else:
                                    frame = frame
                                frame = frame.resize((width, height))
                                frame_array = np.array(frame)
                            
                            # Convert RGB to BGR for OpenCV
                            if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                            else:
                                frame_bgr = frame_array
                            
                            # Ensure frame is uint8 with safe conversion
                            if frame_bgr.dtype != np.uint8:
                                # Use np.max() instead of .max() to handle arrays properly
                                # Also add bounds checking
                                try:
                                    max_val = np.max(frame_bgr)
                                    min_val = np.min(frame_bgr)
                                    
                                    if max_val <= 1.0 and min_val >= 0.0:  # Normalized values [0, 1]
                                        frame_bgr = (frame_bgr * 255).astype(np.uint8)
                                    elif max_val <= 255.0 and min_val >= 0.0:  # Already in [0, 255] range
                                        frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
                                    else:  # Need to normalize
                                        frame_bgr = ((frame_bgr - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                                except Exception as conv_error:
                                    st.warning(f"⚠️ Frame {i} conversion error: {conv_error}, using fallback...")
                                    frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
                            
                            # Validate frame before writing
                            if frame_bgr.shape[:2] != (height, width):
                                st.warning(f"⚠️ Frame {i} has wrong dimensions: {frame_bgr.shape[:2]}, expected: ({height}, {width})")
                                continue
                            
                            if len(frame_bgr.shape) != 3 or frame_bgr.shape[2] != 3:
                                st.warning(f"⚠️ Frame {i} has wrong channel count: {frame_bgr.shape}")
                                continue
                            
                            out.write(frame_bgr)
                            frame_count += 1
                            
                            # Update progress
                            if i % 5 == 0:  # Update every 5 frames
                                progress_placeholder.info(f"🎬 Writing frame {i+1}/{len(frames)}...")
                                
                        except Exception as frame_error:
                            st.warning(f"⚠️ Error processing frame {i}: {frame_error}")
                            continue
                    
                    out.release()
                    progress_placeholder.empty()
                    
                    if frame_count == 0:
                        raise Exception("No frames were successfully written to video")
                    
                    # Verify video file was created
                    if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                        raise Exception("Video file was not created or is empty")
                    
                    # Display video
                    st.success(f"✅ Video generated successfully! ({frame_count} frames)")
                    st.video(video_path)
                    
                    # Download button
                    with open(video_path, 'rb') as f:
                        video_data = f.read()
                    
                    st.download_button(
                        label="📥 Download Video",
                        data=video_data,
                        file_name=f"generated_video_{selected_model_name.replace(' ', '_')}.mp4",
                        mime="video/mp4"
                    )
                    
                    # Generation info
                    with st.expander("📊 Generation Details"):
                        st.json({
                            "model": selected_model_name,
                            "prompt": prompt,
                            "frames": num_frames,
                            "frames_written": frame_count,
                            "fps": fps,
                            "duration": f"{duration}s",
                            "resolution": f"{width}x{height}",
                            "steps": steps,
                            "guidance_scale": guidance,
                            "seed": seed if seed is not None else "Random",
                            "file_size": f"{os.path.getsize(video_path) / 1024 / 1024:.1f} MB"
                        })
                    
                except Exception as video_error:
                    st.error(f"❌ Error creating video: {str(video_error)}")
                    st.error("This might be due to:")
                    st.error("- OpenCV video codec issues")
                    st.error("- Frame format problems")
                    st.error("- Disk space issues")
                
                # Clean up
                try:
                    if 'video_path' in locals() and os.path.exists(video_path):
                        os.unlink(video_path)
                except Exception as cleanup_error:
                    st.warning(f"Could not clean up temporary file: {cleanup_error}")
                    
                # Clear GPU memory after generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                    
            except Exception as e:
                st.error(f"❌ Error generating video: {str(e)}")
                st.error("**Common causes and solutions:**")
                if "CUDA" in str(e):
                    st.error("- **CUDA Issues**: Restart application, update drivers")
                    st.error("- **Memory Issues**: Reduce settings (frames, resolution, steps)")
                    st.error("- **Driver Issues**: Update NVIDIA drivers")
                st.error("- **Model Issues**: Try a different model")
                st.error("- **Network Issues**: Check internet connection for model downloads")
                
                # Clear GPU memory on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

def rag_knowledge_base_page():
    """RAG Knowledge Base Management Page"""
    st.title("📚 Knowledge Base (RAG)")
    
    if not RAG_HANDLER_AVAILABLE:
        st.error("❌ RAG Handler not available. Please ensure rag_handler.py is in the project directory.")
        st.info("The RAG system provides intelligent retrieval-augmented generation capabilities.")
        return
    
    # Initialize RAG handler
    try:
        rag_handler = get_rag_handler(db_path="rag_db.json")
    except Exception as e:
        st.error(f"❌ Error initializing RAG handler: {e}")
        return
    
    # Sidebar with stats
    with st.sidebar:
        st.markdown("### 📊 Database Statistics")
        stats = rag_handler.get_database_stats()
        
        st.metric("Total Documents", stats['total_documents'])
        st.metric("Total Chunks", stats['total_chunks'])
        st.metric("Total Characters", f"{stats['total_characters']:,}")
        
        if stats.get('created'):
            st.caption(f"Created: {stats['created'][:10]}")
        if stats.get('updated'):
            st.caption(f"Updated: {stats['updated'][:10]}")
        
        st.markdown("---")
        
        # Database management
        st.markdown("### 🔧 Database Actions")
        
        if st.button("🔄 Refresh Stats"):
            st.rerun()
        
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.session_state.get('confirm_clear_rag'):
                rag_handler.db.clear_database()
                st.success("✅ Database cleared!")
                st.session_state['confirm_clear_rag'] = False
                st.rerun()
            else:
                st.warning("⚠️ Click again to confirm clearing all documents")
                st.session_state['confirm_clear_rag'] = True
        
        # Export database
        if st.button("💾 Export Database"):
            export_data = json.dumps(rag_handler.db.data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download rag_db.json",
                data=export_data,
                file_name="rag_db_export.json",
                mime="application/json"
            )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Query", "➕ Add Knowledge", "📂 Manage Documents", "📖 Help"])
    
    # Tab 1: Query Knowledge Base
    with tab1:
        st.subheader("🔍 Query Knowledge Base")
        st.markdown("Ask questions and get answers based on your knowledge base using RAG.")
        
        # Query input
        query_input = st.text_area(
            "Enter your question:",
            placeholder="What is Aurora? How does RAG work?",
            height=100,
            key="rag_query_input"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            context_window = st.slider(
                "Context chunks to retrieve:",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of relevant text chunks to use for answering"
            )
        
        with col2:
            if st.button("🔍 Search", type="primary"):
                if query_input.strip():
                    with st.spinner("Searching knowledge base..."):
                        result = rag_handler.query(query_input, context_window=context_window)
                        
                        # Display answer
                        st.markdown("### 💡 Answer:")
                        st.success(result['answer'])
                        
                        # Display sources if available
                        if result.get('sources'):
                            st.markdown("### 📖 Sources:")
                            for idx, source in enumerate(result['sources']):
                                with st.expander(f"Source {idx+1} - Score: {source['score']:.2f}"):
                                    st.markdown(f"**Document ID:** `{source['doc_id']}`")
                                    if source.get('metadata'):
                                        st.json(source['metadata'])
                        
                        # Show retrieval stats
                        if result.get('context_used'):
                            st.info(f"✅ Used {result['num_contexts']} context chunks")
                        else:
                            st.warning("⚠️ No relevant context found in knowledge base")
                else:
                    st.warning("Please enter a question")
    
    # Tab 2: Add Knowledge
    with tab2:
        st.subheader("➕ Add Knowledge to Database")
        
        # Method selector
        add_method = st.radio(
            "Choose how to add knowledge:",
            ["✍️ Manual Entry", "📂 Import from File"],
            horizontal=True
        )
        
        if add_method == "✍️ Manual Entry":
            # Manual text entry
            knowledge_content = st.text_area(
                "Enter knowledge content:",
                placeholder="Enter information you want to add to the knowledge base...",
                height=200,
                key="knowledge_content"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                knowledge_source = st.text_input(
                    "Source (optional):",
                    placeholder="e.g., manual, documentation, notes",
                    value="manual"
                )
            
            with col2:
                knowledge_category = st.text_input(
                    "Category (optional):",
                    placeholder="e.g., features, technical, general"
                )
            
            if st.button("➕ Add to Knowledge Base", type="primary"):
                if knowledge_content.strip():
                    metadata = {}
                    if knowledge_category:
                        metadata['category'] = knowledge_category
                    
                    doc_id = rag_handler.add_knowledge(
                        knowledge_content,
                        source=knowledge_source,
                        metadata=metadata
                    )
                    
                    if doc_id:
                        st.success(f"✅ Knowledge added successfully! Document ID: `{doc_id}`")
                        st.rerun()
                    else:
                        st.error("❌ Failed to add knowledge")
                else:
                    st.warning("Please enter some content")
        
        else:  # Import from file
            uploaded_file = st.file_uploader(
                "Upload a text file:",
                type=['txt', 'md', 'json'],
                help="Upload a text file to import into the knowledge base"
            )
            
            chunk_size = st.slider(
                "Maximum characters per chunk:",
                min_value=500,
                max_value=5000,
                value=1000,
                step=100,
                help="Large files will be split into chunks of this size"
            )
            
            if uploaded_file and st.button("📂 Import File", type="primary"):
                with st.spinner("Importing file..."):
                    try:
                        # Save uploaded file temporarily
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        # Import the file
                        doc_ids = rag_handler.import_from_file(tmp_path, chunk_size=chunk_size)
                        
                        # Clean up
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                        
                        if doc_ids:
                            st.success(f"✅ Imported {len(doc_ids)} document(s) successfully!")
                            st.info(f"Document IDs: {', '.join(doc_ids)}")
                            st.rerun()
                        else:
                            st.error("❌ Failed to import file")
                    except Exception as e:
                        st.error(f"❌ Error importing file: {e}")
    
    # Tab 3: Manage Documents
    with tab3:
        st.subheader("📂 Manage Documents")
        
        # List all documents
        if rag_handler.db.data['documents']:
            st.markdown(f"**Total Documents:** {len(rag_handler.db.data['documents'])}")
            
            # Search/filter documents
            filter_text = st.text_input(
                "Filter documents:",
                placeholder="Search by content or ID...",
                key="filter_docs"
            )
            
            # Display documents
            for idx, doc in enumerate(rag_handler.db.data['documents']):
                doc_id = doc['id']
                content = doc['content']
                metadata = doc.get('metadata', {})
                
                # Apply filter (cache lowercased filter value for efficiency)
                if filter_text:
                    filter_lower = filter_text.lower()
                    if filter_lower not in doc_id.lower() and filter_lower not in content.lower():
                        continue
                
                with st.expander(f"📄 {doc_id[:50]}..." if len(doc_id) > 50 else f"📄 {doc_id}"):
                    st.markdown(f"**Document ID:** `{doc_id}`")
                    st.markdown(f"**Created:** {doc.get('created', 'Unknown')}")
                    
                    if metadata:
                        st.markdown("**Metadata:**")
                        st.json(metadata)
                    
                    st.markdown("**Content Preview:**")
                    content_preview = content[:500] + "..." if len(content) > 500 else content
                    st.text_area("", content_preview, height=150, key=f"preview_{idx}", disabled=True)
                    
                    st.markdown(f"**Chunks:** {len(doc.get('chunks', []))}")
                    st.markdown(f"**Total Length:** {len(content)} characters")
                    
                    # Actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🗑️ Delete", key=f"delete_{doc_id}"):
                            if rag_handler.db.delete_document(doc_id):
                                st.success(f"✅ Deleted document: {doc_id}")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to delete document")
                    
                    with col2:
                        # Export individual document
                        doc_export = json.dumps(doc, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="💾 Export",
                            data=doc_export,
                            file_name=f"{doc_id}.json",
                            mime="application/json",
                            key=f"export_{doc_id}"
                        )
        else:
            st.info("📭 No documents in the knowledge base yet. Add some in the 'Add Knowledge' tab!")
    
    # Tab 4: Help
    with tab4:
        st.subheader("📖 RAG Knowledge Base Help")
        
        st.markdown("""
        ### What is RAG?
        
        **RAG (Retrieval-Augmented Generation)** is a technique that enhances AI responses by:
        1. **Retrieving** relevant information from a knowledge base
        2. **Augmenting** the AI prompt with this context
        3. **Generating** more accurate and informed responses
        
        ### How to Use
        
        #### Adding Knowledge
        1. Go to the "Add Knowledge" tab
        2. Either type content manually or upload a text file
        3. Add optional metadata (source, category)
        4. Click "Add to Knowledge Base"
        
        #### Querying
        1. Go to the "Query" tab
        2. Type your question
        3. Adjust the context window (how many chunks to retrieve)
        4. Click "Search" to get an answer
        
        #### Managing Documents
        1. Go to the "Manage Documents" tab
        2. View all stored documents
        3. Filter/search documents
        4. Delete or export individual documents
        
        ### Tips for Best Results
        
        - **Chunk Size**: Keep documents focused (500-2000 characters)
        - **Metadata**: Use categories and sources for better organization
        - **Context Window**: Start with 3 chunks, adjust based on results
        - **Clear Content**: Write clear, well-structured knowledge entries
        - **Regular Updates**: Keep your knowledge base current
        
        ### Technical Details
        
        - **Storage**: JSON-based database (rag_db.json)
        - **Search**: Keyword-based relevance scoring
        - **Chunking**: Automatic text splitting with overlap
        - **Integration**: Works with Ollama models
        
        ### Example Queries
        
        ```
        "What features does Aurora have?"
        "How do I use the image generation?"
        "What models are supported?"
        "How does voice recognition work?"
        ```
        
        ### Database Management
        
        - **Export**: Save entire database as JSON
        - **Clear**: Remove all documents (use with caution!)
        - **Backup**: Regular exports recommended
        - **Import**: Batch import from text files
        """)
        
        st.markdown("---")
        st.info("💡 **Pro Tip**: Start by adding documentation, FAQs, or important information about your project to create a useful knowledge base!")

def about_page():
    """About page"""
    st.title("ℹ️ About AURORA")
    
    # Load AURORA system configuration
    if AURORA_SYSTEM_AVAILABLE:
        try:
            aurora_system = get_aurora_system()
            
            # Display AURORA identity
            project = aurora_system.config.get("project", {})
            st.markdown(f"# {project.get('name', 'AURORA')}")
            st.markdown(f"### {project.get('full_form', '')}")
            st.markdown(f"**Version:** {project.get('version', '1.0')}")
            st.markdown(f"*{project.get('attribution', '')}*")
            st.markdown("---")
            
            # Display capabilities
            st.markdown(aurora_system.format_capabilities_display())
            
            # Display team information
            st.markdown(aurora_system.format_team_display())
            
            # Display operational policies
            policies = aurora_system.get_policies()
            if policies:
                st.markdown("## 🛡️ Operational Policies")
                for policy_name, policy_text in policies.items():
                    st.markdown(f"**{policy_name.replace('_', ' ').title()}:** {policy_text}")
                st.markdown("---")
            
        except Exception as e:
            st.error(f"Error loading AURORA configuration: {e}")
            # Fallback to basic info
            st.markdown("## 🌅 Welcome to Aurora")
    else:
        st.markdown("## 🌅 Welcome to Aurora")
    
    # Technical features
    st.markdown("""
    ## ✨ Features
    - **💬 Intelligent Chat**: Powered by Ollama models
    - **🎨 Image Generation**: Create multiple images using Stable Diffusion (default: 2 images)
    - **🎬 Video Generation**: Generate videos using AI models
    - **� Knowledge Base (RAG)**: Retrieval-Augmented Generation for intelligent answers
    - **�🗣️ Voice Interaction**: Speech-to-text and text-to-speech with synchronized display
    - **🔍 Web Search**: Wikipedia, Google, YouTube integration
    - **🤖 AI-Powered Search**: Smart search using Ollama
    - **📰 Topic News Search**: Search news by specific topics
    - **🌤️ Real-time Info**: Weather and news updates
    - **🖥️ Agentic Control**: Basic and Vision-guided desktop automation
    
    ### 🎯 How to Use
    1. **Chat**: Select a model and start chatting
    2. **Voice**: Use the microphone for hands-free interaction
    3. **Images**: Switch to the Image Generation page, generate 1-10 images (default: 2)
    4. **Videos**: Create AI-generated videos with custom prompts
    5. **Knowledge Base**: Build and query your own RAG database
    6. **Commands**: Try "web search AI" or "smart search quantum computing"
    
    ## 🛠️ Technical Stack
    - **Framework**: Streamlit + Python
    - **AI Models**: Ollama (Local) + Stable Diffusion + Video Models
    - **Speech**: OpenAI Whisper + Bark TTS
    - **RAG**: JSON-based knowledge database with vector search
    
    ## 📚 Supported Commands
    ```
    Wikipedia: "wikipedia [topic]"
    Weather: "weather in [city]"
    Google: "search google for [term]"
    YouTube: "search youtube for [term]"
    News: "latest news"
    
    AI Features:
    AI Search: "web search [query]"
    Smart Search: "smart search [topic]"
    Topic News: "news about [topic]"
    
    Desktop Control:
    Basic Agent: Toggle for keyboard/mouse control
    Vision Agent: Toggle for autonomous visual tasks
    ```
    
    ### 🔧 System Requirements
    - Python 3.8+
    - Ollama (for chat functionality)
    - GPU recommended (for image/video generation)
    - Microphone (for voice input)
    - CUDA GPU (recommended for video generation)
    """)
    
    # System status
    st.subheader("🖥️ System Status")
    
    # Hardware optimization status
    if HARDWARE_OPTIMIZER_AVAILABLE:
        st.markdown("### 🚀 Hardware Optimization")
        
        try:
            hardware_info = get_hardware_info()
            perf_rec = _hw_optimizer.get_performance_recommendation()
            
            st.success(f"✅ Hardware Optimizer Active: {perf_rec}")
            
            # Show optimized settings summary
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                chat_settings = get_chat_settings()
                st.markdown("**💬 Chat Optimization:**")
                st.info(f"Models: {len(chat_settings.get('recommended_models', []))}")
                st.info(f"Context: {chat_settings.get('context_length', 0)} tokens")
                
            with col_opt2:
                image_settings = get_image_settings()
                st.markdown("**🎨 Image Optimization:**")
                st.info(f"Default: {image_settings.get('default_num_images', 0)} images")
                res = image_settings.get('recommended_resolution', [0, 0])
                st.info(f"Resolution: {res[0]}x{res[1]}")
                
            with col_opt3:
                video_settings = get_video_settings()
                st.markdown("**🎬 Video Optimization:**")
                st.info(f"Frames: {video_settings.get('recommended_frames', 0)}")
                res = video_settings.get('recommended_resolution', [0, 0])
                st.info(f"Resolution: {res[0]}x{res[1]}")
            
            # Hardware refresh button
            if st.button("🔄 Refresh Hardware Detection"):
                _hw_optimizer.force_refresh()
                st.success("Hardware detection refreshed!")
                st.rerun()
                
        except Exception as e:
            st.warning(f"⚠️ Hardware optimizer error: {e}")
    else:
        st.warning("⚠️ Hardware Optimizer not available - using default settings")
    
    # User Preferences Status
    if _preferences_manager:
        st.markdown("### 🎯 User Preferences")
        
        try:
            # Show current saved preferences
            col_pref1, col_pref2, col_pref3 = st.columns(3)
            
            with col_pref1:
                chat_prefs = _preferences_manager.get_category_preferences('chat')
                st.markdown("**💬 Chat Preferences:**")
                
                # Handle dictionary preferences
                last_used_model = chat_prefs.get('last_used_model')
                if isinstance(last_used_model, dict):
                    last_used_model = last_used_model.get('value') or last_used_model.get('optionValue')
                if last_used_model:
                    st.info(f"Model: {last_used_model}")
                    
                if 'preferred_streaming' in chat_prefs:
                    st.info(f"Streaming: {'On' if chat_prefs['preferred_streaming'] else 'Off'}")
                if chat_prefs.get('preferred_tts_engine'):
                    st.info(f"TTS: {chat_prefs['preferred_tts_engine']}")
                
            with col_pref2:
                image_prefs = _preferences_manager.get_category_preferences('image_generation')
                st.markdown("**🎨 Image Preferences:**")
                
                # Handle dictionary preferences
                last_used_model = image_prefs.get('last_used_model')
                if isinstance(last_used_model, dict):
                    last_used_model = last_used_model.get('value') or last_used_model.get('optionValue')
                if last_used_model:
                    st.info(f"Model: {last_used_model}")
                    
                if image_prefs.get('preferred_resolution'):
                    st.info(f"Resolution: {image_prefs['preferred_resolution']}")
                if 'preferred_num_images' in image_prefs:
                    st.info(f"Images: {image_prefs['preferred_num_images']}")
                
            with col_pref3:
                video_prefs = _preferences_manager.get_category_preferences('video_generation')
                st.markdown("**🎬 Video Preferences:**")
                
                # Handle dictionary preferences
                last_used_model = video_prefs.get('last_used_model')
                if isinstance(last_used_model, dict):
                    last_used_model = last_used_model.get('value') or last_used_model.get('optionValue')
                if last_used_model:
                    st.info(f"Model: {last_used_model}")
                    
                if 'preferred_frames' in video_prefs:
                    st.info(f"Frames: {video_prefs['preferred_frames']}")
                if 'preferred_fps' in video_prefs:
                    st.info(f"FPS: {video_prefs['preferred_fps']}")
            
            # Preferences management buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Reset All Preferences"):
                    _preferences_manager.reset_preferences()
                    st.success("All preferences reset to defaults!")
                    st.rerun()
            with col_btn2:
                if st.button("💾 Export Preferences"):
                    prefs_data = _preferences_manager.get_all_preferences()
                    st.download_button(
                        label="📥 Download preferences.json",
                        data=_preferences_manager._save_to_string(),
                        file_name="aurora_preferences.json",
                        mime="application/json"
                    )
            
            st.success("✅ User preferences are being automatically saved")
            
        except Exception as e:
            st.warning(f"⚠️ Preferences manager error: {e}")
    else:
        st.warning("⚠️ User Preferences not available - settings won't be saved")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dependencies:**")
        # Check key dependencies
        deps = {
            "Streamlit": st,
            "Torch": None,
            "Whisper": None,
            "Ollama": None
        }
        
        for dep, module in deps.items():
            try:
                if dep == "Torch":
                    import torch
                    version = torch.__version__
                    st.success(f"✅ {dep} (v{version})")
                elif dep == "Whisper":
                    import whisper
                    st.success(f"✅ {dep}")
                elif dep == "Ollama":
                    import ollama
                    st.success(f"✅ {dep}")
                else:
                    st.success(f"✅ {dep}")
            except ImportError:
                st.error(f"❌ {dep} (not installed)")
            except Exception as e:
                st.error(f"❌ {dep} (error: {str(e)})")
    
    with col2:
        st.markdown("**Services:**")
        # Check services
        services = ["Ollama Server", "GPU Support", "Audio Devices"]
        
        # Ollama check
        try:
            subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            st.success("✅ Ollama Server")
        except:
            st.error("❌ Ollama Server")
        
        # GPU check
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                st.success(f"✅ GPU Support ({gpu_count} GPU(s) - {gpu_name})")
            else:
                st.warning("⚠️ GPU Support (CUDA not available - CPU only)")
        except ImportError:
            st.error("❌ GPU Support (PyTorch not installed)")
        except Exception as e:
            st.error(f"❌ GPU Support (Error: {str(e)})")
        
        # Audio check
        try:
            if AUDIO_AVAILABLE:
                devices = sd.query_devices()
                input_devices = [d for d in devices if d['max_input_channels'] > 0]
                if input_devices:
                    st.success("✅ Audio Devices")
                else:
                    st.warning("⚠️ Audio Devices")
            else:
                st.warning("⚠️ Audio Devices (not available on this platform)")
        except:
            st.error("❌ Audio Devices")
        
        # TTS Engine check
        try:
            from offline_text2speech import tts_manager
            engines = tts_manager.get_available_engines()
            if engines:
                engine_str = ", ".join(engines)
                st.success(f"✅ TTS Engines ({engine_str})")
            else:
                st.error("❌ No TTS Engines")
        except Exception as e:
            st.error(f"❌ TTS Engines (Error: {str(e)})")
    
    # GPU Troubleshooting section
    with st.expander("🔧 GPU Troubleshooting"):
        st.markdown("""
        ### GPU Issues and Solutions
        
        **Common GPU Problems:**
        
        1. **PyTorch not installed:**
           ```bash
           pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
           ```
        
        2. **CUDA not available:**
           - Check if you have an NVIDIA GPU
           - Install CUDA toolkit from NVIDIA website
           - Restart your computer after installation
        
        3. **GPU detected but not working:**
           - Check GPU driver version
           - Verify CUDA version compatibility
           - Try reinstalling PyTorch with CUDA support
        
        **Check GPU Status:**
        """)
        
        # GPU diagnostic button
        if st.button("🔍 Run GPU Diagnostics"):
            st.markdown("**GPU Diagnostic Results:**")
            
            # Check NVIDIA GPU
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ NVIDIA GPU detected")
                    st.code(result.stdout)
                else:
                    st.error("❌ nvidia-smi command failed")
            except FileNotFoundError:
                st.error("❌ nvidia-smi not found (NVIDIA drivers not installed)")
            except Exception as e:
                st.error(f"❌ Error running nvidia-smi: {e}")
            
            # Check PyTorch CUDA
            try:
                import torch
                st.info(f"PyTorch version: {torch.__version__}")
                st.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    st.info(f"CUDA version: {torch.version.cuda}")
                    st.info(f"GPU count: {torch.cuda.device_count()}")
                    for i in range(torch.cuda.device_count()):
                        st.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                else:
                    st.warning("PyTorch CUDA not available")
            except ImportError:
                st.error("PyTorch not installed")
            except Exception as e:
                st.error(f"Error checking PyTorch: {e}")
        
        # GPU Performance Test
        st.markdown("---")
        if st.button("🚀 Test GPU Performance"):
            st.markdown("**GPU Performance Test:**")
            with st.spinner("Running GPU performance test..."):
                success, message = test_gpu_acceleration()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
        
        st.markdown("""
        ### Installation Commands
        
        **For Windows with CUDA 11.8:**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
        
        **For CPU only:**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ```
        
        **Check installed version:**
        ```python
        import torch
        print(torch.__version__)
        print(torch.cuda.is_available())
        ```
        """)
    
    # Bark TTS Troubleshooting section
    with st.expander("🎵 Bark TTS Troubleshooting"):
        st.markdown("""
        ### Bark TTS Issues and Solutions
        
        **Common Bark Problems:**
        
        1. **Bark not installed:**
           ```bash
           pip install git+https://github.com/suno-ai/bark.git
           ```
        
        2. **Missing dependencies:**
           ```bash
           pip install scipy numpy soundfile
           ```
        
        3. **PyTorch weights loading error:**
           - This is automatically handled in the code
           - If issues persist, try reinstalling PyTorch
        
        4. **CUDA out of memory:**
           - Bark requires significant GPU memory
           - Try shorter text inputs
           - Consider using CPU-only mode
        
        **Check Bark Status:**
        """)
        
        # Bark diagnostic button
        if st.button("🔍 Run Bark Diagnostics"):
            st.markdown("**Bark TTS Diagnostic Results:**")
            
            # Check Bark installation
            try:
                import bark
                st.success("✅ Bark package installed")
                
                # Check Bark components
                try:
                    from bark import SAMPLE_RATE, generate_audio, preload_models
                    st.success("✅ Bark components importable")
                    st.info(f"Sample rate: {SAMPLE_RATE}")
                except Exception as e:
                    st.error(f"❌ Error importing Bark components: {e}")
                
                # Check dependencies
                deps = ["scipy", "numpy", "soundfile"]
                for dep in deps:
                    try:
                        __import__(dep)
                        st.success(f"✅ {dep} available")
                    except ImportError:
                        st.error(f"❌ {dep} not installed")
                
                # Check GPU memory for Bark
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        st.info(f"GPU Memory: {gpu_memory:.1f} GB")
                        if gpu_memory < 4:
                            st.warning("⚠️ Less than 4GB GPU memory - Bark may be slow")
                        else:
                            st.success(f"✅ Sufficient GPU memory for Bark")
                    else:
                        st.warning("⚠️ No CUDA GPU - Bark will use CPU (slower)")
                except Exception as e:
                    st.error(f"❌ Error checking GPU: {e}")
                    
            except ImportError:
                st.error("❌ Bark not installed")
                st.info("Install with: pip install git+https://github.com/suno-ai/bark.git")
            except Exception as e:
                st.error(f"❌ Error checking Bark: {e}")
        
        # Bark Performance Test
        st.markdown("---")
        if st.button("🚀 Test Bark TTS"):
            st.markdown("**Bark TTS Test:**")
            with st.spinner("Testing Bark TTS (this may take a while)..."):
                try:
                    from offline_text2speech import tts_manager
                    test_text = "Hello, this is a test of Bark TTS."
                    
                    if tts_manager.bark_available:
                        try:
                            result = tts_manager._speak_with_bark(test_text, return_path=True)
                            if result:
                                st.success("✅ Bark TTS test successful!")
                                st.audio(result)
                                # Clean up
                                try:
                                    os.remove(result)
                                except:
                                    pass
                            else:
                                st.error("❌ Bark TTS test failed")
                        except Exception as e:
                            st.error(f"❌ Bark TTS test error: {e}")
                    else:
                        st.error("❌ Bark TTS not available")
                except Exception as e:
                    st.error(f"❌ Error testing Bark: {e}")
        
        st.markdown("""
        ### Installation Commands
        
        **Install Bark TTS:**
        ```bash
        # Install Bark from GitHub
        pip install git+https://github.com/suno-ai/bark.git
        
        # Install required dependencies
        pip install scipy numpy soundfile
        
        # For GPU support (recommended)
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
        
        **Voice Presets Available:**
        - `v2/en_speaker_0` through `v2/en_speaker_9` (English)
        - `v2/de_speaker_0` through `v2/de_speaker_3` (German)
        - `v2/es_speaker_0` through `v2/es_speaker_9` (Spanish)
        - `v2/fr_speaker_0` through `v2/fr_speaker_9` (French)
        - `v2/hi_speaker_0` through `v2/hi_speaker_9` (Hindi)
        - `v2/it_speaker_0` through `v2/it_speaker_9` (Italian)
        - `v2/ja_speaker_0` through `v2/ja_speaker_8` (Japanese)
        - `v2/ko_speaker_0` through `v2/ko_speaker_8` (Korean)
        - `v2/pl_speaker_0` through `v2/pl_speaker_9` (Polish)
        - `v2/pt_speaker_0` through `v2/pt_speaker_9` (Portuguese)
        - `v2/ru_speaker_0` through `v2/ru_speaker_9` (Russian)
        - `v2/tr_speaker_0` through `v2/tr_speaker_9` (Turkish)
        - `v2/zh_speaker_0` through `v2/zh_speaker_9` (Chinese)
        
        **Tips for better Bark performance:**
        - Keep text under 200 characters for best results
        - Use punctuation to control pacing
        - GPU with 4GB+ VRAM recommended
        - First generation takes longer (model loading)
        """)

# Patch tqdm to do nothing in Streamlit
class StreamlitTqdm:
    def __init__(self, iterable=None, desc=None, total=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []
        self.desc = desc or ""
        self.total = total or len(self.iterable)
        try:
            self.progress = st.progress(0.0, text="...generating speech")
        except:
            self.progress = None
        self.n = 0
        self._iter = iter(self.iterable)
    
    @property
    def progress_percent(self):
        return ((self.n / self.total) * 100) if self.total else 0
    
    def update(self, n=1):
        self.n += n
        if self.progress:
            # Ensure percentage never exceeds 100 to avoid Streamlit progress bar error
            percent = min(self.progress_percent, 100)
            # Convert to 0-1 range for st.progress (it expects values between 0.0 and 1.0)
            progress_value = percent / 100.0
            progress_value = max(0.0, min(progress_value, 1.0))  # Clamp to [0.0, 1.0]
            self.progress.progress(progress_value, text="...generating speech")
    
    def close(self):
        if self.progress:
            self.progress.empty()
    
    def refresh(self):
        """Refresh method for tqdm compatibility - no-op in Streamlit"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update()
        self.close()
    
    def __next__(self):
        item = next(self._iter)
        self.update()
        return item

# Apply tqdm patch only when in Streamlit context
def apply_tqdm_patch():
    """Apply tqdm patch for Streamlit compatibility"""
    try:
        import tqdm
        tqdm.tqdm = StreamlitTqdm
    except Exception:
        pass

# Main application
def main():
    """Main application with navbar navigation"""
    # Configure Streamlit first
    configure_streamlit()
    # Inject CSS
    inject_css()
    # Apply tqdm patch
    apply_tqdm_patch()
    
    # Create navigation items
    nav_items = [
        create_nav_item("chat", "Chat", "💬"),
        create_nav_item("image", "Image Generation", "🎨"), 
        create_nav_item("video", "Video Generation", "🎬"),
        create_nav_item("rag", "Knowledge Base", "📚"),
        create_nav_item("about", "About", "ℹ️")
    ]
    
    # Load logo image
    logo_image = load_logo_as_base64()
    
    # Navigation bar at the top
    selected_page = navbar(
        items=nav_items,
        key="main_navbar",
        logo_image=logo_image,
        logo_text="Aurora",
        selected=st.session_state.get("current_page", "chat")
    )
    
    # Store current page in session state
    if selected_page:
        st.session_state["current_page"] = selected_page
    
    # Use stored page or default to chat
    current_page = st.session_state.get("current_page", "chat")
    
    # Add some spacing after navbar for better layout
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # Sidebar with logo and quick actions
    # Display logo in sidebar
    logo_path = os.path.join(BASE_DIR, '_assets', 'aurora_logo.jpg')
    if os.path.exists(logo_path):
        try:
            from PIL import Image
            logo_img = Image.open(logo_path)
            # Create columns for logo and title
            col1, col2 = st.sidebar.columns([1, 2])
            with col1:
                st.image(logo_img, width=60)
            with col2:
                st.markdown("## Aurora")
        except Exception:
            st.sidebar.title("🌅 Aurora")
    else:
        st.sidebar.title("🌅 Aurora")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Quick Actions")
    
    # Quick actions in sidebar
    if st.sidebar.button("🗑️ Clear Chat History"):
        if "messages" in st.session_state:
            st.session_state.messages = [{
                'role': 'assistant',
                'content': 'Hi! I am Aurora, an intelligent AI assistant. How can I help you today?'
            }]
            st.rerun()
    
    if st.sidebar.button("📝 View Logs"):
        log_path = os.path.join(config.LOGS_DIR, "aurora.log")
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            st.sidebar.text_area("Recent Logs", log_content[-1000:], height=200)
    
    # Route to appropriate page
    if current_page == "chat":
        chat_page()
    elif current_page == "image":
        image_generation_page()
    elif current_page == "video":
        video_generation_page()
    elif current_page == "rag":
        rag_knowledge_base_page()
    elif current_page == "about":
        about_page()

# Run the application
if __name__ == "__main__":
    # Run main app
    main()
else:
    # For streamlit run, call main directly
    try:
        main()
    except Exception as e:
        # If not in Streamlit context, just pass
        pass
