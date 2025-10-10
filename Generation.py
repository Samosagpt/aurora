"""
Enhanced generation module for Samosa GPT with Ollama integration
"""
import ollama
import logging
import re
from typing import Dict, List, Any, Optional
from config import config
from logmanagement import log_manager

logger = logging.getLogger(__name__)

class OllamaManager:
    """Enhanced Ollama integration with error handling and model management"""
    
    def __init__(self):
        self.current_model = None
        self.available_models = []
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Ollama client"""
        try:
            self.client = ollama.Client(host=config.OLLAMA_HOST)
            logger.info(f"Ollama client initialized with host: {config.OLLAMA_HOST}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            self.client = None
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            if not self.client:
                self._initialize_client()
            
            if not self.client:
                return [config.DEFAULT_OLLAMA_MODEL]
            
            models_response = self.client.list()
            models = []
            
            for model in models_response.get('models', []):
                model_name = model.get('name', '').split(':')[0]
                if model_name and model_name not in models:
                    models.append(model_name)
            
            if not models:
                models = [config.DEFAULT_OLLAMA_MODEL]
            
            self.available_models = sorted(models)
            logger.info(f"Found {len(self.available_models)} Ollama models")
            return self.available_models
            
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
            return [config.DEFAULT_OLLAMA_MODEL]
    
    def set_model(self, model_name: str) -> bool:
        """Set the current model"""
        try:
            if model_name in self.get_available_models():
                self.current_model = model_name
                log_manager.append_md_log('Model Changed', model_name)
                logger.info(f"Model set to: {model_name}")
                return True
            else:
                logger.warning(f"Model not found: {model_name}")
                return False
        except Exception as e:
            logger.error(f"Error setting model: {e}")
            return False
    
    def chat_with_memory(self, user_input: str, model_name: str = None, images: List[str] = None) -> str:
        """Enhanced chat with memory and error handling, supports multimodal input
        
        Args:
            user_input: Text prompt from user
            model_name: Ollama model to use
            images: List of base64 encoded images for multimodal models
        """
        try:
            # Use provided model or current model or default
            model_to_use = model_name or self.current_model or config.DEFAULT_OLLAMA_MODEL
            
            if not self.client:
                self._initialize_client()
                if not self.client:
                    return "Error: Could not connect to Ollama. Please ensure Ollama is running."
            
            logger.info(f"Using Ollama model: {model_to_use}")
            log_manager.append_md_log('Model', model_to_use)
            
            # Get conversation history
            history = log_manager.read_json_history()
            
            # Add user input to history
            user_message = {'role': 'user', 'content': user_input}
            
            # Add images if provided (for multimodal models)
            if images and len(images) > 0:
                user_message['images'] = images
                logger.info(f"Sending {len(images)} images with prompt")
            
            history.append(user_message)
            log_manager.append_md_log('User', user_input)
            
            # Generate response
            response = self.client.chat(model=model_to_use, messages=history)
            
            if not response or 'message' not in response:
                raise Exception("Invalid response from Ollama")
            
            raw_msg = response['message']['content']
            
            # Process <think> tags
            processed_msg = self._process_think_tags(raw_msg)
            
            # Add assistant response to history (without think tags for context)
            clean_msg = re.sub(r'<think>.*?</think>', '', raw_msg, flags=re.DOTALL).strip()
            history.append({'role': 'assistant', 'content': clean_msg})
            log_manager.write_json_history(history)
            log_manager.append_md_log('Assistant', processed_msg)
            
            return processed_msg
            
        except Exception as e:
            error_msg = f"Error during chat: {str(e)}"
            logger.error(error_msg)
            log_manager.append_md_log('Error', error_msg)
            
            # Provide helpful error messages
            if "connection" in str(e).lower():
                return "Cannot connect to Ollama. Please ensure Ollama is running and try again."
            elif "model" in str(e).lower():
                return f"Model '{model_to_use}' not found. Please check if the model is installed."
            else:
                return f"An error occurred while processing your request: {str(e)}"
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            if not self.client:
                self._initialize_client()
                if not self.client:
                    return False
            
            logger.info(f"Pulling model: {model_name}")
            log_manager.append_md_log('Model Pull', f"Starting download of {model_name}")
            
            # This is a blocking operation
            self.client.pull(model_name)
            
            log_manager.append_md_log('Model Pull', f"Successfully downloaded {model_name}")
            logger.info(f"Model {model_name} pulled successfully")
            
            # Refresh available models
            self.get_available_models()
            return True
            
        except Exception as e:
            error_msg = f"Error pulling model {model_name}: {str(e)}"
            logger.error(error_msg)
            log_manager.append_md_log('Model Pull Error', error_msg)
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama"""
        try:
            if not self.client:
                self._initialize_client()
                if not self.client:
                    return False
            
            logger.info(f"Deleting model: {model_name}")
            log_manager.append_md_log('Model Delete', f"Starting deletion of {model_name}")
            
            # Use the ollama client to delete the model
            self.client.delete(model_name)
            
            log_manager.append_md_log('Model Delete', f"Successfully deleted {model_name}")
            logger.info(f"Model {model_name} deleted successfully")
            
            # Refresh available models
            self.get_available_models()
            return True
            
        except Exception as e:
            error_msg = f"Error deleting model {model_name}: {str(e)}"
            logger.error(error_msg)
            log_manager.append_md_log('Model Delete Error', error_msg)
            return False

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        try:
            if not self.client:
                return None
            
            response = self.client.show(model_name)
            return response
            
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return None
    
    def _process_think_tags(self, text: str) -> str:
        """Process <think> tags and convert them to dropdown format"""
        def replace_think(match):
            think_content = match.group(1).strip()
            # Use plain text for better Windows compatibility
            return f'<details><summary>💭 Thinking...</summary>\n\n{think_content}\n\n</details>'
        
        # Replace <think>...</think> with dropdown
        processed = re.sub(r'<think>(.*?)</think>', replace_think, text, flags=re.DOTALL)
        return processed
    
    def chat_with_memory_stream(self, user_input: str, model_name: str = None, images: List[str] = None):
        """Enhanced chat with memory and streaming support, supports multimodal input
        
        Args:
            user_input: Text prompt from user
            model_name: Ollama model to use
            images: List of base64 encoded images for multimodal models
        """
        try:
            # Use provided model or current model or default
            model_to_use = model_name or self.current_model or config.DEFAULT_OLLAMA_MODEL
            
            if not self.client:
                self._initialize_client()
                if not self.client:
                    yield "Error: Could not connect to Ollama. Please ensure Ollama is running."
                    return
            
            logger.info(f"Using Ollama model: {model_to_use}")
            log_manager.append_md_log('Model', model_to_use)
            
            # Get conversation history
            history = log_manager.read_json_history()
            
            # Add user input to history
            user_message = {'role': 'user', 'content': user_input}
            
            # Add images if provided (for multimodal models)
            if images and len(images) > 0:
                user_message['images'] = images
                logger.info(f"Sending {len(images)} images with prompt")
            
            history.append(user_message)
            log_manager.append_md_log('User', user_input)
            
            # Generate streaming response
            full_response = ""
            for chunk in self.client.chat(model=model_to_use, messages=history, stream=True):
                if chunk and 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    full_response += content
                    yield content
            
            # Process <think> tags for the full response
            processed_msg = self._process_think_tags(full_response)
            
            # Add assistant response to history (without think tags for context)
            clean_msg = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
            history.append({'role': 'assistant', 'content': clean_msg})
            log_manager.write_json_history(history)
            log_manager.append_md_log('Assistant', processed_msg)
            
        except Exception as e:
            error_msg = f"Error during streaming chat: {str(e)}"
            logger.error(error_msg)
            log_manager.append_md_log('Error', error_msg)
            yield error_msg

# Create global instance
ollama_manager = OllamaManager()

# Backward compatibility functions
def chat_with_memory(user_input: str) -> str:
    """Backward compatibility function"""
    # Try to get model from streamlit_app if available
    try:
        import streamlit_app
        model = getattr(streamlit_app, 'model', None)
        return ollama_manager.chat_with_memory(user_input, model)
    except ImportError:
        return ollama_manager.chat_with_memory(user_input)

def init_model(model_name: str = None) -> bool:
    """Initialize model - backward compatibility"""
    model_to_init = model_name or config.DEFAULT_OLLAMA_MODEL
    return ollama_manager.set_model(model_to_init)