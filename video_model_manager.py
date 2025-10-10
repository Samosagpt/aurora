"""
Video Generation Manager for Aurora with Hugging Face integration
"""
import os
import logging
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download, repo_exists, list_repo_files
from logmanagement import log_manager

logger = logging.getLogger(__name__)

class VideoModelManager:
    """Enhanced video model management with Hugging Face integration"""
    
    def __init__(self):
        self.cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        self.available_models = self._get_default_models()
        self.installed_models = self._get_installed_models()
        self.current_model = None
        
    def _get_default_models(self) -> Dict[str, str]:
        """Get default available models"""
        return {
            "Zeroscope v2 576w": "cerspense/zeroscope_v2_576w",
            "ModelScope T2V": "damo-vilab/text-to-video-ms-1.7b",
            "Text-to-Video Zero": "text-to-video-zero",
            "Stable Video Diffusion": "stabilityai/stable-video-diffusion-img2vid",
            "VideoCrafter2": "VideoCrafter/VideoCrafter2",
            "LaVie": "vchitect/LaVie",
            "Show-1": "showlab/show-1-base",
            "CogVideoX": "THUDM/CogVideoX-2b"
        }
    
    def _get_installed_models(self) -> List[str]:
        """Get list of locally installed models"""
        installed = []
        
        if not self.cache_dir.exists():
            return installed
            
        try:
            # Check for cached models in huggingface hub cache
            for model_name, model_id in self.available_models.items():
                model_path = self._get_model_cache_path(model_id)
                if model_path and model_path.exists():
                    installed.append(model_name)
                    
        except Exception as e:
            logger.error(f"Error checking installed models: {e}")
            
        return installed
    
    def _get_model_cache_path(self, model_id: str) -> Optional[Path]:
        """Get the cache path for a model"""
        try:
            # Convert model ID to cache directory format
            model_cache_name = f"models--{model_id.replace('/', '--')}"
            model_path = self.cache_dir / model_cache_name
            return model_path
        except Exception as e:
            logger.error(f"Error getting model cache path for {model_id}: {e}")
            return None
    
    def get_available_models(self) -> Dict[str, str]:
        """Get available models dictionary"""
        return self.available_models
    
    def get_installed_models(self) -> List[str]:
        """Get list of installed model names"""
        return self._get_installed_models()
    
    def is_model_installed(self, model_name: str) -> bool:
        """Check if a model is installed locally"""
        if model_name not in self.available_models:
            return False
            
        model_id = self.available_models[model_name]
        model_path = self._get_model_cache_path(model_id)
        return model_path and model_path.exists()
    
    def install_model(self, model_name: str) -> bool:
        """Install a model from Hugging Face"""
        try:
            if model_name not in self.available_models:
                logger.error(f"Model {model_name} not found in available models")
                return False
                
            model_id = self.available_models[model_name]
            
            logger.info(f"Installing model: {model_name} ({model_id})")
            log_manager.append_md_log('Video Model Install', f"Starting installation of {model_name}")
            
            # Check if model exists on Hugging Face
            if not repo_exists(model_id):
                logger.error(f"Model {model_id} does not exist on Hugging Face")
                return False
            
            # This will download the model to the cache
            from diffusers import TextToVideoSDPipeline, DiffusionPipeline
            
            # Try to load as TextToVideoSDPipeline first, fallback to DiffusionPipeline
            try:
                if "zeroscope" in model_id.lower():
                    pipeline = TextToVideoSDPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True,
                        cache_dir=str(self.cache_dir)
                    )
                else:
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True,
                        cache_dir=str(self.cache_dir)
                    )
            except Exception:
                # Fallback to generic DiffusionPipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    cache_dir=str(self.cache_dir)
                )
            
            # Clean up pipeline from memory
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            log_manager.append_md_log('Video Model Install', f"Successfully installed {model_name}")
            logger.info(f"Model {model_name} installed successfully")
            
            # Refresh installed models list
            self.installed_models = self._get_installed_models()
            return True
            
        except Exception as e:
            error_msg = f"Error installing model {model_name}: {str(e)}"
            logger.error(error_msg)
            log_manager.append_md_log('Video Model Install Error', error_msg)
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from local cache"""
        try:
            if model_name not in self.available_models:
                logger.error(f"Model {model_name} not found in available models")
                return False
                
            model_id = self.available_models[model_name]
            model_path = self._get_model_cache_path(model_id)
            
            if not model_path or not model_path.exists():
                logger.warning(f"Model {model_name} not found in cache")
                return False
                
            logger.info(f"Deleting model: {model_name}")
            log_manager.append_md_log('Video Model Delete', f"Starting deletion of {model_name}")
            
            # Remove the model directory
            shutil.rmtree(model_path)
            
            log_manager.append_md_log('Video Model Delete', f"Successfully deleted {model_name}")
            logger.info(f"Model {model_name} deleted successfully")
            
            # Refresh installed models list
            self.installed_models = self._get_installed_models()
            return True
            
        except Exception as e:
            error_msg = f"Error deleting model {model_name}: {str(e)}"
            logger.error(error_msg)
            log_manager.append_md_log('Video Model Delete Error', error_msg)
            return False
    
    def get_model_size(self, model_name: str) -> Optional[str]:
        """Get the size of a model (if installed)"""
        try:
            if model_name not in self.available_models:
                return None
                
            model_id = self.available_models[model_name]
            model_path = self._get_model_cache_path(model_id)
            
            if not model_path or not model_path.exists():
                return None
                
            # Calculate directory size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(model_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            # Convert to human readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.1f} TB"
            
        except Exception as e:
            logger.error(f"Error getting model size for {model_name}: {e}")
            return None
    
    def add_custom_model(self, model_name: str, model_id: str) -> bool:
        """Add a custom model to the available models"""
        try:
            # Check if model exists on Hugging Face
            if not repo_exists(model_id):
                logger.error(f"Model {model_id} does not exist on Hugging Face")
                return False
            
            # Add to available models
            self.available_models[model_name] = model_id
            
            log_manager.append_md_log('Custom Video Model Added', f"Added {model_name} ({model_id})")
            logger.info(f"Custom model {model_name} added successfully")
            return True
            
        except Exception as e:
            error_msg = f"Error adding custom model {model_name}: {str(e)}"
            logger.error(error_msg)
            log_manager.append_md_log('Custom Video Model Error', error_msg)
            return False

# Create global instance
video_model_manager = VideoModelManager()
