"""
Hardware Optimizer for Samosa GPT
Detects and caches hardware capabilities to optimize settings for image generation, video generation, and chat.
"""

import json
import logging
import os
import platform
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

# Import optional dependencies
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class HardwareOptimizer:
    def __init__(self, cache_dir: str = None):
        """Initialize hardware optimizer with cache directory"""
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "hardware_cache.json")
        self.hardware_info = None
        self.optimized_settings = None

        # Cache expiry time (24 hours)
        self.cache_expiry = 24 * 60 * 60

        # Load or detect hardware
        self._load_or_detect_hardware()

    def _load_or_detect_hardware(self) -> None:
        """Load hardware info from cache or detect if cache is outdated"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    cached_data = json.load(f)

                # Check if cache is still valid
                cache_time = cached_data.get("timestamp", 0)
                if time.time() - cache_time < self.cache_expiry:
                    print("Loading hardware info from cache...")
                    self.hardware_info = cached_data.get("hardware_info", {})
                    self.optimized_settings = cached_data.get("optimized_settings", {})
                    return
        except Exception as e:
            print(f"Error loading cache: {e}")

        # Detect hardware and optimize settings
        print("Detecting hardware capabilities...")
        self._detect_hardware()
        self._optimize_settings()
        self._save_cache()

    def _detect_hardware(self) -> None:
        """Detect system hardware capabilities"""
        self.hardware_info = {
            "system": self._detect_system(),
            "cpu": self._detect_cpu(),
            "memory": self._detect_memory(),
            "gpu": self._detect_gpu(),
            "storage": self._detect_storage(),
            "detection_time": time.time(),
        }

    def _detect_system(self) -> Dict[str, Any]:
        """Detect system information"""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information"""
        cpu_info = {
            "cores_physical": psutil.cpu_count(logical=False),
            "cores_logical": psutil.cpu_count(logical=True),
        }

        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                cpu_info["frequency_max"] = cpu_freq.max
                cpu_info["frequency_current"] = cpu_freq.current
        except:
            pass

        return cpu_info

    def _detect_memory(self) -> Dict[str, Any]:
        """Detect memory information"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent,
        }

    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information"""
        gpu_info = {"cuda_available": False, "gpu_count": 0, "gpus": [], "total_vram_gb": 0}

        if TORCH_AVAILABLE:
            gpu_info["cuda_available"] = torch.cuda.is_available()
            gpu_info["torch_version"] = torch.__version__

            if torch.cuda.is_available():
                gpu_info["gpu_count"] = torch.cuda.device_count()
                gpu_info["cuda_version"] = torch.version.cuda

                total_vram = 0
                for i in range(gpu_info["gpu_count"]):
                    props = torch.cuda.get_device_properties(i)
                    vram_gb = props.total_memory / (1024**3)
                    total_vram += vram_gb

                    gpu_info["gpus"].append(
                        {
                            "id": i,
                            "name": props.name,
                            "vram_gb": round(vram_gb, 2),
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                    )

                gpu_info["total_vram_gb"] = total_vram

        return gpu_info

    def _detect_storage(self) -> Dict[str, Any]:
        """Detect storage information"""
        try:
            disk_usage = psutil.disk_usage(".")
            return {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "used_percent": round((disk_usage.used / disk_usage.total) * 100, 2),
            }
        except:
            return {"total_gb": 0, "free_gb": 0, "used_percent": 0}

    def _optimize_settings(self) -> None:
        """Optimize settings based on detected hardware"""
        self.optimized_settings = {
            "chat": self._optimize_chat_settings(),
            "image_generation": self._optimize_image_settings(),
            "video_generation": self._optimize_video_settings(),
            "general": self._optimize_general_settings(),
        }

    def _optimize_chat_settings(self) -> Dict[str, Any]:
        """Optimize chat settings based on hardware"""
        cpu_cores = self.hardware_info.get("cpu", {}).get("cores_logical", 4)
        memory_gb = self.hardware_info.get("memory", {}).get("total_gb", 8)
        gpu_available = self.hardware_info.get("gpu", {}).get("cuda_available", False)

        # Recommended models based on memory and GPU
        if memory_gb >= 32 and gpu_available:
            recommended_models = ["llama3.1:70b", "codellama:34b", "samosagpt", "llama3.1:8b"]
            context_length = 8192
        elif memory_gb >= 16 and gpu_available:
            recommended_models = ["llama3.1:8b", "mistral:7b", "codellama:13b", "samosagpt"]
            context_length = 4096
        elif memory_gb >= 8:
            recommended_models = ["llama3.1:8b", "mistral:7b", "samosagpt"]
            context_length = 2048
        else:
            recommended_models = ["samosagpt", "tinyllama"]
            context_length = 1024

        return {
            "recommended_models": recommended_models,
            "context_length": context_length,
            "batch_size": 1,
            "num_threads": min(cpu_cores // 2, 8),
            "use_gpu": gpu_available,
            "streaming_enabled": True,
        }

    def _optimize_image_settings(self) -> Dict[str, Any]:
        """Optimize image generation settings based on hardware"""
        gpu_info = self.hardware_info.get("gpu", {})
        vram_gb = gpu_info.get("total_vram_gb", 0)
        gpu_available = gpu_info.get("cuda_available", False)
        memory_gb = self.hardware_info.get("memory", {}).get("total_gb", 8)

        if not gpu_available:
            # CPU-only settings
            return {
                "default_num_images": 1,
                "max_images": 2,
                "recommended_resolution": [512, 512],
                "max_resolution": [768, 768],
                "inference_steps": 20,
                "guidance_scale": 7.5,
                "use_gpu": False,
                "enable_attention_slicing": True,
                "enable_cpu_offload": True,
                "batch_size": 1,
            }

        # GPU-based optimization
        if vram_gb >= 24:  # High-end GPU (RTX 4090, etc.)
            return {
                "default_num_images": 6,
                "max_images": 20,
                "recommended_resolution": [1024, 1024],
                "max_resolution": [1536, 1536],
                "inference_steps": 50,
                "guidance_scale": 7.5,
                "use_gpu": True,
                "enable_attention_slicing": False,
                "enable_cpu_offload": False,
                "batch_size": 4,
            }
        elif vram_gb >= 16:  # High-end GPU (RTX 4080, 5070 Ti)
            return {
                "default_num_images": 4,
                "max_images": 20,
                "recommended_resolution": [768, 768],
                "max_resolution": [1024, 1024],
                "inference_steps": 30,
                "guidance_scale": 7.5,
                "use_gpu": True,
                "enable_attention_slicing": False,
                "enable_cpu_offload": False,
                "batch_size": 2,
            }
        elif vram_gb >= 12:  # Mid-high GPU (RTX 4070 Ti, 3080)
            return {
                "default_num_images": 3,
                "max_images": 15,
                "recommended_resolution": [768, 768],
                "max_resolution": [1024, 1024],
                "inference_steps": 25,
                "guidance_scale": 7.5,
                "use_gpu": True,
                "enable_attention_slicing": True,
                "enable_cpu_offload": False,
                "batch_size": 2,
            }
        elif vram_gb >= 8:  # Mid-range GPU (RTX 4060 Ti, 3070)
            return {
                "default_num_images": 2,
                "max_images": 10,
                "recommended_resolution": [512, 512],
                "max_resolution": [768, 768],
                "inference_steps": 20,
                "guidance_scale": 7.5,
                "use_gpu": True,
                "enable_attention_slicing": True,
                "enable_cpu_offload": False,
                "batch_size": 1,
            }
        else:  # Low VRAM GPU
            return {
                "default_num_images": 1,
                "max_images": 5,
                "recommended_resolution": [512, 512],
                "max_resolution": [512, 512],
                "inference_steps": 15,
                "guidance_scale": 7.5,
                "use_gpu": True,
                "enable_attention_slicing": True,
                "enable_cpu_offload": True,
                "batch_size": 1,
            }

    def _optimize_video_settings(self) -> Dict[str, Any]:
        """Optimize video generation settings based on hardware"""
        gpu_info = self.hardware_info.get("gpu", {})
        vram_gb = gpu_info.get("total_vram_gb", 0)
        gpu_available = gpu_info.get("cuda_available", False)

        if not gpu_available:
            # CPU-only settings (not recommended for video)
            return {
                "recommended_frames": 8,
                "max_frames": 16,
                "recommended_fps": 4,
                "max_fps": 8,
                "recommended_resolution": [320, 576],
                "max_resolution": [320, 576],
                "inference_steps": 10,
                "guidance_scale": 9.0,
                "use_gpu": False,
                "enable_attention_slicing": True,
                "enable_cpu_offload": True,
                "recommended_duration": 2.0,
            }

        # GPU-based optimization
        if vram_gb >= 24:  # High-end GPU
            return {
                "recommended_frames": 64,
                "max_frames": 128,
                "recommended_fps": 24,
                "max_fps": 30,
                "recommended_resolution": [768, 1024],
                "max_resolution": [1024, 1024],
                "inference_steps": 30,
                "guidance_scale": 9.0,
                "use_gpu": True,
                "enable_attention_slicing": False,
                "enable_cpu_offload": False,
                "recommended_duration": 4.0,
            }
        elif vram_gb >= 16:  # High-end GPU (RTX 5070 Ti)
            return {
                "recommended_frames": 24,
                "max_frames": 48,
                "recommended_fps": 10,
                "max_fps": 30,
                "recommended_resolution": [320, 576],
                "max_resolution": [576, 1024],
                "inference_steps": 20,
                "guidance_scale": 9.0,
                "use_gpu": True,
                "enable_attention_slicing": True,
                "enable_cpu_offload": False,
                "recommended_duration": 3.0,
            }
        elif vram_gb >= 12:  # Mid-high GPU
            return {
                "recommended_frames": 16,
                "max_frames": 32,
                "recommended_fps": 8,
                "max_fps": 24,
                "recommended_resolution": [320, 576],
                "max_resolution": [576, 768],
                "inference_steps": 15,
                "guidance_scale": 9.0,
                "use_gpu": True,
                "enable_attention_slicing": True,
                "enable_cpu_offload": False,
                "recommended_duration": 2.5,
            }
        elif vram_gb >= 8:  # Mid-range GPU
            return {
                "recommended_frames": 12,
                "max_frames": 24,
                "recommended_fps": 6,
                "max_fps": 16,
                "recommended_resolution": [320, 576],
                "max_resolution": [320, 768],
                "inference_steps": 12,
                "guidance_scale": 9.0,
                "use_gpu": True,
                "enable_attention_slicing": True,
                "enable_cpu_offload": False,
                "recommended_duration": 2.0,
            }
        else:  # Low VRAM GPU
            return {
                "recommended_frames": 8,
                "max_frames": 16,
                "recommended_fps": 4,
                "max_fps": 12,
                "recommended_resolution": [320, 576],
                "max_resolution": [320, 576],
                "inference_steps": 10,
                "guidance_scale": 9.0,
                "use_gpu": True,
                "enable_attention_slicing": True,
                "enable_cpu_offload": True,
                "recommended_duration": 2.0,
            }

    def _optimize_general_settings(self) -> Dict[str, Any]:
        """Optimize general settings based on hardware"""
        cpu_cores = self.hardware_info.get("cpu", {}).get("cores_logical", 4)
        memory_gb = self.hardware_info.get("memory", {}).get("total_gb", 8)
        free_storage_gb = self.hardware_info.get("storage", {}).get("free_gb", 10)

        # Cache size based on available memory and storage
        cache_size_mb = min(
            int(memory_gb * 100),  # 100MB per GB of RAM
            int(free_storage_gb * 50),  # 50MB per GB of free storage
            2048,  # Max 2GB cache
        )

        # Performance mode based on overall capabilities
        vram_gb = self.hardware_info.get("gpu", {}).get("total_vram_gb", 0)
        if memory_gb >= 32 and vram_gb >= 16:
            performance_mode = "high_performance"
            max_concurrent_tasks = min(cpu_cores // 2, 4)
        elif memory_gb >= 16 and vram_gb >= 8:
            performance_mode = "balanced"
            max_concurrent_tasks = min(cpu_cores // 4, 3)
        else:
            performance_mode = "power_efficient"
            max_concurrent_tasks = 1

        return {
            "enable_caching": True,
            "cache_size_mb": max(cache_size_mb, 256),  # Minimum 256MB
            "max_concurrent_tasks": max_concurrent_tasks,
            "auto_cleanup": True,
            "performance_mode": performance_mode,
        }

    def _save_cache(self) -> None:
        """Save hardware info and optimized settings to cache"""
        try:
            cache_data = {
                "timestamp": time.time(),
                "hardware_info": self.hardware_info,
                "optimized_settings": self.optimized_settings,
            }

            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)

            print(f"Hardware cache saved to: {self.cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get detected hardware information"""
        return self.hardware_info or {}

    def get_optimized_settings(self, category: str = None) -> Dict[str, Any]:
        """Get optimized settings for a specific category or all categories"""
        if not self.optimized_settings:
            return {}

        if category:
            return self.optimized_settings.get(category, {})

        return self.optimized_settings

    def get_chat_settings(self) -> Dict[str, Any]:
        """Get optimized chat settings"""
        return self.get_optimized_settings("chat")

    def get_image_settings(self) -> Dict[str, Any]:
        """Get optimized image generation settings"""
        return self.get_optimized_settings("image_generation")

    def get_video_settings(self) -> Dict[str, Any]:
        """Get optimized video generation settings"""
        return self.get_optimized_settings("video_generation")

    def get_general_settings(self) -> Dict[str, Any]:
        """Get optimized general settings"""
        return self.get_optimized_settings("general")

    def force_refresh(self) -> None:
        """Force refresh of hardware detection and optimization"""
        print("Forcing hardware detection refresh...")
        self._detect_hardware()
        self._optimize_settings()
        self._save_cache()

    def get_performance_recommendation(self) -> str:
        """Get a human-readable performance recommendation"""
        if not self.hardware_info:
            return "Hardware detection failed"

        gpu_info = self.hardware_info.get("gpu", {})
        memory_gb = self.hardware_info.get("memory", {}).get("total_gb", 0)
        vram_gb = gpu_info.get("total_vram_gb", 0)

        if vram_gb >= 16 and memory_gb >= 32:
            return "🚀 Excellent - Perfect for high-quality image/video generation and large language models"
        elif vram_gb >= 12 and memory_gb >= 16:
            return "✅ Very Good - Great for most AI tasks with good quality settings"
        elif vram_gb >= 8 and memory_gb >= 8:
            return "👍 Good - Suitable for AI tasks with moderate quality settings"
        elif vram_gb >= 4 or memory_gb >= 8:
            return "⚠️ Fair - Can run AI tasks but with reduced quality/speed"
        else:
            return "❌ Limited - Consider upgrading hardware for better AI performance"

    def print_hardware_summary(self) -> None:
        """Print a summary of detected hardware and optimizations"""
        if not self.hardware_info:
            print("❌ No hardware information available")
            return

        print("\n" + "=" * 60)
        print("🖥️  HARDWARE SUMMARY")
        print("=" * 60)

        # System info
        system = self.hardware_info.get("system", {})
        print(
            f"🖥️  System: {system.get('platform', 'Unknown')} {system.get('platform_version', '')}"
        )
        print(f"🏗️  Architecture: {system.get('architecture', 'Unknown')}")
        print(f"🐍 Python: {system.get('python_version', 'Unknown')}")

        # CPU info
        cpu = self.hardware_info.get("cpu", {})
        print(
            f"\n💻 CPU: {cpu.get('cores_physical', 0)} cores / {cpu.get('cores_logical', 0)} threads"
        )
        if "frequency_max" in cpu:
            print(
                f"⚡ Frequency: {cpu.get('frequency_current', 0):.0f} MHz (max: {cpu.get('frequency_max', 0):.0f} MHz)"
            )

        # Memory info
        memory = self.hardware_info.get("memory", {})
        print(
            f"\n🧠 RAM: {memory.get('total_gb', 0):.1f} GB total, {memory.get('available_gb', 0):.1f} GB available ({memory.get('percent_used', 0):.1f}% used)"
        )

        # GPU info
        gpu = self.hardware_info.get("gpu", {})
        if gpu.get("cuda_available"):
            print(f"\n🎮 GPU: {gpu.get('gpu_count', 0)} CUDA device(s)")
            for gpu_device in gpu.get("gpus", []):
                print(
                    f"   📱 {gpu_device.get('name', 'Unknown')}: {gpu_device.get('vram_gb', 0):.1f} GB VRAM"
                )
            print(f"🔥 Total VRAM: {gpu.get('total_vram_gb', 0):.1f} GB")
            print(
                f"🔧 CUDA: {gpu.get('cuda_version', 'Unknown')} | PyTorch: {gpu.get('torch_version', 'Unknown')}"
            )
        else:
            print("\n❌ GPU: No CUDA-capable GPU detected")

        # Storage info
        storage = self.hardware_info.get("storage", {})
        print(
            f"\n💾 Storage: {storage.get('free_gb', 0):.1f} GB free / {storage.get('total_gb', 0):.1f} GB total"
        )

        # Performance recommendation
        print(f"\n📊 Performance: {self.get_performance_recommendation()}")

        # Optimized settings summary
        if self.optimized_settings:
            print(f"\n⚙️  OPTIMIZED SETTINGS")
            print("-" * 40)

            chat = self.optimized_settings.get("chat", {})
            image = self.optimized_settings.get("image_generation", {})
            video = self.optimized_settings.get("video_generation", {})

            print(
                f"💬 Chat: {len(chat.get('recommended_models', []))} models, {chat.get('context_length', 0)} context"
            )
            print(
                f"🎨 Images: {image.get('default_num_images', 0)} default, {image.get('max_images', 0)} max, {image.get('recommended_resolution', [0,0])[0]}x{image.get('recommended_resolution', [0,0])[1]} resolution"
            )
            print(
                f"🎬 Video: {video.get('recommended_frames', 0)} frames, {video.get('recommended_fps', 0)} fps, {video.get('recommended_resolution', [0,0])[0]}x{video.get('recommended_resolution', [0,0])[1]} resolution"
            )

        print("=" * 60)


# Create global instance
_hardware_optimizer = None


def get_hardware_optimizer() -> HardwareOptimizer:
    """Get the global hardware optimizer instance"""
    global _hardware_optimizer
    if _hardware_optimizer is None:
        _hardware_optimizer = HardwareOptimizer()
    return _hardware_optimizer


# Convenience functions
def get_chat_settings() -> Dict[str, Any]:
    """Get optimized chat settings"""
    return get_hardware_optimizer().get_chat_settings()


def get_image_settings() -> Dict[str, Any]:
    """Get optimized image generation settings"""
    return get_hardware_optimizer().get_image_settings()


def get_video_settings() -> Dict[str, Any]:
    """Get optimized video generation settings"""
    return get_hardware_optimizer().get_video_settings()


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information"""
    return get_hardware_optimizer().get_hardware_info()


def print_hardware_summary() -> None:
    """Print hardware summary"""
    get_hardware_optimizer().print_hardware_summary()


# Main execution
if __name__ == "__main__":
    print("🔍 Detecting hardware and optimizing settings...")
    optimizer = HardwareOptimizer()
    optimizer.print_hardware_summary()
