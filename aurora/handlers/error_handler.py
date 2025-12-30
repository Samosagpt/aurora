"""
Production-ready error handling and monitoring for Samosa GPT
"""

import json
import logging
import sys
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil
import streamlit as st

from aurora.core.config import config


class ErrorHandler:
    """Enhanced error handling with monitoring and recovery"""

    def __init__(self):
        self.error_counts = {}
        self.last_errors = []
        self.max_error_history = 100
        self.setup_logging()

    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create error log path if not exists
        error_log = getattr(config, "ERROR_LOG", config.LOGS_DIR / "error.log")
        access_log = getattr(config, "ACCESS_LOG", config.LOGS_DIR / "access.log")

        # Error logger
        self.error_logger = logging.getLogger("samosa.errors")
        error_handler = logging.FileHandler(error_log)
        error_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.ERROR)

        # Access logger
        self.access_logger = logging.getLogger("samosa.access")
        access_handler = logging.FileHandler(access_log)
        access_formatter = logging.Formatter("%(asctime)s - %(message)s")
        access_handler.setFormatter(access_formatter)
        self.access_logger.addHandler(access_handler)
        self.access_logger.setLevel(logging.INFO)

    def handle_error(self, error: Exception, context: str = "", user_message: str = None) -> str:
        """Handle errors with logging and user-friendly messages"""
        error_id = f"ERR_{int(time.time())}"
        error_type = type(error).__name__
        error_message = str(error)

        # Log detailed error
        self.error_logger.error(
            f"Error ID: {error_id} | Context: {context} | Type: {error_type} | Message: {error_message} | Traceback: {traceback.format_exc()}"
        )

        # Track error frequency
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Store error in history
        error_record = {
            "id": error_id,
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message,
            "context": context,
            "traceback": traceback.format_exc(),
        }

        self.last_errors.append(error_record)
        if len(self.last_errors) > self.max_error_history:
            self.last_errors.pop(0)

        # Return user-friendly message
        if user_message:
            return f"{user_message} (Error ID: {error_id})"

        # Default user-friendly messages
        friendly_messages = {
            "ConnectionError": "Unable to connect to the service. Please check your internet connection.",
            "TimeoutError": "The operation took too long. Please try again.",
            "FileNotFoundError": "A required file was not found. Please contact support.",
            "MemoryError": "Insufficient memory. Please try a smaller request.",
            "ImportError": "A required component is missing. Please reinstall the application.",
            "ValueError": "Invalid input provided. Please check your data and try again.",
            "KeyError": "Configuration error. Please contact support.",
        }

        friendly_message = friendly_messages.get(
            error_type, "An unexpected error occurred. Please try again."
        )
        return f"{friendly_message} (Error ID: {error_id})"

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": dict(self.error_counts),
            "recent_errors": self.last_errors[-10:],
            "most_common_error": (
                max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
            ),
        }

    def log_access(self, endpoint: str, user_id: str = "anonymous", duration: float = 0):
        """Log access information"""
        self.access_logger.info(
            f"User: {user_id} | Endpoint: {endpoint} | Duration: {duration:.2f}s"
        )


class SystemMonitor:
    """System monitoring and health checks"""

    def __init__(self):
        self.start_time = datetime.now()
        self.last_health_check = None
        self.health_status = "unknown"
        self.system_stats = {}

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()

            health_data = {
                "timestamp": datetime.now().isoformat(),
                "uptime": str(datetime.now() - self.start_time),
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                },
                "process_memory": {
                    "rss": process_memory.rss,
                    "vms": process_memory.vms,
                    "percent": process.memory_percent(),
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent,
                },
                "cpu_percent": cpu_percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else [0, 0, 0],
            }

            # Determine health status
            if memory.percent > 90 or disk.percent > 95 or cpu_percent > 95:
                self.health_status = "critical"
            elif memory.percent > 75 or disk.percent > 85 or cpu_percent > 80:
                self.health_status = "warning"
            else:
                self.health_status = "healthy"

            health_data["status"] = self.health_status
            self.system_stats = health_data
            self.last_health_check = datetime.now()

            return health_data

        except Exception as e:
            error_handler.handle_error(e, "system_health_check")
            return {"status": "error", "message": str(e)}

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available"""
        dependencies = {
            "ollama": self._check_ollama(),
            "streamlit": self._check_streamlit(),
            "torch": self._check_torch(),
            "whisper": self._check_whisper(),
            "diffusers": self._check_diffusers(),
        }
        return dependencies

    def _check_ollama(self) -> bool:
        """Check Ollama connection"""
        try:
            import requests

            response = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _check_streamlit(self) -> bool:
        """Check Streamlit availability"""
        try:
            import streamlit

            return True
        except ImportError:
            return False

    def _check_torch(self) -> bool:
        """Check PyTorch availability"""
        try:
            import torch

            return True
        except ImportError:
            return False

    def _check_whisper(self) -> bool:
        """Check Whisper availability"""
        try:
            import whisper

            return True
        except ImportError:
            return False

    def _check_diffusers(self) -> bool:
        """Check Diffusers availability"""
        try:
            import diffusers

            return True
        except ImportError:
            return False


class PerformanceMonitor:
    """Monitor performance metrics"""

    def __init__(self):
        self.metrics = {}
        self.request_times = []
        self.max_request_history = 1000

    def time_function(self, func_name: str = None):
        """Decorator to time function execution"""

        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    name = func_name or func.__name__
                    self.record_metric(name, duration)

            return wrapper

        return decorator

    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []

        self.metrics[name].append({"timestamp": datetime.now().isoformat(), "value": value})

        # Keep only recent metrics
        if len(self.metrics[name]) > 100:
            self.metrics[name] = self.metrics[name][-50:]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for name, values in self.metrics.items():
            if values:
                recent_values = [v["value"] for v in values[-20:]]
                stats[name] = {
                    "avg": sum(recent_values) / len(recent_values),
                    "min": min(recent_values),
                    "max": max(recent_values),
                    "count": len(values),
                }
        return stats


def production_error_handler(func):
    """Decorator for production error handling"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = error_handler.handle_error(e, f"Function: {func.__name__}")
            if "st" in globals() and hasattr(st, "error"):
                st.error(error_message)
            else:
                print(f"Error: {error_message}")
            return None

    return wrapper


# Global instances
error_handler = ErrorHandler()
system_monitor = SystemMonitor()
performance_monitor = PerformanceMonitor()


# Health check endpoint data
def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status"""
    return {
        "system": system_monitor.get_system_health(),
        "dependencies": system_monitor.check_dependencies(),
        "errors": error_handler.get_error_stats(),
        "performance": performance_monitor.get_performance_stats(),
        "config": config.get_safe_config(),
        "api_keys": config.validate_api_keys(),
    }
