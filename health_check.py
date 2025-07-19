"""
Health check endpoint for Samosa GPT
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from error_handler import get_health_status
from config import config

class HealthCheck:
    """Comprehensive health check system"""
    
    def __init__(self):
        self.last_check = None
        self.cached_status = None
        self.cache_duration = 30  # Cache for 30 seconds
    
    def get_status(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get health status with caching"""
        current_time = time.time()
        
        # Use cached status if available and recent
        if (not force_refresh and 
            self.cached_status and 
            self.last_check and 
            current_time - self.last_check < self.cache_duration):
            return self.cached_status
        
        # Perform comprehensive health check
        status = self._perform_health_check()
        
        # Cache the result
        self.cached_status = status
        self.last_check = current_time
        
        return status
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "version": getattr(config, 'VERSION', '3.5.0'),
            "environment": getattr(config, 'ENV', 'development'),
            "checks": {}
        }
        
        checks = [
            ("system", self._check_system),
            ("dependencies", self._check_dependencies),
            ("ollama", self._check_ollama),
            ("storage", self._check_storage),
            ("logs", self._check_logs),
            ("permissions", self._check_permissions)
        ]
        
        all_healthy = True
        
        for check_name, check_func in checks:
            try:
                check_result = check_func()
                status["checks"][check_name] = check_result
                
                if not check_result.get("healthy", False):
                    all_healthy = False
                    
            except Exception as e:
                status["checks"][check_name] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                all_healthy = False
        
        # Overall status
        if all_healthy:
            status["status"] = "healthy"
        else:
            status["status"] = "unhealthy"
        
        # Add comprehensive health data
        try:
            health_data = get_health_status()
            status.update(health_data)
        except Exception as e:
            status["health_error"] = str(e)
        
        return status
    
    def _check_system(self) -> Dict[str, Any]:
        """Check system resources"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Memory check (warning if > 80%, critical if > 90%)
            memory_healthy = memory.percent < 80
            memory_status = "healthy"
            if memory.percent > 90:
                memory_status = "critical"
            elif memory.percent > 80:
                memory_status = "warning"
            
            # Disk check (warning if > 85%, critical if > 95%)
            disk_healthy = disk.percent < 85
            disk_status = "healthy"
            if disk.percent > 95:
                disk_status = "critical"
            elif disk.percent > 85:
                disk_status = "warning"
            
            return {
                "healthy": memory_healthy and disk_healthy,
                "memory": {
                    "percent": memory.percent,
                    "status": memory_status,
                    "available_gb": round(memory.available / (1024**3), 2)
                },
                "disk": {
                    "percent": disk.percent,
                    "status": disk_status,
                    "free_gb": round(disk.free / (1024**3), 2)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies"""
        required_packages = [
            'streamlit', 'torch', 'numpy', 'sounddevice', 
            'soundfile', 'requests', 'whisper', 'ollama', 'psutil'
        ]
        
        results = {}
        all_available = True
        
        for package in required_packages:
            try:
                __import__(package)
                results[package] = {"available": True}
            except ImportError as e:
                results[package] = {"available": False, "error": str(e)}
                all_available = False
        
        return {
            "healthy": all_available,
            "packages": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_ollama(self) -> Dict[str, Any]:
        """Check Ollama connection"""
        try:
            import requests
            
            ollama_host = getattr(config, 'OLLAMA_HOST', 'http://localhost:11434')
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    "healthy": True,
                    "connected": True,
                    "models_count": len(models),
                    "host": ollama_host,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "healthy": False,
                    "connected": False,
                    "status_code": response.status_code,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _check_storage(self) -> Dict[str, Any]:
        """Check storage and required directories"""
        required_dirs = [
            config.ASSETS_DIR,
            config.LOGS_DIR,
            getattr(config, 'CACHE_DIR', config.BASE_DIR / '.cache')
        ]
        
        results = {}
        all_accessible = True
        
        for directory in required_dirs:
            try:
                # Check if directory exists and is writable
                directory.mkdir(exist_ok=True)
                test_file = directory / '.health_check'
                test_file.write_text('test')
                test_file.unlink()
                
                results[str(directory)] = {
                    "exists": True,
                    "writable": True
                }
            except Exception as e:
                results[str(directory)] = {
                    "exists": directory.exists(),
                    "writable": False,
                    "error": str(e)
                }
                all_accessible = False
        
        return {
            "healthy": all_accessible,
            "directories": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_logs(self) -> Dict[str, Any]:
        """Check log files and logging system"""
        log_files = [
            config.CONVERSATION_LOG,
            config.ASSISTANT_LOG,
            getattr(config, 'ERROR_LOG', config.LOGS_DIR / 'error.log'),
            getattr(config, 'ACCESS_LOG', config.LOGS_DIR / 'access.log')
        ]
        
        results = {}
        all_accessible = True
        
        for log_file in log_files:
            try:
                # Check if log file is writable
                log_file.parent.mkdir(exist_ok=True)
                log_file.touch()
                
                # Check file size (warn if > 100MB)
                size_mb = log_file.stat().st_size / (1024 * 1024)
                
                results[log_file.name] = {
                    "exists": True,
                    "writable": True,
                    "size_mb": round(size_mb, 2),
                    "large": size_mb > 100
                }
                
                if size_mb > 100:
                    results[log_file.name]["warning"] = "Log file is large (>100MB)"
                    
            except Exception as e:
                results[log_file.name] = {
                    "exists": log_file.exists(),
                    "writable": False,
                    "error": str(e)
                }
                all_accessible = False
        
        return {
            "healthy": all_accessible,
            "log_files": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _check_permissions(self) -> Dict[str, Any]:
        """Check file system permissions"""
        try:
            # Test read/write permissions in key directories
            test_dirs = [config.BASE_DIR, config.LOGS_DIR, config.ASSETS_DIR]
            
            permissions_ok = True
            results = {}
            
            for test_dir in test_dirs:
                try:
                    test_file = test_dir / f'.perm_test_{int(time.time())}'
                    test_file.write_text('permission test')
                    content = test_file.read_text()
                    test_file.unlink()
                    
                    results[str(test_dir)] = {
                        "readable": True,
                        "writable": True
                    }
                except Exception as e:
                    results[str(test_dir)] = {
                        "readable": False,
                        "writable": False,
                        "error": str(e)
                    }
                    permissions_ok = False
            
            return {
                "healthy": permissions_ok,
                "directories": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def save_health_report(self, report_path: Path = None) -> Path:
        """Save health report to file"""
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = config.LOGS_DIR / f"health_report_{timestamp}.json"
        
        status = self.get_status(force_refresh=True)
        
        with open(report_path, 'w') as f:
            json.dump(status, f, indent=2, default=str)
        
        return report_path

# Global health check instance
health_check = HealthCheck()

def get_health_check_status() -> Dict[str, Any]:
    """Get health check status - convenience function"""
    return health_check.get_status()

def perform_startup_health_check() -> bool:
    """Perform startup health check and return success status"""
    try:
        status = health_check.get_status(force_refresh=True)
        is_healthy = status.get("status") == "healthy"
        
        if not is_healthy:
            print("⚠️ Health check failed at startup:")
            for check_name, check_result in status.get("checks", {}).items():
                if not check_result.get("healthy", False):
                    print(f"  ❌ {check_name}: {check_result.get('error', 'Failed')}")
        
        return is_healthy
        
    except Exception as e:
        print(f"❌ Health check system failed: {e}")
        return False
