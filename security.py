"""
Security module for Aurora production deployment
"""
import hashlib
import secrets
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import streamlit as st

logger = logging.getLogger(__name__)

class SecurityManager:
    """Comprehensive security management"""
    
    def __init__(self):
        self.rate_limits = {}
        self.blocked_ips = set()
        self.session_tokens = {}
        self.failed_attempts = {}
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes
    
    def sanitize_input(self, text: str, max_length: int = 10000) -> str:
        """Sanitize user input"""
        if not isinstance(text, str):
            return ""
        
        # Limit length
        text = text[:max_length]
        
        # Remove potentially dangerous characters/patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # JavaScript
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'<iframe[^>]*>.*?</iframe>',  # iframes
            r'<object[^>]*>.*?</object>',  # objects
            r'<embed[^>]*>.*?</embed>',  # embeds
        ]
        
        for pattern in dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()
    
    def validate_file_upload(self, file_path: Path, allowed_extensions: List[str]) -> bool:
        """Validate file uploads"""
        if not file_path.exists():
            return False
        
        # Check extension
        if file_path.suffix.lower().lstrip('.') not in allowed_extensions:
            return False
        
        # Check file size (max 100MB)
        if file_path.stat().st_size > 100 * 1024 * 1024:
            return False
        
        # Check for malicious content in text files
        if file_path.suffix.lower() in ['.txt', '.md', '.py', '.js']:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                if self._contains_malicious_content(content):
                    return False
            except Exception:
                return False
        
        return True
    
    def _contains_malicious_content(self, content: str) -> bool:
        """Check for malicious content patterns"""
        malicious_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def check_rate_limit(self, identifier: str, max_requests: int = 60, window: int = 60) -> bool:
        """Check rate limiting"""
        current_time = time.time()
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Remove old requests outside the window
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier]
            if current_time - req_time < window
        ]
        
        # Check if limit exceeded
        if len(self.rate_limits[identifier]) >= max_requests:
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False
        
        # Add current request
        self.rate_limits[identifier].append(current_time)
        return True
    
    def generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def validate_session_token(self, token: str) -> bool:
        """Validate session token"""
        if not token or token not in self.session_tokens:
            return False
        
        # Check if token has expired (24 hours)
        created_time = self.session_tokens[token]
        if time.time() - created_time > 86400:
            del self.session_tokens[token]
            return False
        
        return True
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "details": details
        }
        
        logger.warning(f"Security Event: {log_entry}")
    
    def check_model_name_safety(self, model_name: str) -> bool:
        """Validate model names for safety"""
        if not model_name:
            return False
        
        # Allow only alphanumeric, hyphens, underscores, dots, and colons
        if not re.match(r'^[a-zA-Z0-9\-_.:\/]+$', model_name):
            return False
        
        # Prevent directory traversal
        if '..' in model_name or model_name.startswith('/') or '\\' in model_name:
            return False
        
        # Limit length
        if len(model_name) > 100:
            return False
        
        return True
    
    def validate_api_endpoint(self, endpoint: str) -> bool:
        """Validate API endpoints"""
        if not endpoint:
            return False
        
        # Must be HTTP/HTTPS
        if not endpoint.startswith(('http://', 'https://')):
            return False
        
        # Block suspicious domains
        blocked_domains = [
            'localhost',
            '127.0.0.1',
            '0.0.0.0',
            'internal',
            'private'
        ]
        
        for domain in blocked_domains:
            if domain in endpoint.lower():
                return False
        
        return True

class InputValidator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_prompt(prompt: str) -> tuple[bool, str]:
        """Validate user prompts"""
        if not prompt or not prompt.strip():
            return False, "Prompt cannot be empty"
        
        if len(prompt) > 10000:
            return False, "Prompt too long (max 10,000 characters)"
        
        # Check for injection attempts
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return False, "Potentially unsafe content detected"
        
        return True, "Valid"
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file paths"""
        if not file_path:
            return False
        
        # Prevent directory traversal
        if '..' in file_path or file_path.startswith(('/', '\\')):
            return False
        
        # Only allow specific directories
        allowed_prefixes = ['assets/', 'logs/', 'cache/', 'uploads/']
        if not any(file_path.startswith(prefix) for prefix in allowed_prefixes):
            return False
        
        return True
    
    @staticmethod
    def validate_config_value(key: str, value: Any) -> bool:
        """Validate configuration values"""
        # Define allowed config keys and their types
        allowed_configs = {
            'debug': bool,
            'max_memory': int,
            'sample_rate': int,
            'channels': int,
            'whisper_model': str,
            'tts_voice': str,
        }
        
        if key not in allowed_configs:
            return False
        
        expected_type = allowed_configs[key]
        if not isinstance(value, expected_type):
            return False
        
        # Additional validation for specific keys
        if key == 'max_memory' and value < 512:
            return False
        
        if key == 'sample_rate' and value not in [16000, 22050, 44100, 48000]:
            return False
        
        return True

class AuditLogger:
    """Audit logging for security events"""
    
    def __init__(self):
        self.audit_log = Path("logs/audit.log")
        self.audit_log.parent.mkdir(exist_ok=True)
        
        # Setup audit logger
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler(self.audit_log)
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_user_action(self, action: str, user_id: str = "anonymous", details: Dict[str, Any] = None):
        """Log user actions"""
        log_data = {
            "action": action,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.logger.info(f"USER_ACTION: {log_data}")
    
    def log_system_event(self, event: str, details: Dict[str, Any] = None):
        """Log system events"""
        log_data = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        self.logger.info(f"SYSTEM_EVENT: {log_data}")
    
    def log_security_alert(self, alert_type: str, details: Dict[str, Any]):
        """Log security alerts"""
        log_data = {
            "alert_type": alert_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        self.logger.warning(f"SECURITY_ALERT: {log_data}")

# Global instances
security_manager = SecurityManager()
input_validator = InputValidator()
audit_logger = AuditLogger()

# Security decorators
def require_valid_input(func):
    """Decorator to validate inputs"""
    def wrapper(*args, **kwargs):
        # Validate string arguments
        for arg in args:
            if isinstance(arg, str):
                if not security_manager.sanitize_input(arg):
                    raise ValueError("Invalid input detected")
        
        # Validate keyword arguments
        for key, value in kwargs.items():
            if isinstance(value, str):
                kwargs[key] = security_manager.sanitize_input(value)
        
        return func(*args, **kwargs)
    return wrapper

def rate_limited(max_requests: int = 60, window: int = 60):
    """Decorator for rate limiting"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Use session ID or IP for rate limiting
            identifier = "default"
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'session_id'):
                identifier = st.session_state.session_id
            
            if not security_manager.check_rate_limit(identifier, max_requests, window):
                raise Exception("Rate limit exceeded. Please try again later.")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
