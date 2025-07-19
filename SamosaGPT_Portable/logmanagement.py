"""
Enhanced logging management for Samosa GPT
"""
import json
import datetime
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO if not config.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / 'samosa.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LogManager:
    """Enhanced logging manager with error handling and backup capabilities"""
    
    def __init__(self):
        self.conversation_log_path = config.CONVERSATION_LOG
        self.md_log_path = config.ASSISTANT_LOG
        self._ensure_log_files_exist()
    
    def _ensure_log_files_exist(self) -> None:
        """Ensure log files exist"""
        try:
            if not self.conversation_log_path.exists():
                self.conversation_log_path.write_text('[]', encoding='utf-8')
            
            if not self.md_log_path.exists():
                self._initialize_md_log()
                
        except Exception as e:
            logger.error(f"Error creating log files: {e}")
    
    def _initialize_md_log(self) -> None:
        """Initialize markdown log with header"""
        header = f"""# Samosa GPT Assistant Log
        
Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Version: {config.VERSION}

## Conversation History

"""
        self.md_log_path.write_text(header, encoding='utf-8')
    
    def read_json_history(self) -> List[Dict[str, Any]]:
        """Read conversation history with error handling"""
        try:
            if self.conversation_log_path.exists():
                content = self.conversation_log_path.read_text(encoding='utf-8')
                return json.loads(content) if content.strip() else []
            return []
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error reading conversation history: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error reading history: {e}")
            return []
    
    def write_json_history(self, history: List[Dict[str, Any]]) -> bool:
        """Write conversation history with backup"""
        try:
            # Create backup
            if self.conversation_log_path.exists():
                backup_path = self.conversation_log_path.with_suffix('.json.backup')
                backup_path.write_text(
                    self.conversation_log_path.read_text(encoding='utf-8'),
                    encoding='utf-8'
                )
            
            # Write new history
            self.conversation_log_path.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            logger.debug("Conversation history saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error writing conversation history: {e}")
            return False
    
    def append_md_log(self, role: str, message: str = "\b") -> None:
        """Append to markdown log with error handling"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = f"**{role}:** *[{timestamp}]* {message}  \n"
            
            # Print to console if debug mode
            if config.DEBUG:
                print(entry.strip())
            
            # Append to file
            with open(self.md_log_path, 'a', encoding='utf-8') as f:
                f.write(entry)
                
        except Exception as e:
            logger.error(f"Error appending to markdown log: {e}")
    
    def clear_json_history(self) -> bool:
        """Clear conversation history"""
        try:
            self.conversation_log_path.write_text('[]', encoding='utf-8')
            self.append_md_log("System", "Conversation history cleared")
            logger.info("Conversation history cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False
    
    def export_logs(self, export_path: Optional[Path] = None) -> Path:
        """Export logs to a specific location"""
        if export_path is None:
            export_path = config.LOGS_DIR / f"export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        export_path.mkdir(exist_ok=True)
        
        try:
            # Copy conversation log
            if self.conversation_log_path.exists():
                (export_path / "conversation_log.json").write_text(
                    self.conversation_log_path.read_text(encoding='utf-8'),
                    encoding='utf-8'
                )
            
            # Copy markdown log
            if self.md_log_path.exists():
                (export_path / "assistant_log.md").write_text(
                    self.md_log_path.read_text(encoding='utf-8'),
                    encoding='utf-8'
                )
            
            logger.info(f"Logs exported to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Error exporting logs: {e}")
            raise

# Create global instance
log_manager = LogManager()

# Backward compatibility functions
def read_json_history() -> List[Dict[str, Any]]:
    return log_manager.read_json_history()

def write_json_history(history: List[Dict[str, Any]]) -> None:
    log_manager.write_json_history(history)

def append_md_log(role: str, message: str = "\b") -> None:
    log_manager.append_md_log(role, message)

def clear_json_history() -> None:
    log_manager.clear_json_history()
    
    