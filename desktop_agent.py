"""
Desktop Agent - Agentic AI with desktop control capabilities
Provides tools for controlling the computer through natural language commands
Includes Tesseract OCR for GUI recognition and precise coordinate detection
"""

import pyautogui
import subprocess
import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import psutil
import win32gui
import win32con
import win32process
from PIL import Image
import io
import base64
import webbrowser
import cv2
import numpy as np

# Try to import pytesseract for OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    # Try to set tesseract path if not in PATH
    try:
        pytesseract.get_tesseract_version()
    except:
        # Common Tesseract installation paths
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getlogin())
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ pytesseract not available. Install with: pip install pytesseract")
    print("⚠️ Also install Tesseract-OCR from: https://github.com/UB-Mannheim/tesseract/wiki")

logger = logging.getLogger(__name__)

class DesktopAgent:
    """Agent capable of controlling desktop operations"""
    
    def __init__(self):
        """Initialize desktop agent"""
        # Safety settings for PyAutoGUI
        pyautogui.FAILSAFE = True  # Move mouse to corner to abort
        pyautogui.PAUSE = 0.5  # Pause between actions
        
        self.screen_size = pyautogui.size()
        self.last_action = None
        self.action_history = []
        
        logger.info(f"Desktop Agent initialized. Screen size: {self.screen_size}")
    
    # ==================== TOOL DEFINITIONS ====================
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools for the AI agent"""
        return [
            {
                "name": "mouse_click",
                "description": "Click the mouse at specific coordinates or on screen element",
                "parameters": {
                    "x": "X coordinate (optional if using element name)",
                    "y": "Y coordinate (optional if using element name)",
                    "button": "Mouse button: 'left', 'right', or 'middle'",
                    "clicks": "Number of clicks (default: 1)"
                }
            },
            {
                "name": "mouse_move",
                "description": "Move mouse to specific coordinates",
                "parameters": {
                    "x": "X coordinate",
                    "y": "Y coordinate",
                    "duration": "Movement duration in seconds (default: 0.2)"
                }
            },
            {
                "name": "type_text",
                "description": "Type text using keyboard",
                "parameters": {
                    "text": "Text to type",
                    "interval": "Interval between keystrokes in seconds (default: 0.01)"
                }
            },
            {
                "name": "press_key",
                "description": "Press keyboard key or key combination",
                "parameters": {
                    "keys": "Key or key combination (e.g., 'enter', 'ctrl+c', 'alt+tab')"
                }
            },
            {
                "name": "take_screenshot",
                "description": "Take a screenshot of the entire screen or a region",
                "parameters": {
                    "region": "Optional tuple (x, y, width, height) for partial screenshot",
                    "save_path": "Optional path to save screenshot"
                }
            },
            {
                "name": "open_application",
                "description": "Open an application or program",
                "parameters": {
                    "app_name": "Application name or path",
                    "args": "Optional command line arguments"
                }
            },
            {
                "name": "open_url",
                "description": "Open a URL in the default web browser (better than open_application for websites)",
                "parameters": {
                    "url": "URL to open (e.g., 'https://github.com' or 'github.com/pulls')"
                }
            },
            {
                "name": "close_application",
                "description": "Close an application by name",
                "parameters": {
                    "app_name": "Application name or window title"
                }
            },
            {
                "name": "list_windows",
                "description": "List all open windows",
                "parameters": {}
            },
            {
                "name": "switch_window",
                "description": "Switch to a specific window",
                "parameters": {
                    "window_title": "Window title or partial title"
                }
            },
            {
                "name": "get_window_info",
                "description": "Get information about the active window",
                "parameters": {}
            },
            {
                "name": "ocr_screen",
                "description": "Perform OCR on screen to extract text and locate GUI elements",
                "parameters": {
                    "region": "Optional tuple (x, y, width, height) for partial OCR",
                    "search_text": "Optional text to search for and locate"
                }
            },
            {
                "name": "find_text_on_screen",
                "description": "Find text on screen using OCR and return its coordinates",
                "parameters": {
                    "text": "Text to find",
                    "confidence": "Minimum confidence level (0-100, default: 60)"
                }
            },
            {
                "name": "click_text",
                "description": "Click on text found via OCR on screen",
                "parameters": {
                    "text": "Text to click on",
                    "button": "Mouse button: 'left', 'right', or 'middle'"
                }
            },
            {
                "name": "recognize_gui_elements",
                "description": "Recognize and locate GUI elements (buttons, text fields, etc.) on screen",
                "parameters": {
                    "region": "Optional tuple (x, y, width, height) to focus on specific area"
                }
            },
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "file_path": "Path to the file"
                }
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "file_path": "Path to the file",
                    "content": "Content to write",
                    "mode": "Write mode: 'w' (overwrite) or 'a' (append)"
                }
            },
            {
                "name": "list_directory",
                "description": "List files and folders in a directory",
                "parameters": {
                    "directory_path": "Path to the directory"
                }
            },
            {
                "name": "search_files",
                "description": "Search for files by name or pattern",
                "parameters": {
                    "search_path": "Directory to search in",
                    "pattern": "File name pattern (e.g., '*.txt')"
                }
            },
            {
                "name": "run_command",
                "description": "Execute a shell command",
                "parameters": {
                    "command": "Command to execute",
                    "shell": "Use shell (default: True)"
                }
            },
            {
                "name": "get_system_info",
                "description": "Get system information (CPU, memory, disk, etc.)",
                "parameters": {}
            },
            {
                "name": "wait",
                "description": "Wait for specified duration",
                "parameters": {
                    "seconds": "Number of seconds to wait"
                }
            }
        ]
    
    # ==================== MOUSE OPERATIONS ====================
    
    def mouse_click(self, x: int = None, y: int = None, button: str = 'left', clicks: int = 1) -> Dict[str, Any]:
        """Click mouse at coordinates"""
        try:
            if x is not None and y is not None:
                pyautogui.click(x, y, clicks=clicks, button=button)
                result = f"Clicked {button} button {clicks} time(s) at ({x}, {y})"
            else:
                pyautogui.click(clicks=clicks, button=button)
                result = f"Clicked {button} button {clicks} time(s) at current position"
            
            self._log_action("mouse_click", {"x": x, "y": y, "button": button, "clicks": clicks})
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Mouse click error: {e}")
            return {"success": False, "error": str(e)}
    
    def mouse_move(self, x: int, y: int, duration: float = 0.2) -> Dict[str, Any]:
        """Move mouse to coordinates"""
        try:
            pyautogui.moveTo(x, y, duration=duration)
            result = f"Moved mouse to ({x}, {y})"
            self._log_action("mouse_move", {"x": x, "y": y, "duration": duration})
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Mouse move error: {e}")
            return {"success": False, "error": str(e)}
    
    def mouse_drag(self, x: int, y: int, duration: float = 0.5, button: str = 'left') -> Dict[str, Any]:
        """Drag mouse to coordinates"""
        try:
            pyautogui.drag(x, y, duration=duration, button=button)
            result = f"Dragged mouse by ({x}, {y})"
            self._log_action("mouse_drag", {"x": x, "y": y, "duration": duration, "button": button})
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Mouse drag error: {e}")
            return {"success": False, "error": str(e)}
    
    def mouse_scroll(self, clicks: int) -> Dict[str, Any]:
        """Scroll mouse wheel"""
        try:
            pyautogui.scroll(clicks)
            direction = "up" if clicks > 0 else "down"
            result = f"Scrolled {abs(clicks)} clicks {direction}"
            self._log_action("mouse_scroll", {"clicks": clicks})
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Mouse scroll error: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== KEYBOARD OPERATIONS ====================
    
    def type_text(self, text: str, interval: float = 0.01) -> Dict[str, Any]:
        """Type text using keyboard"""
        try:
            pyautogui.write(text, interval=interval)
            result = f"Typed text: {text[:50]}..." if len(text) > 50 else f"Typed text: {text}"
            self._log_action("type_text", {"text": text, "interval": interval})
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Type text error: {e}")
            return {"success": False, "error": str(e)}
    
    def press_key(self, keys: str = None, key: str = None) -> Dict[str, Any]:
        """Press keyboard key or combination"""
        try:
            # Accept both 'keys' and 'key' parameter names for flexibility
            key_to_press = keys or key
            if not key_to_press:
                return {"success": False, "error": "No key specified"}
            
            # Handle key combinations (e.g., 'ctrl+c', 'alt+tab')
            if '+' in key_to_press:
                key_list = [k.strip() for k in key_to_press.split('+')]
                pyautogui.hotkey(*key_list)
                result = f"Pressed key combination: {key_to_press}"
            else:
                pyautogui.press(key_to_press)
                result = f"Pressed key: {key_to_press}"
            
            self._log_action("press_key", {"keys": key_to_press})
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Press key error: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== SCREEN OPERATIONS ====================
    
    def take_screenshot(self, region: Tuple[int, int, int, int] = None, save_path: str = None) -> Dict[str, Any]:
        """Take screenshot"""
        try:
            screenshot = pyautogui.screenshot(region=region)
            
            # Always save to a temp location if not specified
            if not save_path:
                import tempfile
                temp_dir = Path("logs/screenshots")
                temp_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(temp_dir / f"screenshot_{int(time.time())}.png")
            
            # Save screenshot
            screenshot.save(save_path)
            
            # Also convert to base64 for optional returning
            buffered = io.BytesIO()
            screenshot.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            self._log_action("take_screenshot", {"region": region, "save_path": save_path})
            
            return {
                "success": True,
                "result": f"Screenshot saved to {save_path}",
                "screenshot_path": save_path,
                "size": screenshot.size,
                "image": img_str
            }
        except Exception as e:
            logger.error(f"Screenshot error: {e}")
            return {"success": False, "error": str(e)}
    
    def locate_on_screen(self, image_path: str, confidence: float = 0.9) -> Dict[str, Any]:
        """Find image on screen"""
        try:
            location = pyautogui.locateOnScreen(image_path, confidence=confidence)
            if location:
                center = pyautogui.center(location)
                result = f"Found image at {center}"
                return {"success": True, "result": result, "location": location, "center": center}
            else:
                return {"success": False, "result": "Image not found on screen"}
        except Exception as e:
            logger.error(f"Locate on screen error: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== OCR & GUI RECOGNITION ====================
    
    def ocr_screen(self, region: Tuple[int, int, int, int] = None, search_text: str = None) -> Dict[str, Any]:
        """Perform OCR on screen to extract text"""
        if not TESSERACT_AVAILABLE:
            return {
                "success": False,
                "error": "Tesseract OCR not available. Install with: pip install pytesseract"
            }
        
        try:
            # Take screenshot
            screenshot = pyautogui.screenshot(region=region)
            
            # Convert to OpenCV format
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Perform OCR with detailed data
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Extract text and positions
            texts = []
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    conf = int(ocr_data['conf'][i])
                    if conf > 0:  # Only include confident detections
                        x = ocr_data['left'][i]
                        y = ocr_data['top'][i]
                        w = ocr_data['width'][i]
                        h = ocr_data['height'][i]
                        
                        # Adjust coordinates if region was specified
                        if region:
                            x += region[0]
                            y += region[1]
                        
                        texts.append({
                            'text': text,
                            'confidence': conf,
                            'position': {'x': x, 'y': y, 'width': w, 'height': h},
                            'center': {'x': x + w//2, 'y': y + h//2}
                        })
            
            # If searching for specific text
            matches = []
            if search_text:
                search_lower = search_text.lower()
                for item in texts:
                    if search_lower in item['text'].lower():
                        matches.append(item)
            
            result = {
                "success": True,
                "result": f"Found {len(texts)} text elements",
                "texts": texts,
                "full_text": ' '.join([t['text'] for t in texts])
            }
            
            if search_text:
                result['matches'] = matches
                result['result'] = f"Found {len(matches)} matches for '{search_text}'"
            
            self._log_action("ocr_screen", {"region": region, "search_text": search_text})
            return result
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return {"success": False, "error": str(e)}
    
    def find_text_on_screen(self, text: str, confidence: int = 60) -> Dict[str, Any]:
        """Find text on screen using OCR and return coordinates"""
        if not TESSERACT_AVAILABLE:
            return {
                "success": False,
                "error": "Tesseract OCR not available"
            }
        
        try:
            # Perform OCR
            ocr_result = self.ocr_screen(search_text=text)
            
            if not ocr_result['success']:
                return ocr_result
            
            # Find best match
            matches = ocr_result.get('matches', [])
            high_conf_matches = [m for m in matches if m['confidence'] >= confidence]
            
            if high_conf_matches:
                best_match = max(high_conf_matches, key=lambda x: x['confidence'])
                return {
                    "success": True,
                    "result": f"Found '{text}' at ({best_match['center']['x']}, {best_match['center']['y']})",
                    "coordinates": best_match['center'],
                    "position": best_match['position'],
                    "confidence": best_match['confidence']
                }
            else:
                return {
                    "success": False,
                    "result": f"Text '{text}' not found with confidence >= {confidence}%",
                    "all_matches": matches
                }
                
        except Exception as e:
            logger.error(f"Find text error: {e}")
            return {"success": False, "error": str(e)}
    
    def click_text(self, text: str, button: str = 'left', confidence: int = 60) -> Dict[str, Any]:
        """Click on text found via OCR"""
        try:
            # Find text on screen
            find_result = self.find_text_on_screen(text, confidence)
            
            if not find_result['success']:
                return find_result
            
            # Click on the coordinates
            coords = find_result['coordinates']
            click_result = self.mouse_click(
                x=coords['x'],
                y=coords['y'],
                button=button
            )
            
            if click_result['success']:
                return {
                    "success": True,
                    "result": f"Clicked on '{text}' at ({coords['x']}, {coords['y']})",
                    "coordinates": coords
                }
            else:
                return click_result
                
        except Exception as e:
            logger.error(f"Click text error: {e}")
            return {"success": False, "error": str(e)}
    
    def recognize_gui_elements(self, region: Tuple[int, int, int, int] = None) -> Dict[str, Any]:
        """Recognize GUI elements (buttons, text fields, etc.) using computer vision"""
        if not TESSERACT_AVAILABLE:
            return {
                "success": False,
                "error": "Tesseract OCR not available"
            }
        
        try:
            # Take screenshot
            screenshot = pyautogui.screenshot(region=region)
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect edges for GUI elements
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours (potential buttons/UI elements)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and analyze contours
            gui_elements = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter out noise (too small) and full screen (too large)
                if 100 < area < (img.shape[0] * img.shape[1] * 0.5):
                    # Adjust coordinates if region specified
                    if region:
                        x += region[0]
                        y += region[1]
                    
                    # Extract text from this region using OCR
                    element_img = gray[y:y+h, x:x+w] if not region else gray[y-region[1]:y-region[1]+h, x-region[0]:x-region[0]+w]
                    element_text = pytesseract.image_to_string(element_img).strip()
                    
                    gui_elements.append({
                        'type': 'button' if 20 < w < 200 and 15 < h < 50 else 'element',
                        'position': {'x': x, 'y': y, 'width': w, 'height': h},
                        'center': {'x': x + w//2, 'y': y + h//2},
                        'text': element_text if element_text else None,
                        'area': area
                    })
            
            # Sort by vertical position (top to bottom)
            gui_elements.sort(key=lambda e: e['position']['y'])
            
            self._log_action("recognize_gui_elements", {"region": region, "found": len(gui_elements)})
            
            return {
                "success": True,
                "result": f"Found {len(gui_elements)} GUI elements",
                "elements": gui_elements[:50]  # Limit to top 50 to avoid huge responses
            }
            
        except Exception as e:
            logger.error(f"GUI recognition error: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== APPLICATION OPERATIONS ====================
    
    def _find_chrome(self) -> str:
        """Find Chrome executable path"""
        possible_paths = [
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
            os.path.expanduser(r'~\AppData\Local\Google\Chrome\Application\chrome.exe'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Fallback: try to start chrome via start command (Windows will find it)
        return 'start chrome'
    
    def open_application(self, app_name: str, args: str = None) -> Dict[str, Any]:
        """Open an application"""
        try:
            # Common application mappings with fallback paths
            app_map = {
                'notepad': 'notepad.exe',
                'calculator': 'calc.exe',
                'paint': 'mspaint.exe',
                'explorer': 'explorer.exe',
                'cmd': 'cmd.exe',
                'powershell': 'powershell.exe',
                'chrome': self._find_chrome(),
                'google chrome': self._find_chrome(),
                'firefox': r'C:\Program Files\Mozilla Firefox\firefox.exe',
                'edge': r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                'microsoft edge': r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            }
            
            # Get actual executable path
            exe_path = app_map.get(app_name.lower(), app_name)
            
            # Handle special case for "start" command on Windows
            if isinstance(exe_path, str) and exe_path.startswith('start '):
                command_str = exe_path
                if args:
                    command_str += f" {args}"
                process = subprocess.Popen(command_str, shell=True)
                result = f"Opened {app_name} via start command (PID: {process.pid})"
                self._log_action("open_application", {"app_name": app_name, "args": args, "pid": process.pid})
                return {"success": True, "result": result, "pid": process.pid}
            
            # Build command for regular applications
            command = [exe_path]
            if args:
                if isinstance(args, str):
                    command.append(args)
                else:
                    command.extend(args)
            
            # Execute
            process = subprocess.Popen(command)
            result = f"Opened {app_name} (PID: {process.pid})"
            
            self._log_action("open_application", {"app_name": app_name, "args": args, "pid": process.pid})
            return {"success": True, "result": result, "pid": process.pid}
        except Exception as e:
            logger.error(f"Open application error: {e}")
            return {"success": False, "error": str(e)}
    
    def open_url(self, url: str) -> Dict[str, Any]:
        """Open a URL in the default web browser
        
        Args:
            url: The URL to open (e.g., 'https://github.com')
        
        Returns:
            Dictionary with success status and result message
        """
        try:
            # Ensure URL has a protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Open URL in default browser (or new tab if browser already open)
            webbrowser.open(url, new=2)  # new=2 opens in new tab if possible
            
            # Wait a moment for browser to open
            time.sleep(2)
            
            # Try to bring browser window to front using win32gui
            try:
                def enum_windows_callback(hwnd, results):
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if title:  # Only windows with titles
                            # Check if window is NOT minimized
                            placement = win32gui.GetWindowPlacement(hwnd)
                            if placement[1] != win32con.SW_SHOWMINIMIZED:
                                results.append((hwnd, title))
                
                windows = []
                win32gui.EnumWindows(enum_windows_callback, windows)
                
                # Look for browser windows (but skip minimized ones)
                browser_keywords = ['chrome', 'edge', 'firefox', 'opera', 'brave', 'whatsapp']
                for hwnd, title in windows:
                    title_lower = title.lower()
                    if any(keyword in title_lower for keyword in browser_keywords):
                        try:
                            # Only bring to foreground, don't change window state
                            logger.info(f"Activating browser window: {title[:50]} (hwnd={hwnd})")
                            win32gui.SetForegroundWindow(hwnd)
                            print(f"✅ Brought browser window to front: {title[:50]}")
                            logger.info(f"Successfully brought browser window to foreground: {title[:50]}")
                            break
                        except Exception as e:
                            print(f"⚠️  Could not activate window '{title[:30]}': {e}")
                            logger.warning(f"Failed to activate browser window '{title[:30]}': {e}")
                            
            except Exception as e:
                print(f"⚠️  Could not bring browser to front: {e}")
                logger.error(f"Error bringing browser window to front: {e}")
            
            result = f"Opened URL: {url}"
            self._log_action("open_url", {"url": url})
            return {"success": True, "result": result, "url": url}
        except Exception as e:
            logger.error(f"Open URL error: {e}")
            return {"success": False, "error": str(e)}
    
    def close_application(self, app_name: str) -> Dict[str, Any]:
        """Close application by name"""
        try:
            closed_count = 0
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if app_name.lower() in proc.info['name'].lower():
                        proc.terminate()
                        closed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            result = f"Closed {closed_count} instance(s) of {app_name}"
            self._log_action("close_application", {"app_name": app_name, "count": closed_count})
            return {"success": True, "result": result, "closed_count": closed_count}
        except Exception as e:
            logger.error(f"Close application error: {e}")
            return {"success": False, "error": str(e)}
    
    def list_windows(self) -> Dict[str, Any]:
        """List all open windows"""
        try:
            windows = []
            
            def enum_windows_callback(hwnd, results):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title:
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        try:
                            process = psutil.Process(pid)
                            process_name = process.name()
                        except:
                            process_name = "Unknown"
                        
                        results.append({
                            "hwnd": hwnd,
                            "title": title,
                            "pid": pid,
                            "process": process_name
                        })
            
            win32gui.EnumWindows(enum_windows_callback, windows)
            
            result = f"Found {len(windows)} open windows"
            self._log_action("list_windows", {"count": len(windows)})
            
            return {"success": True, "result": result, "windows": windows}
        except Exception as e:
            logger.error(f"List windows error: {e}")
            return {"success": False, "error": str(e)}
    
    def switch_window(self, window_title: str) -> Dict[str, Any]:
        """Switch to window by title"""
        try:
            def find_window(title):
                hwnd = win32gui.FindWindow(None, title)
                if hwnd:
                    return hwnd
                
                # Try partial match
                found_hwnd = None
                def callback(hwnd, results):
                    nonlocal found_hwnd
                    if win32gui.IsWindowVisible(hwnd):
                        win_title = win32gui.GetWindowText(hwnd)
                        if title.lower() in win_title.lower():
                            found_hwnd = hwnd
                            return False
                    return True
                
                win32gui.EnumWindows(callback, None)
                return found_hwnd
            
            hwnd = find_window(window_title)
            if hwnd:
                # Check if window is minimized
                placement = win32gui.GetWindowPlacement(hwnd)
                is_minimized = (placement[1] == win32con.SW_SHOWMINIMIZED)
                
                if is_minimized:
                    logger.info(f"Window is minimized, restoring: {window_title} (hwnd={hwnd})")
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                else:
                    logger.info(f"Window is not minimized, just bringing to front: {window_title} (hwnd={hwnd})")
                
                # Bring window to foreground (works for both restored and already-visible windows)
                win32gui.SetForegroundWindow(hwnd)
                result = f"Switched to window: {window_title}"
                self._log_action("switch_window", {"window_title": window_title, "hwnd": hwnd})
                logger.info(f"Successfully switched to window: {window_title}")
                return {"success": True, "result": result}
            else:
                return {"success": False, "result": f"Window not found: {window_title}"}
        except Exception as e:
            logger.error(f"Switch window error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_window_info(self) -> Dict[str, Any]:
        """Get information about active window"""
        try:
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            try:
                process = psutil.Process(pid)
                process_name = process.name()
            except:
                process_name = "Unknown"
            
            rect = win32gui.GetWindowRect(hwnd)
            
            info = {
                "hwnd": hwnd,
                "title": title,
                "pid": pid,
                "process": process_name,
                "position": {"x": rect[0], "y": rect[1]},
                "size": {"width": rect[2] - rect[0], "height": rect[3] - rect[1]}
            }
            
            self._log_action("get_window_info", info)
            return {"success": True, "result": "Retrieved window information", "info": info}
        except Exception as e:
            logger.error(f"Get window info error: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== FILE OPERATIONS ====================
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """Read file contents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = f"Read {len(content)} characters from {file_path}"
            self._log_action("read_file", {"file_path": file_path, "size": len(content)})
            
            return {"success": True, "result": result, "content": content}
        except Exception as e:
            logger.error(f"Read file error: {e}")
            return {"success": False, "error": str(e)}
    
    def write_file(self, file_path: str, content: str, mode: str = 'w') -> Dict[str, Any]:
        """Write content to file"""
        try:
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
            
            result = f"Wrote {len(content)} characters to {file_path}"
            self._log_action("write_file", {"file_path": file_path, "size": len(content), "mode": mode})
            
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Write file error: {e}")
            return {"success": False, "error": str(e)}
    
    def list_directory(self, directory_path: str) -> Dict[str, Any]:
        """List directory contents"""
        try:
            path = Path(directory_path)
            if not path.exists():
                return {"success": False, "error": f"Directory not found: {directory_path}"}
            
            items = []
            for item in path.iterdir():
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": item.stat().st_mtime
                })
            
            result = f"Listed {len(items)} items in {directory_path}"
            self._log_action("list_directory", {"directory_path": directory_path, "count": len(items)})
            
            return {"success": True, "result": result, "items": items}
        except Exception as e:
            logger.error(f"List directory error: {e}")
            return {"success": False, "error": str(e)}
    
    def search_files(self, search_path: str, pattern: str) -> Dict[str, Any]:
        """Search for files"""
        try:
            path = Path(search_path)
            if not path.exists():
                return {"success": False, "error": f"Directory not found: {search_path}"}
            
            files = list(path.glob(f"**/{pattern}"))
            file_paths = [str(f) for f in files]
            
            result = f"Found {len(files)} files matching '{pattern}'"
            self._log_action("search_files", {"search_path": search_path, "pattern": pattern, "count": len(files)})
            
            return {"success": True, "result": result, "files": file_paths}
        except Exception as e:
            logger.error(f"Search files error: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== SYSTEM OPERATIONS ====================
    
    def run_command(self, command: str, shell: bool = True) -> Dict[str, Any]:
        """Execute shell command"""
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
            self._log_action("run_command", {"command": command, "returncode": result.returncode})
            
            return {
                "success": result.returncode == 0,
                "result": f"Command executed with return code {result.returncode}",
                "output": output
            }
        except Exception as e:
            logger.error(f"Run command error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            info = {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "count_logical": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent": disk.percent
                }
            }
            
            self._log_action("get_system_info", info)
            return {"success": True, "result": "Retrieved system information", "info": info}
        except Exception as e:
            logger.error(f"Get system info error: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== UTILITY OPERATIONS ====================
    
    def wait(self, seconds: float) -> Dict[str, Any]:
        """Wait for specified duration"""
        try:
            time.sleep(seconds)
            result = f"Waited {seconds} seconds"
            self._log_action("wait", {"seconds": seconds})
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Wait error: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== HELPER METHODS ====================
    
    def _log_action(self, action_name: str, params: Dict[str, Any]) -> None:
        """Log action to history"""
        action = {
            "action": action_name,
            "params": params,
            "timestamp": time.time()
        }
        self.action_history.append(action)
        self.last_action = action
        logger.info(f"Action: {action_name} - {params}")
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get action history"""
        return self.action_history
    
    def clear_action_history(self) -> None:
        """Clear action history"""
        self.action_history = []
        self.last_action = None


# Global instance
desktop_agent = DesktopAgent()
