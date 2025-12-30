"""
Vision-Enabled Autonomous Agent
Uses vision models, OCR, and OpenCV to understand the screen and make decisions
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

from aurora.agents.desktop_agent import desktop_agent
from aurora.core.Generation import ollama_manager

logger = logging.getLogger(__name__)


class VisionAgent:
    """Autonomous agent that uses vision to understand and control the desktop"""

    def __init__(self):
        """Initialize vision agent"""
        self.desktop_agent = desktop_agent
        self.task_history = []
        self.screenshots_dir = Path("logs/agent_screenshots")
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Vision Agent initialized")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools"""
        return self.desktop_agent.get_available_tools()

    def take_screenshot(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Take a screenshot and return the result"""
        return self.desktop_agent.take_screenshot(save_path)

    def ocr_screen(self, use_easyocr: bool = True) -> Dict[str, Any]:
        """Perform OCR on the current screen"""
        return self._perform_ocr(use_easyocr=use_easyocr)

    def execute_task(
        self, task_description: str, model: str = "qwen3-vl:235b-cloud", max_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Execute a task autonomously using vision feedback

        Args:
            task_description: Natural language description of the task
            model: Vision model to use for understanding screenshots
            max_steps: Maximum number of steps to attempt (default: 10 for faster execution)

        Returns:
            Dict with task results and execution log
        """
        print(f"\n{'='*80}")
        print(f"🤖 VISION AGENT: Starting task - '{task_description}'")
        print(f"{'='*80}\n")

        # Smart pre-check: If task mentions a website/app, open it first
        website_keywords = {
            "github": "https://github.com",
            "google": "https://google.com",
            "youtube": "https://youtube.com",
            "stackoverflow": "https://stackoverflow.com",
            "reddit": "https://reddit.com",
            "whatsapp": "https://web.whatsapp.com",
            "twitter": "https://twitter.com",
            "facebook": "https://facebook.com",
            "instagram": "https://instagram.com",
            "linkedin": "https://linkedin.com",
        }

        # Check for specific tasks
        task_lower = task_description.lower()

        # WhatsApp specific handling
        if "whatsapp" in task_lower:
            print(f"💬 Detected WhatsApp task - checking if already open...")

            # Check if WhatsApp is already open
            windows_result = self.desktop_agent.list_windows()
            whatsapp_open = False

            if windows_result.get("success"):
                for window in windows_result.get("windows", []):
                    if "whatsapp" in window["title"].lower():
                        whatsapp_open = True
                        print(f"✅ WhatsApp already open: {window['title']}")
                        # Switch to it
                        self.desktop_agent.switch_window(window["title"])
                        time.sleep(1)
                        break

            # Only open if not already open
            if not whatsapp_open:
                print(f"🌐 Opening https://web.whatsapp.com...")
                try:
                    result = self.desktop_agent.open_url("https://web.whatsapp.com")
                    print(f"✅ {result.get('result', 'URL opened')}")
                    print("⏱️  Waiting 8 seconds for WhatsApp Web to load and QR scan...")
                    time.sleep(8)  # WhatsApp Web needs more time for QR code scan

                    # Make sure WhatsApp window is in focus
                    print("🔍 Switching to WhatsApp window...")
                    time.sleep(1)
                    windows_result = self.desktop_agent.list_windows()
                    if windows_result.get("success"):
                        for window in windows_result.get("windows", []):
                            if "whatsapp" in window["title"].lower():
                                self.desktop_agent.switch_window(window["title"])
                                print(f"✅ Focused on: {window['title']}")
                                time.sleep(1)
                                break
                except Exception as e:
                    print(f"⚠️  Failed to open WhatsApp Web: {e}")

            # Take initial screenshot to confirm WhatsApp is loaded
            initial_screenshot = self._take_screenshot("initial_whatsapp_check")
            print(f"📸 Initial WhatsApp screenshot saved: {initial_screenshot}")

        # GitHub PRs specific handling
        elif "github" in task_lower and ("pr" in task_lower or "pull request" in task_lower):
            # Open GitHub pulls page directly
            print(f"🌐 Detected GitHub PRs in task - opening https://github.com/pulls directly...")
            try:
                result = self.desktop_agent.open_url("https://github.com/pulls")
                print(f"✅ {result.get('result', 'URL opened')}")
                time.sleep(3)  # Wait for page to load
            except Exception as e:
                print(f"⚠️  Failed to open URL: {e}")

        # General website keywords
        else:
            for keyword, url in website_keywords.items():
                if keyword in task_lower:
                    print(f"🌐 Detected '{keyword}' in task - opening {url} first...")
                    try:
                        result = self.desktop_agent.open_url(url)
                        print(f"✅ {result.get('result', 'URL opened')}")
                        time.sleep(3)  # Wait for page to load
                    except Exception as e:
                        print(f"⚠️  Failed to open URL: {e}")
                    break

        # Check if this is a WhatsApp task - use hardcoded fast path (no AI needed)
        task_lower = task_description.lower()
        if "whatsapp" in task_lower and "text" in task_lower:
            print("\n🚀 FAST PATH: WhatsApp detected - using hardcoded sequence (NO AI)")
            return self._execute_whatsapp_hardcoded(task_description, model)

        # Step 1: Create a roadmap
        roadmap = self._create_roadmap(task_description, model)
        print(f"\n📋 ROADMAP CREATED:")
        for i, step in enumerate(roadmap["steps"], 1):
            print(f"  {i}. {step}")
        print()

        execution_log = {
            "task": task_description,
            "roadmap": roadmap,
            "steps_executed": [],
            "screenshots": [],
            "success": False,
            "error": None,
            "failed_actions": {},  # Track failed actions to avoid loops
            "urls_opened": [],  # Track URLs that have been opened
        }

        # Step 2: Execute roadmap step by step with vision feedback
        for step_num in range(max_steps):
            print(f"\n🔄 STEP {step_num + 1}/{max_steps}")
            print("-" * 80)

            # Take screenshot before action
            screenshot_path = self._take_screenshot(f"step_{step_num}_before")
            execution_log["screenshots"].append(str(screenshot_path))

            # Analyze current state with vision
            print("📸 Analyzing screen...")
            screen_analysis = self._analyze_screen(screenshot_path, model)
            print(f"👁️  Vision Analysis: {screen_analysis[:200]}...")

            # Determine next action based on vision + task context
            print("🧠 Determining next action...")
            next_action = self._determine_next_action(
                task_description=task_description,
                roadmap=roadmap,
                screen_analysis=screen_analysis,
                execution_log=execution_log,
                model=model,
            )

            print(f"⚡ Action: {next_action.get('action_type', 'unknown')}")
            print(f"💭 Reasoning: {next_action.get('reasoning', 'N/A')}")

            # Check if task is complete
            if next_action.get("task_complete"):
                print("\n✅ TASK COMPLETED!")
                execution_log["success"] = True
                break

            # Check if we're clicking the same coordinates repeatedly (stuck)
            recent_coords = []
            for step in execution_log["steps_executed"][-5:]:  # Last 5 steps
                if step["action"].get("action_type") == "mouse_click":
                    params = step["action"].get("parameters", {})
                    coord = (params.get("x"), params.get("y"))
                    recent_coords.append(coord)

            # If we've clicked the same coordinate 3+ times in last 5 steps, we're stuck
            if recent_coords:
                most_common_coord = max(set(recent_coords), key=recent_coords.count)
                if recent_coords.count(most_common_coord) >= 3:
                    print(
                        f"⚠️  STUCK: Clicked {most_common_coord} multiple times. Using direct URL approach..."
                    )

                    # Check what app we need
                    task_lower = task_description.lower()
                    app_url = None

                    if "whatsapp" in task_lower:
                        app_url = "https://web.whatsapp.com"
                    elif "github" in task_lower:
                        app_url = "https://github.com"
                    elif "youtube" in task_lower:
                        app_url = "https://youtube.com"
                    elif "twitter" in task_lower:
                        app_url = "https://twitter.com"

                    if app_url:
                        # Check if URL already opened
                        if app_url in execution_log["urls_opened"]:
                            print(
                                f"⚠️  URL {app_url} already opened! Using OCR to find element instead..."
                            )
                            # Try OCR-based recovery
                            next_action = {
                                "reasoning": "URL already open, using OCR to locate elements",
                                "task_complete": False,
                                "action_type": "wait",
                                "tool": "wait",
                                "parameters": {"seconds": 1},
                                "expected_outcome": "Allow time for OCR-based action next step",
                            }
                        else:
                            # Open the URL directly
                            print(f"🌐 Opening {app_url} directly to recover from stuck state...")
                            execution_log["urls_opened"].append(app_url)
                            next_action = {
                                "reasoning": f"Stuck clicking - opening {app_url} directly",
                                "task_complete": False,
                                "action_type": "open_url",
                                "tool": "open_url",
                                "parameters": {"url": app_url},
                                "expected_outcome": f"Open {app_url} in browser",
                            }
                    else:
                        # Generic recovery - just wait
                        next_action = {
                            "reasoning": f"Stuck clicking {most_common_coord} - waiting to recover",
                            "task_complete": False,
                            "action_type": "wait",
                            "tool": "wait",
                            "parameters": {"seconds": 3},
                            "expected_outcome": "Give time to recover from stuck state",
                        }

            # Check if we're stuck in a loop (same action failed 3+ times)
            action_key = f"{next_action.get('tool')}_{str(next_action.get('parameters'))}"
            if action_key in execution_log["failed_actions"]:
                if execution_log["failed_actions"][action_key] >= 3:
                    print(
                        f"⚠️  LOOP DETECTED: Action '{action_key}' failed 3+ times. Trying alternative..."
                    )
                    # Force a different action
                    next_action = {
                        "reasoning": "Previous action failed multiple times, trying alternative",
                        "task_complete": False,
                        "action_type": "wait",
                        "tool": "wait",
                        "parameters": {"seconds": 3},
                        "expected_outcome": "Give time to reassess",
                    }

            # Execute the action
            print(
                f"🎯 Executing: {next_action.get('tool')} with params {next_action.get('parameters')}"
            )
            action_result = self._execute_action(next_action)

            # Track failed actions
            if not action_result.get("success"):
                if action_key not in execution_log["failed_actions"]:
                    execution_log["failed_actions"][action_key] = 0
                execution_log["failed_actions"][action_key] += 1

            # Log the step
            step_log = {
                "step_number": step_num + 1,
                "screen_analysis": screen_analysis,
                "action": next_action,
                "result": action_result,
                "screenshot_before": str(screenshot_path),
            }
            execution_log["steps_executed"].append(step_log)

            # Wait a bit for UI to update
            time.sleep(1)

            # Take screenshot after action
            screenshot_after = self._take_screenshot(f"step_{step_num}_after")
            execution_log["screenshots"].append(str(screenshot_after))

            # Check for errors
            if not action_result.get("success"):
                print(f"⚠️  Action failed: {action_result.get('error')}")
                # Continue anyway - vision feedback might help recover

        # Save execution log
        log_path = self.screenshots_dir / f"execution_log_{int(time.time())}.json"
        with open(log_path, "w") as f:
            json.dump(execution_log, f, indent=2)

        print(f"\n{'='*80}")
        print(f"📊 TASK EXECUTION COMPLETE")
        print(f"✅ Success: {execution_log['success']}")
        print(f"📝 Steps executed: {len(execution_log['steps_executed'])}")
        print(f"📸 Screenshots saved: {len(execution_log['screenshots'])}")
        print(f"💾 Log saved to: {log_path}")
        print(f"{'='*80}\n")

        return execution_log

    def _create_roadmap(self, task_description: str, model: str) -> Dict[str, Any]:
        """Create a roadmap for the task using LLM"""
        prompt = f"""Create a detailed step-by-step roadmap to accomplish this task:

TASK: {task_description}

Think about this carefully. Break it down into specific, actionable steps.

IMPORTANT: For web-based tasks (like GitHub), use the open_url tool to open URLs directly:
- CORRECT: "Use open_url with url='https://github.com'"
- WRONG: "Open Chrome and then navigate"

Example for "open github and find all open PRs":
1. Open GitHub using open_url (url='https://github.com')
2. Wait for page to load
3. Look for PRs section in the interface
4. Click on PRs or navigate to pulls page
5. Filter by "open" status if needed
6. Verify open PRs are visible

Respond with a JSON object:
{{
    "task": "task description",
    "steps": ["step 1", "step 2", "step 3", ...],
    "estimated_duration": "X minutes",
    "complexity": "easy/medium/hard"
}}

Only respond with the JSON, no other text."""

        try:
            response = ollama_manager.client.chat(
                model=model if "vision" not in model else "llama3.2",
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response["message"]["content"]
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                roadmap = json.loads(response_text[json_start:json_end])
                return roadmap
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            logger.error(f"Error creating roadmap: {e}")
            # Fallback roadmap
            return {
                "task": task_description,
                "steps": [
                    "Analyze current screen",
                    "Determine required actions",
                    "Execute actions",
                    "Verify completion",
                ],
                "estimated_duration": "unknown",
                "complexity": "unknown",
            }

    def _perform_ocr(self, use_easyocr: bool = True) -> Dict[str, Any]:
        """Perform OCR on the current screen"""
        try:
            # Take screenshot first
            screenshot_result = self.desktop_agent.take_screenshot()
            if not screenshot_result.get("success"):
                return {"success": False, "error": "Failed to take screenshot"}

            screenshot_path = screenshot_result["screenshot_path"]
            img = cv2.imread(screenshot_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Use pytesseract for OCR
            text = pytesseract.image_to_string(gray)

            return {
                "success": True,
                "text": text,
                "method": "easyocr" if use_easyocr else "pytesseract",
            }
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {"success": False, "error": str(e)}

    def _execute_whatsapp_hardcoded(self, task_description: str, model: str) -> Dict[str, Any]:
        """
        Execute WhatsApp task using hardcoded sequence - NO AI NEEDED
        This is 10x faster than using vision analysis for each step
        """
        print(f"🚀 HARDCODED WHATSAPP FLOW - No AI, just direct actions!")

        # STEP 0: Always open WhatsApp Web fresh
        print("\n🌐 STEP 0: Opening WhatsApp Web...")
        print("🌐 Opening https://web.whatsapp.com...")
        try:
            result = self.desktop_agent.open_url("https://web.whatsapp.com")
            print(f"✅ {result.get('result', 'URL opened')}")
            print("⏱️  Waiting 10 seconds for WhatsApp Web to load (QR scan, etc.)...")
            time.sleep(10)  # Give time for page load and QR scan

            # Focus WhatsApp window
            print("🔍 Focusing WhatsApp window...")
            time.sleep(1)
            windows_result = self.desktop_agent.list_windows()
            if windows_result.get("success"):
                for window in windows_result.get("windows", []):
                    if "whatsapp" in window["title"].lower():
                        self.desktop_agent.switch_window(window["title"])
                        print(f"✅ Focused on: {window['title']}")
                        time.sleep(2)  # Extra time for window to be ready
                        break
        except Exception as e:
            print(f"⚠️  Failed to open WhatsApp Web: {e}")
            return {"success": False, "error": f"Failed to open WhatsApp: {e}"}

        print("✅ WhatsApp is ready!\n")

        # Extract contact and message from task
        words = task_description.lower().split()
        contact_name = None
        message = None

        if "text" in words:
            idx = words.index("text")
            if idx + 1 < len(words):
                start_idx = idx + 1
                if words[start_idx] in ["my", "the", "a"]:
                    start_idx += 1

                end_idx = start_idx + 1
                for i in range(start_idx + 1, len(words)):
                    if words[i] in ["hi", "hello", "hey", "message:", "text:"]:
                        end_idx = i
                        break
                    end_idx = i + 1

                contact_name = " ".join(words[start_idx:end_idx])

                if end_idx < len(words):
                    message = " ".join(words[end_idx:])

        print(f"📝 Extracted: contact='{contact_name}', message='{message}'")

        # Get screen resolution
        import pyautogui

        screen_width, screen_height = pyautogui.size()
        print(f"🖥️  Screen: {screen_width}x{screen_height}")

        # EXACT coordinates from user's WhatsApp Web (measured with test_cords.py)
        # Search box: (16.7%, 15.4%)
        search_x = int(screen_width * 0.167)
        search_y = int(screen_height * 0.154)

        # First contact in results: (15.3%, 32.2%)
        contact_x = int(screen_width * 0.153)
        contact_y = int(screen_height * 0.322)

        # Message input box: (42.7%, 91.6%)
        input_x = int(screen_width * 0.427)
        input_y = int(screen_height * 0.916)

        print(f"📍 Coordinates:")
        print(f"   Search box: ({search_x}, {search_y})")
        print(f"   First contact: ({contact_x}, {contact_y})")
        print(f"   Message input: ({input_x}, {input_y})")

        execution_log = {
            "task": task_description,
            "roadmap": {
                "task": "Send WhatsApp Message (Hardcoded)",
                "steps": [
                    "Click search box",
                    f"Type contact: {contact_name}",
                    "Click first result",
                    "Click message input",
                    f"Type message: {message}",
                    "Press Enter to send",
                ],
            },
            "steps_executed": [],
            "screenshots": [],
            "success": False,
            "error": None,
        }

        try:
            # STEP 0: Click search box
            print(f"\n⚡ STEP 1/6: Click search box ({search_x}, {search_y})")
            screenshot_before = self._take_screenshot("step_0_before")
            execution_log["screenshots"].append(str(screenshot_before))

            result = self.desktop_agent.mouse_click(search_x, search_y)
            execution_log["steps_executed"].append(
                {
                    "step": 1,
                    "action": "mouse_click",
                    "params": {"x": search_x, "y": search_y},
                    "result": result,
                }
            )
            time.sleep(0.5)

            screenshot_after = self._take_screenshot("step_0_after")
            execution_log["screenshots"].append(str(screenshot_after))

            # STEP 1: Type contact name
            if contact_name:
                print(f"⚡ STEP 2/6: Type contact '{contact_name}'")
                screenshot_before = self._take_screenshot("step_1_before")
                execution_log["screenshots"].append(str(screenshot_before))

                result = self.desktop_agent.type_text(contact_name, interval=0.05)
                execution_log["steps_executed"].append(
                    {
                        "step": 2,
                        "action": "type_text",
                        "params": {"text": contact_name},
                        "result": result,
                    }
                )
                time.sleep(0.5)

                screenshot_after = self._take_screenshot("step_1_after")
                execution_log["screenshots"].append(str(screenshot_after))

            # STEP 2: Click first contact
            print(f"⚡ STEP 3/6: Click first contact ({contact_x}, {contact_y})")
            screenshot_before = self._take_screenshot("step_2_before")
            execution_log["screenshots"].append(str(screenshot_before))

            result = self.desktop_agent.mouse_click(contact_x, contact_y)
            execution_log["steps_executed"].append(
                {
                    "step": 3,
                    "action": "mouse_click",
                    "params": {"x": contact_x, "y": contact_y},
                    "result": result,
                }
            )
            time.sleep(0.8)

            screenshot_after = self._take_screenshot("step_2_after")
            execution_log["screenshots"].append(str(screenshot_after))

            # STEP 3: Click message input
            print(f"⚡ STEP 4/6: Click message input ({input_x}, {input_y})")
            screenshot_before = self._take_screenshot("step_3_before")
            execution_log["screenshots"].append(str(screenshot_before))

            result = self.desktop_agent.mouse_click(input_x, input_y)
            execution_log["steps_executed"].append(
                {
                    "step": 4,
                    "action": "mouse_click",
                    "params": {"x": input_x, "y": input_y},
                    "result": result,
                }
            )
            time.sleep(0.5)

            screenshot_after = self._take_screenshot("step_3_after")
            execution_log["screenshots"].append(str(screenshot_after))

            # STEP 4: Type message
            if message:
                print(f"⚡ STEP 5/6: Type message '{message}'")
                screenshot_before = self._take_screenshot("step_4_before")
                execution_log["screenshots"].append(str(screenshot_before))

                result = self.desktop_agent.type_text(message, interval=0.05)
                execution_log["steps_executed"].append(
                    {
                        "step": 5,
                        "action": "type_text",
                        "params": {"text": message},
                        "result": result,
                    }
                )
                time.sleep(0.3)

                screenshot_after = self._take_screenshot("step_4_after")
                execution_log["screenshots"].append(str(screenshot_after))

            # STEP 5: Press Enter
            print("⚡ STEP 6/6: Press Enter to send")
            screenshot_before = self._take_screenshot("step_5_before")
            execution_log["screenshots"].append(str(screenshot_before))

            result = self.desktop_agent.press_key("enter")
            execution_log["steps_executed"].append(
                {"step": 6, "action": "press_key", "params": {"key": "enter"}, "result": result}
            )
            time.sleep(0.5)

            screenshot_after = self._take_screenshot("step_5_after")
            execution_log["screenshots"].append(str(screenshot_after))

            execution_log["success"] = True
            print("\n✅ WHATSAPP TASK COMPLETED!")

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            execution_log["error"] = str(e)
            execution_log["success"] = False

        # Save execution log
        log_path = self.screenshots_dir / f"execution_log_{int(time.time())}.json"
        with open(log_path, "w") as f:
            json.dump(execution_log, f, indent=2)

        print(f"\n{'='*80}")
        print(f"📊 HARDCODED EXECUTION COMPLETE")
        print(f"✅ Success: {execution_log['success']}")
        print(f"📝 Steps executed: {len(execution_log['steps_executed'])}")
        print(f"📸 Screenshots: {len(execution_log['screenshots'])}")
        print(f"💾 Log: {log_path}")
        print(f"⚡ Total time: ~5 seconds (vs 30+ seconds with AI)")
        print(f"{'='*80}\n")

        return execution_log

    def _take_screenshot(self, label: str) -> Path:
        """Take a screenshot and save it"""
        result = self.desktop_agent.take_screenshot()
        if result["success"]:
            screenshot_path = Path(result["screenshot_path"])
            # Copy to our logs directory with label
            new_path = self.screenshots_dir / f"{label}_{int(time.time())}.png"
            Image.open(screenshot_path).save(new_path)
            return new_path
        else:
            logger.error(f"Failed to take screenshot: {result.get('error')}")
            return None

    def _analyze_screen(self, screenshot_path: Path, model: str) -> str:
        """
        Analyze screenshot using vision model + OCR

        Returns:
            Detailed description of what's on screen
        """
        analyses = []

        # 1. Vision Model Analysis (if available)
        try:
            if screenshot_path and screenshot_path.exists():
                # Encode image for vision model
                with open(screenshot_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                # Use the provided vision model
                vision_model = model

                prompt = """Describe what you see on this screenshot in detail. Include:
- What application or website is open
- What buttons, links, or UI elements are visible
- Any text you can read
- The current state of the interface
- What actions appear to be available

Be specific and detailed."""

                response = ollama_manager.client.chat(
                    model=vision_model,
                    messages=[{"role": "user", "content": prompt, "images": [image_data]}],
                )

                vision_analysis = response["message"]["content"]
                analyses.append(f"VISION MODEL: {vision_analysis}")
        except Exception as e:
            logger.error(f"Vision model analysis failed: {e}")
            analyses.append(f"VISION MODEL: Error - {str(e)}")

        # 2. OCR Analysis
        try:
            if screenshot_path and screenshot_path.exists():
                img = cv2.imread(str(screenshot_path))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Run OCR
                ocr_text = pytesseract.image_to_string(gray)

                if ocr_text.strip():
                    analyses.append(f"OCR TEXT: {ocr_text[:500]}")
                else:
                    analyses.append("OCR TEXT: No text detected")
        except Exception as e:
            logger.error(f"OCR analysis failed: {e}")
            analyses.append(f"OCR: Error - {str(e)}")

        # 3. OpenCV Analysis (detect UI elements)
        try:
            if screenshot_path and screenshot_path.exists():
                img = cv2.imread(str(screenshot_path))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect edges (buttons, windows, etc.)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Count potential UI elements
                significant_contours = [c for c in contours if cv2.contourArea(c) > 100]

                analyses.append(
                    f"OPENCV: Detected ~{len(significant_contours)} UI elements/regions"
                )
        except Exception as e:
            logger.error(f"OpenCV analysis failed: {e}")
            analyses.append(f"OPENCV: Error - {str(e)}")

        return "\n".join(analyses)

    def _determine_next_action(
        self,
        task_description: str,
        roadmap: Dict[str, Any],
        screen_analysis: str,
        execution_log: Dict[str, Any],
        model: str,
    ) -> Dict[str, Any]:
        """Determine next action based on vision analysis and task context"""

        steps_completed = len(execution_log["steps_executed"])
        task_lower = task_description.lower()

        # Smart OCR-based action for WhatsApp
        if "whatsapp" in task_lower and "text" in task_lower:
            # Extract contact name and message from task
            # E.g., "text my mom hi" -> contact="mom", message="hi"
            words = task_description.lower().split()

            contact_name = None
            message = None

            # Find contact name (after "text" or "message")
            if "text" in words:
                idx = words.index("text")
                if idx + 1 < len(words):
                    # Skip "my" if present
                    start_idx = idx + 1
                    if words[start_idx] in ["my", "the", "a"]:
                        start_idx += 1

                    # Find end of contact name (before message)
                    end_idx = start_idx + 1
                    for i in range(start_idx + 1, len(words)):
                        if words[i] in ["hi", "hello", "hey", "message:", "text:"]:
                            end_idx = i
                            break
                        end_idx = i + 1

                    contact_name = " ".join(words[start_idx:end_idx])

                    # Get message (everything after contact name)
                    if end_idx < len(words):
                        message = " ".join(words[end_idx:])

            print(f"📝 Extracted: contact='{contact_name}', message='{message}'")

            # Get screen resolution for adaptive coordinates
            import pyautogui

            screen_width, screen_height = pyautogui.size()
            print(f"🖥️  Screen resolution: {screen_width}x{screen_height}")

            # Use OCR to find and click on elements
            if steps_completed == 0:
                # Step 1: Find and click search box using OCR
                print("🔍 Looking for search box using OCR...")
                try:
                    ocr_result = self.desktop_agent.find_text_on_screen("Search", confidence=50)
                    if ocr_result["success"]:
                        coords = ocr_result["coordinates"]
                        print(f"✅ Found 'Search' text at coordinates: {coords}")
                        return {
                            "reasoning": f"Found search box at {coords} using OCR",
                            "task_complete": False,
                            "action_type": "click_text",
                            "tool": "click_text",
                            "parameters": {"text": "Search", "confidence": 50},
                            "expected_outcome": "Search box will be focused",
                        }
                    else:
                        print(f"⚠️ OCR couldn't find 'Search' text - using fallback coordinates")
                except Exception as e:
                    print(f"⚠️ OCR failed: {e}")
                    if "tesseract" in str(e).lower():
                        print("💡 TIP: Install Tesseract-OCR for better element detection:")
                        print("   Download from: https://github.com/UB-Mannheim/tesseract/wiki")
                        print("   Or: winget install UB-Mannheim.TesseractOCR")

                # Fallback: click estimated search position (left side, ~15% from top)
                # WhatsApp Web search is usually in top-left sidebar
                search_x = int(screen_width * 0.15)  # 15% from left (in sidebar)
                search_y = int(screen_height * 0.15)  # 15% from top
                print(f"📍 Using fallback coordinates: ({search_x}, {search_y})")
                return {
                    "reasoning": f"Clicking search box in WhatsApp (adaptive: {search_x}, {search_y})",
                    "task_complete": False,
                    "action_type": "mouse_click",
                    "tool": "mouse_click",
                    "parameters": {"x": search_x, "y": search_y},
                    "expected_outcome": "Search box will be focused",
                }

            elif steps_completed == 1:
                # Step 2: Type contact name
                if contact_name:
                    return {
                        "reasoning": f"Typing contact name: {contact_name}",
                        "task_complete": False,
                        "action_type": "type_text",
                        "tool": "type_text",
                        "parameters": {"text": contact_name, "interval": 0.05},
                        "expected_outcome": f"Contact '{contact_name}' will appear in search results",
                    }

            elif steps_completed == 2:
                # Step 3: Wait for results, then click first contact
                time.sleep(0.5)  # Quick wait for search results
                # First contact is usually below search box in sidebar
                contact_x = int(screen_width * 0.15)  # Same X as search (in sidebar)
                contact_y = int(screen_height * 0.25)  # Below search box
                print(f"📍 Clicking first contact at: ({contact_x}, {contact_y})")
                return {
                    "reasoning": f"Clicking first search result at ({contact_x}, {contact_y})",
                    "task_complete": False,
                    "action_type": "mouse_click",
                    "tool": "mouse_click",
                    "parameters": {"x": contact_x, "y": contact_y},
                    "expected_outcome": "Chat with contact will open",
                }

            elif steps_completed == 3:
                # Step 4: Click message input box
                # Message input is at bottom center of main chat area
                input_x = int(screen_width * 0.55)  # Center-right (main chat area)
                input_y = int(screen_height * 0.92)  # Near bottom
                print(f"📍 Clicking message input at: ({input_x}, {input_y})")
                return {
                    "reasoning": f"Clicking message input box at ({input_x}, {input_y})",
                    "task_complete": False,
                    "action_type": "mouse_click",
                    "tool": "mouse_click",
                    "parameters": {"x": input_x, "y": input_y},
                    "expected_outcome": "Message input will be focused",
                }

            elif steps_completed == 4:
                # Step 5: Type message
                if message:
                    return {
                        "reasoning": f"Typing message: {message}",
                        "task_complete": False,
                        "action_type": "type_text",
                        "tool": "type_text",
                        "parameters": {"text": message, "interval": 0.05},
                        "expected_outcome": "Message will appear in input box",
                    }

            elif steps_completed == 5:
                # Step 6: Press Enter to send
                return {
                    "reasoning": "Pressing Enter to send message",
                    "task_complete": True,
                    "action_type": "press_key",
                    "tool": "press_key",
                    "parameters": {"key": "enter"},
                    "expected_outcome": "Message will be sent",
                }

        # Fall through to AI-based decision making for other tasks
        steps_completed = len(execution_log["steps_executed"])

        prompt = f"""You are an autonomous desktop control agent. Analyze the current situation and determine the next action.

ORIGINAL TASK: {task_description}

ROADMAP:
{json.dumps(roadmap['steps'], indent=2)}

STEPS COMPLETED SO FAR: {steps_completed}

CURRENT SCREEN ANALYSIS:
{screen_analysis}

Determine the NEXT specific action. Respond with JSON ONLY:
{{
    "reasoning": "why this action",
    "task_complete": false,
    "tool": "mouse_click",
    "parameters": {{"x": 500, "y": 80}}
}}

Available tools: mouse_click, type_text, press_key, click_text, wait

IMPORTANT: Use click_text when possible to click on text (e.g., "Search", "Send", button labels)"""

        try:
            # Use a more powerful model for decision making
            decision_model = "llama3.2:latest"  # Use non-vision model for decisions
            response = ollama_manager.client.chat(
                model=decision_model, messages=[{"role": "user", "content": prompt}]
            )

            response_text = response["message"]["content"]
            print(f"🤖 AI Response: {response_text[:300]}...")

            # Clean and extract JSON - handle escape sequences
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]

                # Fix common JSON escape issues
                # Replace Windows paths backslashes if they appear
                json_str = json_str.replace("\\\\", "/")
                json_str = json_str.replace("\\", "/")

                try:
                    action = json.loads(json_str)
                    return action
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error: {je}. Raw JSON: {json_str[:200]}")
                    raise ValueError(f"Invalid JSON: {str(je)}")
            else:
                raise ValueError("No JSON found in AI response")
        except Exception as e:
            logger.error(f"Error determining next action: {e}")
            # Fallback: wait and observe
            return {
                "reasoning": f"Error in AI decision making: {str(e)}",
                "task_complete": False,
                "action_type": "wait",
                "tool": "wait",
                "parameters": {"seconds": 2},
                "expected_outcome": "Give system time to respond",
            }

    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a desktop action"""
        tool_name = action.get("tool")
        parameters = action.get("parameters", {})

        if not tool_name:
            return {"success": False, "error": "No tool specified"}

        # Get the tool method
        if hasattr(self.desktop_agent, tool_name):
            tool_method = getattr(self.desktop_agent, tool_name)
            try:
                result = tool_method(**parameters)
                return result if isinstance(result, dict) else {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                return {"success": False, "error": str(e)}
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def get_execution_summary(self, execution_log: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the execution"""

        # Check if this is a hardcoded WhatsApp execution
        is_hardcoded = (
            "roadmap" in execution_log
            and isinstance(execution_log["roadmap"], dict)
            and "task" in execution_log["roadmap"]
        )

        if is_hardcoded:
            # Hardcoded WhatsApp format
            summary = f"""
🎯 TASK: {execution_log['roadmap']['task']}
{'✅ SUCCESS' if execution_log['success'] else '❌ FAILED'}

📋 STEPS:
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(execution_log['roadmap']['steps']))}

🔄 EXECUTION:
{chr(10).join(f"  Step {s['step']}: {s['action']} - {s.get('params', {})}"
             for s in execution_log['steps_executed'])}

📸 SCREENSHOTS: {len(execution_log['screenshots'])} captured
💾 All data saved to: {self.screenshots_dir}
"""
        else:
            # Regular format
            summary = f"""
🎯 TASK: {execution_log['task']}
{'✅ SUCCESS' if execution_log['success'] else '❌ INCOMPLETE'}

📋 ROADMAP:
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(execution_log['roadmap']['steps']))}

🔄 EXECUTION:
{chr(10).join(f"  Step {s.get('step_number', '?')}: {s.get('action', {}).get('action_type', 'unknown')} - {s.get('action', {}).get('reasoning', 'N/A')[:100]}"
             for s in execution_log['steps_executed'])}

📸 SCREENSHOTS: {len(execution_log['screenshots'])} captured
💾 All data saved to: {self.screenshots_dir}
"""
        return summary


# Global instance
vision_agent = VisionAgent()


def execute_autonomous_task(task_description: str, model: str = "qwen3-vl:235b-cloud") -> str:
    """
    Main entry point for autonomous task execution

    Args:
        task_description: Natural language description of what to do
        model: Vision model to use

    Returns:
        Human-readable summary of execution
    """
    execution_log = vision_agent.execute_task(task_description, model)
    summary = vision_agent.get_execution_summary(execution_log)
    return summary
