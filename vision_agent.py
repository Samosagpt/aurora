"""
Vision-Enabled Autonomous Agent
Uses vision models, OCR, and OpenCV to understand the screen and make decisions
"""

import json
import logging
import base64
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image
from desktop_agent import desktop_agent
from Generation import ollama_manager

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
        
    def execute_task(self, task_description: str, model: str = "llava:latest", max_steps: int = 20) -> Dict[str, Any]:
        """
        Execute a task autonomously using vision feedback
        
        Args:
            task_description: Natural language description of the task
            model: Vision model to use for understanding screenshots
            max_steps: Maximum number of steps to attempt
            
        Returns:
            Dict with task results and execution log
        """
        print(f"\n{'='*80}")
        print(f"🤖 VISION AGENT: Starting task - '{task_description}'")
        print(f"{'='*80}\n")
        
        # Smart pre-check: If task mentions a website/app, open it first
        website_keywords = {
            'github': 'https://github.com',
            'google': 'https://google.com',
            'youtube': 'https://youtube.com',
            'stackoverflow': 'https://stackoverflow.com',
            'reddit': 'https://reddit.com',
            'whatsapp': 'https://web.whatsapp.com',
            'twitter': 'https://twitter.com',
            'facebook': 'https://facebook.com',
            'instagram': 'https://instagram.com',
            'linkedin': 'https://linkedin.com'
        }
        
        # Check for specific tasks
        task_lower = task_description.lower()
        
        # WhatsApp specific handling
        if 'whatsapp' in task_lower:
            print(f"💬 Detected WhatsApp in task - opening https://web.whatsapp.com...")
            try:
                result = self.desktop_agent.open_url('https://web.whatsapp.com')
                print(f"✅ {result.get('result', 'URL opened')}")
                print("⏱️  Waiting 8 seconds for WhatsApp Web to load and QR scan...")
                time.sleep(8)  # WhatsApp Web needs more time for QR code scan
                
                # Take initial screenshot to confirm WhatsApp is loaded
                initial_screenshot = self._take_screenshot("initial_whatsapp_check")
                print(f"📸 Initial WhatsApp screenshot saved: {initial_screenshot}")
                
            except Exception as e:
                print(f"⚠️  Failed to open WhatsApp Web: {e}")
        
        # GitHub PRs specific handling
        elif 'github' in task_lower and ('pr' in task_lower or 'pull request' in task_lower):
            # Open GitHub pulls page directly
            print(f"🌐 Detected GitHub PRs in task - opening https://github.com/pulls directly...")
            try:
                result = self.desktop_agent.open_url('https://github.com/pulls')
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
        
        # Step 1: Create a roadmap
        roadmap = self._create_roadmap(task_description, model)
        print(f"\n📋 ROADMAP CREATED:")
        for i, step in enumerate(roadmap['steps'], 1):
            print(f"  {i}. {step}")
        print()
        
        execution_log = {
            "task": task_description,
            "roadmap": roadmap,
            "steps_executed": [],
            "screenshots": [],
            "success": False,
            "error": None,
            "failed_actions": {}  # Track failed actions to avoid loops
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
                model=model
            )
            
            print(f"⚡ Action: {next_action.get('action_type', 'unknown')}")
            print(f"💭 Reasoning: {next_action.get('reasoning', 'N/A')}")
            
            # Check if task is complete
            if next_action.get('task_complete'):
                print("\n✅ TASK COMPLETED!")
                execution_log["success"] = True
                break
            
            # Check if we're clicking the same coordinates repeatedly (stuck)
            recent_coords = []
            for step in execution_log['steps_executed'][-5:]:  # Last 5 steps
                if step['action'].get('action_type') == 'mouse_click':
                    params = step['action'].get('parameters', {})
                    coord = (params.get('x'), params.get('y'))
                    recent_coords.append(coord)
            
            # If we've clicked the same coordinate 3+ times in last 5 steps, we're stuck
            if recent_coords:
                most_common_coord = max(set(recent_coords), key=recent_coords.count)
                if recent_coords.count(most_common_coord) >= 3:
                    print(f"⚠️  STUCK: Clicked {most_common_coord} multiple times. Using direct URL approach...")
                    
                    # Check what app we need
                    task_lower = task_description.lower()
                    app_url = None
                    
                    if 'whatsapp' in task_lower:
                        app_url = 'https://web.whatsapp.com'
                    elif 'github' in task_lower:
                        app_url = 'https://github.com'
                    elif 'youtube' in task_lower:
                        app_url = 'https://youtube.com'
                    elif 'twitter' in task_lower:
                        app_url = 'https://twitter.com'
                    
                    if app_url:
                        # Open the URL directly
                        print(f"🌐 Opening {app_url} directly to recover from stuck state...")
                        next_action = {
                            "reasoning": f"Stuck clicking - opening {app_url} directly",
                            "task_complete": False,
                            "action_type": "open_url",
                            "tool": "open_url",
                            "parameters": {"url": app_url},
                            "expected_outcome": f"Open {app_url} in browser"
                        }
                    else:
                        # Generic recovery - just wait
                        next_action = {
                            "reasoning": f"Stuck clicking {most_common_coord} - waiting to recover",
                            "task_complete": False,
                            "action_type": "wait",
                            "tool": "wait",
                            "parameters": {"seconds": 3},
                            "expected_outcome": "Give time to recover from stuck state"
                        }
            
            # Check if we're stuck in a loop (same action failed 3+ times)
            action_key = f"{next_action.get('tool')}_{str(next_action.get('parameters'))}"
            if action_key in execution_log['failed_actions']:
                if execution_log['failed_actions'][action_key] >= 3:
                    print(f"⚠️  LOOP DETECTED: Action '{action_key}' failed 3+ times. Trying alternative...")
                    # Force a different action
                    next_action = {
                        "reasoning": "Previous action failed multiple times, trying alternative",
                        "task_complete": False,
                        "action_type": "wait",
                        "tool": "wait",
                        "parameters": {"seconds": 3},
                        "expected_outcome": "Give time to reassess"
                    }
            
            # Execute the action
            print(f"🎯 Executing: {next_action.get('tool')} with params {next_action.get('parameters')}")
            action_result = self._execute_action(next_action)
            
            # Track failed actions
            if not action_result.get('success'):
                if action_key not in execution_log['failed_actions']:
                    execution_log['failed_actions'][action_key] = 0
                execution_log['failed_actions'][action_key] += 1
            
            # Log the step
            step_log = {
                "step_number": step_num + 1,
                "screen_analysis": screen_analysis,
                "action": next_action,
                "result": action_result,
                "screenshot_before": str(screenshot_path)
            }
            execution_log["steps_executed"].append(step_log)
            
            # Wait a bit for UI to update
            time.sleep(1)
            
            # Take screenshot after action
            screenshot_after = self._take_screenshot(f"step_{step_num}_after")
            execution_log["screenshots"].append(str(screenshot_after))
            
            # Check for errors
            if not action_result.get('success'):
                print(f"⚠️  Action failed: {action_result.get('error')}")
                # Continue anyway - vision feedback might help recover
        
        # Save execution log
        log_path = self.screenshots_dir / f"execution_log_{int(time.time())}.json"
        with open(log_path, 'w') as f:
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
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response['message']['content']
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
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
                    "Verify completion"
                ],
                "estimated_duration": "unknown",
                "complexity": "unknown"
            }
    
    def _perform_ocr(self, use_easyocr: bool = True) -> Dict[str, Any]:
        """Perform OCR on the current screen"""
        try:
            # Take screenshot first
            screenshot_result = self.desktop_agent.take_screenshot()
            if not screenshot_result.get('success'):
                return {"success": False, "error": "Failed to take screenshot"}
            
            screenshot_path = screenshot_result['screenshot_path']
            img = cv2.imread(screenshot_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(gray)
            
            return {
                "success": True,
                "text": text,
                "method": "easyocr" if use_easyocr else "pytesseract"
            }
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _take_screenshot(self, label: str) -> Path:
        """Take a screenshot and save it"""
        result = self.desktop_agent.take_screenshot()
        if result['success']:
            screenshot_path = Path(result['screenshot_path'])
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
                with open(screenshot_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                
                # Use vision-capable model
                vision_model = "llava:latest" if "llava" not in model else model
                
                prompt = """Describe what you see on this screenshot in detail. Include:
- What application or website is open
- What buttons, links, or UI elements are visible
- Any text you can read
- The current state of the interface
- What actions appear to be available

Be specific and detailed."""

                response = ollama_manager.client.chat(
                    model=vision_model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [image_data]
                    }]
                )
                
                vision_analysis = response['message']['content']
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
                
                analyses.append(f"OPENCV: Detected ~{len(significant_contours)} UI elements/regions")
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
        model: str
    ) -> Dict[str, Any]:
        """Determine next action based on vision analysis and task context"""
        
        steps_completed = len(execution_log['steps_executed'])
        
        prompt = f"""You are an autonomous desktop control agent. Analyze the current situation and determine the next action.

ORIGINAL TASK: {task_description}

ROADMAP:
{json.dumps(roadmap['steps'], indent=2)}

STEPS COMPLETED SO FAR: {steps_completed}
{json.dumps([s['action'] for s in execution_log['steps_executed'][-3:]], indent=2) if execution_log['steps_executed'] else '[]'}

CURRENT SCREEN ANALYSIS:
{screen_analysis}

Based on this information, determine the NEXT specific action to take to complete the task.

PREVIOUSLY TRIED COORDINATES (DO NOT REPEAT THESE):
{', '.join([f"({step['action'].get('parameters', {}).get('x', 'N/A')},{step['action'].get('parameters', {}).get('y', 'N/A')})" for step in execution_log['steps_executed'][-5:]]) if execution_log['steps_executed'] else 'None yet'}

CRITICAL RULES FOR ACTION:
1. **BE PROACTIVE**: Click buttons, type in search boxes, navigate pages - don't just wait!
2. **USE VISION ANALYSIS**: The screen analysis tells you what's visible - click on those elements!
3. **AVOID REPETITION**: DO NOT use the same coordinates more than twice! Try DIFFERENT positions!
4. **CLICK COORDINATES**: Use mouse_click with estimated x,y positions (VARY them each time):
   - Top navigation (search, profile): y=50-100, x=100-700 (try different x values!)
   - Left sidebar (contacts, chats): x=100-250, y=200-600
   - Main content area buttons: x=400-800, y=200-600
   - Search boxes: Usually near top, x=300-600, y=50-150
   - Message input (WhatsApp, chat apps): Bottom center, x=400-700, y=700-900
5. **TYPE IN FIELDS**: If you see a search box or input field, click it first, then type_text
6. **PRESS KEYS**: Use press_key for Enter (after typing), Tab (to move focus), etc.
7. **NAVIGATE**: If task mentions "find X", use search or click relevant navigation links
8. **IF STUCK**: If clicking same spot 3+ times, try: press Windows key + type app name, or try different coordinates

PLATFORM-SPECIFIC EXAMPLES:

**For WhatsApp "text mom hi":**
1. Click search box in left sidebar (x=150, y=100)
2. Type "mom" or contact name
3. Click on contact result (x=150, y=200)
4. Click message input box at bottom (x=550, y=850)
5. Type "hi"
6. Press Enter or click send button (x=750, y=850)

**For GitHub "find open PRs":**
1. Click search box at top (x=500, y=80)
2. Type "pulls"
3. Press Enter
4. Click "Open" filter

**For Google "search something":**
1. Click search box (x=500, y=400)
2. Type query
3. Press Enter

AVAILABLE TOOLS:
- mouse_click(x, y, button='left', clicks=1) - Click at screen coordinates
- mouse_move(x, y) - Move mouse cursor
- type_text(text) - Type text (works in focused input field)
- press_key(key) - Press keyboard key: 'enter', 'tab', 'escape', 'backspace', etc.
- open_url(url) - Open website: 'https://github.com', 'github.com/pulls'
- wait(seconds) - Wait (use sparingly, only when page is loading)

RESPOND with JSON ONLY (no markdown, no code blocks):
{{
    "reasoning": "Explain what you see and why this action makes sense",
    "task_complete": false,
    "action_type": "mouse_click",
    "tool": "mouse_click",
    "parameters": {{"x": 500, "y": 80}},
    "expected_outcome": "What should happen after this action"
}}

If task is complete, set "task_complete": true

IMPORTANT: Respond with raw JSON only, no surrounding text or markdown."""

        try:
            # Use a more powerful model for decision making
            decision_model = "llama3.2:latest"  # Use non-vision model for decisions
            response = ollama_manager.client.chat(
                model=decision_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response['message']['content']
            print(f"🤖 AI Response: {response_text[:300]}...")
            
            # Clean and extract JSON - handle escape sequences
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                
                # Fix common JSON escape issues
                # Replace Windows paths backslashes if they appear
                json_str = json_str.replace('\\\\', '/')
                json_str = json_str.replace('\\', '/')
                
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
                "expected_outcome": "Give system time to respond"
            }
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a desktop action"""
        tool_name = action.get('tool')
        parameters = action.get('parameters', {})
        
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
        summary = f"""
🎯 TASK: {execution_log['task']}
{'✅ SUCCESS' if execution_log['success'] else '❌ INCOMPLETE'}

📋 ROADMAP:
{chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(execution_log['roadmap']['steps']))}

🔄 EXECUTION:
{chr(10).join(f"  Step {s['step_number']}: {s['action'].get('action_type', 'unknown')} - {s['action'].get('reasoning', 'N/A')[:100]}" 
             for s in execution_log['steps_executed'])}

📸 SCREENSHOTS: {len(execution_log['screenshots'])} captured
💾 All data saved to: {self.screenshots_dir}
"""
        return summary


# Global instance
vision_agent = VisionAgent()


def execute_autonomous_task(task_description: str, model: str = "llava:latest") -> str:
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
