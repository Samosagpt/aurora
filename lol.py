import os
import re
import time
import base64
import tempfile
import pythoncom
import win32com.client as win32
import json

import speech_recognition as sr
import ollama
import pyttsx3  # NEW: local TTS (Windows SAPI5 by default)
import pyautogui  # For keyboard controls
import win32gui
import win32con

# ------------------ CONFIG ------------------
PPT_PATH = r"C:\Users\ptejaji\Downloads\Aurora_Assistant.pptx"  # change this
MODEL_NAME = "qwen3-vl:235b-cloud"                    # or a specific tag from Ollama library
PHRASE_TIME_LIMIT = 4                       # seconds per mic phrase
EXPORT_WIDTH = 1600
EXPORT_HEIGHT = 900
VOICE_RATE = 180                            # words per minute
RAG_DB_PATH = "bg.json"                     # RAG database file path (JSON format)
# --------------------------------------------

def get_slideshow_view(powerpoint_app):
    pythoncom.PumpWaitingMessages()
    for _ in range(50):
        try:
            windows = powerpoint_app.SlideShowWindows
            if windows.Count > 0:
                return windows(1).View
        except Exception:
            pass
        time.sleep(0.1)
    raise RuntimeError("Could not acquire SlideShowView. Is the slide show running?")

def activate_powerpoint_window():
    """Find and activate the PowerPoint slideshow window."""
    def enum_callback(hwnd, window_list):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            # Look for PowerPoint Slide Show window
            if "PowerPoint" in window_text and ("Slide Show" in window_text or "Slideshow" in window_text):
                window_list.append(hwnd)
        return True
    
    # Wait a bit for the window to appear
    max_attempts = 30
    for attempt in range(max_attempts):
        windows = []
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            # Activate the first matching window
            hwnd = windows[0]
            try:
                # Restore if minimized
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                
                # Bring to front
                win32gui.SetForegroundWindow(hwnd)
                win32gui.BringWindowToTop(hwnd)
                print("✓ PowerPoint window activated")
                return True
            except Exception as e:
                print(f"Warning: Could not activate window: {e}")
                # Try alternative method
                try:
                    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                    win32gui.SetForegroundWindow(hwnd)
                    return True
                except:
                    pass
        
        time.sleep(0.2)
    
    print("Warning: Could not automatically activate PowerPoint window")
    return False

def export_current_slide_as_png(presentation, slide_index, out_dir):
    slide = presentation.Slides(slide_index)
    out_path = os.path.join(out_dir, f"slide_{slide_index}.png")
    slide.Export(out_path, "PNG", EXPORT_WIDTH, EXPORT_HEIGHT)
    return out_path

def extract_slide_text(presentation, slide_index):
    """Extract all text content from a slide including shapes and text boxes."""
    slide = presentation.Slides(slide_index)
    text_content = []
    
    try:
        for shape in slide.Shapes:
            if shape.HasTextFrame:
                if shape.TextFrame.HasText:
                    text_content.append(shape.TextFrame.TextRange.Text)
    except Exception as e:
        print(f"[Warning] Error extracting text from slide {slide_index}: {e}")
    
    return "\n".join(text_content)

def build_rag_database(presentation, output_path=RAG_DB_PATH):
    """Build RAG database by extracting text from all slides and saving to bg.json."""
    print(f"[RAG] Building database from presentation...")
    total_slides = presentation.Slides.Count
    
    rag_data = {
        "presentation": {
            "title": "Aurora Presentation",
            "total_slides": total_slides,
            "generated": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "slides": []
    }
    
    for idx in range(1, total_slides + 1):
        text = extract_slide_text(presentation, idx)
        if text.strip():
            rag_data["slides"].append({
                "number": idx,
                "content": text
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rag_data, f, indent=2, ensure_ascii=False)
    
    print(f"[RAG] Database saved to {output_path} ({total_slides} slides)")
    return output_path

def load_rag_database(filepath=RAG_DB_PATH):
    """Load RAG database from bg.json file."""
    if not os.path.exists(filepath):
        print(f"[RAG] Database file not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to simple dict format: {slide_number: content}
        rag_db = {}
        if "slides" in data:
            for slide in data["slides"]:
                rag_db[slide["number"]] = slide["content"]
        
        print(f"[RAG] Loaded database with {len(rag_db)} slides")
        return rag_db
    except Exception as e:
        print(f"[RAG Error] Failed to load database: {e}")
        return {}

def build_slide_context(presentation, current_index, context_window=2):
    """Build RAG context from current and nearby slides."""
    # Load from RAG database instead of extracting on-the-fly
    rag_db = load_rag_database()
    
    if not rag_db:
        # Fallback to live extraction if database not available
        print("[RAG] Database not found, using live extraction")
        total_slides = presentation.Slides.Count
        start_idx = max(1, current_index - context_window)
        end_idx = min(total_slides, current_index + context_window)
        
        context_parts = []
        for idx in range(start_idx, end_idx + 1):
            text = extract_slide_text(presentation, idx)
            if text.strip():
                marker = " [CURRENT SLIDE]" if idx == current_index else ""
                context_parts.append(f"--- Slide {idx}{marker} ---\n{text}")
        
        return "\n\n".join(context_parts)
    
    # Use RAG database
    total_slides = len(rag_db)
    start_idx = max(1, current_index - context_window)
    end_idx = min(total_slides, current_index + context_window)
    
    context_parts = []
    for idx in range(start_idx, end_idx + 1):
        if idx in rag_db:
            text = rag_db[idx]
            if text.strip():
                marker = " [CURRENT SLIDE]" if idx == current_index else ""
                context_parts.append(f"--- Slide {idx}{marker} ---\n{text}")
    
    return "\n\n".join(context_parts)

def b64_of_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def send_to_ollama_analyze(transcript, image_b64, slide_context=""):
    messages = [
        {
            "role": "user",
            "content": (
                "You are assisting a live presentation.\n"
                "User just said:\n"
                f"{transcript}\n\n"
                f"Slide Text Content (RAG Context):\n{slide_context}\n\n"
                "Analyze the current slide image and respond concisely with:\n"
                "- A brief interpretation of the slide\n"
                "- How it relates to what was said\n"
                "- Any suggestion on the next point to cover"
            ),
            "images": [image_b64],
        }
    ]
    resp = ollama.chat(model=MODEL_NAME, messages=messages)
    return resp.get("message", {}).get("content", "").strip()

def send_to_ollama_describe(image_b64, slide_context=""):
    # Focused prompt for "talk about the slide itself"
    messages = [
        {
            "role": "user",
            "content": (
                f"Slide Text Content (RAG Context):\n{slide_context}\n\n"
                "Describe the visible slide for narration to an audience.\n"
                "Make it 3–5 spoken-style sentences covering: title, key visuals/text, any charts/tables trends, and the main takeaway.\n"
                "Avoid reading long text verbatim; summarize and highlight the most important points.\n"
                "Conclude with one suggested transition to the next idea."
            ),
            "images": [image_b64],
        }
    ]
    resp = ollama.chat(model=MODEL_NAME, messages=messages)
    return resp.get("message", {}).get("content", "").strip()

def parse_command(text):
    # Returns ("next"|"prev"|"goto"|"describe", n or None) or (None, None)
    t = text.lower().strip()
    if re.search(r"\bnext\b", t):
        return ("next", None)
    if re.search(r"\b(prev|previous|back)\b", t):
        return ("prev", None)
    m = re.search(r"\bgo\s*to\s*(?:slide\s*)?(\d+)\b", t)
    if m:
        try:
            n = int(m.group(1))
            return ("goto", n)
        except ValueError:
            pass
    # “talk about the slide” intents
    if re.search(r"\b(explain|describe|summarize|talk about)\b.*\b(slide|this)\b", t) or \
       re.search(r"\bwhat'?s on (this )?slide\b", t):
        return ("describe", None)
    return (None, None)

# ---------- TTS (pyttsx3) ----------
_tts_engine = None
def init_tts():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", VOICE_RATE)

def speak(text):
    try:
        init_tts()
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    except Exception as e:
        print(f"[TTS error] {e}")

# ---------- ASR ----------
def recognize_phrase(recognizer, audio):
    # Use your preferred recognizer here; as a simple default:
    # Google Web Speech (online) or replace with a faster-whisper pipeline.
    # recognizer.recognize_google(audio) is a drop-in option.
    # For local/offline, switch to a faster-whisper integration in your environment.
    return recognizer.recognize_google(audio)

def main():
    print("=" * 60)
    print("  AURORA PRESENTATION - VOICE CONTROLLED")
    print("=" * 60)
    print("\n🎤 Initializing presentation system...")
    
    try:
        powerpoint = win32.Dispatch("PowerPoint.Application")
        powerpoint.Visible = True
        presentation = powerpoint.Presentations.Open(PPT_PATH, WithWindow=True)

        # Check if bg.json exists, if not build it from presentation
        if not os.path.exists(RAG_DB_PATH):
            print("[RAG] No existing database found, building from presentation...")
            build_rag_database(presentation)
        else:
            print(f"[RAG] ✓ Using existing database: {RAG_DB_PATH}")

        # Start slide show and get view
        print("🎬 Starting slideshow...")
        presentation.SlideShowSettings.Run()
        view = get_slideshow_view(powerpoint)
        
        # Activate the PowerPoint window to bring it to front
        print("🔄 Switching to fullscreen presentation...")
        time.sleep(0.5)  # Give window time to fully render
        activate_powerpoint_window()

        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        print("🎤 Calibrating microphone...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

        tmpdir = tempfile.mkdtemp(prefix="ppt_live_")
        
        print("\n" + "=" * 60)
        print("✓ PRESENTATION READY")
        print("=" * 60)
        print("\n📢 Voice Commands Available:")
        print("  • 'next' - Next slide")
        print("  • 'back' or 'previous' - Previous slide")
        print("  • 'go to slide [number]' - Jump to specific slide")
        print("  • 'describe this slide' - AI narrates the slide")
        print("  • Ask any question about the current slide")
        print("\n🎧 Listening for commands...\n")

        while True:
            try:
                with mic as source:
                    audio = recognizer.listen(source, phrase_time_limit=PHRASE_TIME_LIMIT)
                try:
                    transcript = recognize_phrase(recognizer, audio)
                    print(f"\n[You said] {transcript}")
                except Exception:
                    continue
                if not transcript:
                    continue

                cmd, arg = parse_command(transcript)
                if cmd == "next":
                    pyautogui.press('right')  # Use arrow key for next slide
                    print("➡️  Next slide")
                    time.sleep(0.3)  # Small delay for slide transition
                    continue
                elif cmd == "prev":
                    pyautogui.press('left')  # Use arrow key for previous slide
                    print("⬅️  Previous slide")
                    time.sleep(0.3)  # Small delay for slide transition
                    continue
                elif cmd == "goto" and arg is not None:
                    # Type slide number and press Enter to go to specific slide
                    pyautogui.typewrite(str(arg), interval=0.1)
                    pyautogui.press('enter')
                    print(f"🎯 Jump to slide {arg}")
                    time.sleep(0.3)  # Small delay for slide transition
                    continue
                elif cmd == "describe":
                    current_index = view.CurrentShowPosition
                    print(f"🎙️  Describing slide {current_index}...")
                    png_path = export_current_slide_as_png(presentation, current_index, tmpdir)
                    image_b64 = b64_of_file(png_path)
                    # Build RAG context from current and nearby slides
                    slide_context = build_slide_context(presentation, current_index)
                    narration = send_to_ollama_describe(image_b64, slide_context)
                    if narration:
                        print(f"\n📖 [Narration]\n{narration}\n")
                        speak(narration)
                    continue

                # Default multimodal analysis (no control keyword)
                current_index = view.CurrentShowPosition
                print(f"🤔 Analyzing slide {current_index}...")
                png_path = export_current_slide_as_png(presentation, current_index, tmpdir)
                image_b64 = b64_of_file(png_path)
                # Build RAG context from current and nearby slides
                slide_context = build_slide_context(presentation, current_index)
                reply = send_to_ollama_analyze(transcript, image_b64, slide_context)
                if reply:
                    print(f"\n🤖 [AI Response]\n{reply}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Exiting presentation...")
                break
            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue

        try:
            presentation.SlideShowWindow.View.Exit()
        except Exception:
            pass
            
        print("\n✓ Presentation ended successfully.")
        
    except Exception as e:
        print(f"\n❌ Error starting presentation: {e}")
        print("Please ensure:")
        print("  1. PowerPoint is installed")
        print("  2. The PPT file exists at the configured path")
        print(f"  3. Path: {PPT_PATH}")
        input("\nPress Enter to exit...")
        return

if __name__ == "__main__":
    main()
