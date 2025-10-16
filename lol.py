import os
import re
import time
import base64
import tempfile
import json
import webbrowser
from urllib.parse import urlparse, parse_qs

import speech_recognition as sr
import ollama
import pyttsx3
import pyautogui

# ------------------ CONFIG ------------------
GOOGLE_SLIDES_URL = "https://docs.google.com/presentation/d/1jeOFtKZ1hc0FyDyCtIZrCXQokAPHm1jyT0tCy_1w6MA/present?slide=id.p1"
MODEL_NAME = "qwen3-vl:235b-cloud"
PHRASE_TIME_LIMIT = 4
VOICE_RATE = 180
RAG_DB_PATH = "bg.json"
# --------------------------------------------

# --------------------------------------------
# RAG DATABASE FUNCTIONS
# --------------------------------------------
def load_rag_database(db_path=RAG_DB_PATH):
    """Load RAG database from JSON file."""
    if not os.path.exists(db_path):
        return {"documents": []}
    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[RAG Error] Failed to load database: {e}")
        return {"documents": []}

def search_rag_database(db, query, top_k=3):
    """Search RAG database for relevant documents."""
    if not db.get("documents"):
        return []
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_docs = []
    for doc in db["documents"]:
        content_lower = doc.get("content", "").lower()
        matches = sum(1 for word in query_words if word in content_lower)
        score = matches / len(query_words) if query_words else 0
        
        if score > 0:
            scored_docs.append((score, doc))
    
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [{"score": score, **doc} for score, doc in scored_docs[:top_k]]

def build_slide_context(query, rag_db):
    """Build context from RAG database based on query."""
    results = search_rag_database(rag_db, query, top_k=3)
    
    if not results:
        return ""
    
    context_parts = []
    for result in results:
        content = result.get("content", "")
        title = result.get("metadata", {}).get("title", "Slide")
        score = result.get("score", 0)
        if content.strip():
            context_parts.append(f"--- {title} (relevance: {score:.2f}) ---\n{content}")
    
    return "\n\n".join(context_parts)

# --------------------------------------------
# SCREENSHOT CAPTURE FUNCTION
# --------------------------------------------
def capture_slide_screenshot():
    """Capture current screen as screenshot and save to temp file."""
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time() * 1000)
    out_path = os.path.join(temp_dir, f"slide_{timestamp}.png")
    
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save(out_path)
        return out_path
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None

def b64_of_file(path):
    """Convert file to base64 encoding."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def send_to_ollama_analyze(transcript, image_b64, slide_context=""):
    """Send image and transcript to Ollama for analysis."""
    messages = [
        {
            "role": "user",
            "content": (
                "You are assisting a live presentation.\n"
                "User just said:\n"
                f"{transcript}\n\n"
                f"Slide Context (RAG):\n{slide_context}\n\n"
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
    """Send image to Ollama for slide narration."""
    messages = [
        {
            "role": "user",
            "content": (
                f"Slide Context (RAG):\n{slide_context}\n\n"
                "Describe the visible slide for narration to an audience.\n"
                "Make it 3-5 spoken-style sentences covering: title, key visuals/text, any charts/tables trends, and the main takeaway.\n"
                "Avoid reading long text verbatim; summarize and highlight the most important points.\n"
                "Conclude with one suggested transition to the next idea."
            ),
            "images": [image_b64],
        }
    ]
    resp = ollama.chat(model=MODEL_NAME, messages=messages)
    return resp.get("message", {}).get("content", "").strip()

def parse_command(text):
    """Parse voice command from text."""
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
    if re.search(r"\b(explain|describe|summarize|talk about)\b.*\b(slide|this)\b", t) or \
       re.search(r"\bwhat'?s on (this )?slide\b", t):
        return ("describe", None)
    return (None, None)

# ---------- TTS (pyttsx3) ----------
_tts_engine = None
def init_tts():
    """Initialize text-to-speech engine."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", VOICE_RATE)

def speak(text):
    """Speak text using TTS."""
    try:
        init_tts()
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    except Exception as e:
        print(f"[TTS error] {e}")

# ---------- ASR ----------
def recognize_phrase(recognizer, audio):
    """Recognize speech from audio using Google Speech Recognition."""
    return recognizer.recognize_google(audio)

def main():
    print("=" * 60)
    print("  AURORA PRESENTATION - VOICE CONTROLLED")
    print("=" * 60)
    print("\n Initializing presentation system...")
    
    try:
        print(" Opening Google Slides presentation...")
        presentation_url = GOOGLE_SLIDES_URL
        
        if "/edit" in presentation_url:
            presentation_url = presentation_url.replace("/edit", "/present")
        elif "/present" not in presentation_url:
            presentation_url += "/present"
        
        webbrowser.open(presentation_url)
        print(f" Opened: {presentation_url}")
        print(" Note: Please click on the browser window to focus it for keyboard controls")
        
        time.sleep(2)
        pyautogui.press('f11')  # Start presentation mode

        if not os.path.exists(RAG_DB_PATH):
            print(f"[RAG] Warning: No database found at {RAG_DB_PATH}")
            print("[RAG] You can add knowledge to bg.json manually or via the RAG system")
        else:
            print(f"[RAG]  Using existing database: {RAG_DB_PATH}")

        rag_db_data = load_rag_database(RAG_DB_PATH)

        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        print(" Calibrating microphone...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

        tmpdir = tempfile.mkdtemp(prefix="slides_live_")
        
        print("\n" + "=" * 60)
        print(" PRESENTATION READY")
        print("=" * 60)
        print("\n Voice Commands Available:")
        print("   'next' - Next slide")
        print("   'back' or 'previous' - Previous slide")
        print("   'go to slide [number]' - Jump to specific slide")
        print("   'describe this slide' - AI narrates the slide")
        print("   Ask any question about the current slide")
        print("\n Listening for commands...\n")

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
                    pyautogui.press('right')
                    print("  Next slide")
                    time.sleep(0.3)
                    continue
                elif cmd == "prev":
                    pyautogui.press('left')
                    print("  Previous slide")
                    time.sleep(0.3)
                    continue
                elif cmd == "goto" and arg is not None:
                    pyautogui.typewrite(str(arg), interval=0.1)
                    pyautogui.press('enter')
                    print(f" Jump to slide {arg}")
                    time.sleep(0.3)
                    continue
                elif cmd == "describe":
                    print(f"  Describing current slide...")
                    png_path = capture_slide_screenshot()
                    if png_path:
                        image_b64 = b64_of_file(png_path)
                        slide_context = build_slide_context(transcript, rag_db_data)
                        narration = send_to_ollama_describe(image_b64, slide_context)
                        if narration:
                            print(f"\n [Narration]\n{narration}\n")
                            speak(narration)
                    continue

                print(f" Analyzing current slide...")
                png_path = capture_slide_screenshot()
                if png_path:
                    image_b64 = b64_of_file(png_path)
                    slide_context = build_slide_context(transcript, rag_db_data)
                    reply = send_to_ollama_analyze(transcript, image_b64, slide_context)
                    if reply:
                        print(f"\n [AI Response]\n{reply}\n")

            except KeyboardInterrupt:
                print("\n\n Exiting presentation...")
                break
            except Exception as e:
                print(f"  Error: {e}")
                continue

        print("\n Presentation ended successfully.")
        
    except Exception as e:
        print(f"\n Error starting presentation: {e}")
        print("Please ensure:")
        print("  1. A web browser is available")
        print("  2. Internet connection is active")
        print(f"  3. Google Slides URL: {GOOGLE_SLIDES_URL}")
        input("\nPress Enter to exit...")
        return

if __name__ == "__main__":
    main()
