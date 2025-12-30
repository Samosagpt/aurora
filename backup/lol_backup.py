import base64
import json
import os
import re
import tempfile
import time
import webbrowser
from urllib.parse import parse_qs, urlparse

import ollama
import pyautogui  # For keyboard controls
import pyttsx3  # NEW: local TTS (Windows SAPI5 by default)
import speech_recognition as sr

# ------------------ CONFIG ------------------
GOOGLE_SLIDES_URL = "https://docs.google.com/presentation/d/1jeOFtKZ1hc0FyDyCtIZrCXQokAPHm1jyT0tCy_1w6MA/edit?slide=id.p20#slide=id.p20"
MODEL_NAME = "qwen3-vl:235b-cloud"  # or a specific tag from Ollama library
PHRASE_TIME_LIMIT = 4  # seconds per mic phrase
VOICE_RATE = 180  # words per minute
RAG_DB_PATH = "bg.json"  # RAG database file path (JSON format)
# --------------------------------------------


# --------------------------------------------
# RAG DATABASE FUNCTIONS
# --------------------------------------------
def load_rag_database(db_path):
    """Load RAG database from JSON file."""
    if not os.path.exists(db_path):
        return {"documents": []}
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading RAG database: {e}")
        return {"documents": []}


def search_rag_database(db, query, top_k=3):
    """Search RAG database for relevant documents."""
    if not db.get("documents"):
        return []

    query_lower = query.lower()
    query_words = set(query_lower.split())

    scored_docs = []
    for doc in db["documents"]:
        content_lower = doc["content"].lower()
        # Simple word matching score
        matches = sum(1 for word in query_words if word in content_lower)
        score = matches / len(query_words) if query_words else 0

        if score > 0:
            scored_docs.append((score, doc))

    # Sort by score and return top_k
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [{"score": score, **doc} for score, doc in scored_docs[:top_k]]


# --------------------------------------------
# SCREENSHOT CAPTURE FUNCTION
# --------------------------------------------
def capture_slide_screenshot():
    """Capture current screen as screenshot and save to temp file."""
    temp_dir = tempfile.gettempdir()
    timestamp = int(time.time() * 1000)
    out_path = os.path.join(temp_dir, f"slide_{timestamp}.png")

    try:
        # Take screenshot of entire screen
        screenshot = pyautogui.screenshot()
        screenshot.save(out_path)
        return out_path
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rag_data, f, indent=2, ensure_ascii=False)

    print(f"[RAG] Database saved to {output_path} ({total_slides} slides)")
    return output_path


def load_rag_database(filepath=RAG_DB_PATH):
    """Load RAG database from bg.json file."""
    if not os.path.exists(filepath):
        print(f"[RAG] Database file not found: {filepath}")
        return {}

    try:
        with open(filepath, "r", encoding="utf-8") as f:
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
    if re.search(r"\b(explain|describe|summarize|talk about)\b.*\b(slide|this)\b", t) or re.search(
        r"\bwhat'?s on (this )?slide\b", t
    ):
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
                    pyautogui.press("right")  # Use arrow key for next slide
                    print("➡️  Next slide")
                    time.sleep(0.3)  # Small delay for slide transition
                    continue
                elif cmd == "prev":
                    pyautogui.press("left")  # Use arrow key for previous slide
                    print("⬅️  Previous slide")
                    time.sleep(0.3)  # Small delay for slide transition
                    continue
                elif cmd == "goto" and arg is not None:
                    # Type slide number and press Enter to go to specific slide
                    pyautogui.typewrite(str(arg), interval=0.1)
                    pyautogui.press("enter")
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
