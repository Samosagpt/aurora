import speech_recognition as sr

from aurora.audio import offline_sr_whisper as srw
from aurora.audio import offline_text2speech as t2s
from aurora.core.Generation import init_model
from aurora.handlers.prompthandler import handle_query


def main():
    init_model()
    t2s.speak_text("Hello! I am Aurora.")
    recognizer = sr.Recognizer()
    mic = sr.Microphone(device_index=1)

    while True:
        audio = srw.speech_recognition()
        try:
            query = recognizer.recognize_google(audio)
            response = handle_query(query)
            t2s.speak_text(response)
        except Exception:
            t2s.speak_text("Sorry, I didn't catch that.")


if __name__ == "__main__":
    main()
