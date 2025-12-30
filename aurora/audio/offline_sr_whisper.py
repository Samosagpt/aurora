"""
Enhanced offline speech recognition using Whisper
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper

from aurora.core.config import config

logger = logging.getLogger(__name__)


class WhisperSpeechRecognizer:
    """Enhanced Whisper-based speech recognition with better error handling"""

    def __init__(self):
        self.model = None
        self.model_name = config.WHISPER_MODEL
        self.sample_rate = config.SAMPLE_RATE
        self.channels = config.CHANNELS
        self.chunk = config.CHUNK
        self.silence_threshold = config.SILENCE_THRESHOLD
        self.silence_duration = config.SILENCE_DURATION
        self._load_model()

    def _load_model(self) -> bool:
        """Load Whisper model with error handling"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            return False

    def _calculate_rms(self, block: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) for audio level detection"""
        return np.sqrt(np.mean(block**2))

    def _record_audio(self, output_path: Path, verbose: bool = False) -> bool:
        """Record audio with silence detection"""
        try:
            if verbose:
                print("Press Enter to start recording...")
                input("")

            frames = []
            silent_chunks = 0
            max_silent_chunks = int(self.silence_duration * self.sample_rate / self.chunk)
            recording_started = False

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.chunk,
            ) as stream:

                if verbose:
                    print("🎤 Recording... speak now.")

                while True:
                    try:
                        block, _ = stream.read(self.chunk)
                        audio_data = block[:, 0] if self.channels == 1 else block

                        # Calculate audio level
                        audio_level = self._calculate_rms(audio_data)

                        # Start recording when speech is detected
                        if not recording_started and audio_level > self.silence_threshold:
                            recording_started = True
                            if verbose:
                                print("🗣️ Speech detected, recording...")

                        # Only process after recording has started
                        if recording_started:
                            frames.append(audio_data)

                            if audio_level < self.silence_threshold:
                                silent_chunks += 1
                                if silent_chunks > max_silent_chunks:
                                    if verbose:
                                        print("🔇 Silence detected, stopping recording.")
                                    break
                            else:
                                silent_chunks = 0

                    except Exception as e:
                        logger.error(f"Error during audio recording: {e}")
                        return False

            if not frames:
                logger.warning("No audio recorded")
                return False

            # Save audio
            audio_np = np.concatenate(frames)
            sf.write(output_path, audio_np, self.sample_rate)

            if verbose:
                print(f"💾 Audio saved to {output_path}")

            logger.info(f"Audio recorded successfully: {len(frames)} chunks")
            return True

        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return False

    def transcribe_audio(self, audio_path: Path) -> Optional[str]:
        """Transcribe audio file using Whisper"""
        try:
            if not self.model:
                if not self._load_model():
                    return None

            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return None

            logger.info(f"Transcribing audio: {audio_path}")
            result = self.model.transcribe(str(audio_path))

            transcription = result.get("text", "").strip()

            if transcription:
                logger.info(f"Transcription successful: {len(transcription)} characters")
                return transcription
            else:
                logger.warning("Empty transcription result")
                return None

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    def speech_recognition(
        self,
        model_name: Optional[str] = None,
        output_path: Optional[Path] = None,
        silence_threshold: Optional[float] = None,
        silence_duration: Optional[float] = None,
        verbose: bool = False,
    ) -> Optional[str]:
        """Complete speech recognition pipeline"""
        try:
            # Update parameters if provided
            if model_name and model_name != self.model_name:
                self.model_name = model_name
                self._load_model()

            if silence_threshold is not None:
                self.silence_threshold = silence_threshold

            if silence_duration is not None:
                self.silence_duration = silence_duration

            if output_path is None:
                output_path = config.AUDIO_FILE

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Record audio
            if not self._record_audio(output_path, verbose):
                return None

            # Transcribe audio
            transcription = self.transcribe_audio(output_path)

            # Clean up audio file if transcription successful
            if transcription and output_path.exists():
                try:
                    output_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete audio file: {e}")

            return transcription

        except Exception as e:
            logger.error(f"Error in speech recognition pipeline: {e}")
            return None

    def test_microphone(self) -> Tuple[bool, str]:
        """Test microphone availability and functionality"""
        try:
            # List available audio devices
            devices = sd.query_devices()
            input_devices = [d for d in devices if d["max_input_channels"] > 0]

            if not input_devices:
                return False, "No input audio devices found"

            # Test recording a short sample
            test_duration = 2  # seconds
            test_data = sd.rec(
                int(test_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
            )
            sd.wait()

            # Check if we got audio data
            if np.max(np.abs(test_data)) < 0.001:
                return False, "Microphone seems to be muted or not working"

            return True, f"Microphone test successful. Found {len(input_devices)} input devices."

        except Exception as e:
            return False, f"Microphone test failed: {str(e)}"


# Create global instance
speech_recognizer = WhisperSpeechRecognizer()


# Backward compatibility function
def speech_recognition(
    model_name: str = config.WHISPER_MODEL,
    output_path: str = str(config.AUDIO_FILE),
    silence_threshold: float = config.SILENCE_THRESHOLD,
    silence_duration: float = config.SILENCE_DURATION,
    verbose: bool = False,
) -> Optional[str]:
    """Backward compatibility function"""
    return speech_recognizer.speech_recognition(
        model_name=model_name,
        output_path=Path(output_path),
        silence_threshold=silence_threshold,
        silence_duration=silence_duration,
        verbose=verbose,
    )
