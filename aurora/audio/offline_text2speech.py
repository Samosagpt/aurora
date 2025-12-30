"""
Enhanced text-to-speech module using Bark and fallback options
"""

import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, Union

import sounddevice as sd
import soundfile as sf

from aurora.core.config import config

logger = logging.getLogger(__name__)


class TextToSpeechManager:
    """Enhanced TTS manager with multiple backends and error handling"""

    def __init__(self):
        self.bark_available = False
        self.pyttsx3_available = False
        self.current_engine = None
        self._initialize_engines()

    def _initialize_engines(self) -> None:
        """Initialize available TTS engines"""
        # Try to initialize Bark
        try:
            # Fix for PyTorch weights loading issue BEFORE importing bark
            import warnings

            import torch

            # Suppress specific warnings for cleaner output
            warnings.filterwarnings("ignore", category=UserWarning, module="torch")

            # Add numpy compatibility for PyTorch weights loading
            if hasattr(torch.serialization, "add_safe_globals"):
                import numpy

                torch.serialization.add_safe_globals(
                    [
                        numpy.core.multiarray.scalar,
                        numpy.core.multiarray._reconstruct,
                        numpy.ndarray,
                        numpy.dtype,
                    ]
                )

            # Set weights_only to False for Bark compatibility
            original_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return original_load(*args, **kwargs)

            torch.load = patched_load

            # Import bark modules (these need to be installed from GitHub)
            from bark import SAMPLE_RATE, generate_audio, preload_models

            self.bark_available = True
            self.bark_sample_rate = SAMPLE_RATE
            self.generate_audio = generate_audio
            self.preload_models = preload_models

            # Test Bark initialization
            logger.info("Testing Bark TTS engine...")
            try:
                # Try a small preload to verify it works
                self.preload_models()
                logger.info("Bark TTS engine initialized and tested successfully")
            except Exception as test_error:
                logger.warning(f"Bark preload test failed: {test_error}")
                logger.info("Bark available but may have issues - proceeding anyway")

        except ImportError as e:
            logger.warning(f"Bark not available: {e}")
            logger.info("To install Bark: pip install git+https://github.com/suno-ai/bark.git")
        except Exception as e:
            logger.error(f"Error initializing Bark: {e}")
            logger.info("Falling back to pyttsx3 TTS")

        # Try to initialize pyttsx3 as fallback
        try:
            import pyttsx3

            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_available = True
            logger.info("pyttsx3 TTS engine initialized successfully")
        except ImportError as e:
            logger.warning(f"pyttsx3 not available: {e}")
        except Exception as e:
            logger.error(f"Error initializing pyttsx3: {e}")
        # Set current engine
        if self.bark_available:
            self.current_engine = "bark"
        elif self.pyttsx3_available:
            self.current_engine = "pyttsx3"
        else:
            self.current_engine = None
            logger.warning("No TTS engines available")

    def _speak_with_bark(
        self, text: str, voice_preset: str = config.TTS_VOICE_PRESET, return_path: bool = False
    ) -> Optional[str]:
        """Generate speech using Bark"""
        try:
            from scipy.io.wavfile import write as write_wav

            logger.info(f"Generating audio with Bark: {len(text)} characters")

            # Preload models (cached after first call)
            try:
                self.preload_models()
            except Exception as preload_error:
                logger.error(f"Error preloading Bark models: {preload_error}")
                return None

            # Split long text into chunks if needed
            max_length = 200  # Bark works better with shorter text
            if len(text) > max_length:
                # Split into sentences or chunks
                import re

                sentences = re.split(r"[.!?]+", text)
                audio_chunks = []

                for sentence in sentences:
                    if sentence.strip():
                        try:
                            chunk_audio = self.generate_audio(
                                sentence.strip(), history_prompt=voice_preset
                            )
                            audio_chunks.append(chunk_audio)
                        except Exception as chunk_error:
                            logger.error(f"Error generating audio chunk: {chunk_error}")
                            continue

                if audio_chunks:
                    # Concatenate audio chunks
                    import numpy as np

                    audio_array = np.concatenate(audio_chunks)
                else:
                    logger.error("Failed to generate any audio chunks")
                    return None
            else:
                # Generate audio from text
                try:
                    audio_array = self.generate_audio(text, history_prompt=voice_preset)
                except Exception as gen_error:
                    logger.error(f"Error generating audio: {gen_error}")
                    return None

            # Ensure audio array is in correct format
            if audio_array is None or len(audio_array) == 0:
                logger.error("Generated audio array is empty")
                return None

            # Normalize audio to prevent clipping
            import numpy as np

            audio_array = np.clip(audio_array, -1.0, 1.0)

            # Convert to 16-bit PCM
            audio_array = (audio_array * 32767).astype(np.int16)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                write_wav(tmpfile.name, self.bark_sample_rate, audio_array)
                temp_path = tmpfile.name

            logger.info(f"Audio generated and saved to: {temp_path}")

            # Play the audio
            self._play_audio_file(temp_path)

            if return_path:
                return temp_path
            else:
                # Clean up
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Could not delete temp file: {e}")
                return None

        except Exception as e:
            logger.error(f"Error generating speech with Bark: {e}")
            return None

    def _speak_with_pyttsx3(self, text: str) -> bool:
        """Generate speech using pyttsx3 with better error handling"""
        try:
            if not self.pyttsx3_available:
                return False

            logger.info(f"Speaking with pyttsx3: {len(text)} characters")

            # Simple approach - just disable TTS in Streamlit to avoid issues
            try:
                # Check if we're running in Streamlit
                import streamlit as st

                if hasattr(st, "session_state"):
                    logger.info("Running in Streamlit - skipping pyttsx3 to avoid conflicts")
                    return True
            except ImportError:
                pass

            # If not in Streamlit, try normal TTS
            try:
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 0.9)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine

                logger.info("Speech completed successfully with pyttsx3")
                return True

            except Exception as e:
                logger.error(f"Error with pyttsx3: {e}")
                return False

        except Exception as e:
            logger.error(f"Error speaking with pyttsx3: {e}")
            return False

    def _play_audio_file(self, file_path: str) -> bool:
        """Play audio file using sounddevice"""
        try:
            data, samplerate = sf.read(file_path, dtype="float32")
            sd.play(data, samplerate)
            sd.wait()  # Wait until playback is finished
            logger.info("Audio playback completed")
            return True
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
            return False

    def speak_text(
        self,
        text: str,
        voice_preset: str = config.TTS_VOICE_PRESET,
        engine: Optional[str] = None,
        return_path: bool = False,
    ) -> Optional[str]:
        """
        Generate and play speech from text

        Args:
            text: Text to speak
            voice_preset: Voice preset for Bark (ignored for other engines)
            engine: Specific engine to use ('bark', 'pyttsx3', or None for auto)
            return_path: Whether to return the path of generated audio file (Bark only)

        Returns:
            Path to audio file if return_path=True and using Bark, None otherwise
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None

        # Determine which engine to use
        engine_to_use = engine or self.current_engine

        if not engine_to_use:
            logger.error("No TTS engines available")
            return None

        logger.info(f"Speaking text with {engine_to_use}: {len(text)} characters")

        # Log preview without special characters to avoid encoding issues
        preview = text[:50].encode("ascii", errors="ignore").decode("ascii")
        if len(text) > 50:
            preview += "..."
        logger.debug(f"Text preview: '{preview}'")
        if engine_to_use == "bark" and self.bark_available:
            result = self._speak_with_bark(text, voice_preset, return_path)
            if result is not None or not return_path:
                return result

        elif engine_to_use == "pyttsx3" and self.pyttsx3_available:
            if self._speak_with_pyttsx3(text):
                return None

        # Fallback to other available engines
        if engine_to_use != "bark" and self.bark_available:
            logger.info("Falling back to Bark TTS")
            result = self._speak_with_bark(text, voice_preset, return_path)
            if result is not None or not return_path:
                return result

        if engine_to_use != "pyttsx3" and self.pyttsx3_available:
            logger.info("Falling back to pyttsx3 TTS")
            if self._speak_with_pyttsx3(text):
                return None

        logger.error("All TTS engines failed")
        return None

    def get_available_engines(self) -> list[str]:
        """Get list of available TTS engines"""
        engines = []
        if self.bark_available:
            engines.append("bark")
        if self.pyttsx3_available:
            engines.append("pyttsx3")
        return engines

    def test_engines(self) -> dict[str, bool]:
        """Test all available TTS engines"""
        results = {}
        test_text = "Hello, this is a test of the text to speech system."

        for engine in self.get_available_engines():
            try:
                logger.info(f"Testing {engine} engine")
                if engine == "bark":
                    result = self._speak_with_bark(test_text, return_path=True)
                    success = result is not None
                    if result:
                        try:
                            os.remove(result)
                        except:
                            pass
                elif engine == "pyttsx3":
                    success = self._speak_with_pyttsx3(test_text)
                else:
                    success = False

                results[engine] = success
                logger.info(f"{engine} test result: {success}")

            except Exception as e:
                logger.error(f"Error testing {engine}: {e}")
                results[engine] = False

        return results


# Create global instance
tts_manager = TextToSpeechManager()


# Backward compatibility function
def speak_text(
    text: str, voice_preset: str = config.TTS_VOICE_PRESET, return_path: bool = False
) -> Optional[str]:
    """Backward compatibility function"""
    return tts_manager.speak_text(text, voice_preset, return_path=return_path)
