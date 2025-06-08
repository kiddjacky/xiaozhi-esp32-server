import os
import uuid
import asyncio
import traceback
from datetime import datetime
from config.logger import setup_logging
from core.providers.tts.base import TTSProviderBase
from core.providers.tts.dto.dto import SentenceType, ContentType, InterfaceType

TAG = __name__
logger = setup_logging()

class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.interface_type = InterfaceType.NON_STREAM

        # Gemini TTS specific configurations
        self.api_key = config.get("gemini_api_key")
        self.model = config.get("gemini_tts_model", "gemini-2.5-flash-preview-tts")
        self.voice_name = config.get("gemini_voice_name", "Kore")  # Default voice

        if not self.api_key:
            logger.bind(tag=TAG).error("Gemini API key not configured.")
            raise ValueError("Gemini API key not configured.")

        try:
            # Import the new Google GenAI SDK
            from google import genai
            from google.genai import types
            
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=self.api_key)
            
            logger.bind(tag=TAG).info(f"Gemini TTS initialized with model '{self.model}' and voice '{self.voice_name}'")
        except ImportError as e:
            logger.bind(tag=TAG).error(f"Failed to import google.genai: {e}. Please install: pip install google-genai")
            raise
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to initialize Gemini TTS: {str(e)}")
            raise

    async def text_to_speak(self, text, output_file):
        """Generate speech using Gemini TTS API"""
        try:
            logger.bind(tag=TAG).info(f"Generating TTS for text: '{text}' using voice '{self.voice_name}'")

            # Call Gemini TTS API
            response = self.client.models.generate_content(
                model=self.model,
                contents=text,
                config=self.types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=self.types.SpeechConfig(
                        voice_config=self.types.VoiceConfig(
                            prebuilt_voice_config=self.types.PrebuiltVoiceConfig(
                                voice_name=self.voice_name,
                            )
                        )
                    ),
                )
            )

            # Extract audio data
            if (response.candidates and 
                response.candidates[0].content and 
                response.candidates[0].content.parts and
                response.candidates[0].content.parts[0].inline_data):
                
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                
                # Save as WAV file
                self._save_wave_file(output_file, audio_data)
                
                logger.bind(tag=TAG).info(f"Successfully generated TTS audio: {output_file}")
                return True
            else:
                logger.bind(tag=TAG).error("No audio data received from Gemini TTS")
                return False

        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to generate TTS: {str(e)}")
            traceback.print_exc()
            return False

    def _save_wave_file(self, filename, pcm_data, channels=1, rate=24000, sample_width=2):
        """Save PCM data as WAV file"""
        try:
            import wave
            
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(pcm_data)
                
            logger.bind(tag=TAG).debug(f"Saved audio to {filename}")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to save wave file: {str(e)}")
            raise

    async def close(self):
        logger.bind(tag=TAG).info("Closing Gemini TTS provider.")
        pass
