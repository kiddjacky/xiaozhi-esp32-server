import os
import uuid
import json
import queue
import asyncio
import traceback
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig # For specific config types
from config.logger import setup_logging
from core.utils import opus_encoder_utils
from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase
from core.handle.abortHandle import handleAbortMessage
from core.providers.tts.dto.dto import SentenceType, ContentType, InterfaceType

TAG = __name__
logger = setup_logging()

class TTSProvider(TTSProviderBase): # Renamed from GeminiTTSProvider
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)

        self.interface_type = InterfaceType.DUAL_STREAM

        # Gemini specific configurations
        self.api_key = config.get("gemini_api_key")
        self.gemini_model_name = config.get("gemini_tts_model", "gemini-1.5-flash-latest") # Model for generation
        self.gemini_voice_name = config.get("gemini_voice_name", "echo") # Specific voice name e.g., 'Kore', 'Puck', 'echo'

        self.sample_rate_hertz = config.get("sample_rate_hertz", 16000) # Should match Opus encoder and Gemini output if possible
        self.channels = config.get("channels", 1)

        check_model_key("GeminiTTS_API_Key", self.api_key) # Check if API key is provided
        if not self.gemini_voice_name:
            logger.bind(tag=TAG).error("Gemini voice name not configured (gemini_voice_name).")
            raise ValueError("Gemini voice name not configured.")

        try:
            genai.configure(api_key=self.api_key) # Configure once
            self.model = genai.GenerativeModel(self.gemini_model_name)
            logger.bind(tag=TAG).info(f"Gemini AI Model '{self.gemini_model_name}' initialized with voice '{self.gemini_voice_name}'.")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to initialize Gemini AI Model: {str(e)}")
            self.model = None
            raise

        self.enable_two_way = True
        self.tts_text = ""

        self.opus_encoder = opus_encoder_utils.OpusEncoderUtils(
            sample_rate=self.sample_rate_hertz,
            channels=self.channels,
            frame_size_ms=60 # This might need tuning based on typical chunk sizes from Gemini
        )
        logger.bind(tag=TAG).info(f"TTSProvider (Gemini) initialized with Opus encoder: {self.sample_rate_hertz}Hz, {self.channels}ch, 60ms frames.")

    async def open_audio_channels(self, conn):
        try:
            await super().open_audio_channels(conn)
            if not self.model:
                logger.bind(tag=TAG).error("Gemini model not initialized. Cannot open audio channels.")
                raise Exception("Gemini model not initialized.")
            logger.bind(tag=TAG).info("Audio channels opened and TTSProvider (Gemini) is ready.")
        except Exception as e:
            logger.bind(tag=TAG).error(f"Failed to open audio channels: {str(e)}")
            raise

    def tts_text_priority_thread(self):
        logger.bind(tag=TAG).info("Gemini TTS text processing thread started.")
        while not self.conn.stop_event.is_set():
            try:
                message = self.tts_text_queue.get(timeout=1)
                logger.bind(tag=TAG).debug(
                    f"Received TTS task｜{message.sentence_type.name} ｜ {message.content_type.name} | Session ID: {self.conn.sentence_id}"
                )

                if self.conn.client_abort:
                    logger.bind(tag=TAG).info("Client abort received, clearing queue and stopping current synthesis.")
                    while not self.tts_text_queue.empty():
                        try: self.tts_text_queue.get_nowait()
                        except queue.Empty: continue
                    continue

                if message.sentence_type == SentenceType.FIRST:
                    self.tts_audio_first_sentence = True
                    self.before_stop_play_files.clear()
                    logger.bind(tag=TAG).info("First sentence segment, preparing for synthesis.")

                if ContentType.TEXT == message.content_type and message.content_detail:
                    current_text_to_speak = message.content_detail
                    self.tts_text = current_text_to_speak

                    logger.bind(tag=TAG).debug(f"Starting synthesis for text: \"{current_text_to_speak}\"")
                    future = asyncio.run_coroutine_threadsafe(
                        self._synthesize_and_stream_audio(current_text_to_speak),
                        loop=self.conn.loop
                    )
                    future.result()
                    logger.bind(tag=TAG).debug(f"Finished synthesis for text: \"{current_text_to_speak}\"")

                elif ContentType.FILE == message.content_type:
                    logger.bind(tag=TAG).info(
                        f"Adding audio file to playback list: {message.content_file}"
                    )
                    self.before_stop_play_files.append(
                        (message.content_file, message.content_detail)
                    )

                if message.sentence_type == SentenceType.LAST:
                    logger.bind(tag=TAG).info("Last sentence segment processed from input queue.")

            except queue.Empty:
                continue
            except Exception as e:
                logger.bind(tag=TAG).error(
                    f"Error in Gemini TTS text processing: {str(e)}, Type: {type(e).__name__}, Stack: {traceback.format_exc()}"
                )
                self.tts_audio_queue.put((SentenceType.LAST, [], self.tts_text, True))
                continue
        logger.bind(tag=TAG).info("Gemini TTS text processing thread stopped.")

    async def _synthesize_and_stream_audio(self, text_to_speak):
        if not self.model:
            logger.bind(tag=TAG).error("Gemini model is not initialized. Cannot synthesize.")
            self.tts_audio_queue.put((SentenceType.LAST, [], text_to_speak, True))
            return

        try:
            logger.bind(tag=TAG).info(f"Requesting Gemini TTS for: \"{text_to_speak}\" using voice '{self.gemini_voice_name}'")

            generation_config = GenerationConfig(
                response_modalities=["AUDIO"],
                speech_config=SpeechConfig(
                    voice_config=VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(
                            voice_name=self.gemini_voice_name
                        )
                    )
                    # sample_rate_hertz can also be specified in SpeechConfig if API supports it
                    # and if we want to enforce a specific rate from Gemini.
                    # For now, assuming Gemini's default PCM output rate matches self.sample_rate_hertz
                    # or that the Opus encoder can handle resampling if needed (though it's better if they match).
                )
            )

            response_stream = self.model.generate_content(
                contents=[text_to_speak],
                generation_config=generation_config,
                stream=True
            )

            is_first_audio_chunk = True
            for chunk in response_stream:
                if self.conn.client_abort: # Check for abort during streaming
                    logger.bind(tag=TAG).info(f"Client abort detected during TTS streaming for \"{text_to_speak}\". Stopping.")
                    # Clean up any partially sent audio by flushing encoder and sending LAST
                    final_opus_datas = self.opus_encoder.encode_pcm_to_opus(b'', end_of_stream=True)
                    if final_opus_datas:
                        self.tts_audio_queue.put((SentenceType.MIDDLE, final_opus_datas, text_to_speak))
                    self.tts_audio_queue.put((SentenceType.LAST, [], text_to_speak, False)) # Not an error, but aborted
                    return

                # Accessing audio data according to the specified structure
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    pcm_data = chunk.candidates[0].content.parts[0].inline_data.data
                    if pcm_data:
                        opus_datas = self.opus_encoder.encode_pcm_to_opus(pcm_data, end_of_stream=False)
                        if opus_datas: # Only queue if Opus packets were actually produced
                            if is_first_audio_chunk:
                                self.tts_audio_queue.put((SentenceType.FIRST, opus_datas, text_to_speak))
                                is_first_audio_chunk = False
                                logger.bind(tag=TAG).debug(f"Sent FIRST Opus chunk ({len(opus_datas)} packets) for: \"{text_to_speak}\"")
                            else:
                                self.tts_audio_queue.put((SentenceType.MIDDLE, opus_datas, text_to_speak))
                                logger.bind(tag=TAG).debug(f"Sent MIDDLE Opus chunk ({len(opus_datas)} packets) for: \"{text_to_speak}\"")
                    else:
                        logger.bind(tag=TAG).debug("Received chunk with no PCM data.")
                else:
                    logger.bind(tag=TAG).debug(f"Received chunk with unexpected structure or no relevant audio data: {chunk}")

            # After the loop, all PCM data for this segment has been processed.
            # Flush any remaining audio from the Opus encoder.
            final_opus_datas = self.opus_encoder.encode_pcm_to_opus(b'', end_of_stream=True)
            if final_opus_datas:
                # If it was the very first audio produced (e.g. very short text), send as FIRST
                if is_first_audio_chunk and final_opus_datas:
                     self.tts_audio_queue.put((SentenceType.FIRST, final_opus_datas, text_to_speak))
                     logger.bind(tag=TAG).debug(f"Sent final (and FIRST) Opus chunk ({len(final_opus_datas)} packets) for: \"{text_to_speak}\"")
                else: # Otherwise, send as MIDDLE
                     self.tts_audio_queue.put((SentenceType.MIDDLE, final_opus_datas, text_to_speak))
                     logger.bind(tag=TAG).debug(f"Sent final Opus chunk ({len(final_opus_datas)} packets) from flush for: \"{text_to_speak}\"")

            self.tts_audio_queue.put((SentenceType.LAST, [], text_to_speak, False))
            logger.bind(tag=TAG).info(f"Finished streaming TTS for: \"{text_to_speak}\". Sent LAST signal.")

        except Exception as e:
            logger.bind(tag=TAG).error(f"Error during Gemini TTS synthesis/streaming for \"{text_to_speak}\": {str(e)}")
            traceback.print_exc()
            self.tts_audio_queue.put((SentenceType.LAST, [], text_to_speak, True))

    async def close(self):
        logger.bind(tag=TAG).info("Closing TTSProvider (Gemini) resources.")
        self.model = None
        logger.bind(tag=TAG).info("TTSProvider (Gemini) closed.")
