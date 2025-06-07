import unittest
from unittest.mock import MagicMock, patch, call
import queue
import asyncio # Required for async methods if directly called

# Assuming the file is in main.xiaozhi-server.core.providers.tts.gemini_double_stream
# Adjust the import path based on actual project structure and PYTHONPATH
from core.providers.tts.gemini_double_stream import TTSProvider
from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType, ContentType

# Mock google.generativeai before it's imported by the module we're testing
# This is important if genai is used at the module level in gemini_double_stream.py
mock_genai = MagicMock()
# We need to mock the types used from google.generativeai.types as well
mock_genai.types = MagicMock()
mock_genai.types.GenerationConfig = MagicMock
mock_genai.types.SpeechConfig = MagicMock
mock_genai.types.VoiceConfig = MagicMock
mock_genai.types.PrebuiltVoiceConfig = MagicMock

# Patch 'google.generativeai' in the context of the module where it's imported
# This typically means patching it where 'gemini_double_stream.py' looks for it.
# If 'gemini_double_stream.py' does 'import google.generativeai as genai', then
# 'core.providers.tts.gemini_double_stream.genai' is the target.
@patch.dict('sys.modules', {'google.generativeai': mock_genai})
class TestGeminiTTSProvider(unittest.TestCase):

    def setUp(self):
        self.mock_config = {
            "gemini_api_key": "test_api_key",
            "gemini_tts_model": "test-model",
            "gemini_voice_name": "test-voice",
            "sample_rate_hertz": 16000,
            "channels": 1,
            # Add any other config parameters your TTSProvider expects
        }

        # Patch logger before TTSProvider is instantiated
        patcher_logger = patch('core.providers.tts.gemini_double_stream.logger', MagicMock())
        self.addCleanup(patcher_logger.stop)
        patcher_logger.start()

        # Patch opus_encoder_utils before TTSProvider is instantiated
        # so that the OpusEncoderUtils class itself is a mock, and its instance (self.opus_encoder)
        # will also be a mock.
        patcher_opus_encoder = patch('core.providers.tts.gemini_double_stream.opus_encoder_utils.OpusEncoderUtils')
        self.MockOpusEncoderUtils = patcher_opus_encoder.start()
        self.addCleanup(patcher_opus_encoder.stop)

        self.mock_opus_encoder_instance = self.MockOpusEncoderUtils.return_value
        self.mock_opus_encoder_instance.encode_pcm_to_opus = MagicMock(return_value=[b"opus_data_1"])

        # Now instantiate the provider
        self.provider = TTSProvider(config=self.mock_config, delete_audio_file=False)

        # Replace the real genai.GenerativeModel with a mock *after* TTSProvider init if it's too late
        # Or ensure the global mock_genai.GenerativeModel is used.
        # The @patch.dict at class level should handle 'import google.generativeai as genai'
        # Let's verify self.provider.model is indeed a mock from our global mock_genai
        self.mock_model_instance = mock_genai.GenerativeModel.return_value
        self.provider.model = self.mock_model_instance # Ensure it's using the one we can control

        # Mock the tts_audio_queue for direct inspection
        self.provider.tts_audio_queue = queue.Queue()

        # Mock the connection object and its event loop for tts_text_priority_thread
        self.mock_conn = MagicMock()
        self.mock_conn.loop = asyncio.get_event_loop() # Use real loop for run_coroutine_threadsafe if needed
        self.mock_conn.stop_event = MagicMock()
        self.mock_conn.stop_event.is_set.return_value = False # Loop runs initially
        self.mock_conn.client_abort = False
        self.provider.conn = self.mock_conn


    def test_initialization(self):
        self.assertEqual(self.provider.api_key, "test_api_key")
        self.assertEqual(self.provider.gemini_model_name, "test-model")
        self.assertEqual(self.provider.gemini_voice_name, "test-voice")
        mock_genai.configure.assert_called_once_with(api_key="test_api_key")
        mock_genai.GenerativeModel.assert_called_once_with("test-model")
        self.MockOpusEncoderUtils.assert_called_with(
            sample_rate=16000, channels=1, frame_size_ms=60
        )

    def test_synthesize_and_stream_audio_success(self):
        sample_text = "Hello world"

        # Configure the mock model's generate_content method
        mock_audio_chunk_1 = MagicMock()
        mock_audio_chunk_1.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(inline_data=MagicMock(data=b'pcm_data_chunk_1'))]))]
        mock_audio_chunk_2 = MagicMock()
        mock_audio_chunk_2.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(inline_data=MagicMock(data=b'pcm_data_chunk_2'))]))]

        self.mock_model_instance.generate_content.return_value = [mock_audio_chunk_1, mock_audio_chunk_2]

        # Mock opus encoder calls
        self.mock_opus_encoder_instance.encode_pcm_to_opus.side_effect = [
            [b"opus_data_chunk_1"],  # For pcm_data_chunk_1
            [b"opus_data_chunk_2"],  # For pcm_data_chunk_2
            [b"final_opus_data"]     # For flush call (end_of_stream=True)
        ]

        # Run the method to be tested (it's async, so needs await if called directly)
        # In the actual code, it's run via run_coroutine_threadsafe
        async def run_test():
            await self.provider._synthesize_and_stream_audio(sample_text)

        asyncio.run(run_test())

        # Assertions
        self.mock_model_instance.generate_content.assert_called_once()
        args, kwargs = self.mock_model_instance.generate_content.call_args
        self.assertEqual(kwargs['contents'], [sample_text])
        self.assertTrue(kwargs['stream'])
        # Check specific generation_config details if necessary (complex to assert precisely without reconstructing the object)
        self.assertIsNotNone(kwargs['generation_config'])

        # Check opus encoder calls
        expected_opus_calls = [
            call(b'pcm_data_chunk_1', end_of_stream=False),
            call(b'pcm_data_chunk_2', end_of_stream=False),
            call(b'', end_of_stream=True) # Final flush
        ]
        self.mock_opus_encoder_instance.encode_pcm_to_opus.assert_has_calls(expected_opus_calls)

        # Check tts_audio_queue content
        # Order: FIRST, MIDDLE (for chunk2), MIDDLE (for final flush), LAST
        q_item1 = self.provider.tts_audio_queue.get_nowait()
        self.assertEqual(q_item1[0], SentenceType.FIRST)
        self.assertEqual(q_item1[1], [b"opus_data_chunk_1"])
        self.assertEqual(q_item1[2], sample_text)

        q_item2 = self.provider.tts_audio_queue.get_nowait()
        self.assertEqual(q_item2[0], SentenceType.MIDDLE)
        self.assertEqual(q_item2[1], [b"opus_data_chunk_2"])
        self.assertEqual(q_item2[2], sample_text)

        q_item3 = self.provider.tts_audio_queue.get_nowait()
        self.assertEqual(q_item3[0], SentenceType.MIDDLE) # Data from final flush
        self.assertEqual(q_item3[1], [b"final_opus_data"])
        self.assertEqual(q_item3[2], sample_text)

        q_item4 = self.provider.tts_audio_queue.get_nowait()
        self.assertEqual(q_item4[0], SentenceType.LAST)
        self.assertEqual(q_item4[1], [])
        self.assertEqual(q_item4[2], sample_text)
        self.assertFalse(q_item4[3] if len(q_item4) > 3 else False) # Error flag

        self.assertTrue(self.provider.tts_audio_queue.empty())

    def test_synthesize_and_stream_audio_api_error(self):
        sample_text = "Error test"
        self.mock_model_instance.generate_content.side_effect = Exception("Gemini API Error")

        async def run_test():
            await self.provider._synthesize_and_stream_audio(sample_text)

        asyncio.run(run_test())

        # logger.error should have been called (assert this if logger is part of self.provider and mockable)
        # For now, check the queue for an error message
        q_item = self.provider.tts_audio_queue.get_nowait()
        self.assertEqual(q_item[0], SentenceType.LAST)
        self.assertEqual(q_item[1], [])
        self.assertEqual(q_item[2], sample_text)
        self.assertTrue(q_item[3]) # Error flag should be true

        self.assertTrue(self.provider.tts_audio_queue.empty())

    @patch('core.providers.tts.gemini_double_stream.TTSProvider._synthesize_and_stream_audio', new_callable=MagicMock)
    def test_tts_text_priority_thread_simple_run(self, mock_synthesize_method_async):
        # This test is simplified to avoid actual threading for easier testing.
        # We'll check if _synthesize_and_stream_audio is correctly called.

        async def mock_coro(*args, **kwargs):
            # This is the coroutine that _synthesize_and_stream_audio is meant to be.
            # We don't need it to do anything for this specific test, just be awaitable.
            pass

        # Make the mock_synthesize_method_async return our awaitable mock_coro
        mock_synthesize_method_async.return_value = mock_coro()

        # Prepare a message for the input queue
        test_text = "Test message for thread"
        message = TTSMessageDTO(
            sentence_type=SentenceType.FIRST,
            content_type=ContentType.TEXT,
            content_detail=test_text
        )
        self.provider.tts_text_queue.put(message)

        # Simulate stop condition for the thread after one iteration
        def stop_thread_after_one_run(*args, **kwargs):
            if self.provider.tts_text_queue.empty(): # Stop after processing the item
                 self.mock_conn.stop_event.is_set.return_value = True
            return False # Continue if queue not empty or first run

        self.mock_conn.stop_event.is_set.side_effect = stop_thread_after_one_run

        # Patch asyncio.run_coroutine_threadsafe to call our async mock directly
        # This avoids needing a real event loop in a separate thread for this test.
        async def simple_runner(coro, loop):
            await coro # Execute the coroutine
            return MagicMock() # Return a mock future

        with patch('asyncio.run_coroutine_threadsafe', side_effect=simple_runner) as mock_run_coro:
            self.provider.tts_text_priority_thread()

        # Assert that _synthesize_and_stream_audio was called
        mock_synthesize_method_async.assert_called_once_with(test_text)


    def tearDown(self):
        # Ensure any patched objects are stopped.
        # self.addCleanup handles this for patches started with self.
        pass

if __name__ == '__main__':
    unittest.main()
