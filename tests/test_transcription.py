"""
Tests for kavach.transcription.whisper_asr

Run with:
    pytest tests/test_transcription.py -v

Design notes:
  - Whisper model is mocked throughout — we are testing OUR logic
    (guards, speaker tagging, output format), not Whisper's internals.
  - The real Whisper integration is exercised by scripts/test_with_audio.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from kavach.audio.buffer import Speaker, Utterance
from kavach.audio.vad import SpeechSegment
from kavach.transcription.whisper_asr import WhisperASR


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

SAMPLE_RATE = 16_000


def make_segment(
    duration_s: float = 1.0,
    speaker: Speaker = Speaker.CALLER,
    start_time: float = 0.0,
    chunk_index: int = 0,
    rms_amplitude: float = 0.05,  # typical speech-level amplitude
) -> SpeechSegment:
    """Create a SpeechSegment with a sine wave (mimics speech energy)."""
    n_samples = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n_samples)
    audio = (rms_amplitude * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    return SpeechSegment(
        audio=audio,
        speaker=speaker,
        start_time=start_time,
        end_time=start_time + duration_s,
        chunk_index=chunk_index,
    )


def make_silent_segment(duration_s: float = 1.0) -> SpeechSegment:
    """Create a SpeechSegment with near-zero audio (silence leak from VAD)."""
    n_samples = int(duration_s * SAMPLE_RATE)
    audio = np.zeros(n_samples, dtype=np.float32)
    return SpeechSegment(
        audio=audio,
        speaker=Speaker.CALLER,
        start_time=0.0,
        end_time=duration_s,
        chunk_index=0,
    )


def make_tiny_segment(n_samples: int = 50) -> SpeechSegment:
    """Create a SpeechSegment shorter than 0.1 s (too short for Whisper)."""
    audio = (0.05 * np.ones(n_samples)).astype(np.float32)
    return SpeechSegment(
        audio=audio,
        speaker=Speaker.CALLER,
        start_time=0.0,
        end_time=n_samples / SAMPLE_RATE,
        chunk_index=0,
    )


def make_asr_with_mock(transcribe_result: dict) -> WhisperASR:
    """
    Build a WhisperASR whose internal _model is a Mock that returns
    `transcribe_result` when model.transcribe() is called.

    Uses __new__ to bypass __init__ (and the Whisper model download entirely).
    No patching needed — we just inject the mock directly onto the instance.
    """
    mock_model = MagicMock()
    mock_model.transcribe.return_value = transcribe_result

    asr = WhisperASR.__new__(WhisperASR)
    asr.model_name = "tiny"
    asr.device = "cpu"
    asr.language = None
    asr._model = mock_model
    return asr


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestWhisperASRGuards:
    """Tests that WhisperASR's guard clauses work before touching Whisper."""

    def test_too_short_segment_returns_none(self):
        """Segments shorter than 0.1s should be skipped, not crash."""
        asr = make_asr_with_mock({"text": "hello", "language": "en"})
        tiny = make_tiny_segment(n_samples=50)   # ~3ms
        result = asr.transcribe(tiny)
        assert result is None

    def test_silent_segment_returns_none(self):
        """Near-silence (RMS below threshold) should be skipped."""
        asr = make_asr_with_mock({"text": "hello", "language": "en"})
        silent = make_silent_segment(duration_s=1.0)
        result = asr.transcribe(silent)
        assert result is None

    def test_empty_whisper_result_returns_none(self):
        """If Whisper returns empty text, return None (noise/music)."""
        asr = make_asr_with_mock({"text": "", "language": "en"})
        seg = make_segment(duration_s=1.0)
        result = asr.transcribe(seg)
        assert result is None

    def test_whitespace_only_whisper_result_returns_none(self):
        """Whisper sometimes returns only whitespace for inaudible audio."""
        asr = make_asr_with_mock({"text": "   ", "language": "en"})
        seg = make_segment(duration_s=1.0)
        result = asr.transcribe(seg)
        assert result is None

    def test_whisper_exception_returns_none(self):
        """If Whisper raises, return None — never propagate crash to caller."""
        asr = make_asr_with_mock({})
        asr._model.transcribe.side_effect = RuntimeError("CUDA OOM")
        seg = make_segment(duration_s=1.0)
        result = asr.transcribe(seg)
        assert result is None


class TestWhisperASROutput:
    """Tests that valid segments produce correctly formed Utterances."""

    def test_valid_segment_returns_utterance(self):
        asr = make_asr_with_mock({"text": "Hello from SBI bank", "language": "en"})
        seg = make_segment(duration_s=1.0)
        result = asr.transcribe(seg)
        assert isinstance(result, Utterance)

    def test_speaker_tag_comes_from_segment(self):
        """Speaker must be taken from SpeechSegment, never re-detected."""
        asr = make_asr_with_mock({"text": "namaste", "language": "hi"})

        caller_seg = make_segment(speaker=Speaker.CALLER)
        user_seg = make_segment(speaker=Speaker.USER)

        assert asr.transcribe(caller_seg).speaker == Speaker.CALLER
        assert asr.transcribe(user_seg).speaker == Speaker.USER

    def test_text_is_lowercased(self):
        """Utterance.__post_init__ lowercases text — verify end-to-end."""
        asr = make_asr_with_mock({"text": "Please Share Your OTP", "language": "en"})
        seg = make_segment()
        result = asr.transcribe(seg)
        assert result.text == "please share your otp"

    def test_raw_text_preserves_original_case(self):
        original = "Please Share Your OTP"
        asr = make_asr_with_mock({"text": original, "language": "en"})
        seg = make_segment()
        result = asr.transcribe(seg)
        assert result.raw_text == original

    def test_language_set_from_whisper_result(self):
        asr = make_asr_with_mock({"text": "namaste aap kaise hain", "language": "hi"})
        seg = make_segment()
        result = asr.transcribe(seg)
        assert result.language == "hi"

    def test_language_defaults_to_en_if_missing(self):
        """Robustness: Whisper result missing 'language' key → default 'en'."""
        asr = make_asr_with_mock({"text": "some text"})   # no 'language' key
        seg = make_segment()
        result = asr.transcribe(seg)
        assert result.language == "en"

    def test_timestamp_from_segment(self):
        asr = make_asr_with_mock({"text": "hello", "language": "en"})
        seg = make_segment(start_time=12.5)
        result = asr.transcribe(seg)
        assert result.timestamp == 12.5

    def test_chunk_index_from_segment(self):
        asr = make_asr_with_mock({"text": "hello", "language": "en"})
        seg = make_segment(chunk_index=7)
        result = asr.transcribe(seg)
        assert result.chunk_index == 7

    def test_utterance_ready_for_buffer_add(self):
        """The returned Utterance must be accepted by ConversationBuffer.add()."""
        from kavach.audio.buffer import ConversationBuffer

        asr = make_asr_with_mock({"text": "your account is suspended", "language": "en"})
        seg = make_segment()
        result = asr.transcribe(seg)

        buf = ConversationBuffer()
        buf.add(result)   # should not raise
        assert buf.size == 1
        assert "your account is suspended" in buf.caller_text_only()


class TestWhisperASRBatch:
    """Tests for transcribe_batch()."""

    def test_batch_same_length_as_input(self):
        asr = make_asr_with_mock({"text": "hello", "language": "en"})
        segments = [make_segment() for _ in range(5)]
        results = asr.transcribe_batch(segments)
        assert len(results) == 5

    def test_batch_none_for_silent_segments(self):
        asr = make_asr_with_mock({"text": "hello", "language": "en"})
        segments = [
            make_segment(),          # valid
            make_silent_segment(),   # will be filtered
            make_segment(),          # valid
        ]
        results = asr.transcribe_batch(segments)
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

    def test_batch_empty_list_returns_empty_list(self):
        asr = make_asr_with_mock({"text": "hello", "language": "en"})
        assert asr.transcribe_batch([]) == []


class TestWhisperASRRepr:
    def test_repr_contains_model_name(self):
        asr = make_asr_with_mock({"text": "hi", "language": "en"})
        asr.model_name = "tiny"
        assert "tiny" in repr(asr)

    def test_repr_shows_loaded_true(self):
        asr = make_asr_with_mock({"text": "hi", "language": "en"})
        assert "loaded=True" in repr(asr)
