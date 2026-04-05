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

    def test_language_fallback_when_key_missing(self):
        """Robustness: Whisper result missing 'language' key → falls back to
        self.language if set, otherwise 'hi' (safe default for this project)."""
        # Case 1: no language locked — falls back to 'hi'
        asr = make_asr_with_mock({"text": "some text"})   # no 'language' key
        asr.language = None
        seg = make_segment()
        result = asr.transcribe(seg)
        assert result.language == "hi"

        # Case 2: language locked to 'en' — uses that
        asr2 = make_asr_with_mock({"text": "some text"})
        asr2.language = "en"
        result2 = asr2.transcribe(seg)
        assert result2.language == "en"

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


class TestDetectLanguageOnce:
    """Tests for WhisperASR.detect_language_once()."""

    def test_locks_language_from_result(self):
        """detect_language_once() should cache and return the detected language."""
        import sys
        import unittest.mock as um

        asr = make_asr_with_mock({"text": "hi", "language": "en"})
        asr.language = None  # not yet locked

        audio = (0.05 * np.ones(16000 * 5)).astype(np.float32)

        # Patch 'whisper' in sys.modules so the `import whisper` inside the
        # method finds our mock. This avoids AttributeError from module-level patching.
        mock_whisper_module = MagicMock()
        mock_whisper_module.pad_or_trim.return_value = audio[:480000].astype(np.float32)
        mock_whisper_module.log_mel_spectrogram.return_value = MagicMock()
        asr._model.detect_language.return_value = (None, {"hi": 0.9, "en": 0.05})

        with um.patch.dict(sys.modules, {"whisper": mock_whisper_module}):
            lang = asr.detect_language_once(audio)

        assert lang == "hi"
        assert asr.language == "hi"

    def test_does_not_overwrite_if_already_set(self):
        """If self.language is already set, detect_language_once() is a no-op."""
        asr = make_asr_with_mock({"text": "hi", "language": "en"})
        asr.language = "en"  # already locked
        lang = asr.detect_language_once(np.zeros(16000, dtype=np.float32))
        assert lang == "en"
        assert asr.language == "en"
        # detect_language should NOT have been called
        asr._model.detect_language.assert_not_called()

    def test_locked_language_used_in_transcribe(self):
        """After detect_language_once(), transcribe() uses the locked language."""
        asr = make_asr_with_mock({"text": "namaste", "language": "hi"})
        asr.language = "hi"  # as if detect_language_once() already ran
        seg = make_segment()
        result = asr.transcribe(seg)
        # Verify language param passed to model.transcribe was "hi"
        call_kwargs = asr._model.transcribe.call_args
        assert call_kwargs.kwargs.get("language") == "hi"


class TestSpeechAccumulator:
    """Tests for SpeechAccumulator."""

    def _make_asr(self, text: str = "hello from sbi", lang: str = "hi") -> WhisperASR:
        return make_asr_with_mock({"text": text, "language": lang})

    def test_returns_none_below_threshold(self):
        """Segments below min_duration_s should not trigger transcription."""
        asr = self._make_asr()
        acc = SpeechAccumulator(asr, min_duration_s=4.0)
        # 1s segment — well below 4s threshold
        seg = make_segment(duration_s=1.0)
        result = acc.add(seg)
        assert result is None
        asr._model.transcribe.assert_not_called()

    def test_flushes_when_threshold_reached(self):
        """Adding enough segments to reach 4s should trigger transcription."""
        asr = self._make_asr()
        acc = SpeechAccumulator(asr, min_duration_s=4.0)

        results = []
        # 4 × 1s segments = exactly 4s
        for _ in range(4):
            r = acc.add(make_segment(duration_s=1.0))
            results.append(r)

        # Only the last add should have returned an Utterance
        assert results[-1] is not None
        assert isinstance(results[-1], Utterance)
        assert all(r is None for r in results[:-1])

    def test_flush_returns_remaining(self):
        """flush() should transcribe buffered audio even if < min_duration_s."""
        asr = self._make_asr()
        acc = SpeechAccumulator(asr, min_duration_s=4.0)
        acc.add(make_segment(duration_s=1.0))  # below threshold
        result = acc.flush()
        assert result is not None
        assert isinstance(result, Utterance)

    def test_flush_empty_returns_none(self):
        """flush() on an empty accumulator returns None."""
        asr = self._make_asr()
        acc = SpeechAccumulator(asr, min_duration_s=4.0)
        assert acc.flush() is None

    def test_speaker_change_flushes_and_restarts(self):
        """When speaker changes, the old buffer is flushed immediately."""
        asr = self._make_asr()
        acc = SpeechAccumulator(asr, min_duration_s=4.0)

        caller_seg = make_segment(duration_s=1.5, speaker=Speaker.CALLER)
        user_seg   = make_segment(duration_s=0.5, speaker=Speaker.USER)

        result_caller_add = acc.add(caller_seg)  # 1.5s CALLER — no flush yet
        assert result_caller_add is None

        # Adding USER segment triggers flush of buffered CALLER audio
        result_on_switch = acc.add(user_seg)
        assert result_on_switch is not None
        assert result_on_switch.speaker == Speaker.CALLER

    def test_accumulated_audio_is_concatenated(self):
        """Verify combined segment audio length equals sum of inputs."""
        asr = self._make_asr()
        acc = SpeechAccumulator(asr, min_duration_s=4.0)

        seg1 = make_segment(duration_s=2.0)
        seg2 = make_segment(duration_s=2.0)
        acc.add(seg1)
        acc.add(seg2)  # triggers flush

        # Check what audio was actually passed to model.transcribe
        # 4.0s of speech + 0.2s zero-padding added by _flush_internal
        called_audio = asr._model.transcribe.call_args[0][0]
        expected_samples = int(4.0 * SAMPLE_RATE) + int(0.2 * SAMPLE_RATE)
        assert len(called_audio) == expected_samples

    def test_timestamp_from_first_segment(self):
        """The Utterance timestamp should be from the first accumulated segment."""
        asr = self._make_asr()
        acc = SpeechAccumulator(asr, min_duration_s=2.0)

        seg1 = make_segment(duration_s=1.0, start_time=10.0)
        seg2 = make_segment(duration_s=1.0, start_time=11.0)
        acc.add(seg1)
        result = acc.add(seg2)
        assert result is not None
        assert result.timestamp == 10.0

    def test_repr_shows_buffered_duration(self):
        asr = self._make_asr()
        acc = SpeechAccumulator(asr, min_duration_s=4.0)
        acc.add(make_segment(duration_s=1.0))
        assert "buffered=1.00s" in repr(acc)


# Need to import SpeechAccumulator for the tests above
from kavach.transcription.whisper_asr import SpeechAccumulator
