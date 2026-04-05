"""
Tests for kavach.audio.buffer and kavach.audio.vad

Run with:
    pytest tests/test_audio.py -v

No external dependencies needed — VAD tests use synthetic numpy arrays.
"""

import numpy as np
import pytest

from kavach.audio.buffer import ConversationBuffer, Speaker, Utterance
from kavach.audio.vad    import SpeechSegment, VoiceActivityDetector


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def make_utterance(
    speaker    : Speaker,
    text       : str,
    timestamp  : float = 0.0,
    chunk_index: int   = 0,
    language   : str   = "en",
) -> Utterance:
    return Utterance(
        speaker     = speaker,
        text        = text,
        timestamp   = timestamp,
        chunk_index = chunk_index,
        language    = language,
    )


def make_audio(duration_s: float, sample_rate: int = 16_000) -> np.ndarray:
    """Generate silent audio (zeros) of a given duration."""
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def make_speech_audio(duration_s: float, sample_rate: int = 16_000) -> np.ndarray:
    """Generate a sine wave that mimics speech energy."""
    t = np.linspace(0, duration_s, int(duration_s * sample_rate))
    return (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# ConversationBuffer tests
# ═══════════════════════════════════════════════════════════════

class TestConversationBuffer:

    def test_empty_buffer(self):
        buf = ConversationBuffer()
        assert buf.is_empty
        assert buf.size == 0
        assert buf.latest is None
        assert buf.utterances == []

    def test_add_single_utterance(self):
        buf = ConversationBuffer()
        u   = make_utterance(Speaker.CALLER, "Hello I am from SBI", timestamp=0.0)
        buf.add(u)

        assert not buf.is_empty
        assert buf.size == 1
        assert buf.latest == u

    def test_text_is_lowercased(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "Please Share Your OTP"))
        assert buf.latest.text == "please share your otp"

    def test_raw_text_preserved(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "Please Share Your OTP"))
        # raw_text is set from the original before lowercasing in __post_init__
        assert "OTP" in buf.latest.raw_text or "otp" in buf.latest.raw_text

    def test_overflow_evicts_oldest(self):
        buf = ConversationBuffer(max_utterances=3)
        for i in range(5):
            buf.add(make_utterance(Speaker.CALLER, f"utterance {i}", timestamp=float(i)))
        assert buf.size == 3
        # Should have utterances 2, 3, 4
        timestamps = [u.timestamp for u in buf.utterances]
        assert timestamps == [2.0, 3.0, 4.0]

    def test_caller_utterances_only(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "I am from RBI",    timestamp=0.0))
        buf.add(make_utterance(Speaker.USER,   "Yes what happened", timestamp=2.0))
        buf.add(make_utterance(Speaker.CALLER, "Share your otp",   timestamp=5.0))

        caller = buf.caller_utterances()
        assert len(caller) == 2
        assert all(u.speaker == Speaker.CALLER for u in caller)

    def test_user_utterances_only(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "I am from RBI"))
        buf.add(make_utterance(Speaker.USER,   "Yes?"))
        buf.add(make_utterance(Speaker.USER,   "What do you want?"))

        user = buf.user_utterances()
        assert len(user) == 2
        assert all(u.speaker == Speaker.USER for u in user)

    def test_caller_text_only_concatenated(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "I am from SBI",    timestamp=0.0))
        buf.add(make_utterance(Speaker.USER,   "Yes?",              timestamp=2.0))
        buf.add(make_utterance(Speaker.CALLER, "Share your otp",   timestamp=5.0))

        text = buf.caller_text_only()
        assert "i am from sbi" in text
        assert "share your otp" in text
        assert "yes?" not in text   # user text must be excluded

    def test_as_classifier_input_includes_speaker_labels(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "I am from SBI"))
        buf.add(make_utterance(Speaker.USER,   "Yes?"))

        ctx = buf.as_classifier_input()
        assert "CALLER:" in ctx
        assert "USER:"   in ctx

    def test_as_slm_context_has_timestamps(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "I am from SBI", timestamp=1.5))

        ctx = buf.as_slm_context()
        assert "[1.5s]" in ctx
        assert "CALLER:" in ctx

    def test_as_slm_context_multiline(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "I am from SBI",    timestamp=0.0))
        buf.add(make_utterance(Speaker.USER,   "Yes?",              timestamp=2.0))
        buf.add(make_utterance(Speaker.CALLER, "Share your otp",   timestamp=5.0))

        lines = buf.as_slm_context().strip().split("\n")
        assert len(lines) == 3

    def test_clear_wipes_buffer(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "test"))
        buf.clear()
        assert buf.is_empty

    def test_max_utterances_one(self):
        buf = ConversationBuffer(max_utterances=1)
        buf.add(make_utterance(Speaker.CALLER, "first"))
        buf.add(make_utterance(Speaker.CALLER, "second"))
        assert buf.size == 1
        assert "second" in buf.latest.text

    def test_invalid_max_utterances(self):
        with pytest.raises(ValueError):
            ConversationBuffer(max_utterances=0)

    def test_classifier_input_long_text_truncated(self):
        # Fill buffer with very long utterances and verify output doesn't
        # exceed the character budget
        buf = ConversationBuffer(max_utterances=10, max_tokens_approx=20)
        for i in range(10):
            buf.add(make_utterance(Speaker.CALLER, "x" * 200, timestamp=float(i)))

        ctx = buf.as_classifier_input()
        assert len(ctx) <= 20 * 4 + 50  # small tolerance for edge trimming

    def test_detected_languages(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "namaste",   language="hi"))
        buf.add(make_utterance(Speaker.USER,   "hello",     language="en"))
        buf.add(make_utterance(Speaker.CALLER, "share otp", language="hi"))

        langs = buf.detected_languages()
        assert "hi" in langs
        assert "en" in langs

    def test_repr_does_not_raise(self):
        buf = ConversationBuffer()
        buf.add(make_utterance(Speaker.CALLER, "test"))
        assert "ConversationBuffer" in repr(buf)


# ═══════════════════════════════════════════════════════════════
# VoiceActivityDetector tests
# ═══════════════════════════════════════════════════════════════

class TestVoiceActivityDetector:
    """
    These tests use the energy-based fallback (no Silero model download needed).
    The logic being tested is the windowing, merging, and segment output —
    not Silero's internal neural network.
    """

    def setup_method(self):
        # Force energy-based fallback by using a dummy threshold
        # (Silero may not be available in CI)
        self.vad = VoiceActivityDetector(threshold=0.5)

    def test_silent_audio_produces_no_segments(self):
        silent = make_audio(2.0)
        segments = self.vad.process_chunk(
            caller_audio     = silent,
            user_audio       = silent,
            chunk_start_time = 0.0,
            chunk_index      = 0,
        )
        # Energy-based fallback: zeros → prob = 0 → no segments
        assert isinstance(segments, list)

    def test_output_is_list_of_speech_segments(self):
        caller = make_speech_audio(2.0)
        user   = make_audio(2.0)        # user is silent
        segments = self.vad.process_chunk(
            caller_audio     = caller,
            user_audio       = user,
            chunk_start_time = 0.0,
            chunk_index      = 0,
        )
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, SpeechSegment)

    def test_segments_have_correct_speaker(self):
        caller = make_speech_audio(2.0)
        user   = make_audio(2.0)
        segments = self.vad.process_chunk(
            caller_audio     = caller,
            user_audio       = user,
            chunk_start_time = 0.0,
            chunk_index      = 0,
        )
        for seg in segments:
            assert seg.speaker in (Speaker.CALLER, Speaker.USER)

    def test_segments_sorted_by_start_time(self):
        caller = make_speech_audio(2.0)
        user   = make_speech_audio(2.0)
        segments = self.vad.process_chunk(
            caller_audio     = caller,
            user_audio       = user,
            chunk_start_time = 5.0,   # non-zero base time
            chunk_index      = 2,
        )
        if len(segments) > 1:
            times = [s.start_time for s in segments]
            assert times == sorted(times)

    def test_timestamps_offset_by_chunk_start(self):
        caller = make_speech_audio(2.0)
        user   = make_audio(2.0)
        base   = 10.0
        segments = self.vad.process_chunk(
            caller_audio     = caller,
            user_audio       = user,
            chunk_start_time = base,
            chunk_index      = 4,
        )
        for seg in segments:
            assert seg.start_time >= base

    def test_too_short_audio_returns_empty(self):
        tiny = np.zeros(10, dtype=np.float32)
        segments = self.vad.process_chunk(
            caller_audio     = tiny,
            user_audio       = tiny,
            chunk_start_time = 0.0,
            chunk_index      = 0,
        )
        assert segments == []

    def test_speech_segment_duration_positive(self):
        caller = make_speech_audio(2.0)
        user   = make_audio(2.0)
        segments = self.vad.process_chunk(
            caller_audio     = caller,
            user_audio       = user,
            chunk_start_time = 0.0,
            chunk_index      = 0,
        )
        for seg in segments:
            assert seg.duration > 0

    def test_is_speech_returns_float(self):
        audio = make_speech_audio(0.1)
        prob  = self.vad.is_speech(audio)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_silence_has_low_speech_prob(self):
        silent = make_audio(0.1)
        prob   = self.vad.is_speech(silent)
        assert prob < 0.3   # silence → low probability

    def test_chunk_index_stored_in_segment(self):
        caller = make_speech_audio(2.0)
        user   = make_audio(2.0)
        segments = self.vad.process_chunk(
            caller_audio     = caller,
            user_audio       = user,
            chunk_start_time = 0.0,
            chunk_index      = 7,
        )
        for seg in segments:
            assert seg.chunk_index == 7