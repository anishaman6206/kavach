"""
kavach.transcription.whisper_asr
==================================
Whisper-based ASR for transcribing VAD speech segments.

Design decisions:
  - Stateless: takes a SpeechSegment in, returns an Utterance (or None) out.
  - Speaker tag comes from the SpeechSegment — never re-detected here.
  - Language is auto-detected by Whisper per segment (not assumed global).
  - Model is loaded once at construction and reused — caller controls lifetime.
  - Gracefully returns None for empty/noise segments rather than crashing.

Usage:
    from kavach.audio.vad import SpeechSegment
    from kavach.transcription.whisper_asr import WhisperASR

    asr = WhisperASR(model_name="tiny")
    utterance = asr.transcribe(segment)
    if utterance:
        buf.add(utterance)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from kavach.audio.buffer import Speaker, Utterance
from kavach.audio.vad import SpeechSegment

logger = logging.getLogger(__name__)

# Minimum RMS to attempt transcription — below this is almost certainly silence/noise
_MIN_RMS_THRESHOLD = 1e-4

# Whisper's native sample rate
_WHISPER_SAMPLE_RATE = 16_000


class WhisperASR:
    """
    Wraps openai-whisper (tiny model by default) for per-segment transcription.

    The model is loaded once in __init__ and shared across all calls to
    transcribe(). This avoids the ~2s reload overhead on every segment.

    Args:
        model_name : Whisper model size. One of:
                     'tiny', 'base', 'small', 'medium', 'large'.
                     Default is 'tiny' — fast enough for on-device use.
        device     : 'cpu' or 'cuda'. Defaults to 'cpu'.
        language   : Force a specific language ISO code ('hi', 'en', etc.)
                     or None for auto-detection (recommended).
    """

    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        language: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.language = language  # None → auto-detect per segment
        self._model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model. Safe to call multiple times — loads only once."""
        if self._model is not None:
            return
        try:
            import whisper  # openai-whisper package

            logger.info(f"[ASR] Loading Whisper '{self.model_name}' on {self.device}...")
            self._model = whisper.load_model(self.model_name, device=self.device)
            logger.info("[ASR] Whisper loaded successfully.")
        except ImportError:
            raise ImportError(
                "openai-whisper is not installed. Run: pip install openai-whisper"
            )
        except Exception as e:
            raise RuntimeError(f"[ASR] Failed to load Whisper model: {e}") from e

    def transcribe(self, segment: SpeechSegment) -> Optional[Utterance]:
        """
        Transcribe a single speech segment.

        Args:
            segment : A SpeechSegment from VoiceActivityDetector.

        Returns:
            An Utterance ready for ConversationBuffer.add(), or None if:
              - The segment audio is too short (< 0.1s)
              - The audio is below the noise floor (likely silence leak)
              - Whisper returns empty text after stripping
        """
        if self._model is None:
            logger.error("[ASR] Model not loaded — cannot transcribe.")
            return None

        audio = segment.audio

        # Guard: too short to be real speech (Whisper minimum is ~100ms)
        min_samples = int(0.1 * _WHISPER_SAMPLE_RATE)
        if len(audio) < min_samples:
            logger.debug(
                f"[ASR] Segment too short ({len(audio)} samples) — skipping."
            )
            return None

        # Guard: below noise floor — VAD occasionally leaks near-silence
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < _MIN_RMS_THRESHOLD:
            logger.debug(f"[ASR] Segment RMS {rms:.2e} below threshold — skipping.")
            return None

        try:
            result = self._model.transcribe(
                audio,
                language=self.language,   # None → Whisper auto-detects
                fp16=False,               # fp16 off for CPU compatibility
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"[ASR] Whisper transcription failed: {e}")
            return None

        raw_text: str = result.get("text", "").strip()
        detected_language: str = result.get("language", "en") or "en"

        # Empty transcript (music, noise, inaudible) — skip
        if not raw_text:
            logger.debug("[ASR] Whisper returned empty transcript — skipping.")
            return None

        return Utterance(
            speaker=segment.speaker,
            text=raw_text,           # Utterance.__post_init__ will lowercase this
            timestamp=segment.start_time,
            chunk_index=segment.chunk_index,
            raw_text=raw_text,
            language=detected_language,
        )

    def transcribe_batch(
        self, segments: list[SpeechSegment]
    ) -> list[Optional[Utterance]]:
        """
        Transcribe a list of segments in order.

        Returns a list of the same length — None entries for skipped segments.
        """
        return [self.transcribe(seg) for seg in segments]

    def __repr__(self) -> str:
        loaded = self._model is not None
        return (
            f"WhisperASR(model='{self.model_name}', "
            f"device='{self.device}', "
            f"language={self.language!r}, "
            f"loaded={loaded})"
        )
