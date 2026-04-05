"""
kavach.transcription.whisper_asr
==================================
Whisper-based ASR for transcribing VAD speech segments.

Design decisions:
  - Stateless per-segment transcription via WhisperASR.transcribe().
  - Speaker tag always comes from the SpeechSegment — never re-detected.
  - Language is detected ONCE on the first 30s of audio via
    detect_language_once(), then locked for the rest of the call.
    This prevents per-segment hallucination on short Hindi clips.
  - Model defaults to 'small' — better multilingual accuracy than 'tiny',
    still runs on CPU in reasonable time.
  - SpeechAccumulator buffers short VAD segments until >= min_duration_s
    of audio is collected before calling Whisper, reducing hallucination
    caused by very short (~0.3s) segments being transcribed in isolation.

Usage:
    from kavach.audio.vad import SpeechSegment
    from kavach.transcription.whisper_asr import WhisperASR, SpeechAccumulator

    asr = WhisperASR(model_name="small")

    # Detect language once from the first 30s of raw call audio
    asr.detect_language_once(first_30s_audio)

    # Use accumulator to batch short segments before transcribing
    acc = SpeechAccumulator(asr, min_duration_s=4.0)

    for segment in vad_segments:
        utterance = acc.add(segment)      # returns Utterance when buffer >= 4s
        if utterance:
            buf.add(utterance)

    # End of call — flush any remaining buffered audio
    utterance = acc.flush()
    if utterance:
        buf.add(utterance)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from kavach.audio.buffer import Speaker, Utterance
from kavach.audio.vad import SpeechSegment

logger = logging.getLogger(__name__)

# Minimum RMS to attempt transcription — below this is almost certainly silence/noise
_MIN_RMS_THRESHOLD = 1e-4

# Whisper's native sample rate
_WHISPER_SAMPLE_RATE = 16_000

# How many seconds of audio Whisper's language detector uses
_LANG_DETECT_WINDOW_S = 30


class WhisperASR:
    """
    Wraps openai-whisper for per-segment transcription.

    Key improvements over naive per-segment transcription:
      - detect_language_once() locks the language for the whole call,
        preventing the model from hallucinating a different language
        for each short segment.
      - Default model is 'small' (better Hindi/multilingual accuracy).

    Args:
        model_name : Whisper model size. One of:
                     'tiny', 'base', 'small', 'medium', 'large'.
                     Default is 'small' — recommended for Hindi/Indian languages.
        device     : 'cpu' or 'cuda'. Defaults to 'cpu'.
        language   : Force a specific language ISO code ('hi', 'en', etc.),
                     or None to use detect_language_once() / auto-detect.
    """

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        language: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.language = language  # None until detect_language_once() is called
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

    def detect_language_once(self, audio: np.ndarray) -> str:
        """
        Detect the dominant language from the first 30 seconds of call audio
        and cache it. All subsequent transcribe() calls will use this language.

        Call this once before processing segments — pass the raw audio array
        (full call or first 30s). Does nothing if self.language is already set.

        Args:
            audio : float32 numpy array at 16kHz. Only the first 30s is used.

        Returns:
            Detected ISO 639-1 language code, e.g. 'hi', 'en', 'te'.
        """
        if self.language is not None:
            logger.debug(
                f"[ASR] Language already set to '{self.language}' — skipping detection."
            )
            return self.language

        if self._model is None:
            logger.warning("[ASR] Model not loaded — cannot detect language.")
            return "hi"  # safe default for this project

        try:
            import whisper

            # Trim to detection window (Whisper pads/trims to 30s internally)
            window_samples = _LANG_DETECT_WINDOW_S * _WHISPER_SAMPLE_RATE
            detection_audio = audio[:window_samples].astype(np.float32)

            # Whisper's language detector works on a mel spectrogram
            detection_audio = whisper.pad_or_trim(detection_audio)
            mel = whisper.log_mel_spectrogram(detection_audio).to(self._model.device)
            _, probs = self._model.detect_language(mel)
            detected = max(probs, key=probs.get)

            self.language = detected
            logger.info(f"[ASR] Language detected and locked: '{detected}'")
            return detected

        except Exception as e:
            logger.warning(f"[ASR] Language detection failed ({e}) — defaulting to 'hi'.")
            self.language = "hi"
            return "hi"

    def transcribe(self, segment: SpeechSegment) -> Optional[Utterance]:
        """
        Transcribe a single speech segment.

        Uses self.language (set by detect_language_once) to avoid per-segment
        language hallucination on short Hindi clips.

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
                language=self.language,   # locked after detect_language_once()
                fp16=False,               # fp16 off for CPU compatibility
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"[ASR] Whisper transcription failed: {e}")
            return None

        raw_text: str = result.get("text", "").strip()
        detected_language: str = result.get("language", None) or self.language or "hi"

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
        self, segments: List[SpeechSegment]
    ) -> List[Optional[Utterance]]:
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


class SpeechAccumulator:
    """
    Buffers short VAD segments until enough audio is collected before
    calling Whisper. This dramatically reduces hallucination caused by
    transcribing very short (~0.3s) segments in isolation.

    How it works:
      - Call add(segment) for each VAD speech segment.
      - When accumulated audio >= min_duration_s, the buffer is flushed:
        all buffered audio is concatenated and transcribed as one call.
      - If a new segment has a different speaker than the current buffer,
        the buffer is flushed first to avoid mixing speakers.
      - Call flush() at end of call to transcribe any remaining audio.

    Args:
        asr           : A WhisperASR instance (with model already loaded).
        min_duration_s: Minimum seconds of speech before calling Whisper.
                        Default is 4.0s — balances latency vs. accuracy.

    Usage:
        acc = SpeechAccumulator(asr, min_duration_s=4.0)
        for seg in vad_segments:
            utterance = acc.add(seg)
            if utterance:
                buf.add(utterance)
        # End of call
        utterance = acc.flush()
        if utterance:
            buf.add(utterance)
    """

    def __init__(
        self,
        asr: WhisperASR,
        min_duration_s: float = 4.0,
    ) -> None:
        self._asr = asr
        self._min_duration_s = min_duration_s
        self._min_samples = int(min_duration_s * _WHISPER_SAMPLE_RATE)
        self._segments: List[SpeechSegment] = []
        self._accumulated_samples: int = 0

    @property
    def buffered_duration_s(self) -> float:
        """Seconds of audio currently buffered (not yet sent to Whisper)."""
        return self._accumulated_samples / _WHISPER_SAMPLE_RATE

    def add(self, segment: SpeechSegment) -> Optional[Utterance]:
        """
        Add a VAD speech segment to the accumulation buffer.

        Returns an Utterance if the buffer reached min_duration_s (or if
        the speaker changed and the old buffer was flushed), else None.

        A speaker change triggers an immediate flush of the existing buffer
        before the new segment is added, preventing speaker mixing.
        """
        # Speaker changed — flush existing buffer, then start fresh
        if self._segments and self._segments[0].speaker != segment.speaker:
            flushed = self._flush_internal()
            self._segments = [segment]
            self._accumulated_samples = len(segment.audio)
            return flushed

        self._segments.append(segment)
        self._accumulated_samples += len(segment.audio)

        if self._accumulated_samples >= self._min_samples:
            return self._flush_internal()

        return None

    def flush(self) -> Optional[Utterance]:
        """
        Force-flush remaining buffered audio regardless of duration.
        Call at end of call session to avoid losing the last utterance.
        """
        return self._flush_internal()

    def _flush_internal(self) -> Optional[Utterance]:
        """Concatenate buffered audio, transcribe, reset buffer."""
        if not self._segments:
            return None

        combined_audio = np.concatenate(
            [seg.audio for seg in self._segments]
            + [np.zeros(int(0.2 * _WHISPER_SAMPLE_RATE), dtype=np.float32)]
        )
        first = self._segments[0]
        last = self._segments[-1]

        combined_segment = SpeechSegment(
            audio=combined_audio,
            speaker=first.speaker,
            start_time=first.start_time,
            end_time=last.end_time,
            chunk_index=first.chunk_index,
        )

        self._segments = []
        self._accumulated_samples = 0

        return self._asr.transcribe(combined_segment)

    def __repr__(self) -> str:
        return (
            f"SpeechAccumulator("
            f"min_duration_s={self._min_duration_s}, "
            f"buffered={self.buffered_duration_s:.2f}s, "
            f"segments={len(self._segments)})"
        )
