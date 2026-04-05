"""
kavach.audio.vad
=================
Voice Activity Detection using Silero VAD.

Strips silence from both CALLER and USER audio channels before
passing to Whisper — saves 40-60% of ASR compute on a typical call.

Design decisions:
  - Processes BOTH channels (CALLER earpiece + USER mic) independently
  - Returns SpeechSegment list — each tagged with Speaker enum
  - Silero VAD model is loaded once and reused across chunks
  - Falls back gracefully if torch.hub is unavailable (mock mode for testing)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from kavach.audio.buffer import Speaker

# ─────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────

@dataclass
class SpeechSegment:
    """
    A segment of confirmed speech audio, ready for Whisper.

    Attributes:
        audio       : raw PCM audio as float32 numpy array [-1, 1]
        speaker     : CALLER or USER
        start_time  : seconds from call start (global timestamp)
        end_time    : seconds from call start
        chunk_index : which processing chunk this came from
    """
    audio       : np.ndarray
    speaker     : Speaker
    start_time  : float
    end_time    : float
    chunk_index : int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:
        return (
            f"SpeechSegment({self.speaker.value}, "
            f"{self.start_time:.2f}s–{self.end_time:.2f}s, "
            f"dur={self.duration:.2f}s)"
        )


# ─────────────────────────────────────────────
# VAD
# ─────────────────────────────────────────────

class VoiceActivityDetector:
    """
    Wraps Silero VAD for dual-channel phone call audio.

    Phone calls give us two separate audio streams:
      - CALLER channel : audio received from the network (the other person)
      - USER channel   : audio from the device microphone (the phone owner)

    We run VAD independently on each channel, so we always know who is
    speaking. This is far simpler than general speaker diarization.

    Usage:
        vad = VoiceActivityDetector(threshold=0.5)
        segments = vad.process_chunk(
            caller_audio=caller_pcm,   # np.ndarray, float32, 16kHz
            user_audio=user_pcm,       # np.ndarray, float32, 16kHz
            chunk_start_time=0.0,
            chunk_index=0,
        )
        for seg in segments:
            print(seg.speaker, seg.duration)
    """

    SAMPLE_RATE      : int   = 16_000   # Hz — Whisper & Silero both expect 16kHz
    WINDOW_SIZE_MS   : int   = 32       # Silero expects 32ms windows (512 samples @ 16kHz)
    MIN_SPEECH_MS    : int   = 250      # ignore bursts shorter than this
    MIN_SILENCE_MS   : int   = 300      # gap needed to mark a new turn

    def __init__(
        self,
        threshold : float = 0.5,
        device    : str   = "cpu",
    ) -> None:
        """
        Args:
            threshold : Silero confidence threshold (0–1).
                        Higher = less sensitive (fewer false positives).
                        Lower  = more sensitive (catches quiet speech).
                        0.5 is the recommended default.
            device    : 'cpu' or 'cuda'. Use cpu for on-device mobile.
        """
        self.threshold = threshold
        self.device    = device
        self._model    = None
        self._utils    = None
        self._loaded   = False

        self._load_model()

    def _load_model(self) -> None:
        """
        Load Silero VAD from torch.hub.
        Safe to call multiple times — only loads once.
        """
        if self._loaded:
            return
        try:
            import torch
            self._model, self._utils = torch.hub.load(
                repo_or_dir = "snakers4/silero-vad",
                model       = "silero_vad",
                force_reload = False,
                verbose      = False,
            )
            self._model.to(self.device)
            self._loaded = True
        except Exception as e:
            # In test environments without internet / torch.hub,
            # fall back to energy-based VAD (less accurate but functional)
            print(f"[VAD] Silero load failed ({e}). Using energy-based fallback.")
            self._loaded = False

    def is_speech(self, audio: np.ndarray) -> float:
        """
        Return Silero speech probability for a single audio chunk.
        Falls back to RMS energy threshold if Silero is unavailable.

        Args:
            audio : float32 numpy array at 16kHz

        Returns:
            Probability 0.0–1.0 (≥ threshold = speech)
        """
        if self._loaded and self._model is not None:
            import torch
            tensor = torch.from_numpy(audio).float()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                prob = self._model(tensor, self.SAMPLE_RATE).item()
            return float(prob)
        else:
            # Energy-based fallback
            rms = float(np.sqrt(np.mean(audio ** 2)))
            # Typical speech RMS on a phone call is 0.02–0.15
            return min(rms / 0.05, 1.0)

    def process_chunk(
        self,
        caller_audio    : np.ndarray,
        user_audio      : np.ndarray,
        chunk_start_time: float,
        chunk_index     : int,
    ) -> List[SpeechSegment]:
        """
        Run VAD on one audio chunk from both channels.

        Args:
            caller_audio     : PCM float32 array at 16kHz (CALLER channel)
            user_audio       : PCM float32 array at 16kHz (USER channel)
            chunk_start_time : absolute timestamp (seconds from call start)
            chunk_index      : sequential chunk number

        Returns:
            List of SpeechSegment — one per detected speech region.
            May be empty if both channels are silent.
        """
        segments: List[SpeechSegment] = []

        for audio, speaker in [
            (caller_audio, Speaker.CALLER),
            (user_audio,   Speaker.USER),
        ]:
            detected = self._detect_segments(
                audio       = audio,
                speaker     = speaker,
                base_time   = chunk_start_time,
                chunk_index = chunk_index,
            )
            segments.extend(detected)

        # Sort by start_time across both channels
        segments.sort(key=lambda s: s.start_time)
        return segments

    def _detect_segments(
        self,
        audio       : np.ndarray,
        speaker     : Speaker,
        base_time   : float,
        chunk_index : int,
    ) -> List[SpeechSegment]:
        """
        Slide a 96ms window over the audio and collect speech regions.
        Merges adjacent speech windows separated by < MIN_SILENCE_MS.
        """
        window_samples  = int(self.SAMPLE_RATE * self.WINDOW_SIZE_MS / 1000)
        min_speech_samp = int(self.SAMPLE_RATE * self.MIN_SPEECH_MS  / 1000)
        min_silence_samp= int(self.SAMPLE_RATE * self.MIN_SILENCE_MS / 1000)

        if len(audio) < window_samples:
            return []

        # Slide window — collect (start, end, is_speech) tuples
        windows: List[Tuple[int, int, bool]] = []
        for start in range(0, len(audio) - window_samples, window_samples):
            end    = start + window_samples
            chunk  = audio[start:end]
            prob   = self.is_speech(chunk)
            windows.append((start, end, prob >= self.threshold))

        # Merge consecutive speech windows, filter by min_speech_duration
        segments: List[SpeechSegment] = []
        in_speech   = False
        speech_start= 0
        silence_len = 0

        for start_samp, end_samp, is_speech_window in windows:
            if is_speech_window:
                if not in_speech:
                    speech_start = start_samp
                    in_speech    = True
                silence_len = 0
            else:
                if in_speech:
                    silence_len += (end_samp - start_samp)
                    if silence_len >= min_silence_samp:
                        # End of speech region
                        speech_end   = end_samp - silence_len
                        speech_audio = audio[speech_start:speech_end]
                        if len(speech_audio) >= min_speech_samp:
                            t_start = base_time + speech_start / self.SAMPLE_RATE
                            t_end   = base_time + speech_end   / self.SAMPLE_RATE
                            segments.append(SpeechSegment(
                                audio       = speech_audio,
                                speaker     = speaker,
                                start_time  = t_start,
                                end_time    = t_end,
                                chunk_index = chunk_index,
                            ))
                        in_speech   = False
                        silence_len = 0

        # Handle speech that runs to end of chunk
        if in_speech:
            speech_audio = audio[speech_start:]
            if len(speech_audio) >= min_speech_samp:
                t_start = base_time + speech_start / self.SAMPLE_RATE
                t_end   = base_time + len(audio)   / self.SAMPLE_RATE
                segments.append(SpeechSegment(
                    audio       = speech_audio,
                    speaker     = speaker,
                    start_time  = t_start,
                    end_time    = t_end,
                    chunk_index = chunk_index,
                ))

        return segments