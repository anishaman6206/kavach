"""
kavach.transcription.gemini_asr
=================================
Gemini Flash audio transcription backend.

Purpose: offline ground-truth transcription for test audio files and
quality comparison against Whisper. NOT for production pipeline —
audio is sent to Google's servers, so privacy requirements rule it out
for live call monitoring.

Typical use:
    python scripts/compare_asr.py --audio data/samples/scam_call.mp3

Design decisions:
  - Same external interface as WhisperASR so both backends are swappable
    in compare_asr.py without changing call sites.
  - Audio is sent as inline base64-encoded WAV for files < 20 MB.
    Larger files use the Gemini Files API (upload → reference → delete).
  - Language is detected once from the first 30 s by asking Gemini,
    then locked for the rest of the session — mirrors WhisperASR behaviour.
  - Uses google-genai (new SDK), not the deprecated google-generativeai.
  - API key loaded from configs/config.yaml → api_keys.gemini.
  - Controlled by gemini_asr: true/false toggle in config.

Usage:
    from kavach.transcription.gemini_asr import GeminiASR

    asr = GeminiASR()                          # loads key from config.yaml
    asr = GeminiASR(api_key="AIza...")          # or pass directly

    # Detect dominant language from first 30 s of audio
    lang = asr.detect_language_once(audio_np)  # returns 'hi', 'en', etc.

    # Transcribe a VAD speech segment (same as WhisperASR)
    utterance = asr.transcribe(segment)        # returns Utterance | None

    # Or transcribe raw audio directly (useful in compare script)
    text = asr.transcribe_raw(audio_np)        # returns str
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np

from kavach.audio.buffer import Speaker, Utterance
from kavach.audio.vad import SpeechSegment

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_MIN_RMS_THRESHOLD = 1e-4
_MAX_INLINE_BYTES = 20 * 1024 * 1024        # 20 MB — above this use Files API
_LANG_DETECT_WINDOW_S = 30
_DEFAULT_MODEL = "gemini-2.5-flash"

# System instruction sent with every transcription request
_TRANSCRIPTION_SYSTEM = (
    "You are a professional transcription service. "
    "Transcribe the audio exactly as spoken. "
    "Rules:\n"
    "- Preserve the original language: Hindi in Devanagari script, "
    "  English in Latin script, Tamil in Tamil script, etc.\n"
    "- Do NOT translate under any circumstances.\n"
    "- Do NOT add punctuation that was not implied by speech.\n"
    "- Do NOT add explanatory text — output ONLY the transcription.\n"
    "- If the audio contains code-switching (e.g., Hindi + English), "
    "  transcribe each word in its original script."
)

_TRANSCRIPTION_PROMPT = "Transcribe this audio:"

_LANG_DETECT_SYSTEM = (
    "You are a language identification service. "
    "Listen to the audio and reply with ONLY the ISO 639-1 language code "
    "of the dominant spoken language — e.g. 'hi' for Hindi, 'en' for English, "
    "'te' for Telugu, 'ta' for Tamil, 'mr' for Marathi, 'bn' for Bengali. "
    "Reply with the two-letter code only, nothing else."
)


def _load_api_key(config_path: Optional[Path] = None) -> Optional[str]:
    """
    Load the Gemini API key from config.yaml.
    Falls back to the GEMINI_API_KEY environment variable if config is missing.
    """
    # 1. Try config.yaml
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent.parent / "configs" / "config.yaml"
        )
    if config_path.exists():
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            key = cfg.get("api_keys", {}).get("gemini", "")
            if key and key != "YOUR_GEMINI_API_KEY_HERE":
                return key
        except Exception as e:
            logger.warning(f"[GeminiASR] Could not read config.yaml: {e}")

    # 2. Fallback: environment variable
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key:
        return env_key

    return None


def _is_gemini_enabled(config_path: Optional[Path] = None) -> bool:
    """Check the gemini_asr: true/false toggle in config.yaml."""
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent.parent / "configs" / "config.yaml"
        )
    if not config_path.exists():
        return True  # no config → assume enabled
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return bool(cfg.get("asr", {}).get("gemini_asr", True))
    except Exception:
        return True


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """
    Convert a float32 numpy audio array to raw WAV bytes (PCM 16-bit mono).
    Gemini accepts audio/wav as inline data or via Files API.
    """
    buf = io.BytesIO()
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


class GeminiASR:
    """
    Gemini Flash audio transcription — same interface as WhisperASR.

    This is an OFFLINE quality-comparison tool only. Audio is sent to
    Google's servers, so it must never be used in the live call pipeline.

    Args:
        api_key     : Gemini API key. If None, loaded from config.yaml or
                      GEMINI_API_KEY env var.
        model_name  : Gemini model to use. Default: 'gemini-2.5-flash'.
        language    : Force a specific ISO 639-1 language code, or None to
                      use detect_language_once() / let Gemini decide.
        config_path : Path to config.yaml. If None, uses default location.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = _DEFAULT_MODEL,
        language: Optional[str] = None,
        config_path: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self._config_path = config_path
        self._client = None

        resolved_key = api_key or _load_api_key(config_path)
        if not resolved_key:
            raise ValueError(
                "[GeminiASR] No API key found. Set api_keys.gemini in config.yaml "
                "or export GEMINI_API_KEY=<key>."
            )

        if not _is_gemini_enabled(config_path):
            raise RuntimeError(
                "[GeminiASR] Gemini ASR is disabled (asr.gemini_asr: false in config.yaml). "
                "Set it to true to use this backend."
            )

        self._load_client(resolved_key)

    def _load_client(self, api_key: str) -> None:
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            logger.info(f"[GeminiASR] Client initialised (model: {self.model_name}).")
        except ImportError:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            )

    # ── Core audio → text helpers ─────────────────────────────────────────

    def _send_audio(self, wav_bytes: bytes, prompt: str, system: str) -> str:
        """
        Send WAV bytes to Gemini and return the text response.
        Uses inline data if < 20 MB, otherwise uploads via Files API.
        """
        from google.genai import types

        if len(wav_bytes) <= _MAX_INLINE_BYTES:
            audio_part = types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[prompt, audio_part],
                config=types.GenerateContentConfig(system_instruction=system),
            )
        else:
            # Large file — upload via Files API, then reference it
            response = self._send_via_files_api(wav_bytes, prompt, system)

        return (response.text or "").strip()

    def _send_via_files_api(self, wav_bytes: bytes, prompt: str, system: str) -> object:
        """Upload audio through Files API for files >= 20 MB."""
        from google.genai import types

        tmp_path = None
        uploaded_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                tmp_path = f.name

            logger.info(f"[GeminiASR] Uploading {len(wav_bytes)/1e6:.1f} MB via Files API...")
            uploaded_file = self._client.files.upload(
                file=tmp_path,
                config={"mime_type": "audio/wav"},
            )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[prompt, uploaded_file],
                config=types.GenerateContentConfig(system_instruction=system),
            )
            return response

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if uploaded_file is not None:
                try:
                    self._client.files.delete(name=uploaded_file.name)
                    logger.debug("[GeminiASR] Uploaded file deleted from Files API.")
                except Exception:
                    pass

    # ── Public interface (mirrors WhisperASR) ─────────────────────────────

    def transcribe_raw(self, audio: np.ndarray) -> str:
        """
        Transcribe a raw numpy audio array directly.

        This is the primary method used in compare_asr.py — sends the full
        audio chunk to Gemini in one shot (better quality than segment-by-segment).

        Args:
            audio : float32 numpy array at 16 kHz.

        Returns:
            Transcribed text string (empty string if audio is silence/noise).
        """
        if self._client is None:
            raise RuntimeError("[GeminiASR] Client not initialised.")

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < _MIN_RMS_THRESHOLD:
            logger.debug(f"[GeminiASR] Audio RMS {rms:.2e} below threshold — skipping.")
            return ""

        prompt = _TRANSCRIPTION_PROMPT
        if self.language:
            prompt = (
                f"Transcribe this audio. The spoken language is '{self.language}'. "
                "Preserve original script (Devanagari for Hindi, etc.):"
            )

        try:
            wav_bytes = _audio_to_wav_bytes(audio)
            text = self._send_audio(wav_bytes, prompt, _TRANSCRIPTION_SYSTEM)
            logger.debug(f"[GeminiASR] transcribe_raw → {text[:80]!r}")
            return text
        except Exception as e:
            logger.warning(f"[GeminiASR] Transcription failed: {e}")
            return ""

    def detect_language_once(self, audio: np.ndarray) -> str:
        """
        Detect the dominant language from the first 30 s of audio.
        Caches result in self.language — no-op if already set.

        Args:
            audio : float32 numpy array at 16 kHz. Only first 30 s used.

        Returns:
            ISO 639-1 language code, e.g. 'hi', 'en', 'te'.
        """
        if self.language is not None:
            logger.debug(
                f"[GeminiASR] Language already set to '{self.language}' — skipping."
            )
            return self.language

        window_samples = _LANG_DETECT_WINDOW_S * _SAMPLE_RATE
        detection_audio = audio[:window_samples].astype(np.float32)

        try:
            wav_bytes = _audio_to_wav_bytes(detection_audio)
            result = self._send_audio(
                wav_bytes,
                "What language is spoken in this audio?",
                _LANG_DETECT_SYSTEM,
            )
            # Normalise: strip punctuation, lowercase, take first token
            detected = result.strip().lower().split()[0].rstrip(".,;:")
            if len(detected) == 2 and detected.isalpha():
                self.language = detected
                logger.info(f"[GeminiASR] Language detected and locked: '{detected}'")
                return detected
            else:
                logger.warning(
                    f"[GeminiASR] Unexpected language response '{result}' — defaulting to 'hi'."
                )
        except Exception as e:
            logger.warning(f"[GeminiASR] Language detection failed ({e}) — defaulting to 'hi'.")

        self.language = "hi"
        return "hi"

    def transcribe(self, segment: SpeechSegment) -> Optional[Utterance]:
        """
        Transcribe a single VAD speech segment.
        Same signature as WhisperASR.transcribe() — drop-in compatible.

        Args:
            segment : SpeechSegment from VoiceActivityDetector.

        Returns:
            Utterance ready for ConversationBuffer, or None if skipped.
        """
        audio = segment.audio

        min_samples = int(0.1 * _SAMPLE_RATE)
        if len(audio) < min_samples:
            logger.debug(f"[GeminiASR] Segment too short ({len(audio)} samples) — skipping.")
            return None

        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms < _MIN_RMS_THRESHOLD:
            logger.debug(f"[GeminiASR] Segment RMS {rms:.2e} below threshold — skipping.")
            return None

        raw_text = self.transcribe_raw(audio)
        if not raw_text:
            return None

        return Utterance(
            speaker=segment.speaker,
            text=raw_text,
            timestamp=segment.start_time,
            chunk_index=segment.chunk_index,
            raw_text=raw_text,
            language=self.language or "hi",
        )

    def transcribe_batch(
        self, segments: List[SpeechSegment]
    ) -> List[Optional[Utterance]]:
        """
        Transcribe a list of segments in order.
        Returns a list of the same length — None for skipped segments.
        """
        return [self.transcribe(seg) for seg in segments]

    def __repr__(self) -> str:
        return (
            f"GeminiASR(model='{self.model_name}', "
            f"language={self.language!r}, "
            f"connected={self._client is not None})"
        )
