"""
scripts/compare_asr.py
=======================
Side-by-side quality comparison of Whisper vs. Gemini ASR.

Runs both backends on the same audio window and prints their transcripts
next to each other so you can judge accuracy, Hindi script quality,
hallucination rate, etc.

Whisper approach  : VAD → SpeechAccumulator → WhisperASR (segment-by-segment)
Gemini approach   : full audio window sent in one shot → GeminiASR.transcribe_raw()

This difference is intentional — Gemini has a much larger context window so it
benefits from seeing the whole chunk at once rather than short segments.

Usage:
    python scripts/compare_asr.py --audio data/samples/scam_call.mp3
    python scripts/compare_asr.py --audio data/samples/scam_call.mp3 --duration 60
    python scripts/compare_asr.py --audio data/samples/scam_call.mp3 --model tiny

Requirements:
    pip install librosa openai-whisper google-genai pyyaml
    configs/config.yaml must have api_keys.gemini set
"""

import argparse
import os
import sys
import textwrap

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Force UTF-8 output on Windows so Devanagari prints correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SAMPLE_RATE = 16_000
COLUMN_WIDTH = 60       # characters per column in side-by-side display


def load_audio(path: str, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration_s)
        return audio.astype(np.float32)
    except ImportError:
        print("[ERROR] librosa not installed. Run: pip install librosa")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not load audio: {e}")
        sys.exit(1)


def run_whisper(audio: np.ndarray, model_name: str, min_acc: float) -> str:
    """
    Full Whisper pipeline: VAD → SpeechAccumulator → WhisperASR.
    Returns concatenated transcript of all utterances.
    """
    from kavach.audio.vad import VoiceActivityDetector
    from kavach.transcription.whisper_asr import WhisperASR, SpeechAccumulator

    CHUNK_S = 2.5
    chunk_samples = int(CHUNK_S * SAMPLE_RATE)
    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples

    vad = VoiceActivityDetector(threshold=0.5)
    asr = WhisperASR(model_name=model_name, device="cpu")

    print("  [Whisper] Detecting language from first 30 s...")
    lang = asr.detect_language_once(audio)
    print(f"  [Whisper] Language locked: '{lang}'")

    acc = SpeechAccumulator(asr, min_duration_s=min_acc)
    utterances = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_samples
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        segments = vad.process_chunk(
            caller_audio=chunk,
            user_audio=np.zeros_like(chunk),
            chunk_start_time=chunk_idx * CHUNK_S,
            chunk_index=chunk_idx,
        )
        for seg in segments:
            utt = acc.add(seg)
            if utt:
                utterances.append(f"[{utt.timestamp:.1f}s] {utt.raw_text}")

    # flush tail
    utt = acc.flush()
    if utt:
        utterances.append(f"[{utt.timestamp:.1f}s] {utt.raw_text} [flush]")

    return "\n".join(utterances) if utterances else "(no speech detected)"


def run_gemini(audio: np.ndarray) -> str:
    """
    Gemini transcription: full audio sent in one shot.
    Returns the raw text response from the API.
    """
    from kavach.transcription.gemini_asr import GeminiASR

    print("  [Gemini] Initialising client...")
    asr = GeminiASR()

    print("  [Gemini] Detecting language from first 30 s...")
    lang = asr.detect_language_once(audio)
    print(f"  [Gemini] Language locked: '{lang}'")

    print("  [Gemini] Sending full audio chunk to API...")
    text = asr.transcribe_raw(audio)
    return text if text else "(no speech detected)"


def side_by_side(left_header: str, right_header: str, left: str, right: str) -> None:
    """Print two text blocks side by side in aligned columns."""
    w = COLUMN_WIDTH
    sep = " │ "

    def wrap(text: str) -> list:
        lines = []
        for para in text.splitlines():
            if not para.strip():
                lines.append("")
                continue
            lines.extend(textwrap.wrap(para, width=w) or [""])
        return lines

    left_lines = wrap(left)
    right_lines = wrap(right)
    total = max(len(left_lines), len(right_lines))

    # Header
    print(f"\n{'─'*(w*2 + len(sep))}")
    print(f"{left_header:<{w}}{sep}{right_header}")
    print(f"{'─'*(w*2 + len(sep))}")

    for i in range(total):
        l = left_lines[i] if i < len(left_lines) else ""
        r = right_lines[i] if i < len(right_lines) else ""
        print(f"{l:<{w}}{sep}{r}")

    print(f"{'─'*(w*2 + len(sep))}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Whisper vs Gemini ASR side-by-side."
    )
    parser.add_argument(
        "--audio", required=True,
        help="Path to audio file (MP3, WAV, M4A, …)"
    )
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Seconds of audio to compare (default: 30)"
    )
    parser.add_argument(
        "--model", type=str, default="small",
        help="Whisper model size: tiny/base/small/medium (default: small)"
    )
    parser.add_argument(
        "--accumulate", type=float, default=4.0,
        help="Whisper accumulation window in seconds (default: 4.0)"
    )
    parser.add_argument(
        "--whisper-only", action="store_true",
        help="Run only Whisper (skip Gemini API call — useful offline)"
    )
    parser.add_argument(
        "--gemini-only", action="store_true",
        help="Run only Gemini (skip Whisper model load)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"[ERROR] File not found: {args.audio}")
        sys.exit(1)

    print("\n" + "="*65)
    print("  KAVACH — ASR Quality Comparison: Whisper vs. Gemini")
    print("="*65)
    print(f"  Audio   : {args.audio}")
    print(f"  Duration: first {args.duration:.0f} s")
    if not args.gemini_only:
        print(f"  Whisper : {args.model} model, {args.accumulate}s accumulation")
    print()

    print("[1] Loading audio...")
    audio = load_audio(args.audio, args.duration)
    actual_s = len(audio) / SAMPLE_RATE
    print(f"  Loaded {actual_s:.1f} s of audio ({len(audio):,} samples @ {SAMPLE_RATE} Hz)\n")

    whisper_text = ""
    gemini_text = ""

    if not args.gemini_only:
        print("[2] Running Whisper pipeline...")
        try:
            whisper_text = run_whisper(audio, args.model, args.accumulate)
            print("  [Whisper] Done.\n")
        except Exception as e:
            whisper_text = f"[ERROR] {e}"
            print(f"  [Whisper] FAILED: {e}\n")

    if not args.whisper_only:
        print("[3] Running Gemini pipeline...")
        try:
            gemini_text = run_gemini(audio)
            print("  [Gemini] Done.\n")
        except Exception as e:
            gemini_text = f"[ERROR] {e}"
            print(f"  [Gemini] FAILED: {e}\n")

    # ── Side-by-side output ───────────────────────────────────────────────
    if args.gemini_only:
        print("\n── Gemini transcript ──")
        print(gemini_text)
    elif args.whisper_only:
        print("\n── Whisper transcript ──")
        print(whisper_text)
    else:
        whisper_header = f"WHISPER ({args.model})"
        gemini_header  = "GEMINI (gemini-2.5-flash)"
        side_by_side(whisper_header, gemini_header, whisper_text, gemini_text)

    print("Comparison complete.")


if __name__ == "__main__":
    main()
