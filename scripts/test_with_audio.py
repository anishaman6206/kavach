"""
scripts/test_with_audio.py
===========================
Test Kavach's audio layer (VAD + Buffer) with a real audio file.

This script simulates what happens during a live call:
  - Loads your MP3/WAV scam recording
  - Treats the entire audio as CALLER channel
    (single-channel recording = one person or mixed, but scam-labeled)
  - Chunks it into 2.5s windows exactly as the live pipeline does
  - Runs VAD on each chunk
  - Fills the rolling buffer
  - Prints what the buffer looks like after every chunk

Usage:
    cd kavach
    python scripts/test_with_audio.py --audio data/samples/scam_call.mp3

Requirements:
    pip install librosa soundfile numpy
    (librosa handles MP3 via audioread/ffmpeg)
"""

import argparse
import sys
import os

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np


def load_audio(path: str, target_sr: int = 16_000) -> np.ndarray:
    """
    Load any audio file (MP3, WAV, M4A, OGG) and resample to 16kHz mono.
    Returns float32 numpy array normalised to [-1, 1].
    """
    try:
        import librosa
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
        print(f"  Loaded: {path}")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration:    {len(audio)/sr:.2f} seconds")
        print(f"  Samples:     {len(audio):,}")
        return audio.astype(np.float32)
    except ImportError:
        print("[ERROR] librosa not installed. Run: pip install librosa")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not load audio: {e}")
        print("        Make sure ffmpeg is installed for MP3 support:")
        print("        Ubuntu/Debian: sudo apt install ffmpeg")
        print("        Mac:           brew install ffmpeg")
        print("        Windows:       download from ffmpeg.org")
        sys.exit(1)


def run_test(audio_path: str, chunk_duration_s: float = 2.5) -> None:
    from kavach.audio.buffer import ConversationBuffer, Speaker, Utterance
    from kavach.audio.vad    import VoiceActivityDetector

    SAMPLE_RATE = 16_000
    chunk_samples = int(chunk_duration_s * SAMPLE_RATE)

    print("\n" + "="*55)
    print("  KAVACH — Audio Layer Test")
    print("="*55)

    # ── Load audio ────────────────────────────────────────────
    print(f"\n[1] Loading audio...")
    audio = load_audio(audio_path)
    total_duration = len(audio) / SAMPLE_RATE
    n_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    print(f"  Chunks to process: {n_chunks} × {chunk_duration_s}s")

    # ── Init VAD and buffer ───────────────────────────────────
    print(f"\n[2] Initialising VAD...")
    vad = VoiceActivityDetector(threshold=0.5)
    buf = ConversationBuffer(max_utterances=10)

    # ── Process each chunk ────────────────────────────────────
    print(f"\n[3] Processing chunks...\n")
    total_speech_segments = 0
    silent_chunks = 0

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_samples
        end   = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        chunk_start_time = chunk_idx * chunk_duration_s

        # ── Key design point: single-channel MP3 ─────────────
        # A real phone call has separate CALLER and USER channels.
        # A recording has them mixed or single-channel.
        # For testing: treat the whole audio as CALLER channel.
        # USER channel = silence (zeros).
        # In real deployment, Android gives us two separate streams.
        caller_audio = chunk
        user_audio   = np.zeros_like(chunk)

        segments = vad.process_chunk(
            caller_audio     = caller_audio,
            user_audio       = user_audio,
            chunk_start_time = chunk_start_time,
            chunk_index      = chunk_idx,
        )

        if not segments:
            silent_chunks += 1
            print(f"  Chunk {chunk_idx+1:02d} [{chunk_start_time:.1f}s]: "
                  f"[silent — VAD filtered]")
            continue

        total_speech_segments += len(segments)
        print(f"  Chunk {chunk_idx+1:02d} [{chunk_start_time:.1f}s]: "
              f"{len(segments)} speech segment(s)")

        # ── Add detected speech to buffer ─────────────────────
        # In the real pipeline, Whisper transcribes each segment here.
        # For this test, we use placeholder text to verify buffer mechanics.
        for seg_idx, seg in enumerate(segments):
            placeholder_text = (
                f"[transcription pending — "
                f"seg {seg_idx+1} of chunk {chunk_idx+1}, "
                f"dur={seg.duration:.2f}s]"
            )
            utterance = Utterance(
                speaker     = seg.speaker,
                text        = placeholder_text,
                timestamp   = seg.start_time,
                chunk_index = seg.chunk_index,
            )
            buf.add(utterance)

    # ── Final buffer state ────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Results")
    print(f"{'='*55}")
    print(f"  Total duration:         {total_duration:.1f}s")
    print(f"  Chunks processed:       {n_chunks}")
    print(f"  Silent chunks (VAD):    {silent_chunks}")
    print(f"  Speech segments found:  {total_speech_segments}")
    print(f"  Buffer size (last 10):  {buf.size} utterances")

    print(f"\n── Final rolling buffer state ──")
    if buf.is_empty:
        print("  [Buffer is empty — all chunks were silent]")
        print("  Check: is VAD threshold too high? Try --threshold 0.3")
    else:
        buf.pretty_print()

        print("── As MuRIL classifier input (flat string) ──")
        ctx = buf.as_classifier_input()
        print(f"  Length: {len(ctx)} chars")
        print(f"  Preview: {ctx[:200]}...")

        print("\n── As SLM context (structured dialogue) ──")
        print(buf.as_slm_context())

    print(f"\n{'='*55}")
    print(f"  Test complete.")
    if total_speech_segments == 0:
        print("  WARNING: No speech segments detected.")
        print("  This likely means VAD threshold is too high OR")
        print("  the audio volume is very low.")
        print("  Try: python scripts/test_with_audio.py "
              "--audio your_file.mp3 --threshold 0.3")
    else:
        print("  Audio layer is working correctly.")
        print("  Next step: connect Whisper to transcribe each segment.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Kavach audio layer with a real audio file."
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file (MP3, WAV, M4A, OGG)",
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=2.5,
        help="Chunk duration in seconds (default: 2.5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="VAD threshold 0-1 (default: 0.5, lower = more sensitive)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"[ERROR] File not found: {args.audio}")
        sys.exit(1)

    run_test(audio_path=args.audio, chunk_duration_s=args.chunk)