"""
scripts/run_pipeline.py
========================
End-to-end Kavach demo pipeline on a real audio file.

Ties together every module:
    Audio → VAD → ASR (Gemini or Whisper) → ConversationBuffer
    → HeuristicDetector → MuRILClassifier → GeminiSLM (gated) → RiskScorer
    → live risk timeline

The ASR accumulates speech across 2.5s VAD chunks. A transcription is
triggered when accumulated speech reaches 10s (--accumulate flag), not on
every chunk — this keeps Gemini API calls reasonable and gives the LLM
enough context per segment.

Usage:
    python scripts/run_pipeline.py --audio data/samples/scam_call.mp3
    python scripts/run_pipeline.py --audio data/samples/scam_call.mp3 --first-60s
    python scripts/run_pipeline.py --audio data/samples/scam_call.mp3 --mode whisper
    python scripts/run_pipeline.py --audio data/samples/scam_call.mp3 --accumulate 5

Flags:
    --audio       : path to audio file (MP3, WAV, M4A, …)
    --mode        : "gemini" (default) | "whisper" — ASR backend
    --first-Ns    : process only first N seconds (e.g. --first-60s)
    --accumulate  : seconds of speech to accumulate before transcribing (default 10)
    --vad-threshold: Silero VAD sensitivity 0–1 (default 0.5)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SAMPLE_RATE  = 16_000
CHUNK_S      = 2.5
CHUNK_SAMPLES = int(CHUNK_S * SAMPLE_RATE)

_ALERT_ICONS = {
    "SAFE":     "   ",
    "CAUTION":  "⚠️ ",
    "ALERT":    "🔴 ",
    "CRITICAL": "🚨 ",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str, duration_s: float | None = None) -> np.ndarray:
    try:
        import librosa
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=duration_s)
        return audio.astype(np.float32)
    except ImportError:
        print("[ERROR] librosa not installed. Run: pip install librosa")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not load audio: {e}")
        sys.exit(1)


def load_config() -> dict:
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _fmt_level(level: str) -> str:
    icons = {"SAFE": "[ SAFE ]", "CAUTION": "[CAUTION]",
             "ALERT": "[ ALERT]", "CRITICAL": "[CRITIC!]"}
    return icons.get(level, level)


# ─────────────────────────────────────────────────────────────────────────────
# GeminiASR accumulator (10s window instead of Whisper's 4s)
# ─────────────────────────────────────────────────────────────────────────────

class _GeminiAccumulator:
    """
    Buffers VAD segments until >= min_duration_s of audio is collected,
    then calls GeminiASR.transcribe_raw() on the concatenated audio.
    Returns an Utterance when flushing, None otherwise.
    Mirrors SpeechAccumulator's API so the pipeline loop is identical.
    """
    def __init__(self, asr, min_duration_s: float = 10.0):
        from kavach.audio.vad import SpeechSegment
        self._asr = asr
        self._min_s = min_duration_s
        self._min_samples = int(min_duration_s * SAMPLE_RATE)
        self._segments = []
        self._acc_samples = 0

    def add(self, segment):
        from kavach.audio.buffer import Speaker, Utterance

        if self._segments and self._segments[0].speaker != segment.speaker:
            flushed = self._flush()
            self._segments = [segment]
            self._acc_samples = len(segment.audio)
            return flushed

        self._segments.append(segment)
        self._acc_samples += len(segment.audio)

        if self._acc_samples >= self._min_samples:
            return self._flush()
        return None

    def flush(self):
        return self._flush()

    def _flush(self):
        from kavach.audio.buffer import Speaker, Utterance
        if not self._segments:
            return None
        audio = np.concatenate([s.audio for s in self._segments])
        first = self._segments[0]
        self._segments = []
        self._acc_samples = 0

        text = self._asr.transcribe_raw(audio)
        if not text:
            return None
        return Utterance(
            speaker=first.speaker,
            text=text,
            timestamp=first.start_time,
            chunk_index=first.chunk_index,
            raw_text=text,
            language=self._asr.language or "hi",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    audio_path: str,
    mode: str = "gemini",
    duration_s: float | None = None,
    accumulate_s: float = 10.0,
    vad_threshold: float = 0.5,
    ui: str = "none",
) -> None:
    from kavach.audio.buffer        import ConversationBuffer
    from kavach.audio.vad           import VoiceActivityDetector
    from kavach.detection.heuristics import HeuristicDetector
    from kavach.detection.classifier import MuRILClassifier
    from kavach.detection.slm        import GeminiSLM
    from kavach.fusion.risk_scorer   import RiskScorer

    cfg = load_config()
    api_key = cfg.get("api_keys", {}).get("gemini", "")

    # ── Terminal UI (optional) ────────────────────────────────────────────────
    terminal_ui = None
    if ui == "terminal":
        from kavach.ui.terminal_ui import KavachTerminalUI
        terminal_ui = KavachTerminalUI()

    # ── Banner (plain mode only — terminal UI takes over the screen) ──────────
    if terminal_ui is None:
        print("\n" + "=" * 65)
        print("  KAVACH — End-to-End Scam Detection Pipeline")
        print("=" * 65)
        print(f"  Audio  : {audio_path}")
        print(f"  Mode   : {mode.upper()} ASR")
        if duration_s:
            print(f"  Window : first {duration_s:.0f}s")
        print(f"  Accum  : {accumulate_s}s per transcription call")
        print()

    def _log(msg: str) -> None:
        if terminal_ui is None:
            print(msg)

    # ── Load audio ────────────────────────────────────────────────────────────
    _log("[1] Loading audio...")
    audio = load_audio(audio_path, duration_s)
    total_s = len(audio) / SAMPLE_RATE
    n_chunks = (len(audio) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    _log(f"    {total_s:.1f}s loaded ({n_chunks} chunks × {CHUNK_S}s)\n")

    # ── Init modules ─────────────────────────────────────────────────────────
    _log("[2] Initialising modules...")

    vad = VoiceActivityDetector(threshold=vad_threshold)
    _log(f"    VAD           : Silero (threshold={vad_threshold})")

    buf = ConversationBuffer(max_utterances=10)

    if mode == "gemini":
        if not api_key:
            print("[ERROR] No Gemini API key in config.yaml. Use --mode whisper.")
            sys.exit(1)
        from kavach.transcription.gemini_asr import GeminiASR
        asr = GeminiASR(api_key=api_key)
        _log(f"    ASR           : GeminiASR (gemini-2.5-flash)")
        _log(f"    Lang detect   : detecting from first 30s...")
        asr.detect_language_once(audio)
        _log(f"    Language      : {asr.language}")
        accumulator = _GeminiAccumulator(asr, min_duration_s=accumulate_s)
    else:
        from kavach.transcription.whisper_asr import WhisperASR, SpeechAccumulator
        asr = WhisperASR(model_name="small", device="cpu")
        _log(f"    ASR           : WhisperASR (small)")
        _log(f"    Lang detect   : detecting from first 30s...")
        asr.detect_language_once(audio)
        _log(f"    Language      : {asr.language}")
        accumulator = SpeechAccumulator(asr, min_duration_s=accumulate_s)

    heuristics  = HeuristicDetector()
    _log(f"    Heuristics    : HeuristicDetector (regex tier detector)")

    clf = MuRILClassifier(threshold=0.35)
    _log(f"    Classifier    : MuRIL (zero-shot, threshold=0.35)")

    if api_key:
        slm = GeminiSLM(api_key=api_key)
        _log(f"    SLM           : GeminiSLM (gemini-2.5-flash)")
    else:
        slm = None
        _log(f"    SLM           : DISABLED (no API key)")

    scorer = RiskScorer(cfg)
    _log(f"    RiskScorer    : weights h=0.20 c=0.30 s=0.50, decay=0.90")

    # ── Timeline header (plain mode only) ────────────────────────────────────
    if terminal_ui is None:
        print(f"\n{'─'*65}")
        print(f"  {'TIME':>5}  {'LEVEL':^9}  {'SCORE':^6}  TIERS  EXPLANATION")
        print(f"{'─'*65}")

    # ── Start UI after all modules are loaded ─────────────────────────────────
    if terminal_ui is not None:
        terminal_ui.start()

    # ── Processing loop ───────────────────────────────────────────────────────
    pipeline_start = time.perf_counter()
    total_utterances = 0
    total_slm_calls  = 0
    highest_result   = None
    prev_level       = "SAFE"

    for chunk_idx in range(n_chunks):
        start    = chunk_idx * CHUNK_SAMPLES
        end      = min(start + CHUNK_SAMPLES, len(audio))
        chunk    = audio[start:end]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        chunk_t = chunk_idx * CHUNK_S

        segments = vad.process_chunk(
            caller_audio=chunk,
            user_audio=np.zeros_like(chunk),
            chunk_start_time=chunk_t,
            chunk_index=chunk_idx,
        )

        for seg in segments:
            utt = accumulator.add(seg)
            if utt is None:
                continue

            buf.add(utt)
            total_utterances += 1

            # ── Detection ────────────────────────────────────────────────────
            h_result  = heuristics.analyze(buf.caller_text_only())
            c_result  = clf.predict(buf.as_classifier_input())

            slm_result = None
            if slm is not None and c_result.escalate_to_slm:
                slm_result = slm.analyze(
                    slm_context=buf.as_slm_context(),
                    heuristic_score=h_result.heuristic_score,
                    classifier_score=c_result.p_scam,
                )
                total_slm_calls += 1

            risk = scorer.update(h_result, c_result, slm_result)
            elapsed_so_far = time.perf_counter() - pipeline_start

            if highest_result is None or risk.final_score > highest_result.final_score:
                highest_result = risk

            if terminal_ui is not None:
                terminal_ui.update(risk, buf, elapsed_s=utt.timestamp)
                if risk.alert_level in ("ALERT", "CRITICAL"):
                    terminal_ui.show_alert(risk)
            else:
                # Plain mode: print row only when level changes or on first utterance
                level_changed = risk.alert_level != prev_level
                if level_changed or total_utterances == 1:
                    tiers_str = "+".join(str(t) for t in risk.tiers_seen_this_call) or "—"
                    expl = risk.explanation[:42] + "…" if len(risk.explanation) > 42 else risk.explanation
                    print(
                        f"  {utt.timestamp:>5.0f}s  "
                        f"{_fmt_level(risk.alert_level)}  "
                        f"{risk.final_score:>5.3f}  "
                        f"[{tiers_str:^5}]  {expl}"
                    )
                    prev_level = risk.alert_level

    # Flush accumulator tail
    utt = accumulator.flush()
    if utt is not None:
        buf.add(utt)
        total_utterances += 1
        h_result  = heuristics.analyze(buf.caller_text_only())
        c_result  = clf.predict(buf.as_classifier_input())
        slm_result = None
        if slm is not None and c_result.escalate_to_slm:
            slm_result = slm.analyze(buf.as_slm_context(), h_result.heuristic_score, c_result.p_scam)
            total_slm_calls += 1
        risk = scorer.update(h_result, c_result, slm_result)
        if highest_result is None or risk.final_score > highest_result.final_score:
            highest_result = risk
        if terminal_ui is not None:
            terminal_ui.update(risk, buf, elapsed_s=utt.timestamp)
        elif risk.alert_level != prev_level:
            tiers_str = "+".join(str(t) for t in risk.tiers_seen_this_call) or "—"
            expl = risk.explanation[:42] + "…" if len(risk.explanation) > 42 else risk.explanation
            print(
                f"  {utt.timestamp:>5.0f}s  "
                f"{_fmt_level(risk.alert_level)}  "
                f"{risk.final_score:>5.3f}  "
                f"[{tiers_str:^5}]  {expl}"
            )

    elapsed = time.perf_counter() - pipeline_start

    # ── Stop UI / print final summary ─────────────────────────────────────────
    if terminal_ui is not None:
        terminal_ui.stop()
    else:
        print(f"{'─'*65}")
        print(f"\n  ── FINAL SUMMARY ──────────────────────────────────────────")
        if highest_result:
            print(f"  Verdict         : {highest_result.alert_level}")
            print(f"  Highest score   : {highest_result.final_score:.3f}")
            print(f"  Tiers seen      : {highest_result.tiers_seen_this_call or '(none)'}")
            print(f"  Explanation     : {highest_result.explanation}")
        else:
            print("  Verdict         : SAFE (no speech detected)")
        print(f"  Audio duration  : {total_s:.1f}s")
        print(f"  Utterances      : {total_utterances}")
        print(f"  SLM calls       : {total_slm_calls}")
        print(f"  Pipeline time   : {elapsed:.1f}s")
        print(f"  {'='*57}")
    scorer.reset()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_first_ns(args_list: list[str]) -> tuple[list[str], float | None]:
    """Extract --first-Ns flag (e.g. --first-60s → 60.0) from argv."""
    import re
    cleaned = []
    duration = None
    for arg in args_list:
        m = re.fullmatch(r"--first-(\d+(?:\.\d+)?)s", arg)
        if m:
            duration = float(m.group(1))
        else:
            cleaned.append(arg)
    return cleaned, duration


if __name__ == "__main__":
    raw_args = sys.argv[1:]
    clean_args, first_n_s = _parse_first_ns(raw_args)

    parser = argparse.ArgumentParser(
        description="Kavach end-to-end scam detection pipeline."
    )
    parser.add_argument("--audio",         required=True,
                        help="Path to audio file (MP3, WAV, …)")
    parser.add_argument("--mode",          default="gemini",
                        choices=["gemini", "whisper"],
                        help="ASR backend: gemini (default) | whisper")
    parser.add_argument("--accumulate",    type=float, default=10.0,
                        help="Seconds of speech to accumulate before transcribing (default 10)")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="Silero VAD threshold 0–1 (default 0.5)")
    parser.add_argument("--ui",            default="terminal",
                        choices=["terminal", "none"],
                        help="UI mode: terminal (rich live dashboard) | none (plain print, default for CI)")
    args = parser.parse_args(clean_args)

    if not os.path.exists(args.audio):
        print(f"[ERROR] File not found: {args.audio}")
        sys.exit(1)

    run_pipeline(
        audio_path=args.audio,
        mode=args.mode,
        duration_s=first_n_s,
        accumulate_s=args.accumulate,
        vad_threshold=args.vad_threshold,
        ui=args.ui,
    )
