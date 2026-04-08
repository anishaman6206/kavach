"""
scripts/run_web_dashboard.py
=============================
One-command launcher for the Kavach web dashboard.

Usage:
    python scripts/run_web_dashboard.py --audio data/samples/scam_call.mp3 --first-60s
    python scripts/run_web_dashboard.py --audio data/samples/scam_call.mp3 --mode whisper --slm-mode ollama

What it does:
  1. Starts a FastAPI/uvicorn server (background thread) on port 8000
  2. Opens http://localhost:8000 in the default browser
  3. POSTs to /analyze with the given audio path
  4. Keeps the server alive until Ctrl+C

Flags:
    --audio       : path to audio file (MP3, WAV, …)          [required]
    --mode        : "gemini" (default) | "whisper"            [ASR backend]
    --slm-mode    : "ollama" (default) | "gemini"             [SLM backend]
    --first-Ns    : process only first N seconds  (e.g. --first-60s)
    --accumulate  : seconds of speech per ASR call (default 10)
    --port        : server port (default 8000)
    --no-browser  : skip auto-opening the browser
"""

import argparse
import os
import re
import sys
import time
import threading
import webbrowser
from pathlib import Path

# ── sys.path so kavach package is importable ──────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _parse_first_ns(args_list: list) -> tuple:
    """Extract --first-Ns (e.g. --first-60s → 60.0) from argv."""
    cleaned  = []
    duration = None
    for arg in args_list:
        m = re.fullmatch(r"--first-(\d+(?:\.\d+)?)s", arg)
        if m:
            duration = float(m.group(1))
        else:
            cleaned.append(arg)
    return cleaned, duration


def _wait_for_server(port: int, timeout: float = 30.0) -> bool:
    """Poll until the server responds to /health or timeout expires."""
    import urllib.request
    url      = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return True
        except Exception:
            time.sleep(0.25)
    return False


def _trigger_analyze(
    port: int,
    audio_path: str,
    first_n_seconds,
    mode: str,
    slm_mode: str,
    accumulate_s: float,
) -> None:
    """POST /analyze to kick off the pipeline."""
    import json
    import urllib.request

    payload = json.dumps({
        "audio_path":      audio_path,
        "first_n_seconds": first_n_seconds,
        "mode":            mode,
        "slm_mode":        slm_mode,
        "accumulate_s":    accumulate_s,
    }).encode()

    req = urllib.request.Request(
        url     = f"http://localhost:{port}/analyze",
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            print(f"[kavach] Pipeline started: {body}")
    except Exception as exc:
        print(f"[kavach] Failed to trigger /analyze: {exc}")


def main() -> None:
    raw_args = sys.argv[1:]
    clean_args, first_n_s = _parse_first_ns(raw_args)

    parser = argparse.ArgumentParser(
        description="Kavach web dashboard launcher."
    )
    parser.add_argument("--audio",      required=True,
                        help="Path to audio file (MP3, WAV, …)")
    parser.add_argument("--mode",       default="gemini",
                        choices=["gemini", "whisper"],
                        help="ASR backend (default: gemini)")
    parser.add_argument("--slm-mode",   default="ollama",
                        choices=["ollama", "gemini"],
                        help="SLM backend (default: ollama)")
    parser.add_argument("--accumulate", type=float, default=10.0,
                        help="Seconds of speech per transcription call (default 10)")
    parser.add_argument("--port",       type=int, default=8000,
                        help="Server port (default 8000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open the browser")
    args = parser.parse_args(clean_args)

    if not os.path.exists(args.audio):
        print(f"[ERROR] Audio file not found: {args.audio}")
        sys.exit(1)

    audio_path = os.path.abspath(args.audio)

    # ── Start uvicorn in a daemon thread ──────────────────────────────────────
    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    config = uvicorn.Config(
        app       = "kavach.ui.web_server:app",
        host      = "0.0.0.0",
        port      = args.port,
        log_level = "warning",
    )
    server = uvicorn.Server(config)

    server_thread = threading.Thread(target=server.run, daemon=True, name="uvicorn")
    server_thread.start()

    print(f"[kavach] Starting server on http://localhost:{args.port} ...")

    # ── Wait for server to be ready ───────────────────────────────────────────
    if not _wait_for_server(args.port, timeout=30.0):
        print("[ERROR] Server did not start within 30 seconds.")
        sys.exit(1)

    print(f"[kavach] Server ready.")

    # ── Open browser ──────────────────────────────────────────────────────────
    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        print(f"[kavach] Opening browser at {url}")
        webbrowser.open(url)

    # ── Brief pause so browser connects WebSocket before pipeline starts ──────
    time.sleep(1.5)

    # ── Trigger the analysis ──────────────────────────────────────────────────
    print(f"[kavach] Sending audio to pipeline: {audio_path}")
    if first_n_s:
        print(f"[kavach] Processing first {first_n_s:.0f}s of audio")
    _trigger_analyze(args.port, audio_path, first_n_s, args.mode, args.slm_mode, args.accumulate)

    # ── Keep server alive until Ctrl+C ────────────────────────────────────────
    print("[kavach] Dashboard running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[kavach] Shutting down.")
        server.should_exit = True


if __name__ == "__main__":
    main()
