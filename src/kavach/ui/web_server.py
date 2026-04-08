"""
kavach.ui.web_server
=====================
FastAPI backend for the Kavach web dashboard.

Runs the full detection pipeline in a background thread and streams
per-decision-point events to the browser via WebSocket.

Endpoints:
    GET  /          → serves static/index.html
    GET  /health    → {"status": "ok"}
    POST /analyze   → starts pipeline in background thread
    WS   /ws        → streams JSON events until {"type": "done"}

WebSocket message schema (one per decision point):
    {
      "timestamp":        float,          # seconds from call start
      "alert_level":      str,            # SAFE | CAUTION | ALERT | CRITICAL
      "final_score":      float,          # 0.0–1.0
      "tiers_detected":   List[int],      # e.g. [1, 3]
      "explanation":      str,
      "component_scores": {
          "heuristic": float,
          "classifier": float,
          "slm":        float,
      },
      "utterances": [
          {"speaker": str, "text": str, "timestamp": float},
          ...
      ]
    }

Terminal messages:
    {"type": "done"}              — pipeline finished
    {"type": "error", "message": str}  — pipeline raised an exception
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional

# ── sys.path so kavach package is importable when run as a module ──────────────
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Kavach Web Dashboard", version="1.0.0")

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

# Mount /static — serves index.html and any future assets
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── Global event-bus state ────────────────────────────────────────────────────
# All WebSocket connections subscribe via a queue. The pipeline thread
# emits events into every active queue (fan-out). One pipeline runs at a time.

_active_loop: Optional[asyncio.AbstractEventLoop] = None
_active_queues: List[asyncio.Queue] = []
_queue_lock = threading.Lock()

SAMPLE_RATE   = 16_000
CHUNK_S       = 2.5
CHUNK_SAMPLES = int(CHUNK_S * SAMPLE_RATE)


# ── Thread → asyncio bridge ───────────────────────────────────────────────────

def _emit(event: dict) -> None:
    """
    Thread-safe: called from the pipeline background thread to push an event
    to every connected WebSocket queue.
    """
    if _active_loop is None:
        return
    with _queue_lock:
        queues = list(_active_queues)
    for q in queues:
        _active_loop.call_soon_threadsafe(q.put_nowait, event)


# ── Gemini accumulator (mirrors run_pipeline._GeminiAccumulator) ──────────────

class _GeminiAccumulator:
    """
    Buffers VAD segments until ≥ min_duration_s of audio is collected,
    then calls GeminiASR.transcribe_raw() on the concatenated audio.
    Returns an Utterance when flushing, None while still accumulating.
    """

    def __init__(self, asr, min_duration_s: float = 10.0) -> None:
        self._asr          = asr
        self._min_s        = min_duration_s
        self._min_samples  = int(min_duration_s * SAMPLE_RATE)
        self._segments: list = []
        self._acc_samples  = 0

    def add(self, segment):
        if self._segments and self._segments[0].speaker != segment.speaker:
            flushed = self._flush()
            self._segments     = [segment]
            self._acc_samples  = len(segment.audio)
            return flushed

        self._segments.append(segment)
        self._acc_samples += len(segment.audio)

        if self._acc_samples >= self._min_samples:
            return self._flush()
        return None

    def flush(self):
        return self._flush()

    def _flush(self):
        from kavach.audio.buffer import Utterance
        if not self._segments:
            return None
        audio = np.concatenate([s.audio for s in self._segments])
        first = self._segments[0]
        self._segments    = []
        self._acc_samples = 0

        text = self._asr.transcribe_raw(audio)
        if not text:
            return None
        return Utterance(
            speaker     = first.speaker,
            text        = text,
            timestamp   = first.start_time,
            chunk_index = first.chunk_index,
            raw_text    = text,
            language    = self._asr.language or "hi",
        )


# ── Pipeline thread ───────────────────────────────────────────────────────────

def _load_config() -> dict:
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _run_pipeline(
    audio_path: str,
    first_n_seconds: Optional[float],
    mode: str,
    accumulate_s: float,
    slm_mode: str = "ollama",
) -> None:
    """
    Background thread — loads audio, runs the full Kavach pipeline,
    emits one WebSocket event per decision point, then emits {"type":"done"}.
    """
    try:
        import librosa
        from kavach.audio.buffer         import ConversationBuffer
        from kavach.audio.vad            import VoiceActivityDetector
        from kavach.detection.heuristics import HeuristicDetector
        from kavach.detection.classifier import MuRILClassifier
        from kavach.fusion.risk_scorer   import RiskScorer

        cfg     = _load_config()
        api_key = cfg.get("api_keys", {}).get("gemini", "")

        # ── Load audio ────────────────────────────────────────────────────────
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=first_n_seconds)
        audio    = audio.astype(np.float32)
        n_chunks = (len(audio) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES

        # ── Init pipeline modules ─────────────────────────────────────────────
        vad = VoiceActivityDetector(threshold=0.5)
        buf = ConversationBuffer(max_utterances=10)

        if mode == "gemini" and api_key:
            from kavach.transcription.gemini_asr import GeminiASR
            asr = GeminiASR(api_key=api_key)
            asr.detect_language_once(audio)
            accumulator = _GeminiAccumulator(asr, min_duration_s=accumulate_s)
        else:
            from kavach.transcription.whisper_asr import WhisperASR, SpeechAccumulator
            asr = WhisperASR(model_name="small", device="cpu")
            asr.detect_language_once(audio)
            accumulator = SpeechAccumulator(asr, min_duration_s=accumulate_s)

        heuristics = HeuristicDetector()
        clf        = MuRILClassifier(threshold=0.35)

        # ── Initialize SLM based on slm_mode ───────────────────────────────────
        slm = None
        if slm_mode == "gemini" and api_key:
            from kavach.detection.slm import GeminiSLM
            slm = GeminiSLM(api_key=api_key)
        elif slm_mode == "ollama":
            from kavach.detection.slm import OllamaLlamaSLM
            slm = OllamaLlamaSLM(host=cfg.get("slm", {}).get("ollama_host", "http://localhost:11434"))

        scorer     = RiskScorer(cfg)

        # ── Processing loop ───────────────────────────────────────────────────
        for chunk_idx in range(n_chunks):
            start = chunk_idx * CHUNK_SAMPLES
            end   = min(start + CHUNK_SAMPLES, len(audio))
            chunk = audio[start:end]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

            chunk_t  = chunk_idx * CHUNK_S
            segments = vad.process_chunk(
                caller_audio    = chunk,
                user_audio      = np.zeros_like(chunk),
                chunk_start_time= chunk_t,
                chunk_index     = chunk_idx,
            )

            for seg in segments:
                utt = accumulator.add(seg)
                if utt is None:
                    continue

                buf.add(utt)

                h_result  = heuristics.analyze(buf.caller_text_only())
                c_result  = clf.predict(buf.as_classifier_input())

                slm_result = None
                if slm is not None and c_result.escalate_to_slm:
                    slm_result = slm.analyze(
                        slm_context      = buf.as_slm_context(),
                        heuristic_score  = h_result.heuristic_score,
                        classifier_score = c_result.p_scam,
                    )

                risk = scorer.update(h_result, c_result, slm_result)

                utterances = [
                    {
                        "speaker":   u.speaker.value,
                        "text":      u.text,
                        "timestamp": u.timestamp,
                    }
                    for u in buf.utterances
                ]

                _emit({
                    "timestamp":       utt.timestamp,
                    "alert_level":     risk.alert_level,
                    "final_score":     risk.final_score,
                    "tiers_detected":  risk.tiers_seen_this_call,
                    "explanation":     risk.explanation,
                    "component_scores": risk.component_scores,
                    "utterances":      utterances,
                })

        # ── Flush accumulator tail ────────────────────────────────────────────
        utt = accumulator.flush()
        if utt is not None:
            buf.add(utt)
            h_result  = heuristics.analyze(buf.caller_text_only())
            c_result  = clf.predict(buf.as_classifier_input())
            slm_result = None
            if slm is not None and c_result.escalate_to_slm:
                slm_result = slm.analyze(
                    slm_context      = buf.as_slm_context(),
                    heuristic_score  = h_result.heuristic_score,
                    classifier_score = c_result.p_scam,
                )
            risk = scorer.update(h_result, c_result, slm_result)

            utterances = [
                {"speaker": u.speaker.value, "text": u.text, "timestamp": u.timestamp}
                for u in buf.utterances
            ]
            _emit({
                "timestamp":        utt.timestamp,
                "alert_level":      risk.alert_level,
                "final_score":      risk.final_score,
                "tiers_detected":   risk.tiers_seen_this_call,
                "explanation":      risk.explanation,
                "component_scores": risk.component_scores,
                "utterances":       utterances,
            })

        scorer.reset()

    except Exception as exc:
        _emit({"type": "error", "message": str(exc)})

    finally:
        _emit({"type": "done"})


# ── HTTP routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the single-page dashboard."""
    index_path = _STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


class AnalyzeRequest(BaseModel):
    audio_path:      str
    first_n_seconds: Optional[float] = None
    mode:            str             = "gemini"
    slm_mode:        str             = "ollama"
    accumulate_s:    float           = 10.0


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> dict:
    """
    Start the pipeline in a background thread.  Returns immediately;
    results stream via WebSocket /ws.
    """
    global _active_loop
    _active_loop = asyncio.get_running_loop()

    thread = threading.Thread(
        target  = _run_pipeline,
        args    = (
            request.audio_path,
            request.first_n_seconds,
            request.mode,
            request.accumulate_s,
            request.slm_mode,
        ),
        daemon  = True,
        name    = "kavach-pipeline",
    )
    thread.start()
    return {"status": "started", "audio_path": request.audio_path}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """
    Browser connects here. Receives every event emitted by the pipeline
    thread until {"type": "done"} or {"type": "error"}.
    """
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue()

    with _queue_lock:
        _active_queues.append(queue)

    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
            if event.get("type") in ("done", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        with _queue_lock:
            if queue in _active_queues:
                _active_queues.remove(queue)
