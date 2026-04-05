# Kavach — Architecture Decision Record

## Core design principles

1. **On-device first** — no audio or transcript leaves the user's phone in production
2. **Speaker-aware** — all scam signal detection applies to CALLER speech only
3. **Context-aware** — decisions made on full conversation buffer, never single sentences
4. **Two-stage gating** — cheap classifier gates expensive SLM (80/20 split)
5. **Theory-driven** — Manipulation Funnel embedded as code, not just training signal

## The Manipulation Funnel

All effective banking scam calls follow this 3-tier sequence:

```
Tier 1: Authority / Impersonation
  "I'm calling from SBI / RBI / police..."
        ↓
Tier 2: Urgency / Threat
  "Your account will be frozen / you will be arrested..."
        ↓
Tier 3: Credential Extraction
  "Please share your OTP / PIN / account number..."
```

A call requires Tier 1 + Tier 3 (minimum) to be flagged. Tier 2 elevates confidence.

## Module responsibilities

### audio/
- `capture.py` — wraps OS audio API, outputs dual-channel (USER / CALLER) PCM stream
- `vad.py` — Silero VAD, filters silence, outputs speech-only segments
- `buffer.py` — rolling buffer of last N utterances, speaker-tagged

### transcription/
- `whisper_asr.py` — Whisper.cpp inference, outputs tagged utterance strings

### detection/
- `heuristics.py` — regex Tier signal detector, outputs heuristic_score (0–0.5)
- `classifier.py` — MuRIL fine-tuned binary classifier, outputs P(scam)
- `slm.py` — Gemini Flash / Llama 3.2 context reasoner, outputs structured verdict

### fusion/
- `risk_scorer.py` — weighted fusion, tier accumulation, temporal decay

### ui/
- `alert.py` — alert state machine, notification dispatcher

## Data flow

```
raw_audio_stream
    │
    ▼ (VAD)
speech_segments [caller_channel, user_channel]
    │
    ▼ (Whisper)
tagged_utterance: { speaker: "CALLER", text: "...", timestamp: t }
    │
    ▼ (append)
rolling_buffer: [utterance_1, ..., utterance_10]
    │
    ├──► heuristics(caller_utterances_only) → h_score
    │
    ├──► muril(full_buffer_text) → p_scam
    │         │
    │         └── if p_scam > 0.35:
    │                 slm(full_buffer) → {verdict, tiers, reason, confidence}
    │
    ▼
risk_fusion(h_score, p_scam, slm_verdict) → final_risk_score
    │
    ▼
alert_system(final_risk_score) → UI state
```

## Latency budget (per 2.5s audio chunk)

| Layer | Latency | Runs |
|---|---|---|
| VAD | ~5ms | Always |
| Whisper tiny | ~700ms | Always |
| Heuristics | ~3ms | Always |
| MuRIL (quantized) | ~120ms | Always |
| SLM (Gemini Flash API) | ~800ms | ~20% of chunks |
| Risk fusion | ~5ms | Always |
| **Total without SLM** | **~833ms** | 80% |
| **Total with SLM** | **~1,633ms** | 20% |

## Model choices

| Task | Demo (BTP) | Production |
|---|---|---|
| ASR | Whisper.cpp tiny | IndicConformer (AI4Bharat) |
| Classifier | MuRIL-base fine-tuned | MuRIL-base fine-tuned |
| SLM | Gemini Flash API | Llama 3.2 3B (4-bit, on-device) |

## Commit convention

```
feat: add VAD module using Silero
fix: correct speaker channel assignment in buffer
docs: update architecture with revised heuristic role
test: add unit tests for rolling buffer edge cases
refactor: extract tier detection patterns to config
```
