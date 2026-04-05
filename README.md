# Kavach — Real-Time AI Banking Scam Call Detection

> **Kavach** (कवच) means *armour* in Hindi. This system protects users from voice phishing (vishing) attacks during live phone calls, entirely on-device.

## What it does

Kavach listens to incoming calls in real time, transcribes speech using on-device ASR, and uses a two-stage AI pipeline to detect banking scam patterns — flagging the call with an alert before any financial harm occurs.

## Architecture overview

```
Live call audio
      │
      ▼
[VAD] Silero — strips silence
      │
      ▼
[ASR] Whisper.cpp — speech → text, speaker tagged
      │
      ▼
[Rolling Buffer] — last 10 utterances, CALLER/USER labelled
      │
      ├──► [Heuristic layer] — Manipulation Funnel tier signals
      │
      ├──► [MuRIL classifier] — fast binary gate (120ms)
      │
      └──► [Llama 3.2 / Gemini Flash] — context reasoning (selective)
                    │
                    ▼
           [Risk fusion] — score + tier accumulation
                    │
                    ▼
           [Alert UI] — real-time risk bar + warnings
```

## Manipulation Funnel model

Scam calls follow a predictable 3-tier pattern:
- **Tier 1** — Authority / Impersonation (fake RBI officer, bank, police)
- **Tier 2** — Urgency / Threat (arrest, account freeze, legal action)
- **Tier 3** — Credential Extraction (OTP, PIN, account number, Aadhaar)

A call is flagged only when **Tier 1 + Tier 3** are both present in the CALLER's speech, evaluated with full conversational context — not keyword matching.

## Tech stack

| Component | Technology |
|---|---|
| Voice Activity Detection | Silero VAD |
| Speech-to-Text | Whisper.cpp (tiny, quantized) → IndicConformer (production) |
| Fast classifier | MuRIL (google/muril-base-cased, fine-tuned) |
| Context reasoning | Llama 3.2 3B (on-device) / Gemini Flash (demo) |
| Languages | Hindi, English, Telugu, Tamil, Marathi, Bengali, Kannada |

## Project structure

```
kavach/
├── src/kavach/
│   ├── audio/          # capture, VAD, rolling buffer
│   ├── transcription/  # Whisper ASR + speaker tagging
│   ├── detection/      # heuristics, MuRIL classifier, SLM
│   ├── fusion/         # risk score computation
│   └── ui/             # alert system
├── data/               # audio samples for testing
├── models/             # model weights (not tracked in git)
├── tests/              # unit tests per module
├── notebooks/          # EDA and training notebooks
├── scripts/            # data collection, training, evaluation
└── configs/            # all configuration in one place
```

## Setup

```bash
git clone https://github.com/anishaman6206/kavach.git
cd kavach
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp configs/config.yaml.example configs/config.yaml
# Add your API keys to configs/config.yaml
```

## Running the demo

```bash
python scripts/run_pipeline.py --audio data/samples/test_call.wav
```

## License

MIT
