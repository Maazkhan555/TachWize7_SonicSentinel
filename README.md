# TachWize7_SonicSentinel

**SonicSentinel AI** — Real-Time Acoustic Intelligence Platform

A full-stack audio intelligence system with live microphone monitoring, sound event detection, speech transcription, anomaly detection, and forensic file analysis.

---

## Features

### Live Detector (Open-Source Pipeline)
- **Sound Event Detection** — AST AudioSet-527 (527 sound classes)
- **Voice Activity Detection** — Silero VAD
- **Speech Transcription** — OpenAI Whisper base (local, no API)
- **Acoustic Anomaly Detection** — Spectral z-score vs rolling baseline
- **Signal Analysis** — RMS, peak, spectral centroid, dominant frequency, ZCR
- **Real-time timeline** — every chunk logged with timestamp and event type

### Upload & Analyze (Custom Model)
- **SonicSentinel CNN** — trained on ~50,000 FSD50K recordings, 193 classes
- **AST fusion** — dual-model agreement scoring
- **Forensic analysis** — waveform, spectrogram, signal features, confidence breakdown

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.1 + Flask-SocketIO 5.5 |
| Live Audio | Silero VAD + AST + Whisper (all local) |
| Upload Model | Custom PyTorch CNN (FSD50K 193-class) |
| Audio Processing | librosa, soundfile, torchaudio |
| Frontend | Vanilla JS + Socket.IO client |
| Deployment | Railway |

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/Maazkhan555/TachWize7_SonicSentinel.git
cd TachWize7_SonicSentinel

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download open-source models (first run only)
python download_models.py

# 5. Add SonicSentinel checkpoint
# Place sonicsentinel_fsd50k_193class_best.pt in:
# SonicSentinel_Persistent 0.1/models/

# 6. Run
python app.py
```

Open http://localhost:5000

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server port (set automatically by Railway) |
| `FLASK_ENV` | `production` | Set to `development` for debug mode |
| `LIVE_WINDOW_SEC` | `5.0` | Live analysis window size in seconds |
| `LIVE_STRIDE_SEC` | `2.0` | Stride between analysis windows |
| `LIVE_ANOMALY_BASELINE` | `30` | Chunks to build anomaly baseline |
| `LIVE_ANOMALY_Z` | `2.5` | Anomaly z-score threshold |

---

## Model Notes

### Open-source models (auto-downloaded)
- **Silero VAD** — loaded via `torch.hub`, cached locally
- **AST** — `MIT/ast-finetuned-audioset-10-10-0.4593` from HuggingFace
- **Whisper base** — ~139 MB, downloaded on first run

### Custom model (not in repo — too large for GitHub)
- `sonicsentinel_fsd50k_193class_best.pt` — trained on FSD50K, 193 classes
- Must be uploaded to Railway as a volume or provided via `MODEL_PATH`

### Not available
- **Pyannote diarization** — requires HuggingFace token + gated model license
- **Emotion2vec** — not available for Python 3.13 via pip

---

## Deployment on Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select this repository
4. Railway auto-detects `nixpacks.toml` and builds
5. Add environment variables in Railway dashboard if needed
6. For the SonicSentinel model: add a Railway Volume and upload the `.pt` file

---

## Architecture

```
LIVE DETECTOR                    UPLOAD & ANALYZE
─────────────                    ────────────────
Browser Mic                      File Upload
    │                                │
WebSocket (Socket.IO)            REST POST /api/analyze
    │                                │
Flask Backend                    Flask Backend
    │                                │
┌───┴────────────┐            ┌──────┴──────────┐
│ Silero VAD     │            │ SonicSentinel   │
│ AST AudioSet   │            │ CNN (193-class) │
│ Whisper base   │            │ AST (527-class) │
│ Anomaly Detect │            │ Fusion Engine   │
└───────────────┘            └────────────────┘
    │                                │
Event Timeline               Analysis Result
```

---

## License

MIT — see LICENSE file.
