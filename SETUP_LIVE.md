# SETUP_LIVE.md — Live Detector Open-Source Audio Intelligence

## What is installed and working

| Component | Model | Status |
|-----------|-------|--------|
| Voice Activity Detection | Silero VAD (snakers4/silero-vad) | ✅ Auto-downloads via torch.hub |
| Sound Event Detection | AST MIT/ast-finetuned-audioset-10-10-0.4593 | ✅ Cached from HuggingFace |
| Speech Transcription | OpenAI Whisper base (local) | ✅ Downloaded on first run |
| Signal Analysis | librosa + numpy | ✅ Already installed |
| Speaker Diarization | pyannote.audio 3.x | ⚠️ Requires credentials (see below) |
| Vocal Emotion | emotion2vec | ❌ Not available for Python 3.13 |

---

## Running the application

```bash
python app.py
```

The app runs on http://localhost:5000

On first startup, models load in a background thread. The Live Detector
dashboard shows each model's status. Wait for all models to show READY
before starting a session.

---

## Speaker Diarization Setup (Optional)

Speaker diarization uses pyannote.audio which requires:

### Step 1 — Create a Hugging Face account
https://huggingface.co/join

### Step 2 — Accept the model license
Visit and accept the license at:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

### Step 3 — Get your token
https://huggingface.co/settings/tokens

### Step 4 — Install pyannote.audio
```bash
pip install pyannote.audio
```

### Step 5 — Set your token
Set the environment variable before running:
```bash
# Windows
set HF_TOKEN=hf_your_token_here
python app.py

# Linux/macOS
HF_TOKEN=hf_your_token_here python app.py
```

NEVER hardcode your token in source code.

---

## Hardware requirements

- CPU: Any modern CPU. Whisper base runs in ~2-5s per 5s window on CPU.
- RAM: ~2 GB minimum for all models loaded simultaneously.
- GPU: Optional. If CUDA is available, AST and Whisper use it automatically.

## Troubleshooting

**Models show LOADING for a long time**
- First run downloads models. Check your internet connection.
- Silero VAD downloads ~2 MB from GitHub.
- Whisper base downloads ~139 MB from OpenAI CDN.
- AST downloads ~90 MB from HuggingFace.

**"Audio chunk too small" error**
- The browser needs ~5 seconds of audio before sending the first chunk.
- This is normal — wait for the buffer to fill.

**Transcription shows "(No speech detected)"**
- VAD did not detect speech in that window. This is correct behavior.
- Speak louder or closer to the microphone.

**Socket.IO connection fails**
- Make sure you run `python app.py` (not `flask run`).
- The app uses socketio.run() which handles WebSocket upgrades.

---

## Privacy

- All processing is local. No audio is sent to external services.
- Microphone capture stops immediately when you click STOP.
- No audio is saved to disk unless you explicitly add that feature.
- Speaker labels (Speaker 01, Speaker 02) are anonymous session IDs.
  They do not identify real people.
