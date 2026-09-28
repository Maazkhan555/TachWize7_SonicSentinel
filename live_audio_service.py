"""
live_audio_service.py — Open-Source Live Audio Intelligence Pipeline
=====================================================================
Hardware: CPU-only, 8 GB RAM, 4 cores, Python 3.13
Models loaded:
  A. Silero VAD          — torch.hub, no pip install needed
  B. AST AudioSet-527    — transformers AutoFeatureExtractor + ASTForAudioClassification
                           (uses direct model API — NOT pipeline, which was removed in transformers 5.x)
  C. Whisper base        — openai-whisper, local CPU inference
  D. Anomaly detection   — librosa spectral features + rolling z-score baseline (no extra model)

NOT loaded (documented):
  - Pyannote diarization: requires HF token + gated model license
  - Emotion2vec: not available for Python 3.13 via pip

This module is ONLY used by the Live Detector routes.
It does NOT import model_service, audio_processor, or FusionEngine.
"""

from __future__ import annotations

import io
import os
import time
import uuid
import logging
import threading
import tempfile
import numpy as np
from datetime import datetime, timezone
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
LIVE_SR  = 16_000
VAD_SR   = 16_000
AST_SR   = 16_000

ANALYSIS_WINDOW_SEC    = float(os.environ.get("LIVE_WINDOW_SEC",  "5.0"))
ANALYSIS_STRIDE_SEC    = float(os.environ.get("LIVE_STRIDE_SEC",  "2.0"))
WHISPER_MIN_SPEECH_SEC = float(os.environ.get("LIVE_WHISPER_MIN", "1.5"))

# Anomaly detection: rolling baseline window (number of chunks to keep)
ANOMALY_BASELINE_SIZE = int(os.environ.get("LIVE_ANOMALY_BASELINE", "30"))
ANOMALY_Z_THRESHOLD   = float(os.environ.get("LIVE_ANOMALY_Z", "2.5"))

# ── Model registry ─────────────────────────────────────────────────────────────
_registry: dict[str, dict] = {
    "vad":        {"name": "Silero VAD",                   "status": "loading",               "device": "cpu",  "error": None, "last_ms": None},
    "sound_event":{"name": "AST (AudioSet-527)",           "status": "loading",               "device": "cpu",  "error": None, "last_ms": None},
    "transcribe": {"name": "Whisper base",                 "status": "loading",               "device": "cpu",  "error": None, "last_ms": None},
    "anomaly":    {"name": "Spectral Anomaly Detector",    "status": "ready",                 "device": "cpu",  "error": None, "last_ms": None},
    "diarize":    {"name": "Pyannote Speaker Diarization", "status": "requires_credentials",  "device": None,   "error": "Requires HF token + gated model. See SETUP_LIVE.md.", "last_ms": None},
    "emotion":    {"name": "Emotion2Vec",                  "status": "unavailable",           "device": None,   "error": "Not available for Python 3.13 via pip.", "last_ms": None},
}

# ── Model handles ──────────────────────────────────────────────────────────────
_vad_model       = None
_vad_utils       = None
_ast_extractor   = None   # AutoFeatureExtractor
_ast_model       = None   # ASTForAudioClassification
_ast_id2label    = None   # dict[int, str]
_whisper_model   = None
_load_lock       = threading.Lock()
_models_loaded   = False


def _update_registry(key: str, **kwargs):
    if key in _registry:
        _registry[key].update(kwargs)


def load_all_models() -> None:
    """Load all open-source models once at startup. Thread-safe."""
    global _models_loaded
    with _load_lock:
        if _models_loaded:
            return
        _load_vad()
        _load_ast()
        _load_whisper()
        _models_loaded = True
        logger.info("[LIVE] All loadable models initialized.")


def _load_vad():
    global _vad_model, _vad_utils
    try:
        import torch
        logger.info("[LIVE-VAD] Loading Silero VAD from torch hub …")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _vad_model.eval()
        _update_registry("vad", status="ready", device="cpu")
        logger.info("[LIVE-VAD] Silero VAD ready.")
    except Exception as e:
        _update_registry("vad", status="offline", error=str(e))
        logger.exception("[LIVE-VAD] Failed to load: %s", e)


def _load_ast():
    """
    Load AST using direct model API.
    Uses ASTFeatureExtractor (the concrete class) as primary import,
    with AutoFeatureExtractor as fallback — avoids import race with
    the FusionEngine thread that loads transformers simultaneously.
    """
    global _ast_extractor, _ast_model, _ast_id2label
    MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
    try:
        import torch
        # Use concrete class names — more reliable than Auto* in threaded startup
        try:
            from transformers import ASTFeatureExtractor, ASTForAudioClassification
            _FeatureExtractorCls = ASTFeatureExtractor
        except ImportError:
            from transformers import AutoFeatureExtractor, ASTForAudioClassification
            _FeatureExtractorCls = AutoFeatureExtractor

        logger.info("[LIVE-AST] Loading feature extractor …")
        _ast_extractor = _FeatureExtractorCls.from_pretrained(MODEL_ID)

        logger.info("[LIVE-AST] Loading ASTForAudioClassification …")
        _ast_model = ASTForAudioClassification.from_pretrained(MODEL_ID)
        _ast_model.eval()

        _ast_id2label = _ast_model.config.id2label  # {0: 'Speech', 1: 'Music', …}

        # Warm-up with 1s silence
        dummy = np.zeros(LIVE_SR, dtype=np.float32)
        inputs = _ast_extractor(dummy, sampling_rate=AST_SR, return_tensors="pt")
        with torch.no_grad():
            _ = _ast_model(**inputs).logits

        _update_registry("sound_event", status="ready", device="cpu")
        logger.info("[LIVE-AST] AST ready. %d classes.", len(_ast_id2label))
    except Exception as e:
        _update_registry("sound_event", status="offline", error=str(e))
        logger.exception("[LIVE-AST] Failed to load: %s", e)


def _load_whisper():
    global _whisper_model
    try:
        import torch
        import whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[LIVE-WHISPER] Loading Whisper base on %s …", device)
        _whisper_model = whisper.load_model("base", device=device)
        _update_registry("transcribe", status="ready", device=device)
        logger.info("[LIVE-WHISPER] Whisper base ready on %s.", device)
    except Exception as e:
        _update_registry("transcribe", status="offline", error=str(e))
        logger.exception("[LIVE-WHISPER] Failed to load: %s", e)


def get_model_status() -> dict:
    return {k: dict(v) for k, v in _registry.items()}


# ── Audio decoding ─────────────────────────────────────────────────────────────

def decode_audio_bytes(raw: bytes, content_type: str = "") -> Optional[np.ndarray]:
    """
    Decode raw audio bytes (WAV from browser) to float32 mono 16 kHz numpy array.
    Returns None on failure.
    """
    suffix = ".wav"
    ct = (content_type or "").lower()
    if "ogg" in ct:
        suffix = ".ogg"
    elif "mp4" in ct or "aac" in ct:
        suffix = ".mp4"
    elif "webm" in ct:
        suffix = ".webm"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(tmp_fd)
        with open(tmp_path, "wb") as fh:
            fh.write(raw)

        import soundfile as sf
        try:
            audio, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        except Exception:
            import librosa
            audio, sr = librosa.load(tmp_path, sr=None, mono=False)
            audio = audio.astype(np.float32)

        # to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1) if audio.shape[1] <= audio.shape[0] else audio.mean(axis=0)

        # resample to 16 kHz
        if sr != LIVE_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=LIVE_SR)

        return audio.astype(np.float32)
    except Exception as e:
        logger.warning("[LIVE] decode_audio_bytes failed: %s", e)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Signal analysis ────────────────────────────────────────────────────────────

def compute_signal_features(audio: np.ndarray, sr: int = LIVE_SR) -> dict:
    """Compute signal features using librosa/numpy. No ML model involved."""
    if audio is None or len(audio) == 0:
        return {}
    try:
        import librosa
        arr = np.clip(audio, -1.0, 1.0)
        rms         = float(np.sqrt(np.mean(arr ** 2)))
        peak        = float(np.max(np.abs(arr)))
        silence_pct = float(np.mean(np.abs(arr) < 0.01) * 100)
        zcr         = float(np.mean(np.abs(np.diff(np.sign(arr)))) / 2) if len(arr) > 1 else 0.0

        sc       = float(librosa.feature.spectral_centroid(y=arr, sr=sr)[0].mean())
        sb       = float(librosa.feature.spectral_bandwidth(y=arr, sr=sr)[0].mean())
        rolloff  = float(librosa.feature.spectral_rolloff(y=arr, sr=sr)[0].mean())
        fft      = np.abs(np.fft.rfft(arr))
        freqs    = np.fft.rfftfreq(len(arr), 1.0 / sr)
        dom_freq = float(freqs[np.argmax(fft[1:]) + 1]) if len(fft) > 1 else 0.0
        duration = round(len(arr) / sr, 3)

        frame_size = max(1, len(arr) // 10)
        frame_rms  = [
            float(np.sqrt(np.mean(arr[i * frame_size:(i + 1) * frame_size] ** 2)))
            for i in range(10)
        ]
        energy_delta = round(max(frame_rms) - min(frame_rms), 4) if frame_rms else 0.0
        transient    = energy_delta > 0.15 or (rms > 0.25 and peak > 0.55)

        return {
            "rms":                round(rms, 4),
            "peak":               round(peak, 4),
            "spectral_centroid":  round(sc, 1),
            "spectral_bandwidth": round(sb, 1),
            "spectral_rolloff":   round(rolloff, 1),
            "dominant_freq":      round(dom_freq, 1),
            "zcr":                round(zcr, 4),
            "silence_pct":        round(silence_pct, 1),
            "duration_sec":       duration,
            "energy_delta":       energy_delta,
            "transient":          transient,
        }
    except Exception as e:
        logger.warning("[LIVE-SIGNAL] Feature extraction failed: %s", e)
        return {}


# ── VAD ────────────────────────────────────────────────────────────────────────

def run_vad(audio: np.ndarray, sr: int = LIVE_SR) -> dict:
    if _vad_model is None:
        return {"status": "offline", "speech_detected": False, "segments": [], "speech_ratio": 0.0}
    try:
        import torch
        t0 = time.perf_counter()

        if sr != VAD_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=VAD_SR)

        tensor = torch.from_numpy(audio).float()
        get_ts = _vad_utils[0]
        timestamps = get_ts(
            tensor, _vad_model,
            sampling_rate=VAD_SR,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

        ms = round((time.perf_counter() - t0) * 1000, 1)
        _update_registry("vad", last_ms=ms)

        total_speech = sum(t["end"] - t["start"] for t in timestamps)
        speech_ratio = total_speech / max(len(audio), 1)

        segments = [
            {
                "start_sec": round(t["start"] / VAD_SR, 3),
                "end_sec":   round(t["end"]   / VAD_SR, 3),
                "duration":  round((t["end"] - t["start"]) / VAD_SR, 3),
            }
            for t in timestamps
        ]

        return {
            "status":          "ready",
            "speech_detected": len(timestamps) > 0,
            "speech_ratio":    round(speech_ratio, 3),
            "segments":        segments,
            "latency_ms":      ms,
        }
    except Exception as e:
        logger.exception("[LIVE-VAD] Inference error: %s", e)
        _update_registry("vad", status="offline", error=str(e))
        return {"status": "error", "speech_detected": False, "segments": [], "error": str(e)}


# ── Sound Event Detection ──────────────────────────────────────────────────────

def run_sound_event_detection(audio: np.ndarray, sr: int = LIVE_SR) -> dict:
    """
    Run AST directly via AutoFeatureExtractor + ASTForAudioClassification.
    Always runs — no silence gate.
    """
    if _ast_model is None or _ast_extractor is None:
        return {"status": "offline", "predictions": [], "top_label": None, "top_confidence": 0.0}
    try:
        import torch
        t0 = time.perf_counter()

        if sr != AST_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=AST_SR)

        inputs = _ast_extractor(
            audio.astype(np.float32),
            sampling_rate=AST_SR,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = _ast_model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        top_k = min(10, len(probs))
        top   = probs.topk(top_k)

        ms = round((time.perf_counter() - t0) * 1000, 1)
        _update_registry("sound_event", last_ms=ms)

        predictions = [
            {
                "label":      _ast_id2label[idx.item()],
                "confidence": round(float(score.item()), 4),
            }
            for idx, score in zip(top.indices, top.values)
        ]

        top_pred = predictions[0] if predictions else {"label": "Unknown", "confidence": 0.0}

        return {
            "status":         "ready",
            "predictions":    predictions,
            "top_label":      top_pred["label"],
            "top_confidence": top_pred["confidence"],
            "latency_ms":     ms,
        }
    except Exception as e:
        logger.exception("[LIVE-AST] Inference error: %s", e)
        _update_registry("sound_event", status="offline", error=str(e))
        return {"status": "error", "predictions": [], "top_label": None, "top_confidence": 0.0, "error": str(e)}


# ── Transcription ──────────────────────────────────────────────────────────────

def run_transcription(audio: np.ndarray, sr: int = LIVE_SR,
                      speech_ratio: float = 0.0) -> dict:
    if _whisper_model is None:
        return {"status": "offline", "text": "", "segments": []}

    if speech_ratio < 0.10:
        return {
            "status":   "skipped",
            "reason":   "Insufficient speech detected by VAD",
            "text":     "",
            "segments": [],
        }

    try:
        t0 = time.perf_counter()

        if sr != LIVE_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=LIVE_SR)

        result = _whisper_model.transcribe(
            audio.astype(np.float32),
            language=None,
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,
        )

        ms   = round((time.perf_counter() - t0) * 1000, 1)
        _update_registry("transcribe", last_ms=ms)

        text = (result.get("text") or "").strip()
        raw_segs = result.get("segments") or []

        segments = [
            {
                "start":       round(float(s.get("start", 0)), 2),
                "end":         round(float(s.get("end", 0)), 2),
                "text":        (s.get("text") or "").strip(),
                "avg_logprob": round(float(s.get("avg_logprob", -1.0)), 3),
            }
            for s in raw_segs
            if (s.get("text") or "").strip()
        ]

        uncertain = any(s["avg_logprob"] < -1.0 for s in segments)

        return {
            "status":     "ready",
            "text":       text,
            "segments":   segments,
            "uncertain":  uncertain,
            "language":   result.get("language", "unknown"),
            "latency_ms": ms,
        }
    except Exception as e:
        logger.exception("[LIVE-WHISPER] Transcription error: %s", e)
        _update_registry("transcribe", status="offline", error=str(e))
        return {"status": "error", "text": "", "segments": [], "error": str(e)}


# ── Anomaly Detection (spectral z-score, no extra model needed) ────────────────

class AnomalyDetector:
    """
    Rolling spectral baseline anomaly detector.
    Computes a z-score of the current chunk's spectral features
    against a rolling window of recent chunks.
    No external model required — uses librosa spectral features.
    """

    def __init__(self, baseline_size: int = ANOMALY_BASELINE_SIZE,
                 z_threshold: float = ANOMALY_Z_THRESHOLD):
        self.baseline_size = baseline_size
        self.z_threshold   = z_threshold
        self._baseline: deque = deque(maxlen=baseline_size)
        self._lock = threading.Lock()

    def score(self, signal: dict) -> dict:
        """
        Compute anomaly score for the current chunk.
        Returns anomaly_score (0-1), is_anomaly (bool), z_score (float),
        baseline_size (int), threshold (float).
        """
        # Feature vector: rms, spectral_centroid, zcr, energy_delta
        features = np.array([
            signal.get("rms", 0.0),
            signal.get("spectral_centroid", 0.0) / 8000.0,  # normalize to 0-1
            signal.get("zcr", 0.0),
            signal.get("energy_delta", 0.0),
        ], dtype=np.float32)

        with self._lock:
            baseline_list = list(self._baseline)

        if len(baseline_list) < 5:
            # Not enough baseline — add to baseline, no anomaly
            with self._lock:
                self._baseline.append(features)
            return {
                "status":         "baseline_building",
                "anomaly_score":  0.0,
                "is_anomaly":     False,
                "z_score":        0.0,
                "baseline_chunks": len(baseline_list) + 1,
                "threshold":      self.z_threshold,
                "message":        f"Building baseline ({len(baseline_list)+1}/{self.baseline_size} chunks)",
            }

        baseline_arr = np.stack(baseline_list)
        mean = baseline_arr.mean(axis=0)
        std  = baseline_arr.std(axis=0) + 1e-8

        z_scores  = np.abs((features - mean) / std)
        max_z     = float(z_scores.max())
        mean_z    = float(z_scores.mean())

        # Anomaly score: sigmoid-like mapping of max z-score to 0-1
        anomaly_score = float(1.0 / (1.0 + np.exp(-(max_z - self.z_threshold))))
        is_anomaly    = max_z > self.z_threshold

        # Update baseline with current chunk
        with self._lock:
            self._baseline.append(features)

        return {
            "status":          "ready",
            "anomaly_score":   round(anomaly_score, 4),
            "is_anomaly":      is_anomaly,
            "z_score":         round(max_z, 3),
            "mean_z_score":    round(mean_z, 3),
            "baseline_chunks": len(baseline_list),
            "threshold":       self.z_threshold,
            "message":         "ANOMALY DETECTED" if is_anomaly else "Normal",
        }

    def reset(self):
        with self._lock:
            self._baseline.clear()


# ── Session management ─────────────────────────────────────────────────────────

class LiveSession:
    """
    Holds state for one live monitoring session.
    Thread-safe event history, rolling audio buffer, per-session anomaly detector.
    """

    def __init__(self, session_id: str):
        self.session_id   = session_id
        self.started_at   = datetime.now(timezone.utc).isoformat()
        self.ended_at     = None
        self.active       = True
        self._events: deque = deque(maxlen=500)
        self._lock        = threading.Lock()
        self._chunk_count = 0
        self.anomaly_detector = AnomalyDetector()

    def add_event(self, event: dict) -> None:
        with self._lock:
            self._events.appendleft(event)

    def get_events(self, limit: int = 100) -> list:
        with self._lock:
            return list(self._events)[:limit]

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
        self.anomaly_detector.reset()

    def end(self) -> None:
        self.active   = False
        self.ended_at = datetime.now(timezone.utc).isoformat()

    def increment_chunk(self) -> int:
        self._chunk_count += 1
        return self._chunk_count


# Active sessions
_sessions: dict[str, LiveSession] = {}
_sessions_lock = threading.Lock()


def create_session() -> LiveSession:
    sid = str(uuid.uuid4())
    session = LiveSession(sid)
    with _sessions_lock:
        _sessions[sid] = session
    logger.info("[LIVE] Session created: %s", sid)
    return session


def get_session(session_id: str) -> Optional[LiveSession]:
    with _sessions_lock:
        return _sessions.get(session_id)


def end_session(session_id: str) -> None:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session:
            session.end()
            logger.info("[LIVE] Session ended: %s  events=%d", session_id, len(session._events))


# ── Main pipeline ──────────────────────────────────────────────────────────────

def analyze_chunk(raw_bytes: bytes, content_type: str = "",
                  session: Optional[LiveSession] = None) -> dict:
    """
    Full open-source pipeline for one audio chunk from the browser.

    1. Decode audio bytes → float32 16 kHz mono
    2. Signal analysis (librosa)
    3. VAD (Silero)
    4. Sound event detection (AST direct API) — always runs
    5. Transcription (Whisper) — only if VAD detects speech
    6. Anomaly detection (spectral z-score)
    7. Build timestamped event dict
    8. Store in session history
    """
    t_start  = time.perf_counter()
    ts       = datetime.now(timezone.utc).isoformat()
    event_id = str(uuid.uuid4())[:8].upper()

    # ── 1. Decode ──────────────────────────────────────────────────────────
    audio = decode_audio_bytes(raw_bytes, content_type)
    if audio is None or len(audio) < LIVE_SR * 0.25:
        return {
            "success":   False,
            "error":     "Audio too short or could not be decoded",
            "event_id":  event_id,
            "timestamp": ts,
        }

    # ── 2. Signal analysis ─────────────────────────────────────────────────
    signal = compute_signal_features(audio, sr=LIVE_SR)

    # ── 3. VAD ────────────────────────────────────────────────────────────
    vad             = run_vad(audio, sr=LIVE_SR)
    speech_ratio    = vad.get("speech_ratio", 0.0)
    speech_detected = vad.get("speech_detected", False)

    # ── 4. Sound event detection (always runs) ─────────────────────────────
    sed = run_sound_event_detection(audio, sr=LIVE_SR)

    # ── 5. Transcription ──────────────────────────────────────────────────
    transcription = run_transcription(audio, sr=LIVE_SR, speech_ratio=speech_ratio)

    # ── 6. Anomaly detection ───────────────────────────────────────────────
    anomaly = {}
    if session and signal:
        anomaly = session.anomaly_detector.score(signal)
    elif signal:
        # No session — use a module-level detector
        anomaly = _module_anomaly_detector.score(signal)

    # ── 7. Build event ────────────────────────────────────────────────────
    total_ms = round((time.perf_counter() - t_start) * 1000, 1)

    event = {
        "success":    True,
        "event_id":   event_id,
        "timestamp":  ts,
        "session_id": session.session_id if session else None,

        "signal": signal,

        "vad": {
            "speech_detected": speech_detected,
            "speech_ratio":    speech_ratio,
            "segments":        vad.get("segments", []),
            "status":          vad.get("status", "offline"),
            "latency_ms":      vad.get("latency_ms"),
        },

        "sound_event": {
            "top_label":      sed.get("top_label"),
            "top_confidence": sed.get("top_confidence", 0.0),
            "predictions":    sed.get("predictions", []),
            "status":         sed.get("status", "offline"),
            "latency_ms":     sed.get("latency_ms"),
        },

        "transcription": {
            "text":       transcription.get("text", ""),
            "segments":   transcription.get("segments", []),
            "uncertain":  transcription.get("uncertain", False),
            "language":   transcription.get("language", ""),
            "status":     transcription.get("status", "offline"),
            "latency_ms": transcription.get("latency_ms"),
        },

        "anomaly": anomaly,

        "diarization": {
            "status":  _registry["diarize"]["status"],
            "message": _registry["diarize"]["error"],
        },
        "emotion": {
            "status":  _registry["emotion"]["status"],
            "message": _registry["emotion"]["error"],
        },

        "total_latency_ms": total_ms,
        "model_status":     get_model_status(),
    }

    # ── 8. Store in session ───────────────────────────────────────────────
    if session:
        session.add_event(event)
        session.increment_chunk()

    logger.info(
        "[LIVE] chunk | speech=%s sed=%s conf=%.0f%% anomaly=%s total_ms=%.0f",
        speech_detected,
        sed.get("top_label", "—"),
        (sed.get("top_confidence") or 0) * 100,
        anomaly.get("is_anomaly", False),
        total_ms,
    )

    return event


# Module-level anomaly detector for sessionless calls
_module_anomaly_detector = AnomalyDetector()
