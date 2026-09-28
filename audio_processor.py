"""
audio_processor.py — Centralized audio preprocessing pipeline for SonicSentinel.

All audio sources (upload, microphone blob, live chunk) go through the same pipeline:
  raw bytes / file path
  → mono
  → 16,000 Hz
  → 5-second crop / pad
  → Log-Mel Spectrogram (64 bands, n_fft=1024, hop=256, f_min=20, f_max=8000)
  → Amplitude → dB
  → per-sample normalization
  → torch.Tensor [1, 1, 64, T]
"""

import io
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Constants matching the trained model ──────────────────────────────────────
SAMPLE_RATE   = 16_000
DURATION_SEC  = 5
N_SAMPLES     = SAMPLE_RATE * DURATION_SEC   # 80 000
N_MELS        = 64
N_FFT         = 1024
HOP_LENGTH    = 256
F_MIN         = 20.0
F_MAX         = 8_000.0


def _load_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Load audio from raw bytes using soundfile, fallback to librosa."""
    import soundfile as sf
    buf = io.BytesIO(data)
    try:
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        buf.seek(0)
        import librosa
        audio, sr = librosa.load(buf, sr=None, mono=False)
        audio = audio.astype(np.float32)
    return audio, sr


def _load_audio_path(path: str) -> tuple[np.ndarray, int]:
    """Load audio from a file path."""
    import soundfile as sf
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa
        audio, sr = librosa.load(path, sr=None, mono=False)
        audio = audio.astype(np.float32)
    return audio, sr


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=1) if audio.shape[1] <= audio.shape[0] else audio.mean(axis=0)
    return audio


def _resample(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    if orig_sr == SAMPLE_RATE:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)


def _crop_or_pad(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= N_SAMPLES:
        return audio[:N_SAMPLES]
    pad = N_SAMPLES - len(audio)
    return np.pad(audio, (0, pad), mode="constant")


def _mel_spectrogram_db(audio: np.ndarray) -> np.ndarray:
    """Compute log-Mel spectrogram matching training pipeline."""
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def _normalize(spec: np.ndarray) -> np.ndarray:
    mean = spec.mean()
    std  = spec.std() + 1e-8
    return (spec - mean) / std


def preprocess_audio(source, is_bytes: bool = False,
                     sample_rate: int | None = None) -> tuple[torch.Tensor, dict]:
    """
    Preprocess audio from a file path or raw bytes.

    Returns:
        tensor: shape [1, 1, 64, T] ready for the model
        info:   dict with duration_sec, sample_rate, rms, peak, zcr, etc.
    """
    try:
        if isinstance(source, np.ndarray):
            audio_raw = np.asarray(source, dtype=np.float32)
            orig_sr = sample_rate or SAMPLE_RATE
        elif is_bytes:
            audio_raw, orig_sr = _load_audio_bytes(source)
        else:
            audio_raw, orig_sr = _load_audio_path(source)
    except Exception as e:
        raise ValueError(f"Cannot load audio: {e}") from e

    audio_mono = _to_mono(audio_raw)
    duration   = len(audio_mono) / orig_sr

    audio_16k  = _resample(audio_mono, orig_sr)
    audio_5s   = _crop_or_pad(audio_16k)

    # ── Signal features (computed on the 5-second window) ─────────────────
    rms          = float(np.sqrt(np.mean(audio_5s ** 2)))
    peak         = float(np.max(np.abs(audio_5s)))
    silence_pct  = float(np.mean(np.abs(audio_5s) < 0.01) * 100)
    zcr          = float(np.mean(np.abs(np.diff(np.sign(audio_5s)))) / 2)

    # Spectral features via librosa
    try:
        import librosa
        sc  = librosa.feature.spectral_centroid(y=audio_5s, sr=SAMPLE_RATE)[0].mean()
        sb  = librosa.feature.spectral_bandwidth(y=audio_5s, sr=SAMPLE_RATE)[0].mean()
        sr_ = librosa.feature.spectral_rolloff(y=audio_5s, sr=SAMPLE_RATE)[0].mean()
        fft = np.abs(np.fft.rfft(audio_5s))
        freqs = np.fft.rfftfreq(len(audio_5s), 1 / SAMPLE_RATE)
        dom_freq = float(freqs[np.argmax(fft)])
    except Exception:
        sc = sb = sr_ = dom_freq = 0.0

    # Energy change detection (split into 10 frames)
    frame_size = N_SAMPLES // 10
    frame_rms  = [float(np.sqrt(np.mean(audio_5s[i*frame_size:(i+1)*frame_size]**2)))
                  for i in range(10)]
    energy_delta = max(frame_rms) - min(frame_rms) if frame_rms else 0.0

    info = {
        "duration_sec":     round(duration, 2),
        "sample_rate":      orig_sr,
        "rms":              round(rms, 4),
        "peak":             round(peak, 4),
        "silence_pct":      round(silence_pct, 1),
        "zcr":              round(zcr, 4),
        "spectral_centroid": round(float(sc), 1),
        "spectral_bandwidth": round(float(sb), 1),
        "spectral_rolloff":  round(float(sr_), 1),
        "dominant_freq":    round(dom_freq, 1),
        "energy_delta":     round(energy_delta, 4),
    }

    spec   = _mel_spectrogram_db(audio_5s)
    spec_n = _normalize(spec)
    tensor = torch.from_numpy(spec_n).unsqueeze(0).unsqueeze(0)  # [1,1,64,T]
    return tensor, info


def analyze_audio_scene(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                        speech_detected: bool = False) -> dict:
    """Estimate dominant acoustic scene and whether a steady background hum is present."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1) if arr.shape[1] > arr.shape[0] else arr.mean(axis=0)
    if arr.size == 0:
        return {
            "primary_sound": "Background silence",
            "background_noise": False,
            "alert_like": False,
            "speech_activity": False,
            "danger_level": "low",
            "summary": "No usable signal detected.",
        }

    arr = np.clip(arr, -1.0, 1.0)
    rms = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.max(np.abs(arr)))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(arr))))) if arr.size > 1 else 0.0

    if arr.size > 1:
        spectrum = np.abs(np.fft.rfft(arr))
        freqs = np.fft.rfftfreq(arr.size, 1.0 / sample_rate)
        if spectrum.size > 1:
            dominant_idx = int(np.argmax(spectrum[1:]) + 1)
            dominant_freq = float(freqs[dominant_idx])
            low_band = spectrum[(freqs >= 20) & (freqs <= 200)]
            mid_band = spectrum[(freqs >= 200) & (freqs <= 2000)]
            low_share = float(np.sum(low_band) / max(np.sum(spectrum), 1e-8))
            mid_share = float(np.sum(mid_band) / max(np.sum(spectrum), 1e-8))
        else:
            dominant_freq = 0.0
            low_share = 0.0
            mid_share = 0.0
    else:
        dominant_freq = 0.0
        low_share = 0.0
        mid_share = 0.0

    fan_like = (
        dominant_freq <= 120
        and rms > 0.01
        and rms < 0.45
        and peak < 0.75
        and zcr < 0.18
        and (dominant_freq < 80 or low_share > 0.03)
    )

    alert_like = (
        not speech_detected
        and not fan_like
        and (
            peak > 0.55
            or rms > 0.30
            or (dominant_freq > 400 and zcr > 0.12 and peak > 0.15)
        )
    )

    if speech_detected:
        primary_sound = "Speech / vocal activity"
        background_noise = False
        danger_level = "elevated" if rms > 0.12 else "low"
        summary = "Speech or vocal activity identified by the sound classifier; high volume alone is not an alarm."
    elif fan_like:
        primary_sound = "Fan / background hum"
        background_noise = True
        danger_level = "low"
        summary = "Background fan or ventilation hum detected with steady low-frequency acoustic energy."
    elif alert_like:
        primary_sound = "Urgent alarm / impact burst"
        background_noise = False
        danger_level = "high" if rms > 0.30 or peak > 0.70 else "elevated"
        summary = "High-energy transient signal detected with burst-like characteristics suggestive of alarms, impacts, or sudden acoustic events."
    elif dominant_freq > 300:
        primary_sound = "Speech / vocal activity"
        background_noise = False
        danger_level = "elevated" if rms > 0.12 else "low"
        summary = "Human voice or tonal vocal sound detected above the environmental baseline."
    elif dominant_freq > 60:
        primary_sound = "Mechanical / engine / workspace rumble"
        background_noise = True
        danger_level = "elevated" if rms > 0.15 else "low"
        summary = "Mechanical or engine-like acoustic energy detected with moderate low-frequency rumble."
    else:
        primary_sound = "Ambient environmental sound"
        background_noise = True if rms < 0.12 else False
        danger_level = "low"
        summary = "Ambient environmental sound detected with low threat energy and no strong transient pattern."

    return {
        "primary_sound": primary_sound,
        "background_noise": background_noise,
        "alert_like": alert_like,
        "speech_activity": speech_detected,
        "danger_level": danger_level,
        "summary": summary,
        "dominant_frequency_hz": round(dominant_freq, 1),
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "zcr": round(zcr, 4),
        "low_band_share": round(low_share, 4),
        "mid_band_share": round(mid_share, 4),
    }


def classify_energy(rms: float, energy_delta: float) -> str:
    """Return a human-readable energy label from signal features."""
    if rms < 0.005:
        return "SILENCE"
    if rms < 0.02:
        return "LOW ENERGY"
    if energy_delta > 0.15 or rms > 0.25:
        return "SUDDEN IMPACT"
    if rms > 0.12:
        return "HIGH ENERGY"
    if rms > 0.05:
        return "ELEVATED"
    return "NORMAL"


def assess_audio_quality(audio_info: dict) -> dict:
    """Estimate recording usability from measured signal features."""
    duration = float(audio_info.get("duration_sec", 0) or 0)
    rms = float(audio_info.get("rms", 0) or 0)
    peak = float(audio_info.get("peak", 0) or 0)
    silence_pct = float(audio_info.get("silence_pct", 100) or 0)
    issues = []

    if peak >= 0.995:
        issues.append("clipping")
    if silence_pct >= 90:
        issues.append("mostly silent")
    if rms < 0.005:
        issues.append("low signal strength")
    if duration < 0.25:
        issues.append("too short")

    if duration < 0.25 or silence_pct >= 99.5 or rms < 0.001:
        quality = "Unusable"
    elif peak >= 0.995 or silence_pct >= 90 or rms < 0.005:
        quality = "Poor"
    elif silence_pct >= 70 or rms < 0.015:
        quality = "Acceptable"
    else:
        quality = "Good"
    return {"label": quality, "issues": issues}


def compute_dashboard_state(label: str, confidence: float, rms: float,
                             high_risk_classes: set) -> str:
    """
    Map model output + signal features to a dashboard state.
    States: IDLE | LISTENING | NORMAL | ELEVATED | HIGH_ACTIVITY | ALERT
    """
    if label in high_risk_classes and confidence >= 0.60:
        return "ALERT"
    if rms > 0.12 or (label in high_risk_classes and confidence >= 0.40):
        return "HIGH_ACTIVITY"
    if rms > 0.05 or confidence >= 0.70:
        return "ELEVATED"
    if rms > 0.01:
        return "NORMAL"
    return "LISTENING"
