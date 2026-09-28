import numpy as np

from audio_processor import analyze_audio_scene, assess_audio_quality


def test_fan_noise_is_detected_as_background_noise():
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    fan = 0.4 * np.sin(2 * np.pi * 48 * t)
    noise = 0.08 * np.random.default_rng(0).normal(size=t.shape[0])
    audio = fan + noise

    result = analyze_audio_scene(audio, sr)

    assert result["primary_sound"]
    assert "fan" in result["primary_sound"].lower() or "background" in result["primary_sound"].lower()
    assert result["background_noise"] is True
    assert "fan" in result["summary"].lower() or "background" in result["summary"].lower()


def test_alarm_style_signal_is_flagged_as_alert_like():
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    alarm = 0.9 * (np.sin(2 * np.pi * 1200 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)))
    audio = alarm.astype(np.float32)

    result = analyze_audio_scene(audio, sr)

    assert result["alert_like"] is True or result["danger_level"] in {"elevated", "high"}
    assert result["summary"]


def test_loud_speech_prediction_is_not_reported_as_an_alarm():
    sr = 16000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    speech_like = 0.75 * np.sin(2 * np.pi * 220 * t) * (0.55 + 0.45 * np.sin(2 * np.pi * 3 * t))

    result = analyze_audio_scene(speech_like.astype(np.float32), sr, speech_detected=True)

    assert result["primary_sound"] == "Speech / vocal activity"
    assert result["alert_like"] is False
    assert "not an alarm" in result["summary"]


def test_audio_quality_marks_silence_and_clipping():
    silent = assess_audio_quality({"duration_sec": 2.0, "rms": 0.0, "peak": 0.0, "silence_pct": 100.0})
    clipped = assess_audio_quality({"duration_sec": 2.0, "rms": 0.2, "peak": 1.0, "silence_pct": 5.0})

    assert silent["label"] == "Unusable"
    assert "mostly silent" in silent["issues"]
    assert clipped["label"] == "Poor"
    assert "clipping" in clipped["issues"]
