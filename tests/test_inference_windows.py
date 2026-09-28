import numpy as np

from app import _build_audio_windows


def test_short_audio_stays_a_single_window():
    sample_rate = 16_000
    audio = np.zeros(sample_rate * 2, dtype=np.float32)

    windows = _build_audio_windows(audio, sample_rate)

    assert len(windows) == 1
    assert windows[0][0] == 0
    assert len(windows[0][1]) == sample_rate * 2


def test_long_audio_uses_model_sized_windows_across_clip():
    sample_rate = 16_000
    audio = np.zeros(sample_rate * 30, dtype=np.float32)

    windows = _build_audio_windows(audio, sample_rate)

    assert len(windows) == 6
    assert all(len(window) == sample_rate * 5 for _, window in windows)
    assert windows[0][0] == 0
    assert windows[-1][0] + len(windows[-1][1]) / sample_rate == 30