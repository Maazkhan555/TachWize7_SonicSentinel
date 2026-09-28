"""
test_live_pipeline.py — Tests for the open-source Live Detector pipeline.
Run with: python test_live_pipeline.py
"""
import io
import struct
import numpy as np
import sys

PASS = []
FAIL = []

def ok(name):
    PASS.append(name)
    print(f"  PASS  {name}")

def fail(name, reason):
    FAIL.append(name)
    print(f"  FAIL  {name}: {reason}")

def make_wav(audio_f32, sr=16000):
    samples = (np.clip(audio_f32, -1, 1) * 32767).astype('<i2')
    buf = io.BytesIO()
    data_size = len(samples) * 2
    buf.write(b'RIFF'); buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVEfmt '); buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sr, sr*2, 2, 16))
    buf.write(b'data'); buf.write(struct.pack('<I', data_size))
    buf.write(samples.tobytes())
    return buf.getvalue()

print("\n=== Loading models ===")
from live_audio_service import (
    load_all_models, get_model_status,
    compute_signal_features, run_vad,
    run_sound_event_detection, run_transcription,
    analyze_chunk, decode_audio_bytes,
)
load_all_models()
status = get_model_status()
print("Model status:")
for k, v in status.items():
    print(f"  {k}: {v['status']}")

sr = 16000
t = np.linspace(0, 3.0, sr * 3, dtype=np.float32)
tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
silence = np.zeros(sr * 3, dtype=np.float32)
tone_wav = make_wav(tone)
silence_wav = make_wav(silence)

print("\n=== Test A: Signal features ===")
try:
    sig = compute_signal_features(tone)
    assert sig['rms'] > 0.1, f"RMS too low: {sig['rms']}"
    assert abs(sig['dominant_freq'] - 440.0) < 5, f"Dom freq wrong: {sig['dominant_freq']}"
    assert sig['duration_sec'] > 0
    ok("Signal features — tone")
except Exception as e:
    fail("Signal features — tone", e)

try:
    sig = compute_signal_features(silence)
    assert sig['rms'] == 0.0
    assert sig['silence_pct'] == 100.0
    ok("Signal features — silence")
except Exception as e:
    fail("Signal features — silence", e)

print("\n=== Test B: Audio decoding ===")
try:
    decoded = decode_audio_bytes(tone_wav, 'audio/wav')
    assert decoded is not None
    assert len(decoded) > 0
    assert decoded.dtype == np.float32
    ok("decode_audio_bytes — WAV")
except Exception as e:
    fail("decode_audio_bytes — WAV", e)

print("\n=== Test C: VAD ===")
try:
    vad = run_vad(tone)
    assert vad['status'] in ('ready', 'error')
    if vad['status'] == 'ready':
        assert isinstance(vad['speech_detected'], bool)
        assert 0.0 <= vad['speech_ratio'] <= 1.0
        ok("VAD — tone (speech_detected=" + str(vad['speech_detected']) + ")")
    else:
        fail("VAD — tone", vad.get('error'))
except Exception as e:
    fail("VAD — tone", e)

try:
    vad_sil = run_vad(silence)
    if vad_sil['status'] == 'ready':
        assert vad_sil['speech_detected'] == False, "Silence should not be speech"
        ok("VAD — silence correctly classified as non-speech")
    else:
        fail("VAD — silence", vad_sil.get('error'))
except Exception as e:
    fail("VAD — silence", e)

print("\n=== Test D: Sound Event Detection ===")
try:
    sed = run_sound_event_detection(tone)
    assert sed['status'] in ('ready', 'error')
    if sed['status'] == 'ready':
        assert sed['top_label'] is not None
        assert 0.0 <= sed['top_confidence'] <= 1.0
        assert len(sed['predictions']) > 0
        ok(f"SED — tone: '{sed['top_label']}' {round(sed['top_confidence']*100,1)}%")
    else:
        fail("SED — tone", sed.get('error'))
except Exception as e:
    fail("SED — tone", e)

print("\n=== Test E: Transcription ===")
try:
    tr_skip = run_transcription(tone, speech_ratio=0.0)
    assert tr_skip['status'] == 'skipped'
    ok("Transcription — correctly skipped when speech_ratio=0")
except Exception as e:
    fail("Transcription — skip logic", e)

print("\n=== Test F: Full pipeline (analyze_chunk) ===")
try:
    result = analyze_chunk(tone_wav, content_type='audio/wav')
    assert result['success'] == True
    assert 'signal' in result
    assert 'vad' in result
    assert 'sound_event' in result
    assert 'transcription' in result
    assert 'diarization' in result
    assert 'emotion' in result
    assert result['total_latency_ms'] > 0
    ok(f"analyze_chunk — success, latency={result['total_latency_ms']}ms")
    ok(f"analyze_chunk — SED: {result['sound_event']['top_label']}")
    ok(f"analyze_chunk — VAD: speech={result['vad']['speech_detected']}")
    ok(f"analyze_chunk — ASR: {result['transcription']['status']}")
except Exception as e:
    fail("analyze_chunk", e)

try:
    result_sil = analyze_chunk(silence_wav, content_type='audio/wav')
    assert result_sil['success'] == True
    ok("analyze_chunk — silence handled without crash")
except Exception as e:
    fail("analyze_chunk — silence", e)

print("\n=== Test G: Failure handling ===")
try:
    result_bad = analyze_chunk(b'\x00' * 100, content_type='audio/wav')
    assert result_bad['success'] == False
    ok("analyze_chunk — tiny chunk returns success=False")
except Exception as e:
    fail("analyze_chunk — tiny chunk", e)

print("\n=== Test H: Diarization / emotion status ===")
try:
    assert status['diarize']['status'] == 'requires_credentials'
    ok("Diarization — correctly reports requires_credentials")
except Exception as e:
    fail("Diarization status", e)

try:
    assert status['emotion']['status'] == 'unavailable'
    ok("Emotion — correctly reports unavailable")
except Exception as e:
    fail("Emotion status", e)

print(f"\n{'='*50}")
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("Failed tests:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All tests passed.")
