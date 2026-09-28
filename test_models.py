import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

import numpy as np

# Synthetic 5s audio at 16kHz (440 Hz sine wave)
sr = 16000
audio = np.sin(2 * np.pi * 440 * np.arange(sr * 5) / sr).astype(np.float32)

print("=" * 60)
print("TEST 1: SonicSentinel CNN")
print("=" * 60)
from services.models.sonicsentinel_model import SonicSentinelModel
ss = SonicSentinelModel()
ss.load()
print(f"Loaded: {ss.loaded}, Error: {ss.error}")
if ss.loaded:
    r = ss.predict(audio)
    print(f"Prediction: {r['prediction']} ({r['confidence']*100:.1f}%) in {r['latency_ms']}ms")
    print(f"Top 3: {[(p['label'], f\"{p['confidence']*100:.1f}%\") for p in r['top_predictions'][:3]]}")

print("\n" + "=" * 60)
print("TEST 2: Teacher Machine")
print("=" * 60)
from services.models.teacher_model import TeacherMachineModel
tm = TeacherMachineModel()
tm.load()
print(f"Loaded: {tm.loaded}, Error: {tm.error}")
if tm.loaded:
    r = tm.predict(audio)
    print(f"Prediction: {r['prediction']} ({r['confidence']*100:.1f}%) in {r['latency_ms']}ms")
    print(f"Top preds: {[(p['label'], f\"{p['confidence']*100:.1f}%\") for p in r['top_predictions']]}")

print("\n" + "=" * 60)
print("TEST 3: YAMNet")
print("=" * 60)
from services.models.third_party_model import YAMNetModel
yn = YAMNetModel()
yn.load()
print(f"Loaded: {yn.loaded}, Error: {yn.error}")
if yn.loaded:
    r = yn.predict(audio)
    print(f"Prediction: {r['prediction']} ({r['confidence']*100:.1f}%) in {r['latency_ms']}ms")
    print(f"Top 3: {[(p['label'], f\"{p['confidence']*100:.1f}%\") for p in r['top_predictions'][:3]]}")

print("\n" + "=" * 60)
print("TEST 4: Fusion Engine")
print("=" * 60)
from services.fusion_engine import FusionEngine
fe = FusionEngine.get()
result = fe.predict(audio)
f = result['fusion']
print(f"Fusion label:    {f['label']}")
print(f"Fusion conf:     {f['confidence']*100:.1f}%")
print(f"Agreement:       {f['agreement_str']}")
print(f"Online:          {f['online_count']}/{f['total_count']}")
print(f"State hint:      {f['state_hint']}")
