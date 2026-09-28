import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("=== YAMNET CHECK ===")
try:
    import tensorflow_hub as hub
    print("tensorflow_hub OK:", hub.__version__)
except Exception as e:
    print("tensorflow_hub MISSING:", e)

try:
    import tensorflow_hub as hub
    m = hub.load("https://tfhub.dev/google/yamnet/1")
    print("YAMNet model loaded OK")
    import numpy as np
    dummy = np.zeros(16000, dtype='float32')
    scores, emb, spec = m(dummy)
    print("YAMNet inference OK, scores shape:", scores.shape)
except Exception as e:
    print("YAMNet FAILED:", repr(e))

print("\n=== TM WEIGHT SHAPES ===")
import json, numpy as np
with open("tm-my-audio-model/model.json") as f:
    mj = json.load(f)
manifest = mj["weightsManifest"][0]["weights"]
for w in manifest:
    print("  ", w["name"], w["shape"])

print("\n=== TM ARCHITECTURE FLATTEN SIZE ===")
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
inp = tf.keras.Input(shape=(43, 232, 1))
x = tf.keras.layers.Conv2D(8,  (2,8), activation="relu", padding="valid")(inp)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(32, (2,4), activation="relu", padding="valid")(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(32, (2,4), activation="relu", padding="valid")(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
x = tf.keras.layers.Conv2D(32, (2,4), activation="relu", padding="valid")(x)
x = tf.keras.layers.MaxPooling2D((1,2))(x)
x = tf.keras.layers.Flatten()(x)
print("  Flatten size with input (43,232,1):", x.shape[-1])
print("  dense_1/kernel in weights.bin:", [w["shape"] for w in manifest if w["name"]=="dense_1/kernel"])

print("\n=== SS-CNN CLASS LABELS ===")
import torch
ckpt = torch.load(
    "SonicSentinel_Persistent 0.1/models/sonicsentinel_fsd50k_193class_best.pt",
    map_location="cpu", weights_only=False
)
if isinstance(ckpt, dict):
    print("  checkpoint keys:", list(ckpt.keys()))
    for k in ("classes", "class_names", "labels", "idx_to_class", "class_to_idx"):
        if k in ckpt:
            print(f"  {k}:", ckpt[k][:5] if hasattr(ckpt[k], '__getitem__') else ckpt[k])
else:
    print("  checkpoint is raw state_dict, no class info embedded")
