"""
services/models/teacher_model.py

Teachable Machine audio model — TF.js format loaded into Keras.

Real architecture (verified from weights.bin):
  Input: (batch, N_MELS, 232, 1)  where N_MELS must be in 31-38 (gives flatten=704)
  Conv2D(8,[2,8],relu,valid) → MaxPool(2,2)
  Conv2D(32,[2,4],relu,valid) → MaxPool(2,2)
  Conv2D(32,[2,4],relu,valid) → MaxPool(2,2)
  Conv2D(32,[2,4],relu,valid) → MaxPool(1,2)
  Flatten(704) → Dropout(0.25) → Dense(2000,relu) → Dense(2,softmax)

The model.json header claims 43 mel bands but the dense_1/kernel shape [704,2000]
proves the model was trained with 31-38 mel bands. We use 40 mel bands in librosa
and slice to TM_N_MELS_ACTUAL=40 — but since 40 gives 1056 we must use 38.
Verified: mels=38, T=232 → flatten=704. ✓
"""

import os
import time
import logging
import json
import numpy as np

from .base_model import BaseAudioModel

logger = logging.getLogger(__name__)

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "tm-my-audio-model")
MODEL_JSON  = os.path.join(MODEL_DIR, "model.json")
WEIGHTS_BIN = os.path.join(MODEL_DIR, "weights.bin")

# Preprocessing constants — verified against actual weight shapes
TM_SR          = 16_000
TM_N_FFT       = 1024
TM_HOP         = 512
TM_N_MELS      = 38    # actual trained mel bands (model.json header is wrong)
TM_FMIN        = 0.0
TM_FMAX        = 8_000.0
TM_TIME_FRAMES = 232


class TeacherMachineModel(BaseAudioModel):

    MODEL_ID   = "teacher_machine"
    MODEL_NAME = "Teachable Machine"

    def __init__(self):
        self.model  = None
        self.labels = ["Background Noise", "Class 2"]
        self.loaded = False
        self.error  = None

    def load(self) -> None:
        logger.info("[TM] Loading...")

        if not os.path.isfile(MODEL_JSON) or not os.path.isfile(WEIGHTS_BIN):
            self.error = "TM model files not found in tm-my-audio-model/"
            logger.error("[TM] %s", self.error)
            return

        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
        except ImportError as e:
            self.error = f"TensorFlow not installed: {e}"
            logger.error("[TM] %s", self.error)
            return

        try:
            # Load labels from metadata
            meta_path = os.path.join(MODEL_DIR, "metadata.json")
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                self.labels = meta.get("wordLabels", self.labels)
            logger.info("[TM] Labels: %s", self.labels)

            # Parse weight manifest
            with open(MODEL_JSON) as f:
                model_json = json.load(f)
            manifest = model_json["weightsManifest"][0]["weights"]

            # Load binary weights
            with open(WEIGHTS_BIN, "rb") as f:
                raw = f.read()

            weights_dict = {}
            offset = 0
            for entry in manifest:
                name  = entry["name"]
                shape = entry["shape"]
                n     = 1
                for d in shape:
                    n *= d
                nbytes = n * 4
                arr = np.frombuffer(raw[offset:offset + nbytes], dtype=np.float32).reshape(shape)
                weights_dict[name] = arr
                offset += nbytes
            logger.info("[TM] Parsed %d weight tensors from weights.bin", len(weights_dict))

            # Build Keras model with CORRECT input shape (TM_N_MELS=38, not 43)
            inp = tf.keras.Input(shape=(TM_N_MELS, TM_TIME_FRAMES, 1), name="input")
            x = tf.keras.layers.Conv2D(8,  (2, 8), activation="relu", padding="valid", name="conv2d_1")(inp)
            x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_1")(x)
            x = tf.keras.layers.Conv2D(32, (2, 4), activation="relu", padding="valid", name="conv2d_2")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_2")(x)
            x = tf.keras.layers.Conv2D(32, (2, 4), activation="relu", padding="valid", name="conv2d_3")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name="max_pooling2d_3")(x)
            x = tf.keras.layers.Conv2D(32, (2, 4), activation="relu", padding="valid", name="conv2d_4")(x)
            x = tf.keras.layers.MaxPooling2D((1, 2), name="max_pooling2d_4")(x)
            x = tf.keras.layers.Flatten(name="flatten_1")(x)
            x = tf.keras.layers.Dropout(0.25, name="dropout_1")(x)
            x = tf.keras.layers.Dense(2000, activation="relu", name="dense_1")(x)
            out = tf.keras.layers.Dense(len(self.labels), activation="softmax", name="NewHeadDense")(x)
            model = tf.keras.Model(inputs=inp, outputs=out)

            # Assign weights
            assigned = 0
            for layer in model.layers:
                k_key = f"{layer.name}/kernel"
                b_key = f"{layer.name}/bias"
                if k_key in weights_dict and b_key in weights_dict:
                    layer.set_weights([weights_dict[k_key], weights_dict[b_key]])
                    assigned += 1
            logger.info("[TM] Assigned weights to %d layers", assigned)

            self.model  = model
            self.loaded = True
            logger.info("[TM] Loaded successfully. Input: (%d, %d, 1), Classes: %s",
                        TM_N_MELS, TM_TIME_FRAMES, self.labels)

        except Exception as e:
            self.error = str(e)
            logger.exception("[TM] Load failed: %s", e)

    def predict(self, audio: np.ndarray) -> dict:
        if not self.loaded:
            logger.warning("[TM] Predict called but model not loaded: %s", self.error)
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME,
                                      self.error or "Model not loaded")
        try:
            logger.info("[TM] Inference started, audio shape: %s", audio.shape)
            t0   = time.perf_counter()
            spec = self._preprocess(audio)
            probs = self.model(spec, training=False).numpy()[0]
            ms   = (time.perf_counter() - t0) * 1000

            top_idx   = probs.argsort()[::-1]
            top_preds = [
                {"label": self.labels[i], "confidence": round(float(probs[i]), 4)}
                for i in top_idx
            ]
            logger.info("[TM] Prediction: %s (%.1f%%) in %.1fms",
                        top_preds[0]["label"], top_preds[0]["confidence"] * 100, ms)
            return {
                "model_id":        self.MODEL_ID,
                "model_name":      self.MODEL_NAME,
                "prediction":      top_preds[0]["label"],
                "confidence":      top_preds[0]["confidence"],
                "top_predictions": top_preds,
                "latency_ms":      round(ms, 1),
                "status":          "ready",
                "error":           None,
            }
        except Exception as e:
            logger.exception("[TM] Inference error: %s", e)
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME, str(e))

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        import librosa
        # Compute mel spectrogram with TM_N_MELS bands
        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=TM_SR,
            n_fft=TM_N_FFT,
            hop_length=TM_HOP,
            n_mels=TM_N_MELS,
            fmin=TM_FMIN,
            fmax=TM_FMAX,
        )
        log_mel = np.log(mel + 1e-6).astype(np.float32)  # (TM_N_MELS, T)

        # Crop or pad time axis to TM_TIME_FRAMES
        T = log_mel.shape[1]
        if T >= TM_TIME_FRAMES:
            log_mel = log_mel[:, :TM_TIME_FRAMES]
        else:
            log_mel = np.pad(log_mel, ((0, 0), (0, TM_TIME_FRAMES - T)))

        return log_mel[np.newaxis, :, :, np.newaxis]  # (1, TM_N_MELS, TM_TIME_FRAMES, 1)

    def get_info(self) -> dict:
        return {
            "model_id":    self.MODEL_ID,
            "model_name":  self.MODEL_NAME,
            "framework":   "TensorFlow/Keras (TF.js weights)",
            "num_classes": len(self.labels),
            "classes":     self.labels,
            "input_shape": f"(1, {TM_N_MELS}, {TM_TIME_FRAMES}, 1)",
            "loaded":      self.loaded,
            "error":       self.error,
        }
