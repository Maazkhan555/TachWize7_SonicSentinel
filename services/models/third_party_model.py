"""
services/models/third_party_model.py

YAMNet — Google's pretrained audio event classifier (521 AudioSet classes).
Loaded locally via tensorflow_hub. No API, no internet required after first download.
Input: float32 waveform at 16 kHz mono.
"""

import time
import logging
import numpy as np

from .base_model import BaseAudioModel

logger = logging.getLogger(__name__)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_SR  = 16_000


class YAMNetModel(BaseAudioModel):

    MODEL_ID   = "yamnet"
    MODEL_NAME = "YAMNet (AudioSet-521)"

    def __init__(self):
        self.model   = None
        self.classes = []
        self.loaded  = False
        self.error   = None

    def load(self) -> None:
        logger.info("[YAMNET] Loading...")

        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
        except ImportError as e:
            self.error = f"TensorFlow not installed: {e}"
            logger.error("[YAMNET] %s", self.error)
            return

        try:
            import tensorflow_hub as hub
        except ImportError as e:
            self.error = f"tensorflow_hub not installed: {e}"
            logger.error("[YAMNET] %s", self.error)
            return

        try:
            logger.info("[YAMNET] Loading from TF Hub (cached after first download)...")
            self.model = hub.load(YAMNET_URL)
            logger.info("[YAMNET] Model graph loaded")

            # Load class names from the model's bundled asset file
            import csv
            class_map_path = self.model.class_map_path().numpy().decode("utf-8")
            with tf.io.gfile.GFile(class_map_path) as f:
                reader = csv.DictReader(f)
                self.classes = [row["display_name"] for row in reader]

            # Warm-up inference to catch any runtime issues now
            dummy = np.zeros(YAMNET_SR, dtype=np.float32)
            scores, _, _ = self.model(dummy)
            logger.info("[YAMNET] Warm-up OK, scores shape: %s", scores.shape)

            self.loaded = True
            logger.info("[YAMNET] Loaded successfully. %d classes.", len(self.classes))

        except Exception as e:
            self.error = str(e)
            logger.exception("[YAMNET] Load failed: %s", e)

    def predict(self, audio: np.ndarray) -> dict:
        if not self.loaded:
            logger.warning("[YAMNET] Predict called but model not loaded: %s", self.error)
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME,
                                      self.error or "Model not loaded")
        try:
            import tensorflow as tf
            logger.info("[YAMNET] Inference started, audio length: %d samples", len(audio))
            t0 = time.perf_counter()

            waveform    = audio.astype(np.float32)
            scores, _, _ = self.model(waveform)
            # scores: (num_frames, 521) — mean across frames for clip-level prediction
            mean_scores = tf.reduce_mean(scores, axis=0).numpy()
            ms = (time.perf_counter() - t0) * 1000

            top_k   = min(5, len(mean_scores))
            top_idx = mean_scores.argsort()[::-1][:top_k]
            top_preds = [
                {"label": self.classes[i], "confidence": round(float(mean_scores[i]), 4)}
                for i in top_idx
            ]

            logger.info("[YAMNET] Prediction: %s (%.1f%%) in %.1fms",
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
            logger.exception("[YAMNET] Inference error: %s", e)
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME, str(e))

    def get_info(self) -> dict:
        return {
            "model_id":    self.MODEL_ID,
            "model_name":  self.MODEL_NAME,
            "framework":   "TensorFlow Hub",
            "num_classes": len(self.classes),
            "loaded":      self.loaded,
            "error":       self.error,
        }
