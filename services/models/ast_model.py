"""
services/models/ast_model.py

Audio Spectrogram Transformer (AST) adapter.
Model: MIT/ast-finetuned-audioset-10-10-0.4593
Loaded ONCE at startup via HuggingFace Transformers.

Uses the model's own feature extractor, sample-rate, normalization,
label mapping, and input shape — nothing is manually invented.
"""

import time
import logging
import numpy as np

from .base_model import BaseAudioModel

logger = logging.getLogger(__name__)

MODEL_ID   = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SR     = 16_000   # the feature extractor will resample if needed


class ASTAudioAdapter(BaseAudioModel):

    MODEL_ID   = "ast"
    MODEL_NAME = "AST (AudioSet-527)"

    def __init__(self):
        self.model            = None
        self.feature_extractor = None
        self.id2label         = {}
        self.loaded           = False
        self.error            = None
        self._target_sr       = AST_SR

    # ── load ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        logger.info("[AST] Loading MIT/ast-finetuned-audioset-10-10-0.4593 ...")
        try:
            from transformers import ASTForAudioClassification, AutoFeatureExtractor
            import torch
        except ImportError as e:
            self.error = f"transformers or torch not installed: {e}"
            logger.error("[AST] %s", self.error)
            return

        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            # Read the sample rate the feature extractor expects
            if hasattr(self.feature_extractor, "sampling_rate"):
                self._target_sr = self.feature_extractor.sampling_rate
            logger.info("[AST] Feature extractor loaded. Target SR: %d Hz", self._target_sr)

            self.model = ASTForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            self.model.eval()

            # Move to GPU if available
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self._device)

            # Read the real label mapping from the model config
            self.id2label = self.model.config.id2label  # {0: "Speech", 1: "Music", ...}
            logger.info("[AST] id2label loaded: %d classes", len(self.id2label))

            # Warm-up inference
            dummy = np.zeros(self._target_sr, dtype=np.float32)
            self._run_ast(dummy)
            logger.info("[AST] Warm-up OK. Device: %s", self._device)

            self.loaded = True
            logger.info("[AST] Loaded successfully. %d classes.", len(self.id2label))

        except Exception as e:
            self.error = str(e)
            logger.exception("[AST] Load failed: %s", e)

    # ── health_check ──────────────────────────────────────────────────────

    def health_check(self) -> dict:
        if not self.loaded:
            return {"model": self.MODEL_NAME, "status": "offline", "error": self.error}
        try:
            dummy = np.zeros(self._target_sr, dtype=np.float32)
            result = self._run_ast(dummy)
            return {
                "model":  self.MODEL_NAME,
                "status": "online",
                "top1":   result["top_predictions"][0]["label"] if result["top_predictions"] else "—",
            }
        except Exception as e:
            return {"model": self.MODEL_NAME, "status": "offline", "error": str(e)}

    # ── preprocess ────────────────────────────────────────────────────────

    def preprocess(self, audio: np.ndarray, orig_sr: int = None) -> np.ndarray:
        """
        Resample to the feature extractor's required sample rate.
        Returns float32 mono waveform at self._target_sr.
        """
        sr = orig_sr or self._target_sr
        if sr != self._target_sr:
            import librosa
            audio = librosa.resample(audio.astype(np.float32),
                                     orig_sr=sr, target_sr=self._target_sr)
        return audio.astype(np.float32)

    # ── predict ───────────────────────────────────────────────────────────

    def predict(self, audio: np.ndarray) -> dict:
        """
        Run inference on a float32 mono waveform (any SR — will be resampled).
        audio is assumed to be at 16 kHz (same as SonicSentinel pipeline).
        """
        if not self.loaded:
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME,
                                      self.error or "Model not loaded")
        try:
            t0 = time.perf_counter()
            # Resample if the feature extractor wants a different SR
            wave = self.preprocess(audio, orig_sr=16_000)
            result = self._run_ast(wave)
            ms = (time.perf_counter() - t0) * 1000
            result["latency_ms"] = round(ms, 1)
            return result
        except Exception as e:
            logger.exception("[AST] Inference error: %s", e)
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME, str(e))

    # ── predict_window ────────────────────────────────────────────────────

    def predict_window(self, audio: np.ndarray, orig_sr: int = 16_000) -> dict:
        """Run inference on a single window (resamples if needed)."""
        if not self.loaded:
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME,
                                      self.error or "Model not loaded")
        try:
            t0   = time.perf_counter()
            wave = self.preprocess(audio, orig_sr=orig_sr)
            result = self._run_ast(wave)
            ms = (time.perf_counter() - t0) * 1000
            result["latency_ms"] = round(ms, 1)
            return result
        except Exception as e:
            logger.exception("[AST] predict_window error: %s", e)
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME, str(e))

    # ── top_predictions ───────────────────────────────────────────────────

    def top_predictions(self, audio: np.ndarray, k: int = 5) -> list:
        """Return top-k predictions as a list of {label, confidence} dicts."""
        result = self.predict(audio)
        return result.get("top_predictions", [])[:k]

    # ── internal ──────────────────────────────────────────────────────────

    def _run_ast(self, wave: np.ndarray) -> dict:
        """
        Core inference: feature extraction → model forward → softmax → top-5.
        Uses the model's own feature extractor — no manual preprocessing.
        """
        import torch

        inputs = self.feature_extractor(
            wave,
            sampling_rate=self._target_sr,
            return_tensors="pt",
        )
        # Move inputs to same device as model
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, num_classes)

        probs   = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        top_k   = min(5, len(probs))
        top_idx = probs.argsort()[::-1][:top_k]

        top_preds = [
            {
                "label":      self.id2label[int(i)],
                "confidence": round(float(probs[i]), 4),
            }
            for i in top_idx
        ]

        return {
            "model_id":        self.MODEL_ID,
            "model_name":      self.MODEL_NAME,
            "prediction":      top_preds[0]["label"],
            "confidence":      top_preds[0]["confidence"],
            "top_predictions": top_preds,
            "latency_ms":      0.0,   # filled by caller
            "status":          "ready",
            "error":           None,
        }

    # ── get_info ──────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        return {
            "model_id":    self.MODEL_ID,
            "model_name":  self.MODEL_NAME,
            "framework":   "HuggingFace Transformers (PyTorch)",
            "hf_model":    "MIT/ast-finetuned-audioset-10-10-0.4593",
            "num_classes": len(self.id2label),
            "target_sr":   self._target_sr,
            "loaded":      self.loaded,
            "error":       self.error,
        }
