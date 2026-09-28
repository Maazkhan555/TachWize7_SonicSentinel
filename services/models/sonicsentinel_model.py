"""
services/models/sonicsentinel_model.py

Adapter wrapping the existing SonicSentinel PyTorch CNN (193-class FSD50K).
Delegates to the already-working model_service.ModelService singleton so we
don't duplicate model-loading code.
"""

import time
import logging
import numpy as np

from .base_model import BaseAudioModel

logger = logging.getLogger(__name__)


class SonicSentinelModel(BaseAudioModel):

    MODEL_ID   = "sonicsentinel"
    MODEL_NAME = "SonicSentinel CNN"

    def __init__(self):
        self._svc    = None
        self.loaded  = False
        self.error   = None

    def load(self) -> None:
        try:
            from model_service import ModelService
            self._svc   = ModelService.get()
            self.loaded = self._svc.loaded
            self.error  = self._svc.error
            if self.loaded:
                logger.info("SonicSentinel adapter ready.")
            else:
                logger.warning("SonicSentinel model not loaded: %s", self.error)
        except Exception as e:
            self.error = str(e)
            logger.exception("SonicSentinel adapter load error")

    def predict(self, audio: np.ndarray) -> dict:
        """Run inference from a raw float32 waveform array."""
        if not self.loaded or self._svc is None:
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME,
                                      self.error or "Model not loaded")
        try:
            import torch
            from audio_processor import _crop_or_pad, _mel_spectrogram_db, _normalize
            t0     = time.perf_counter()
            a5s    = _crop_or_pad(audio)
            spec   = _mel_spectrogram_db(a5s)
            spec_n = _normalize(spec)
            tensor = torch.from_numpy(spec_n).unsqueeze(0).unsqueeze(0)
            result = self._svc.predict(tensor)
            ms     = (time.perf_counter() - t0) * 1000
            return self._build_result(result, ms)
        except Exception as e:
            logger.exception("SonicSentinel inference error")
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME, str(e))

    def predict_from_path(self, path: str) -> dict:
        """Run inference from a file path (uses full preprocess_audio pipeline)."""
        if not self.loaded or self._svc is None:
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME,
                                      self.error or "Model not loaded")
        try:
            import time as _time
            from audio_processor import preprocess_audio
            t0 = _time.perf_counter()
            tensor, audio_info = preprocess_audio(path, is_bytes=False)
            result = self._svc.predict(tensor)
            ms = (_time.perf_counter() - t0) * 1000
            r = self._build_result(result, ms)
            r["audio_info"] = audio_info
            return r
        except Exception as e:
            logger.exception("SonicSentinel predict_from_path error")
            return self._empty_result(self.MODEL_ID, self.MODEL_NAME, str(e))

    def _build_result(self, result: dict, ms: float) -> dict:
        return {
            "model_id":        self.MODEL_ID,
            "model_name":      self.MODEL_NAME,
            "prediction":      result["label"],
            "confidence":      result["confidence"],
            "top_predictions": result["top_predictions"],
            "latency_ms":      round(ms, 1),
            "status":          "ready",
            "error":           None,
        }

    def get_info(self) -> dict:
        info = self._svc.status() if self._svc else {}
        return {
            "model_id":    self.MODEL_ID,
            "model_name":  self.MODEL_NAME,
            "framework":   "PyTorch",
            "num_classes": 193,
            "loaded":      self.loaded,
            "device":      info.get("device", "—"),
            "error":       self.error,
        }
