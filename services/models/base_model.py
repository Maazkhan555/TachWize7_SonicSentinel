"""
services/models/base_model.py

Abstract base class that every model adapter must implement.
All adapters return a normalized prediction dict so the fusion engine
and dashboard can work regardless of the underlying ML framework.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAudioModel(ABC):

    @abstractmethod
    def load(self) -> None:
        """Load model weights / graph into memory. Called once at startup."""

    @abstractmethod
    def predict(self, audio: Any) -> dict:
        """
        Run inference on preprocessed audio.

        Parameters
        ----------
        audio : numpy ndarray, shape (N,) at 16 kHz, float32, mono

        Returns
        -------
        dict with keys:
            model_id        str   short identifier
            model_name      str   human-readable name
            prediction      str   top predicted class label
            confidence      float 0.0–1.0
            top_predictions list  [{"label": str, "confidence": float}, ...]
            latency_ms      float inference time in milliseconds
            status          str   "ready" | "error" | "unavailable"
            error           str | None
        """

    @abstractmethod
    def get_info(self) -> dict:
        """Return static metadata about the model."""

    # ── Shared helper ──────────────────────────────────────────────────────
    @staticmethod
    def _empty_result(model_id: str, model_name: str, error: str) -> dict:
        return {
            "model_id":        model_id,
            "model_name":      model_name,
            "prediction":      "—",
            "confidence":      0.0,
            "top_predictions": [],
            "latency_ms":      0.0,
            "status":          "error",
            "error":           error,
        }
