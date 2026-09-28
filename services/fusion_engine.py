"""
services/fusion_engine.py

Runs TWO independent real AI models on the same audio and fuses their predictions.

MODEL 1: SonicSentinel CNN (FSD50K 193-class, local PyTorch checkpoint)
MODEL 2: AST — MIT/ast-finetuned-audioset-10-10-0.4593 (HuggingFace Transformers)

Agreement is computed via semantic normalization — raw label strings are NEVER
compared directly because the two models use different label systems.

Architecture:
    SAME AUDIO
         |
    +----+----+
    |         |
    v         v
 SS-CNN      AST
    |         |
    v         v
  Top-5     Top-5
    |         |
    +----+----+
         |
    FUSION ENGINE
    (semantic agreement + weighted confidence)
         |
    FINAL EVENT
"""

import logging
import os
from typing import Optional
import numpy as np

from .models.sonicsentinel_model import SonicSentinelModel
from .models.ast_model           import ASTAudioAdapter
from .semantic_normalizer        import normalize_label, normalize_predictions, labels_agree

logger = logging.getLogger(__name__)

# Confidence weights — only applied when both models are ready
_WEIGHTS = {
    "sonicsentinel": 0.50,
    "ast":           0.50,
}

TOTAL_MODELS = 2
MIN_AUTO_CONFIDENCE = float(os.environ.get("SONICSENTINEL_MIN_CONFIDENCE", "0.40"))
MIN_TOP_TWO_MARGIN = float(os.environ.get("SONICSENTINEL_MIN_MARGIN", "0.08"))


def _decision_metadata(label: str, normalized: str, confidence: float,
                       predictions: list[dict], disagreement: bool = False) -> dict:
    scores = sorted(
        (float(item.get("confidence", 0.0)) for item in predictions),
        reverse=True,
    )
    margin = scores[0] - scores[1] if len(scores) > 1 else (scores[0] if scores else 0.0)
    unknown = normalized == "UNKNOWN" or label.strip().lower() in {"other", "unknown", "unknown sound", "—"}
    requires_review = (
        unknown
        or disagreement
        or confidence < MIN_AUTO_CONFIDENCE
        or margin < MIN_TOP_TWO_MARGIN
    )
    return {
        "label": "Unknown sound" if unknown else label,
        "requires_review": requires_review,
        "decision_status": "MANUAL_REVIEW" if requires_review else "CLASSIFIED",
        "top_two_margin": round(margin, 4),
    }


class FusionEngine:
    _instance: Optional["FusionEngine"] = None

    def __init__(self):
        self.models = {
            "sonicsentinel": SonicSentinelModel(),
            "ast":           ASTAudioAdapter(),
        }

    @classmethod
    def get(cls) -> "FusionEngine":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_all()
        return cls._instance

    def _load_all(self):
        for name, model in self.models.items():
            try:
                model.load()
            except Exception as e:
                logger.error("[FUSION] Unexpected error loading %s: %s", name, e)
        online = sum(1 for m in self.models.values() if m.loaded)
        logger.info("[FUSION] %d/%d models online", online, TOTAL_MODELS)

    def predict(self, audio: np.ndarray, tmp_path: str = None) -> dict:
        """
        Run both models on the same audio. Returns per-model results + fusion.

        audio    : float32 ndarray (N,) at 16 kHz
        tmp_path : temp file path for SonicSentinel (uses librosa file loading)
        """
        logger.info("[FUSION] Running inference on %d samples", len(audio))
        results = {}

        for name, model in self.models.items():
            try:
                if name == "sonicsentinel" and tmp_path:
                    results[name] = model.predict_from_path(tmp_path)
                else:
                    results[name] = model.predict(audio)
            except Exception as e:
                logger.exception("[FUSION] Unhandled error from %s: %s", name, e)
                results[name] = model._empty_result(
                    model.MODEL_ID, model.MODEL_NAME, str(e)
                )

        # Annotate each result with normalized labels
        for name, res in results.items():
            if res.get("status") == "ready":
                res["normalized_prediction"] = normalize_label(res.get("prediction", ""))
                res["top_predictions"] = normalize_predictions(
                    res.get("top_predictions", [])
                )

        fusion = self._fuse(results)
        logger.info(
            "[FUSION] Result: %s (%.1f%%) agreement=%s online=%d/%d",
            fusion["label"], fusion["confidence"] * 100,
            fusion["agreement"], fusion["online_count"], TOTAL_MODELS,
        )
        return {"models": results, "fusion": fusion}

    def status(self) -> dict:
        return {
            name: {"loaded": m.loaded, "error": m.error, **m.get_info()}
            for name, m in self.models.items()
        }

    # ── Fusion logic ───────────────────────────────────────────────────────

    def _fuse(self, results: dict) -> dict:
        ready   = {k: v for k, v in results.items() if v.get("status") == "ready"}
        n_ready = len(ready)
        n_total = TOTAL_MODELS

        if n_ready == 0:
            return {
                "label":         "Unknown sound",
                "normalized":    "UNKNOWN",
                "confidence":    0.0,
                "requires_review": True,
                "decision_status": "MANUAL_REVIEW",
                "top_two_margin": 0.0,
                "agreement":     "none",
                "agreement_str": f"0/{n_total} ONLINE",
                "state_hint":    "IDLE",
                "top_labels":    [],
                "online_count":  0,
                "total_count":   n_total,
                "evidence":      [],
            }

        # Single model online — use it directly, no fusion math
        if n_ready == 1:
            only     = next(iter(ready.values()))
            norm     = only.get("normalized_prediction") or normalize_label(only["prediction"])
            predictions = only.get("top_predictions", [])
            decision = _decision_metadata(
                only["prediction"], norm, float(only.get("confidence", 0.0)), predictions
            )
            evidence = [{
                "model":      only["model_name"],
                "raw":        only["prediction"],
                "normalized": norm,
                "confidence": only["confidence"],
            }]
            return {
                **decision,
                "normalized":    norm,
                "confidence":    only["confidence"],
                "agreement":     "single",
                "agreement_str": f"1/{n_total} ONLINE",
                "state_hint":    _conf_to_state(only["confidence"]),
                "top_labels":    [
                    {**prediction, "label": "Unknown sound"}
                    if normalize_label(prediction.get("label", "")) == "UNKNOWN"
                    else prediction
                    for prediction in predictions
                ],
                "online_count":  1,
                "total_count":   n_total,
                "evidence":      evidence,
            }

        # Both models ready — semantic agreement check
        ss_res  = ready.get("sonicsentinel", {})
        ast_res = ready.get("ast", {})

        ss_raw  = ss_res.get("prediction", "—")
        ast_raw = ast_res.get("prediction", "—")
        ss_norm = ss_res.get("normalized_prediction") or normalize_label(ss_raw)
        ast_norm = ast_res.get("normalized_prediction") or normalize_label(ast_raw)

        evidence = [
            {
                "model":      ss_res.get("model_name", "SonicSentinel CNN"),
                "raw":        ss_raw,
                "normalized": ss_norm,
                "confidence": ss_res.get("confidence", 0.0),
            },
            {
                "model":      ast_res.get("model_name", "AST"),
                "raw":        ast_raw,
                "normalized": ast_norm,
                "confidence": ast_res.get("confidence", 0.0),
            },
        ]

        semantic_agree = (ss_norm == ast_norm) and ss_norm != "UNKNOWN"

        # Weighted confidence fusion across top predictions
        label_scores: dict[str, float] = {}
        display_labels: dict[str, str] = {}
        display_label_confidence: dict[str, float] = {}
        total_weight = sum(_WEIGHTS.get(k, 0.5) for k in ready)

        for model_id, res in ready.items():
            w = _WEIGHTS.get(model_id, 0.5) / total_weight
            for pred in res.get("top_predictions", []):
                raw_label = pred.get("label", "—")
                confidence = float(pred.get("confidence", 0.0))
                normalized = pred.get("normalized") or normalize_label(raw_label)
                category_key = normalized if normalized != "UNKNOWN" else f"RAW:{raw_label.casefold()}"
                label_scores[category_key] = label_scores.get(category_key, 0.0) + confidence * w
                if confidence > display_label_confidence.get(category_key, -1.0):
                    display_labels[category_key] = raw_label
                    display_label_confidence[category_key] = confidence

        sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        top_category, top_conf = sorted_labels[0]
        top_label = display_labels[top_category]

        # Agreement string — based on semantic normalization, not raw string equality
        if semantic_agree:
            agreement     = "full"
            agreement_str = f"2/{n_total} AGREE — {ss_norm}"
        else:
            agreement     = "split"
            agreement_str = f"MODEL DISAGREEMENT — SS:{ss_norm} / AST:{ast_norm}"

        # State hint
        if agreement == "split":
            state_hint = "MODEL_DISAGREEMENT"
        else:
            ss_state = ss_res.get("dashboard_state")
            state_hint = ss_state or _conf_to_state(top_conf)

        # Normalized label for the fused result
        fused_norm = ss_norm if semantic_agree else (
            top_category if not top_category.startswith("RAW:") else "UNKNOWN"
        )
        decision = _decision_metadata(
            top_label,
            fused_norm,
            top_conf,
            [{"confidence": score} for _, score in sorted_labels[:2]],
            disagreement=agreement == "split",
        )

        return {
            **decision,
            "normalized":    fused_norm,
            "confidence":    round(top_conf, 4),
            "agreement":     agreement,
            "agreement_str": agreement_str,
            "state_hint":    state_hint,
            "top_labels":    [
                {
                    "label": "Unknown sound" if category.startswith("RAW:") else display_labels[category],
                    "confidence": round(score, 4),
                }
                for category, score in sorted_labels[:5]
            ],
            "online_count":  n_ready,
            "total_count":   n_total,
            "evidence":      evidence,
        }


def _conf_to_state(conf: float) -> str:
    if conf >= 0.80:
        return "HIGH_ACTIVITY"
    if conf >= 0.55:
        return "ELEVATED"
    return "NORMAL"
