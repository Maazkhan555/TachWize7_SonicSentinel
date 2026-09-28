from services.fusion_engine import FusionEngine


def _fuse_single(label, confidence, top_predictions):
    engine = object.__new__(FusionEngine)
    return engine._fuse({
        "sonicsentinel": {
            "status": "ready",
            "prediction": label,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "model_name": "SonicSentinel CNN",
        }
    })


def test_other_prediction_becomes_unknown_and_requires_review():
    result = _fuse_single("Other", 0.22, [
        {"label": "Other", "confidence": 0.22},
        {"label": "Rain", "confidence": 0.18},
    ])

    assert result["label"] == "Unknown sound"
    assert result["normalized"] == "UNKNOWN"
    assert result["requires_review"] is True
    assert result["decision_status"] == "MANUAL_REVIEW"
    assert result["evidence"][0]["raw"] == "Other"
    assert result["top_labels"][0]["label"] == "Unknown sound"


def test_clear_high_confidence_prediction_is_classified():
    result = _fuse_single("Siren", 0.91, [
        {"label": "Siren", "confidence": 0.91},
        {"label": "Alarm", "confidence": 0.03},
    ])

    assert result["label"] == "Siren"
    assert result["requires_review"] is False
    assert result["decision_status"] == "CLASSIFIED"


def test_close_top_predictions_require_review():
    result = _fuse_single("Siren", 0.62, [
        {"label": "Siren", "confidence": 0.62},
        {"label": "Alarm", "confidence": 0.58},
    ])

    assert result["requires_review"] is True
    assert result["top_two_margin"] == 0.04


def test_semantically_equivalent_labels_share_fusion_confidence():
    engine = object.__new__(FusionEngine)
    result = engine._fuse({
        "sonicsentinel": {
            "status": "ready",
            "prediction": "Speech",
            "normalized_prediction": "SPEECH",
            "confidence": 0.55,
            "top_predictions": [
                {"label": "Speech", "confidence": 0.55, "normalized": "SPEECH"},
                {"label": "Music", "confidence": 0.10, "normalized": "MUSIC"},
            ],
            "model_name": "SonicSentinel CNN",
        },
        "ast": {
            "status": "ready",
            "prediction": "Conversation",
            "normalized_prediction": "SPEECH",
            "confidence": 0.52,
            "top_predictions": [
                {"label": "Conversation", "confidence": 0.52, "normalized": "SPEECH"},
                {"label": "Music", "confidence": 0.12, "normalized": "MUSIC"},
            ],
            "model_name": "AST",
        },
    })

    assert result["agreement"] == "full"
    assert result["confidence"] == 0.535
    assert result["requires_review"] is False