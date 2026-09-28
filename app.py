"""
SonicSentinel AI — Flask Backend
Real PyTorch inference, real audio preprocessing, real signal analysis.
"""

import os
import uuid
import time
import json
import csv
import io
import logging
import tempfile
import string
import random
from collections import deque
from datetime import datetime, timezone

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "sonicsentinel-secret-2090"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", logger=False, engineio_logger=False)

# ── Upload config ─────────────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "aac", "webm", "opus"}
MAX_UPLOAD_MB      = 50
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

EVENT_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "event_history.json")


def _ensure_store_dir():
    os.makedirs(os.path.dirname(EVENT_STORE_PATH), exist_ok=True)


def _load_persisted_events() -> list[dict]:
    _ensure_store_dir()
    try:
        if not os.path.exists(EVENT_STORE_PATH):
            return []
        with open(EVENT_STORE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
            if isinstance(payload, list):
                return payload
    except Exception:
        logger.warning("Unable to read event store; starting empty.")
    return []


def _persist_events(events: deque | list) -> None:
    _ensure_store_dir()
    payload = list(events)
    with open(EVENT_STORE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


# ── In-memory event history (capped at 200) ───────────────────────────────────
event_history: deque = deque(_load_persisted_events(), maxlen=200)

# ── Load live audio service (open-source pipeline, separate from upload) ────
try:
    from live_audio_service import (
        load_all_models as _live_load_models,
        analyze_chunk   as _live_analyze_chunk,
        create_session  as _live_create_session,
        end_session     as _live_end_session,
        get_session     as _live_get_session,
        get_model_status as _live_model_status,
    )
    import threading as _threading
    _threading.Thread(target=_live_load_models, daemon=True).start()
    _live_available = True
    logger.info("Live audio service initializing in background thread.")
except ImportError as e:
    _live_available = False
    logger.warning("Live audio service unavailable (%s).", e)
except Exception as e:
    _live_available = False
    logger.exception("Live audio service startup error: %s", e)

# ── Load models at startup ────────────────────────────────────────────────────
try:
    from model_service import HIGH_RISK_CLASSES
    from audio_processor import (
        preprocess_audio,
        classify_energy,
        assess_audio_quality,
        compute_dashboard_state,
        analyze_audio_scene,
    )
    from services.fusion_engine import FusionEngine
    import librosa, numpy as _np
    _ml_available = True
    fusion = FusionEngine.get()
    logger.info("Fusion engine ready. Models: %s",
                {k: v["loaded"] for k, v in fusion.status().items()})
except ImportError as e:
    _ml_available = False
    fusion = None
    logger.warning("ML stack unavailable (%s). Inference disabled.", e)
except Exception as e:
    _ml_available = False
    fusion = None
    logger.exception("ML stack startup error: %s", e)


# ── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def rand_id(n=6):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def _load_audio_waveform(tmp_path: str):
    """Load 16 kHz mono float32 waveform for TM + YAMNet."""
    audio, _ = librosa.load(tmp_path, sr=16000, mono=True)
    return audio.astype(_np.float32)


def _build_audio_windows(audio: _np.ndarray, sample_rate: int = 16000,
                         window_seconds: float = 5.0, stride_seconds: float = 2.5,
                         max_windows: int = 6) -> list[tuple[float, _np.ndarray]]:
    """Create evenly distributed, overlapping model-length windows for a clip."""
    window_samples = max(1, round(sample_rate * window_seconds))
    if len(audio) <= window_samples:
        return [(0.0, audio)]

    last_start = len(audio) - window_samples
    stride_samples = max(1, round(sample_rate * stride_seconds))
    starts = list(range(0, last_start + 1, stride_samples))
    if starts[-1] != last_start:
        starts.append(last_start)

    if len(starts) > max_windows:
        indices = _np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[index] for index in sorted(set(indices.tolist()))]

    return [
        (start / sample_rate, audio[start:start + window_samples])
        for start in starts
    ]


def _run_inference(tmp_path: str, filename: str, source: str = "upload") -> dict:
    """Full multi-model pipeline: load → fuse → scene analysis → build event dict."""
    audio_wave = _load_audio_waveform(tmp_path)
    windows = _build_audio_windows(audio_wave)
    window_results = []
    for start_seconds, window in windows:
        window_result = fusion.predict(
            window,
            tmp_path=tmp_path if len(windows) == 1 else None,
        )
        window_results.append((start_seconds, window, window_result))

    best_start, analyzed_audio, fused = max(
        window_results,
        key=lambda item: float(item[2].get("fusion", {}).get("confidence", 0.0)),
    )
    window_labels = {
        item[2].get("fusion", {}).get("normalized", "UNKNOWN")
        for item in window_results
        if item[2].get("fusion", {}).get("normalized") not in {None, "UNKNOWN"}
    }
    window_disagreement = len(window_labels) > 1
    fused = {**fused, "fusion": dict(fused["fusion"])}
    if window_disagreement:
        fused["fusion"].update({
            "requires_review": True,
            "decision_status": "MANUAL_REVIEW",
            "window_disagreement": True,
        })

    ss_result  = fused["models"].get("sonicsentinel", {})
    audio_info = ss_result.get("audio_info") or {}
    if not audio_info:
        _, audio_info = preprocess_audio(analyzed_audio, sample_rate=16000)

    f          = fused["fusion"]
    label      = f["label"]
    confidence = f["confidence"]
    speech_detected = f.get("normalized") == "SPEECH"
    audio_quality = assess_audio_quality(audio_info)
    requires_review = bool(f.get("requires_review", False)) or audio_quality["label"] in {"Poor", "Unusable"}
    scene = analyze_audio_scene(
        analyzed_audio,
        sample_rate=16000,
        speech_detected=speech_detected,
    )
    energy_lbl = classify_energy(audio_info.get("rms", 0), audio_info.get("energy_delta", 0))
    if speech_detected and energy_lbl == "SUDDEN IMPACT":
        energy_lbl = "HIGH ENERGY"
    if requires_review and scene.get("alert_like"):
        scene = {
            **scene,
            "primary_sound": "Unclassified high-energy transient",
            "summary": "A transient was detected, but the model result is uncertain. Manual review is required before treating it as an alarm.",
            "alert_like": False,
            "possible_transient": True,
            "danger_level": "elevated",
        }
    dash_state = f["state_hint"] or compute_dashboard_state(
        label, confidence, audio_info.get("rms", 0), HIGH_RISK_CLASSES
    )
    if scene.get("alert_like"):
        dash_state = "ALERT" if confidence >= 0.55 or label in HIGH_RISK_CLASSES else "ELEVATED"
    elif requires_review:
        dash_state = "MODEL_DISAGREEMENT" if f.get("agreement") == "split" else "UNCERTAIN"
    elif scene.get("background_noise") and confidence < 0.5:
        dash_state = "LISTENING"
    quality_acceptable = audio_quality["label"] in {"Good", "Acceptable"}
    is_alert = quality_acceptable and not requires_review and (
        (label in HIGH_RISK_CLASSES and confidence >= 0.60)
        or scene.get("alert_like")
    )
    total_ms = sum(v.get("latency_ms", 0) for v in fused["models"].values())
    normalized_label = f.get("normalized", "UNKNOWN")
    if is_alert and normalized_label in {"GUNSHOT", "SCREAM"}:
        severity = "Critical"
    elif is_alert:
        severity = "High"
    elif requires_review:
        severity = "Medium"
    elif scene.get("background_noise"):
        severity = "Informational"
    else:
        severity = "Low"

    logger.info("[INFERENCE] %s | label=%s conf=%.1f%% state=%s agreement=%s online=%d/3 scene=%s",
                source, label, confidence * 100, dash_state,
                f.get("agreement"), f.get("online_count", 0), scene.get("primary_sound"))

    event = {
        "audioId":         f"EVT-{rand_id()}",
        "filename":        filename,
        "source":          source,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "source_duration_sec": round(len(audio_wave) / 16000, 2),
        "label":           label,
        "normalized_label": normalized_label,
        "confidence":      confidence,
        "top_predictions": f["top_labels"],
        "audio_features":  audio_info,
        "audio_quality":   audio_quality,
        "analysis_windows": len(windows),
        "selected_window_start_sec": round(best_start, 2),
        "window_predictions": [
            {
                "start_sec": round(start, 2),
                "label": result.get("fusion", {}).get("label", "Unknown sound"),
                "normalized": result.get("fusion", {}).get("normalized", "UNKNOWN"),
                "confidence": result.get("fusion", {}).get("confidence", 0.0),
            }
            for start, _, result in window_results
        ],
        "energy_label":    energy_lbl,
        "dashboard_state": dash_state,
        "is_alert":        is_alert,
        "alert_status":    "active" if is_alert else "none",
        "severity":        severity,
        "requires_review": requires_review,
        "review_status":   "pending" if requires_review else "not_required",
        "review_decision": None,
        "reviewer_comments": "",
        "decision_status": "MANUAL_REVIEW" if requires_review else f.get("decision_status", "CLASSIFIED"),
        "recommended_action": (
            "Improve the recording and review the result."
            if audio_quality["label"] in {"Poor", "Unusable"}
            else "Review the audio and model evidence."
            if requires_review else "No immediate action required."
        ),
        "inference_ms":    round(total_ms, 1),
        "device":          ss_result.get("device", "cpu"),
        "models":          fused["models"],
        "fusion":          f,
        "scene_analysis":  scene,
        "background_noise": scene.get("background_noise", False),
        "primary_sound":   scene.get("primary_sound", "Ambient sound"),
        "audio_summary":   scene.get("summary", ""),
    }
    event_history.appendleft(event)
    _persist_events(event_history)
    return event


# ── Public pages ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email    = request.form.get("email", "")
        password = request.form.get("password", "")
        if email == "wrong@email.com":
            error = "Invalid credentials. Please check your email and password."
        else:
            session["user"] = email
            return redirect(url_for("live"))
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        return redirect(url_for("live"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── Dashboard pages ───────────────────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    return redirect(url_for("live"))


@app.route("/live")
def live():
    return render_template("dashboard/live.html", active="live")


@app.route("/analyze")
def analyze():
    return render_template("dashboard/analyze.html", active="analyze")


@app.route("/history")
def history():
    return render_template("dashboard/history.html", active="history")


@app.route("/alerts")
def alerts():
    return render_template("dashboard/alerts.html", active="alerts")


@app.route("/review")
def review():
    return render_template("dashboard/review.html", active="review")


@app.route("/reports")
def reports():
    return render_template("dashboard/reports.html", active="reports")


@app.route("/admin")
def admin():
    return render_template("dashboard/admin.html", active="admin")


# ── API: auth helpers ────────────────────────────────────────────────────────

@app.route("/api/auth/check-email", methods=["POST"])
def api_check_email():
    """Always returns available=True (no real user DB)."""
    return jsonify({"available": True})


# ── API: system status ────────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    if fusion:
        model_status = fusion.status()
    else:
        model_status = {"loaded": False, "error": "ML stack not installed"}
    return jsonify({
        "system":  "online",
        "models":  model_status,
        "history_count": len(event_history),
    })


# ── API: upload & analyze ─────────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No audio file provided"}), 400

    file = request.files["audio"]
    if not file or not file.filename:
        return jsonify({"success": False, "error": "Empty file"}), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"success": False,
                        "error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 415

    if not _ml_available or not fusion:
        return jsonify({"success": False,
                        "error": "Model not available. Check server logs."}), 503

    suffix = "." + filename.rsplit(".", 1)[1].lower()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(tmp_fd)
        file.save(tmp_path)
        event = _run_inference(tmp_path, filename, source="upload")
        return jsonify({"success": True, **event})
    except ValueError as e:
        logger.warning("Audio load error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception as e:
        logger.exception("Inference error")
        return jsonify({"success": False, "error": "Inference failed. See server logs."}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── API: live microphone chunk ────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accepts a raw audio blob (webm/wav/ogg) from the browser MediaRecorder.
    Returns prediction JSON.
    """
    if not _ml_available or not fusion:
        return jsonify({"success": False, "error": "Model not available"}), 503

    data = request.get_data()
    if not data or len(data) < 512:
        return jsonify({"success": False, "error": "Audio chunk too small"}), 400

    # Detect format from Content-Type or default to webm
    ct     = request.content_type or ""
    suffix = ".webm"
    if "wav"  in ct: suffix = ".wav"
    elif "ogg" in ct: suffix = ".ogg"
    elif "mp4" in ct or "aac" in ct: suffix = ".mp4"

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(tmp_fd)
        with open(tmp_path, "wb") as f:
            f.write(data)
        event = _run_inference(tmp_path, f"mic_chunk{suffix}", source="live")
        return jsonify({"success": True, **event})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception as e:
        logger.exception("Live predict error")
        return jsonify({"success": False, "error": "Inference failed"}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── API: event history ────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def api_history():
    limit  = min(int(request.args.get("limit", 50)), 200)
    events = list(event_history)[:limit]
    return jsonify(events)


@app.route("/api/history/clear", methods=["POST", "DELETE"])
def api_history_clear():
    event_history.clear()
    _persist_events(event_history)
    return jsonify({"success": True, "count": 0})


# ── Legacy /api/events kept for existing pages ────────────────────────────────

@app.route("/api/events", methods=["GET"])
def api_events_get():
    return api_history()


@app.route("/api/events", methods=["POST"])
def api_events_post():
    return api_analyze()


def _find_event(audio_id: str) -> dict | None:
    return next((event for event in event_history if event.get("audioId") == audio_id), None)


def _event_needs_review(event: dict) -> bool:
    if event.get("review_status") == "reviewed":
        return False
    fusion_result = event.get("fusion") or {}
    normalized = event.get("normalized_label") or fusion_result.get("normalized")
    return bool(
        event.get("requires_review")
        or normalized == "UNKNOWN"
        or fusion_result.get("agreement") == "split"
        or float(event.get("confidence", 0.0) or 0.0) < 0.40
    )


@app.route("/api/events/<audio_id>/review", methods=["POST"])
def api_event_review(audio_id: str):
    event = _find_event(audio_id)
    if event is None:
        return jsonify({"success": False, "error": "Event not found"}), 404
    payload = request.get_json(silent=True) or {}
    decision = str(payload.get("decision", "")).lower()
    if decision not in {"confirmed", "false_positive", "corrected"}:
        return jsonify({"success": False, "error": "Unsupported review decision"}), 400
    event["review_status"] = "reviewed"
    event["review_decision"] = decision
    event["reviewer_comments"] = str(payload.get("comments", ""))[:2000]
    if decision == "corrected" and payload.get("corrected_label"):
        event["reviewed_label"] = str(payload["corrected_label"])[:120]
    event["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    _persist_events(event_history)
    return jsonify({"success": True, "event": event})


@app.route("/api/events/<audio_id>/alert", methods=["POST"])
def api_event_alert_action(audio_id: str):
    event = _find_event(audio_id)
    if event is None:
        return jsonify({"success": False, "error": "Event not found"}), 404
    payload = request.get_json(silent=True) or {}
    action = str(payload.get("action", "")).lower()
    if action not in {"acknowledged", "dismissed", "escalated"}:
        return jsonify({"success": False, "error": "Unsupported alert action"}), 400
    if not event.get("is_alert"):
        return jsonify({"success": False, "error": "Event is not an alert"}), 409
    event["alert_status"] = action
    event["alert_action_at"] = datetime.now(timezone.utc).isoformat()
    _persist_events(event_history)
    return jsonify({"success": True, "event": event})


@app.route("/api/events/export.csv", methods=["GET"])
def api_events_export():
    fields = [
        "audioId", "timestamp", "source", "filename", "label", "normalized_label",
        "confidence", "is_alert", "severity", "audio_quality", "model_agreement",
        "alert_status", "requires_review", "review_status", "review_decision",
        "reviewed_label", "recommended_action", "inference_ms",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    rows = [{
        **event,
        "audio_quality": (event.get("audio_quality") or {}).get("label", "Not assessed"),
        "model_agreement": (event.get("fusion") or {}).get("agreement_str", "—"),
    } for event in event_history]
    writer.writerows(rows)
    payload = io.BytesIO(output.getvalue().encode("utf-8"))
    return send_file(payload, mimetype="text/csv", as_attachment=True, download_name="sonicsentinel-events.csv")


# ── API: stats ────────────────────────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def api_stats():
    events = list(event_history)
    total  = len(events)
    alerts = sum(1 for e in events if e.get("is_alert"))
    avg_c  = (sum(e.get("confidence", 0) for e in events) / total) if total else 0
    category_counts = {}
    quality_counts = {}
    for event in events:
        category = event.get("label") or "Unknown sound"
        category_counts[category] = category_counts.get(category, 0) + 1
        quality = (event.get("audio_quality") or {}).get("label", "Not assessed")
        quality_counts[quality] = quality_counts.get(quality, 0) + 1
    return jsonify({
        "totalEvents":    total,
        "criticalAlerts": alerts,
        "falsePositives": sum(1 for event in events if event.get("review_decision") == "false_positive"),
        "pendingReviews": sum(1 for event in events if _event_needs_review(event)),
        "modelDisagreements": sum(1 for event in events if event.get("fusion", {}).get("agreement") == "split"),
        "avgConfidence":  f"{avg_c*100:.1f}%",
        "categoryCounts": category_counts,
        "audioQualityCounts": quality_counts,
    })


@app.route("/api/models/training-history", methods=["GET"])
def api_training_history():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    runs = [
        {
            "id": "active",
            "name": "Active SonicSentinel model",
            "status": "active",
            "history_file": "training_history.csv",
            "history_path": os.path.join(project_dir, "training_history.csv"),
            "checkpoint_path": os.path.join(project_dir, "SonicSentinel_Persistent 0.1", "models", "sonicsentinel_fsd50k_193class_best.pt"),
        },
        {
            "id": "fast_candidate",
            "name": "Fast candidate",
            "status": "candidate",
            "history_file": "sonicsentinel_fast_history.csv",
            "history_path": os.path.join(project_dir, "sonicsentinel_fast_history.csv"),
            "checkpoint_path": os.path.join(project_dir, "sonicsentinel_fast_best.pt"),
        },
    ]
    results = []
    for run in runs:
        history_path = run.pop("history_path")
        checkpoint_path = run.pop("checkpoint_path")
        if not os.path.isfile(history_path):
            continue
        with open(history_path, newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        epochs = []
        for row in rows:
            try:
                epochs.append({
                    "epoch": int(row["epoch"]),
                    "train_accuracy": float(row["train_accuracy"]),
                    "val_accuracy": float(row["val_accuracy"]),
                    "train_loss": float(row["train_loss"]),
                    "val_loss": float(row["val_loss"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
        best_val = max((epoch["val_accuracy"] for epoch in epochs), default=None)
        results.append({
            **run,
            "epochs": epochs,
            "best_val_accuracy": best_val,
            "checkpoint_bytes": os.path.getsize(checkpoint_path) if os.path.isfile(checkpoint_path) else None,
        })
    return jsonify({"runs": results})


# ── API: model health ────────────────────────────────────────────────────────

@app.route("/api/models/health", methods=["GET"])
def api_models_health():
    """
    Returns real health status for each model.
    A model is only reported ONLINE if inference was successfully tested at startup.
    """
    if not fusion:
        return jsonify({
            "sonicsentinel": {"status": "offline", "error": "ML stack not loaded"},
            "ast":           {"status": "offline", "error": "ML stack not loaded"},
            "fusion":        {"status": "offline"},
        })

    st = fusion.status()
    ss  = st.get("sonicsentinel", {})
    ast = st.get("ast", {})

    def _model_health(info: dict) -> dict:
        loaded = info.get("loaded", False)
        return {
            "status":      "online" if loaded else "offline",
            "model_name":  info.get("model_name", "—"),
            "framework":   info.get("framework", "—"),
            "num_classes": info.get("num_classes", 0),
            "device":      info.get("device", "—"),
            "error":       info.get("error"),
        }

    fusion_online = ss.get("loaded", False) or ast.get("loaded", False)
    return jsonify({
        "sonicsentinel": _model_health(ss),
        "ast":           _model_health(ast),
        "fusion":        {"status": "online" if fusion_online else "offline"},
    })


# ── API: model self-test ──────────────────────────────────────────────────────

@app.route("/api/models/test", methods=["GET"])
def api_models_test():
    """
    Loads a known local audio file (or generates a synthetic tone) and runs
    SonicSentinel + AST + Fusion. Returns full diagnostic output.
    """
    if not _ml_available or not fusion:
        return jsonify({"success": False, "error": "ML stack not available"}), 503

    import tempfile, os
    import numpy as _np2

    # Try to find a real test file first
    test_dirs = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests", "audio"),
    ]
    test_path = None
    for d in test_dirs:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    test_path = os.path.join(d, fn)
                    break
        if test_path:
            break

    # Fall back to a synthetic 440 Hz sine tone (1 second)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    used_synthetic = False
    try:
        if test_path:
            import shutil
            shutil.copy(test_path, tmp_path)
            test_filename = os.path.basename(test_path)
        else:
            import soundfile as _sf
            sr = 16_000
            t  = _np2.linspace(0, 1.0, sr, dtype=_np2.float32)
            tone = (0.3 * _np2.sin(2 * _np2.pi * 440 * t)).astype(_np2.float32)
            _sf.write(tmp_path, tone, sr)
            test_filename = "synthetic_440hz_tone.wav"
            used_synthetic = True

        audio_wave, _ = librosa.load(tmp_path, sr=16_000, mono=True)
        audio_wave = audio_wave.astype(_np2.float32)

        fused = fusion.predict(audio_wave, tmp_path=tmp_path)
        ss_r  = fused["models"].get("sonicsentinel", {})
        ast_r = fused["models"].get("ast", {})
        f     = fused["fusion"]

        _, audio_info = preprocess_audio(tmp_path, is_bytes=False)

        return jsonify({
            "success":       True,
            "test_file":     test_filename,
            "synthetic":     used_synthetic,
            "input": {
                "duration_sec": audio_info.get("duration_sec"),
                "sample_rate":  audio_info.get("sample_rate"),
                "rms":          audio_info.get("rms"),
            },
            "sonicsentinel": {
                "status":     ss_r.get("status"),
                "prediction": ss_r.get("prediction"),
                "normalized": ss_r.get("normalized_prediction"),
                "confidence": ss_r.get("confidence"),
                "top5":       ss_r.get("top_predictions", [])[:5],
                "latency_ms": ss_r.get("latency_ms"),
                "error":      ss_r.get("error"),
            },
            "ast": {
                "status":     ast_r.get("status"),
                "prediction": ast_r.get("prediction"),
                "normalized": ast_r.get("normalized_prediction"),
                "confidence": ast_r.get("confidence"),
                "top5":       ast_r.get("top_predictions", [])[:5],
                "latency_ms": ast_r.get("latency_ms"),
                "error":      ast_r.get("error"),
            },
            "fusion": {
                "event":      f.get("label"),
                "normalized": f.get("normalized"),
                "confidence": f.get("confidence"),
                "agreement":  f.get("agreement"),
                "agreement_str": f.get("agreement_str"),
                "evidence":   f.get("evidence", []),
            },
        })
    except Exception as e:
        logger.exception("Model self-test error")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False,
                    "error": f"File too large. Max {MAX_UPLOAD_MB} MB."}), 413


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# ── Socket.IO events for Live Detector ──────────────────────────────────────
# These are ONLY used by the Live Detector page.
# They do not touch the upload/analyze pipeline.

@socketio.on("connect")
def on_connect():
    logger.info("[LIVE-WS] Client connected: %s", request.sid)
    emit("connected", {"status": "ok", "sid": request.sid})


@socketio.on("disconnect")
def on_disconnect():
    logger.info("[LIVE-WS] Client disconnected: %s", request.sid)


@socketio.on("start_session")
def on_start_session(data):
    """Browser requests a new live session."""
    if not _live_available:
        emit("session_error", {"error": "Live audio service not available"})
        return
    try:
        sess = _live_create_session()
        join_room(sess.session_id)
        emit("session_started", {
            "session_id": sess.session_id,
            "started_at": sess.started_at,
            "model_status": _live_model_status(),
        })
        logger.info("[LIVE-WS] Session started: %s", sess.session_id)
    except Exception as e:
        logger.exception("[LIVE-WS] start_session error")
        emit("session_error", {"error": str(e)})


@socketio.on("stop_session")
def on_stop_session(data):
    """Browser stops the live session."""
    sid = (data or {}).get("session_id")
    if sid:
        _live_end_session(sid)
        leave_room(sid)
    emit("session_stopped", {"session_id": sid})


@socketio.on("audio_chunk")
def on_audio_chunk(data):
    """
    Browser sends a base64-encoded audio chunk.
    data = {"session_id": str, "audio": bytes, "content_type": str}
    """
    if not _live_available:
        emit("analysis_error", {"error": "Live audio service not available"})
        return
    try:
        session_id   = (data or {}).get("session_id")
        raw_audio    = (data or {}).get("audio", b"")
        content_type = (data or {}).get("content_type", "audio/wav")

        if isinstance(raw_audio, str):
            import base64
            raw_audio = base64.b64decode(raw_audio)

        if not raw_audio or len(raw_audio) < 512:
            emit("analysis_error", {"error": "Audio chunk too small"})
            return

        sess = _live_get_session(session_id) if session_id else None
        result = _live_analyze_chunk(raw_audio, content_type=content_type, session=sess)
        emit("analysis_result", result)
    except Exception as e:
        logger.exception("[LIVE-WS] audio_chunk error")
        emit("analysis_error", {"error": str(e)})


@socketio.on("get_model_status")
def on_get_model_status(data):
    status = _live_model_status() if _live_available else {}
    emit("model_status", status)


# ── API: live model status (REST fallback) ────────────────────────────────────

@app.route("/api/live/status", methods=["GET"])
def api_live_status():
    if not _live_available:
        return jsonify({"available": False, "error": "Live audio service not loaded"})
    return jsonify({"available": True, "models": _live_model_status()})


@app.route("/api/live/analyze", methods=["POST"])
def api_live_analyze():
    """
    REST fallback for browsers that cannot use Socket.IO.
    Accepts raw audio bytes, returns analysis JSON.
    """
    if not _live_available:
        return jsonify({"success": False, "error": "Live audio service not available"}), 503

    data = request.get_data()
    if not data or len(data) < 512:
        return jsonify({"success": False, "error": "Audio chunk too small"}), 400

    content_type = request.content_type or "audio/wav"
    try:
        result = _live_analyze_chunk(data, content_type=content_type, session=None)
        return jsonify(result)
    except Exception as e:
        logger.exception("[LIVE-REST] analyze error")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    socketio.run(app, host="0.0.0.0", port=port, debug=debug, allow_unsafe_werkzeug=True)
