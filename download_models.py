"""
download_models.py
==================
Run at Railway build time to pre-download all required models.
This avoids cold-start delays and ensures models are cached in the image.

Usage:
    python download_models.py
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_silero_vad():
    logger.info("Downloading Silero VAD...")
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        logger.info("Silero VAD: OK")
        return True
    except Exception as e:
        logger.error("Silero VAD failed: %s", e)
        return False


def download_ast():
    logger.info("Downloading AST AudioSet model...")
    try:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        fe = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        logger.info("AST: OK — %d classes", len(model.config.id2label))
        return True
    except Exception as e:
        logger.error("AST failed: %s", e)
        return False


def download_whisper():
    logger.info("Downloading Whisper base...")
    try:
        import whisper
        model = whisper.load_model("base", device="cpu")
        logger.info("Whisper base: OK")
        return True
    except Exception as e:
        logger.error("Whisper failed: %s", e)
        return False


def check_sonicsentinel():
    """Check if the SonicSentinel checkpoint exists (must be uploaded separately)."""
    candidates = [
        os.path.join("SonicSentinel_Persistent 0.1", "models", "sonicsentinel_fsd50k_193class_best.pt"),
        "sonicsentinel_fsd50k_193class_best.pt",
    ]
    for path in candidates:
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / 1e6
            logger.info("SonicSentinel checkpoint found: %s (%.1f MB)", path, size_mb)
            return True
    logger.warning("SonicSentinel checkpoint NOT found. Upload & Analyze will be unavailable.")
    logger.warning("Upload the .pt file to Railway volume or set MODEL_PATH env var.")
    return False


if __name__ == "__main__":
    results = {
        "silero_vad": download_silero_vad(),
        "ast":        download_ast(),
        "whisper":    download_whisper(),
        "sonicsentinel": check_sonicsentinel(),
    }

    logger.info("=" * 50)
    for name, ok in results.items():
        status = "OK" if ok else "MISSING/FAILED"
        logger.info("  %-20s %s", name, status)
    logger.info("=" * 50)

    # Exit 0 even if SonicSentinel is missing — live detector still works
    failed_required = [k for k, v in results.items() if not v and k != "sonicsentinel"]
    if failed_required:
        logger.error("Required models failed: %s", failed_required)
        sys.exit(1)

    logger.info("Model download complete.")
