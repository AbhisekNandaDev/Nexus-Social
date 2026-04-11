"""
Audio transcription via faster-whisper (local, no cloud API).

Design notes
────────────
• The WhisperModel is expensive to load (~1-3 s depending on size).
  It is kept as a module-level singleton so it is loaded once and reused
  across all requests for the lifetime of the process.

• faster-whisper accepts a video/audio file path directly and decodes
  the audio track internally via ffmpeg — no separate audio extraction step.

• VAD (voice-activity detection) filtering is enabled so silent segments
  are skipped, keeping transcription fast on videos with sparse speech.

• compute_type="int8" gives the best CPU throughput; switch to "float16"
  for CUDA.
"""
from __future__ import annotations

import threading
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ── singleton ──────────────────────────────────────────────────────────────────
_model = None
_model_lock = threading.Lock()
_loaded_model_size: Optional[str] = None
_loaded_device: Optional[str] = None


def _get_model(model_size: str, device: str):
    """Return the cached WhisperModel, loading it on first call."""
    global _model, _loaded_model_size, _loaded_device

    # Fast path — already loaded with matching config
    if _model is not None and _loaded_model_size == model_size and _loaded_device == device:
        return _model

    with _model_lock:
        # Re-check inside lock in case another thread loaded it first
        if _model is not None and _loaded_model_size == model_size and _loaded_device == device:
            return _model

        from faster_whisper import WhisperModel  # imported lazily so the module loads fast

        compute = "int8" if device == "cpu" else "float16"
        logger.info(
            "Loading Whisper model | size=%s device=%s compute_type=%s",
            model_size, device, compute,
        )
        _model = WhisperModel(model_size, device=device, compute_type=compute)
        _loaded_model_size = model_size
        _loaded_device = device
        logger.info("Whisper model ready | size=%s", model_size)

    return _model


# ── public API ─────────────────────────────────────────────────────────────────

def transcribe(
    video_path: str,
    model_size: str = "base",
    device: str = "cpu",
    language: Optional[str] = None,
) -> dict:
    """Transcribe the audio track of a video (or audio) file.

    Args:
        video_path:  Path to the media file.  faster-whisper decodes the
                     audio via ffmpeg internally — no pre-extraction needed.
        model_size:  Whisper model variant (tiny / base / small / medium /
                     large-v3).  Defaults to "base".
        device:      "cpu" or "cuda".
        language:    ISO-639-1 code (e.g. "en") or None for auto-detect.

    Returns a dict with:
        text                 – full transcript string, or None if no speech
        language             – detected ISO-639-1 language code
        language_probability – confidence of the language detection (0-1)
    """
    model = _get_model(model_size, device)

    logger.info(
        "Transcription started | model=%s device=%s language=%s path=%s",
        model_size, device, language or "auto", video_path,
    )

    try:
        segments, info = model.transcribe(
            video_path,
            language=language,
            beam_size=5,
            vad_filter=True,       # skip silent regions — significant speed-up
            vad_parameters={"min_silence_duration_ms": 500},
        )

        # Consume the generator (segments are lazy)
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        full_text: Optional[str] = " ".join(text_parts) or None

        logger.info(
            "Transcription complete | language=%s (%.2f) length=%s chars",
            info.language,
            info.language_probability,
            len(full_text) if full_text else 0,
        )

        return {
            "text": full_text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
        }

    except Exception as exc:
        logger.warning("Transcription failed (non-fatal) | error=%s", exc)
        return {"text": None, "language": None, "language_probability": None}
