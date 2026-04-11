"""
Low-level video frame utilities.

All heavy maths live here so the pipeline layer stays readable.
No LLM or DeepFace calls — pure OpenCV + NumPy.
"""
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Tunable constants ──────────────────────────────────────────────────────────
SAMPLE_FPS: float = 2.0          # frames extracted per video second
PIXEL_DIFF_THRESHOLD: float = 0.05   # 5 % mean absolute diff → scene changed
SSIM_THRESHOLD: float = 0.85     # structural similarity above this → same scene
DRIFT_INTERVAL_SECONDS: float = 5.0  # how often to run the cumulative drift check
MAX_LLM_CALLS: int = 15          # hard cap on LLM invocations per video

# ── Frame helpers ──────────────────────────────────────────────────────────────

def to_gray_small(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to 128×128 grayscale — cheap comparison surface."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)


def pixel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference normalised to [0, 1].

    Both inputs must be same-shape uint8 arrays (e.g. 128×128 grayscale).
    Returns 0.0 (identical) … 1.0 (maximum difference).
    """
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))
    return float(diff / 255.0)


def ssim_global(a: np.ndarray, b: np.ndarray) -> float:
    """Global (single-window) SSIM between two same-shape grayscale images.

    Uses the standard SSIM formula (Wang et al. 2004) applied to the whole
    image at once.  Fast enough for 128×128 thumbnails.
    Returns −1.0 (completely different) … 1.0 (identical).
    """
    C1 = (0.01 * 255) ** 2   # ≈  6.5  — luminance stability constant
    C2 = (0.03 * 255) ** 2   # ≈ 58.5  — contrast stability constant

    f1 = a.astype(np.float64)
    f2 = b.astype(np.float64)

    mu1, mu2 = f1.mean(), f2.mean()
    sigma1_sq = f1.var()
    sigma2_sq = f2.var()
    sigma12 = np.mean((f1 - mu1) * (f2 - mu2))

    numerator   = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(np.clip(numerator / denominator, -1.0, 1.0))


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes ready for the LLM."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed for frame")
    return buf.tobytes()


def get_video_meta(path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = total_frames / fps
    return {
        "video_fps": round(fps, 2),
        "total_frames": total_frames,
        "video_duration": round(duration, 2),
    }
