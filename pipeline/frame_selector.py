"""7-stage cascade frame selector for video content moderation.

Stages (cheap → expensive):
    1. Sample at 2 fps
    2. pHash near-duplicate removal        (~1 µs/frame)
    3. Color histogram distance             (~0.1 ms/frame)
    4. MS-SSIM with adaptive threshold      (~1 ms/frame)
    5. Optical flow camera motion filter    (~5 ms/frame)
    6. Shot boundary detection (TransNetV2 or manual drift)
    7. Diversity selection (CLIP KMeans or pHash AgglomerativeClustering)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIG: dict = {
    "sampling_fps": 2,
    "phash_threshold": 5,
    "histogram_threshold": 0.3,
    "ssim_threshold": 0.85,
    "optical_flow_threshold": 2.0,
    "max_llm_calls": 15,
    "use_transnet": True,
    "use_clip": True,
    "drift_check_interval": 5,
}


# ── dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class RawFrame:
    frame_bgr: np.ndarray
    frame_index: int
    timestamp: float
    phash: object = field(default=None)
    histogram: Optional[np.ndarray] = field(default=None)
    # Histogram distance from previous frame — higher = more visually distinct.
    # Set by _filter_histogram and used by diversity selection to prefer
    # frames that represent genuine content changes.
    histogram_score: float = field(default=0.0)


@dataclass
class FrameCandidate:
    frame_bytes: bytes
    frame_index: int
    timestamp: float
    selection_reason: str
    cluster_id: int = -1


@dataclass
class FrameSelectionResult:
    selected_frames: List[FrameCandidate]
    total_frames_in_video: int
    total_frames_sampled: int
    frames_after_phash: int
    frames_after_histogram: int
    frames_after_msssim: int
    frames_after_optical_flow: int
    frames_after_shot_detection: int
    frames_after_diversity: int
    needs_review: bool
    video_metadata: dict  # fps, duration, resolution, codec, total_frames


# ── helpers ────────────────────────────────────────────────────────────────────

def _to_jpeg(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf)


# ── FrameSelector ──────────────────────────────────────────────────────────────

class FrameSelector:
    def __init__(self, config: dict):
        cfg = {**_DEFAULT_CONFIG, **config}
        self.sampling_fps: float = float(cfg["sampling_fps"])
        self.phash_threshold: int = int(cfg["phash_threshold"])
        self.histogram_threshold: float = float(cfg["histogram_threshold"])
        self.ssim_threshold: float = float(cfg["ssim_threshold"])
        self.optical_flow_threshold: float = float(cfg["optical_flow_threshold"])
        self.max_llm_calls: int = int(cfg["max_llm_calls"])
        self.use_transnet: bool = bool(cfg["use_transnet"])
        self.use_clip: bool = bool(cfg["use_clip"])
        self.drift_check_interval: float = float(cfg["drift_check_interval"])

    # ── Stage 1: sample ───────────────────────────────────────────────────────

    def _sample_frames(self, video_path: str) -> Tuple[List[RawFrame], dict]:
        cap = cv2.VideoCapture(video_path)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()

        frame_step = max(1, int(round(video_fps / self.sampling_fps)))

        frames: List[RawFrame] = []
        last_bgr: Optional[np.ndarray] = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            last_bgr = frame
            if frame_idx % frame_step == 0:
                frames.append(RawFrame(
                    frame_bgr=frame.copy(),
                    frame_index=frame_idx,
                    timestamp=round(frame_idx / video_fps, 3),
                ))
            frame_idx += 1
        cap.release()

        # Always include last frame if not already captured
        if last_bgr is not None and (not frames or frames[-1].frame_index != frame_idx - 1):
            frames.append(RawFrame(
                frame_bgr=last_bgr.copy(),
                frame_index=frame_idx - 1,
                timestamp=round((frame_idx - 1) / video_fps, 3),
            ))

        meta = {
            "fps": video_fps,
            "duration": duration,
            "resolution": (width, height),
            "total_frames": total_frames,
            "codec": codec,
        }
        logger.info("Stage 1 (sample) | frames=%d total_video=%d fps=%.1f dur=%.1fs",
                    len(frames), total_frames, video_fps, duration)
        return frames, meta

    # ── Stage 2: pHash ────────────────────────────────────────────────────────

    def _ensure_phash(self, frames: List[RawFrame]) -> None:
        try:
            from imagehash import phash as compute_phash  # type: ignore
        except ImportError:
            return
        for f in frames:
            if f.phash is None:
                pil = Image.fromarray(cv2.cvtColor(f.frame_bgr, cv2.COLOR_BGR2RGB))
                f.phash = compute_phash(pil, hash_size=16)

    def _filter_phash(self, frames: List[RawFrame]) -> List[RawFrame]:
        if len(frames) < 2:
            return frames
        try:
            from imagehash import phash as compute_phash  # type: ignore
        except ImportError:
            logger.debug("imagehash not installed — skipping pHash stage")
            return frames

        self._ensure_phash(frames)
        last_idx = frames[-1].frame_index
        survivors: List[RawFrame] = [frames[0]]

        for f in frames[1:]:
            if f.frame_index == last_idx:
                survivors.append(f)
                continue
            dist = f.phash - survivors[-1].phash
            if dist >= self.phash_threshold:
                survivors.append(f)

        logger.info("Stage 2 (pHash) | in=%d out=%d threshold=%d",
                    len(frames), len(survivors), self.phash_threshold)
        return survivors

    # ── Stage 3: histogram ────────────────────────────────────────────────────

    def _compute_histogram(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _ensure_histograms(self, frames: List[RawFrame]) -> None:
        for f in frames:
            if f.histogram is None:
                f.histogram = self._compute_histogram(f.frame_bgr)

    def _filter_histogram(self, frames: List[RawFrame]) -> List[RawFrame]:
        """
        Soft histogram gate — scores every frame, hard-drops only near-identical ones.

        Previous behaviour: drop if dist < threshold (0.30).
        New behaviour:
          - Always store histogram_score on each frame (used by diversity selection
            to prefer frames with larger colour changes).
          - Hard-drop only if dist < threshold * 0.5 (truly identical colour).
          - Keep borderline frames (threshold*0.5 ≤ dist < threshold) — they may
            contain content that looks similar in palette but is visually distinct
            (e.g. two different people against the same yellow wall).
        """
        if len(frames) < 2:
            return frames
        self._ensure_histograms(frames)
        last_idx = frames[-1].frame_index
        hard_drop_floor = self.histogram_threshold * 0.5
        survivors: List[RawFrame] = [frames[0]]

        for f in frames[1:]:
            if f.frame_index == last_idx:
                f.histogram_score = 0.0
                survivors.append(f)
                continue
            dist = float(cv2.compareHist(survivors[-1].histogram, f.histogram, cv2.HISTCMP_CHISQR))
            f.histogram_score = dist  # store for diversity selection
            if dist >= hard_drop_floor:
                survivors.append(f)
            else:
                logger.debug("Histogram hard-drop | ts=%.2fs dist=%.3f floor=%.3f",
                             f.timestamp, dist, hard_drop_floor)

        logger.info("Stage 3 (histogram) | in=%d out=%d hard_floor=%.2f (config_threshold=%.2f)",
                    len(frames), len(survivors), hard_drop_floor, self.histogram_threshold)
        return survivors

    # ── Stage 4: MS-SSIM with adaptive threshold ──────────────────────────────

    def _to_gray_small(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (128, 128))

    def _ssim(self, a: np.ndarray, b: np.ndarray) -> float:
        try:
            from skimage.metrics import structural_similarity as sk_ssim  # type: ignore
            # Handle both old (multichannel) and new (channel_axis) skimage APIs
            try:
                return float(sk_ssim(a, b, channel_axis=None, gaussian_weights=True,
                                     sigma=1.5, use_sample_covariance=False, data_range=255))
            except TypeError:
                return float(sk_ssim(a, b, multichannel=False, gaussian_weights=True,
                                     sigma=1.5, use_sample_covariance=False, data_range=255))
        except ImportError:
            # Pure numpy single-scale SSIM fallback (Wang et al. 2004)
            a = a.astype(np.float64)
            b = b.astype(np.float64)
            c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            mu1, mu2 = np.mean(a), np.mean(b)
            num = (2 * mu1 * mu2 + c1) * (2 * np.mean((a - mu1) * (b - mu2)) + c2)
            den = (mu1 ** 2 + mu2 ** 2 + c1) * (np.var(a) + np.var(b) + c2)
            return float(num / den)

    def _filter_msssim(self, frames: List[RawFrame], all_frames: List[RawFrame]) -> List[RawFrame]:
        if len(frames) < 2:
            return frames

        # Compute adaptive threshold from up to 30 consecutive sampled frame pairs
        sample = all_frames[:30] if len(all_frames) > 30 else all_frames
        scores: List[float] = []
        gray_cache: Dict[int, np.ndarray] = {f.frame_index: self._to_gray_small(f.frame_bgr)
                                              for f in sample}
        for i in range(1, len(sample)):
            scores.append(self._ssim(gray_cache[sample[i - 1].frame_index],
                                     gray_cache[sample[i].frame_index]))

        if scores:
            adaptive = float(np.mean(scores) - 2 * np.std(scores))
            threshold = max(0.5, min(adaptive, self.ssim_threshold))
        else:
            threshold = self.ssim_threshold
        logger.debug("SSIM adaptive threshold=%.3f (base=%.3f)", threshold, self.ssim_threshold)

        last_idx = frames[-1].frame_index

        # Precompute grays for survivors; reuse cache where possible
        def get_gray(f: RawFrame) -> np.ndarray:
            if f.frame_index not in gray_cache:
                gray_cache[f.frame_index] = self._to_gray_small(f.frame_bgr)
            return gray_cache[f.frame_index]

        survivors: List[RawFrame] = [frames[0]]
        last_sent_gray = get_gray(frames[0])  # compare against last frame kept

        for f in frames[1:]:
            if f.frame_index == last_idx:
                survivors.append(f)
                last_sent_gray = get_gray(f)
                continue
            curr_gray = get_gray(f)
            sim = self._ssim(curr_gray, last_sent_gray)
            if sim <= threshold:
                survivors.append(f)
                last_sent_gray = curr_gray

        logger.info("Stage 4 (MS-SSIM) | in=%d out=%d threshold=%.3f",
                    len(frames), len(survivors), threshold)
        return survivors

    # ── Stage 5: optical flow ─────────────────────────────────────────────────

    def _filter_optical_flow(self, frames: List[RawFrame]) -> List[RawFrame]:
        if len(frames) < 2:
            return frames

        last_idx = frames[-1].frame_index

        def to_gray256(f: np.ndarray) -> np.ndarray:
            return cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (256, 256))

        survivors: List[RawFrame] = [frames[0]]
        prev_gray = to_gray256(frames[0].frame_bgr)

        for f in frames[1:]:
            if f.frame_index == last_idx:
                survivors.append(f)
                prev_gray = to_gray256(f.frame_bgr)
                continue

            curr_gray = to_gray256(f.frame_bgr)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_mag = float(np.mean(magnitude))
            moving = magnitude > 0.5
            angle_std = float(np.std(np.arctan2(flow[..., 1], flow[..., 0])[moving])) \
                if np.any(moving) else 0.0

            # Coherent motion = high magnitude + low angular variance → camera pan/zoom
            is_camera_motion = mean_mag > self.optical_flow_threshold and angle_std < 0.5
            if not is_camera_motion:
                survivors.append(f)
            else:
                logger.debug("Flow: camera motion | ts=%.2fs mean_mag=%.2f angle_std=%.2f",
                             f.timestamp, mean_mag, angle_std)
            prev_gray = curr_gray

        logger.info("Stage 5 (optical flow) | in=%d out=%d", len(frames), len(survivors))
        return survivors

    # ── Stage 6: shot boundary detection ─────────────────────────────────────

    def _detect_shot_boundaries(
        self,
        video_path: str,
        frames: List[RawFrame],
        all_frames: List[RawFrame],
    ) -> List[RawFrame]:
        if len(frames) <= 1:
            return frames
        if self.use_transnet:
            try:
                from transnetv2 import TransNetV2  # type: ignore
                return self._transnet_shot_filter(video_path, frames)
            except ImportError:
                logger.debug("TransNetV2 not available — using manual drift fallback")
        return self._manual_drift_filter(frames, all_frames)

    def _transnet_shot_filter(self, video_path: str, frames: List[RawFrame]) -> List[RawFrame]:
        from transnetv2 import TransNetV2  # type: ignore

        model = TransNetV2()
        _, single_frame_pred, _ = model.predict_video(video_path)
        # TransNetV2 outputs per-frame probabilities at 25 fps
        transnet_fps = 25.0
        boundary_ts: List[float] = [i / transnet_fps for i, p in enumerate(single_frame_pred)
                                     if p > 0.5]
        logger.info("TransNetV2 found %d shot boundaries", len(boundary_ts))

        if not boundary_ts:
            return frames

        # Force-include frames within 0.5s of a shot boundary
        forced: set = {0, len(frames) - 1}
        for i, f in enumerate(frames):
            if any(abs(f.timestamp - bt) <= 0.5 for bt in boundary_ts):
                forced.add(i)

        # Also keep frames >3s since last kept frame (gradual scene coverage)
        last_kept_ts = frames[0].timestamp
        survivors_idx: List[int] = sorted(forced)
        for i, f in enumerate(frames):
            if i in forced:
                last_kept_ts = f.timestamp
                continue
            if (f.timestamp - last_kept_ts) >= 3.0:
                survivors_idx.append(i)
                last_kept_ts = f.timestamp

        survivors = [frames[i] for i in sorted(set(survivors_idx))]
        logger.info("Stage 6 (TransNet) | in=%d out=%d", len(frames), len(survivors))
        return survivors

    def _manual_drift_filter(self, frames: List[RawFrame], all_frames: List[RawFrame]) -> List[RawFrame]:
        """ADWIN-style adaptive change detection."""
        if len(frames) < 2:
            return frames

        self._ensure_histograms(frames)
        self._ensure_histograms(all_frames)

        # Build running stats from all sampled frames
        distances: List[float] = []
        for i in range(1, len(all_frames)):
            d = cv2.compareHist(all_frames[i - 1].histogram, all_frames[i].histogram,
                                cv2.HISTCMP_CHISQR)
            distances.append(d)

        if distances:
            change_threshold = float(np.mean(distances) + 2 * np.std(distances))
        else:
            change_threshold = self.histogram_threshold * 2

        forced: set = {0, len(frames) - 1}
        last_included_idx = 0
        last_drift_ts = -self.drift_check_interval

        for i in range(1, len(frames)):
            curr = frames[i]
            prev = frames[i - 1]
            d = cv2.compareHist(prev.histogram, curr.histogram, cv2.HISTCMP_CHISQR)

            if d > change_threshold:
                forced.add(i)
                last_included_idx = i
                continue

            if (curr.timestamp - last_drift_ts) >= self.drift_check_interval:
                last_drift_ts = curr.timestamp
                drift = cv2.compareHist(frames[last_included_idx].histogram,
                                        curr.histogram, cv2.HISTCMP_CHISQR)
                if drift > self.histogram_threshold * 0.7:
                    forced.add(i)
                    last_included_idx = i

        survivors = [frames[i] for i in sorted(forced)]
        logger.info("Stage 6 (manual drift) | in=%d out=%d change_threshold=%.3f",
                    len(frames), len(survivors), change_threshold)
        return survivors

    # ── Stage 7: diversity selection ──────────────────────────────────────────

    def _select_diverse_frames(self, frames: List[RawFrame]) -> List[FrameCandidate]:
        if not frames:
            return []

        if self.use_clip:
            try:
                import open_clip  # type: ignore
                import torch        # type: ignore
                return self._clip_diversity(frames, open_clip, torch)
            except ImportError:
                logger.debug("open_clip/torch not available — using pHash clustering")

        return self._phash_diversity(frames)

    def _clip_diversity(self, frames: List[RawFrame], open_clip, torch) -> List[FrameCandidate]:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not hasattr(FrameSelector, "_clip_model"):
            logger.info("Loading CLIP ViT-B/32...")
            m, _, p = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k")
            FrameSelector._clip_model = m.to(device).eval()
            FrameSelector._clip_preprocess = p
            logger.info("CLIP loaded")

        model = FrameSelector._clip_model
        preprocess = FrameSelector._clip_preprocess

        embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for f in frames:
                pil = Image.fromarray(cv2.cvtColor(f.frame_bgr, cv2.COLOR_BGR2RGB))
                t = preprocess(pil).unsqueeze(0).to(device)
                emb = model.encode_image(t)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy()[0])

        emb_matrix = np.stack(embeddings)
        return self._kmeans_select(frames, emb_matrix, "diversity_centroid")

    def _phash_diversity(self, frames: List[RawFrame]) -> List[FrameCandidate]:
        if len(frames) <= self.max_llm_calls:
            return [FrameCandidate(_to_jpeg(f.frame_bgr), f.frame_index, f.timestamp,
                                   "diversity_centroid") for f in frames]

        self._ensure_phash(frames)
        n = len(frames)
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(frames[i].phash - frames[j].phash)
                dist_matrix[i, j] = dist_matrix[j, i] = d

        try:
            from sklearn.cluster import AgglomerativeClustering  # type: ignore
            n_clusters = min(n, self.max_llm_calls)
            labels = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average"
            ).fit_predict(dist_matrix)
        except ImportError:
            logger.debug("sklearn not available — evenly spacing frames")
            return self._evenly_space(frames)

        candidates: List[FrameCandidate] = []
        n_clusters = min(n, self.max_llm_calls)
        for cid in range(n_clusters):
            indices = [i for i, l in enumerate(labels) if l == cid]
            if not indices:
                continue
            indices.sort(key=lambda i: frames[i].timestamp)
            f = frames[indices[len(indices) // 2]]
            candidates.append(FrameCandidate(_to_jpeg(f.frame_bgr), f.frame_index,
                                             f.timestamp, "diversity_centroid", cid))

        logger.info("Stage 7 (pHash cluster) | in=%d out=%d", len(frames), len(candidates))
        return candidates

    def _kmeans_select(self, frames: List[RawFrame], embeddings: np.ndarray,
                       reason: str) -> List[FrameCandidate]:
        from sklearn.cluster import KMeans  # type: ignore

        n_clusters = min(len(frames), self.max_llm_calls)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        candidates: List[FrameCandidate] = []
        for cid in range(n_clusters):
            indices = [i for i, l in enumerate(labels) if l == cid]
            if not indices:
                continue
            best = min(indices, key=lambda i: np.linalg.norm(embeddings[i] - km.cluster_centers_[cid]))
            f = frames[best]
            candidates.append(FrameCandidate(_to_jpeg(f.frame_bgr), f.frame_index,
                                             f.timestamp, reason, cid))

        logger.info("Stage 7 (CLIP KMeans) | in=%d out=%d", len(frames), len(candidates))
        return candidates

    def _evenly_space(self, frames: List[RawFrame]) -> List[FrameCandidate]:
        n = self.max_llm_calls
        step = len(frames) / n
        target = [frames[int(i * step)] for i in range(n)]
        return [FrameCandidate(_to_jpeg(f.frame_bgr), f.frame_index, f.timestamp,
                               "diversity_centroid") for f in target]

    # ── targeted resampling for flagged time windows ───────────────────────────

    def resample_flagged_windows(
        self,
        video_path: str,
        windows: List[Tuple[float, float]],
        fps: float = 5.0,
    ) -> List[FrameCandidate]:
        """
        Extract frames from specific time windows at a higher fps (default 5).
        Called after transcription identifies dangerous audio segments so we
        get more visual coverage of those moments.

        Returns FrameCandidate objects (already JPEG-encoded) ready to be
        merged into the existing selection list.
        """
        if not windows:
            return []

        cap = cv2.VideoCapture(video_path)
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(round(video_fps / fps)))

        candidates: List[FrameCandidate] = []
        seen_indices: set = set()

        for start_ts, end_ts in windows:
            start_idx = max(0, int(start_ts * video_fps))
            end_idx   = min(total_frames - 1, int(end_ts * video_fps))

            idx = start_idx
            while idx <= end_idx:
                if idx in seen_indices:
                    idx += frame_step
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ret, frame = cap.read()
                if not ret:
                    break
                seen_indices.add(idx)
                candidates.append(FrameCandidate(
                    frame_bytes=_to_jpeg(frame),
                    frame_index=idx,
                    timestamp=round(idx / video_fps, 3),
                    selection_reason="targeted_resample",
                ))
                idx += frame_step

        cap.release()
        logger.info("Targeted resample | windows=%d frames=%d fps=%.0f",
                    len(windows), len(candidates), fps)
        return candidates

    # ── public entry point ────────────────────────────────────────────────────

    def select_frames(self, video_path: str) -> FrameSelectionResult:
        t0 = time.monotonic()

        all_frames, meta = self._sample_frames(video_path)
        total_sampled = len(all_frames)

        after_phash = self._filter_phash(all_frames)
        after_hist = self._filter_histogram(after_phash)
        after_ssim = self._filter_msssim(after_hist, all_frames)
        after_flow = self._filter_optical_flow(after_ssim)
        after_shots = self._detect_shot_boundaries(video_path, after_flow, all_frames)
        selected = self._select_diverse_frames(after_shots)

        needs_review = len(selected) >= self.max_llm_calls
        elapsed = time.monotonic() - t0

        logger.info(
            "Frame selection complete | sampled=%d → phash=%d → histogram=%d → "
            "msssim=%d → flow=%d → shots=%d → diverse=%d | needs_review=%s | elapsed=%.2fs",
            total_sampled, len(after_phash), len(after_hist), len(after_ssim),
            len(after_flow), len(after_shots), len(selected), needs_review, elapsed,
        )

        return FrameSelectionResult(
            selected_frames=selected,
            total_frames_in_video=meta["total_frames"],
            total_frames_sampled=total_sampled,
            frames_after_phash=len(after_phash),
            frames_after_histogram=len(after_hist),
            frames_after_msssim=len(after_ssim),
            frames_after_optical_flow=len(after_flow),
            frames_after_shot_detection=len(after_shots),
            frames_after_diversity=len(selected),
            needs_review=needs_review,
            video_metadata=meta,
        )
