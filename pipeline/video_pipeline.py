"""
Video classification pipeline orchestrator.

Processing flow:
    1. Frame selection (7-stage cascade → FrameSelector)
    2. Audio transcription (parallel with step 3 → AudioPipeline)
    3. Per-frame LLM + DeepFace classification (ImagePipeline)
    4. Aggregation (safety override → transcript safety → weighted avg → FrameAggregator)
    5. Embedding generation (EmbeddingGenerator)
    6. Cleanup temp files
"""
from __future__ import annotations

import concurrent.futures
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pipeline.audio_pipeline import AudioPipeline, TranscriptionResult
from pipeline.aggregator import FrameAggregator
from pipeline.embedding import EmbeddingGenerator
from pipeline.frame_selector import FrameCandidate, FrameSelector, FrameSelectionResult
from pipeline.image_pipeline import ImagePipeline
from utils.common_functions import load_config
from utils.logger import get_logger
from utils.ollama_llm_provider import OllamaProvider

_base_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_base_dir)
_config = load_config(os.path.join(_project_dir, "config", "config.yml"))

logger = get_logger(__name__)


@dataclass
class VideoClassificationResult:
    # Safety
    nudity_level: str
    nsfw_subcategories: List[str]
    violence_level: str
    violence_type: List[str]
    self_harm_level: str
    self_harm_type: List[str]
    age_group: str
    risk: str
    classification_confidence: float

    # Content
    content_description: str
    display_tags: List[str]
    mood: str
    scene_type: str
    text_in_image: Optional[str]
    objects_detected: List[str]
    people_count: object

    # DeepFace
    deepface_age: Optional[float]
    deepface_age_group: Optional[str]

    # Audio
    transcript: Optional[str]
    transcript_language: Optional[str]
    transcript_safety_flags: List[dict]
    has_audio: bool

    # Embedding (768-dim, for pgvector)
    embedding: List[float]

    # Video metadata
    video_duration_seconds: float
    video_fps: float
    video_resolution: Tuple[int, int]

    # Processing metadata
    total_frames_sampled: int
    frames_analyzed: int
    llm_calls_used: int
    needs_review: bool
    processing_time_seconds: float
    frame_selection_breakdown: dict

    # Secondary classifications (confidence distribution per field)
    secondary_classifications: dict = field(default_factory=dict)


class VideoPipeline:
    def __init__(self):
        video_cfg = _config.get("video_processing", {})
        self.frame_selector = FrameSelector(video_cfg)
        self.audio_pipeline = AudioPipeline()
        self.aggregator = FrameAggregator()

        # Shared LLM client — one instance reused across all frame classification threads
        model = _config["image_classification"]["model"]
        think = _config["image_classification"].get("think", False)
        self._llm = OllamaProvider(model, think=think)

        # Wire the same client into aggregator for transcript safety analysis
        self.aggregator._ollama = self._llm

    # ── danger keyword list for targeted resampling ───────────────────────────

    _DANGER_KEYWORDS: frozenset = frozenset({
        "kill", "murder", "shoot", "stab", "bomb", "weapon", "gun", "knife",
        "blood", "gore", "torture", "rape", "assault", "threat", "attack",
        "suicide", "self-harm", "cut myself", "cutting", "overdose", "hang",
        "drug", "cocaine", "heroin", "meth", "poison",
        "naked", "nude", "porn", "explicit", "sex",
    })

    # ── helpers ───────────────────────────────────────────────────────────────

    def _transcribe(self, video_path: str) -> TranscriptionResult:
        t_cfg = _config.get("transcription", {})
        if not t_cfg.get("enabled", False):
            logger.debug("Transcription disabled — skipping")
            return TranscriptionResult(full_text="", segments=[], language="",
                                       has_audio=False, duration=0.0)
        return self.audio_pipeline.transcribe(
            video_path,
            model_size=t_cfg.get("model", "base"),
            device=t_cfg.get("device", "cpu"),
            language=t_cfg.get("language"),
        )

    def _find_flagged_windows(
        self, transcript: TranscriptionResult
    ) -> List[Tuple[float, float]]:
        """
        Scan transcript segments for danger keywords.
        Returns merged time windows (start, end) with a 2-second buffer on each side.
        Only works reliably for English; non-English transcripts get no resampling
        (the LLM transcript safety check still catches dangerous content later).
        """
        if not transcript.segments or transcript.language not in ("en", ""):
            return []

        raw: List[Tuple[float, float]] = []
        for seg in transcript.segments:
            text_lower = seg.text.lower()
            if any(kw in text_lower for kw in self._DANGER_KEYWORDS):
                raw.append((max(0.0, seg.start - 2.0), seg.end + 2.0))

        if not raw:
            return []

        # Merge overlapping windows
        raw.sort()
        merged: List[Tuple[float, float]] = [raw[0]]
        for start, end in raw[1:]:
            if start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        logger.info("Flagged windows | count=%d segments=%s", len(merged),
                    [(round(s, 1), round(e, 1)) for s, e in merged])
        return merged

    def _get_transcript_context(
        self,
        transcript: TranscriptionResult,
        ts_start: float,
        ts_end: float,
        window_secs: float = 3.0,
    ) -> Optional[str]:
        """Return transcript text covering [ts_start - window, ts_end + window]."""
        if not transcript.segments:
            return None
        lo = ts_start - window_secs
        hi = ts_end + window_secs
        parts = [seg.text.strip() for seg in transcript.segments
                 if seg.end >= lo and seg.start <= hi and seg.text.strip()]
        return " ".join(parts) if parts else None

    def _classify_batch(
        self, batch: List[FrameCandidate], transcript: TranscriptionResult
    ) -> dict:
        """Classify one batch of 1-3 frames with transcript context."""
        frames_bytes = [c.frame_bytes for c in batch]
        timestamps   = [c.timestamp   for c in batch]
        ts_start     = batch[0].timestamp
        ts_end       = batch[-1].timestamp
        ctx          = self._get_transcript_context(transcript, ts_start, ts_end)

        result = ImagePipeline.classify_batch(
            frames=frames_bytes,
            llm_client=self._llm,
            timestamps=timestamps,
            transcript_context=ctx,
        )
        # Tag result with the middle frame's metadata
        mid = batch[len(batch) // 2]
        result["timestamp"]        = mid.timestamp
        result["selection_reason"] = mid.selection_reason
        logger.debug("Batch classified | ts=%.1f–%.1fs nudity=%s conf=%s",
                     ts_start, ts_end, result.get("nudity_level"), result.get("confidence"))
        return result

    # ── public entry points ───────────────────────────────────────────────────

    def process(self, video_path: str, caption: Optional[str] = None) -> VideoClassificationResult:
        """Full pipeline from a file path."""
        t0 = time.monotonic()
        logger.info("Video pipeline started | path=%s", video_path)

        # ── Step 1: frame selection + transcription in parallel ───────────────
        # Both are I/O + CPU bound (OpenCV / FFmpeg) and independent.
        # We need the transcript BEFORE classification so we can:
        #   a) inject audio context into each LLM batch call
        #   b) resample dangerous windows at 5 fps
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            future_sel = pool.submit(self.frame_selector.select_frames, video_path)  # type: ignore[arg-type]
            future_tx  = pool.submit(self._transcribe, video_path)                   # type: ignore[arg-type]
            selection: FrameSelectionResult = future_sel.result()
            transcript: TranscriptionResult = future_tx.result()

        if not selection.selected_frames:
            raise ValueError("No frames passed the content-change filters")

        # ── Step 2: targeted resampling of dangerous audio windows ────────────
        flagged_windows = self._find_flagged_windows(transcript)
        extra_frames: List[FrameCandidate] = []
        if flagged_windows:
            extra_frames = self.frame_selector.resample_flagged_windows(
                video_path, flagged_windows, fps=5.0
            )

        # Merge extra frames; deduplicate by frame_index; keep chronological order
        all_selected = list(selection.selected_frames)
        if extra_frames:
            existing_indices = {c.frame_index for c in all_selected}
            new_extra = [c for c in extra_frames if c.frame_index not in existing_indices]
            all_selected = sorted(all_selected + new_extra, key=lambda c: c.timestamp)
            logger.info("After targeted resample | total_frames=%d (+%d new)",
                        len(all_selected), len(new_extra))

        # ── Step 3: batch classification with transcript context ──────────────
        # Group frames into batches of 3 — one LLM call per batch.
        # Batching gives the model cross-frame context (3× fewer LLM calls).
        cap = self.frame_selector.max_llm_calls
        to_classify = all_selected[:cap]  # type: ignore[index]
        needs_review = len(all_selected) > cap or selection.needs_review

        batch_size  = _config.get("video_processing", {}).get("batch_size", 3)
        batches: List[List[FrameCandidate]] = [
            to_classify[i : i + batch_size]          # type: ignore[index]
            for i in range(0, len(to_classify), batch_size)
        ]

        frame_results: List[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batches) or 1) as pool:
            futures = {
                pool.submit(self._classify_batch, b, transcript): b  # type: ignore[arg-type]
                for b in batches
            }
            for future in concurrent.futures.as_completed(futures):
                batch = futures[future]
                try:
                    frame_results.append(future.result())
                except Exception as exc:
                    logger.warning("Batch classification failed | ts=%.1fs error=%s",
                                   batch[0].timestamp if batch else -1, exc)

        frame_results.sort(key=lambda r: r.get("timestamp", 0))

        if not frame_results:
            raise ValueError("LLM returned no usable results for any selected frame")

        logger.info("Frame classification done | batches=%d results=%d needs_review=%s",
                    len(batches), len(frame_results), needs_review)

        aggregated = self.aggregator.aggregate(frame_results, transcript)
        aggregated["needs_review"] = needs_review

        # Embedding: encode selected frames visually via SigLIP
        try:
            frame_dicts = [
                {"frame_bytes": c.frame_bytes, "selection_reason": c.selection_reason}
                for c in to_classify
            ]
            embedding = EmbeddingGenerator.generate_for_video(frame_dicts)
        except Exception as exc:
            logger.warning("Embedding generation failed | error=%s", exc)
            embedding = []

        meta = selection.video_metadata
        elapsed = int((time.monotonic() - t0) * 100) / 100

        logger.info(
            "Video pipeline complete | dur=%.1fs classified=%d "
            "transcript_chars=%d needs_review=%s elapsed=%.1fs",
            meta.get("duration", 0), len(frame_results),
            len(transcript.full_text), needs_review, elapsed,
        )

        return VideoClassificationResult(
            nudity_level=aggregated.get("nudity_level", "safe"),
            nsfw_subcategories=aggregated.get("nsfw_subcategories", []),
            violence_level=aggregated.get("violence_level", "none"),
            violence_type=aggregated.get("violence_type", []),
            self_harm_level=aggregated.get("self_harm_level", "none"),
            self_harm_type=aggregated.get("self_harm_type", []),
            age_group=aggregated.get("age_group", "unknown"),
            risk=aggregated.get("risk", "allow"),
            classification_confidence=aggregated.get("confidence", 0.5),
            content_description=aggregated.get("content_description", ""),
            display_tags=aggregated.get("display_tags", []),
            mood=aggregated.get("mood", "neutral"),
            scene_type=aggregated.get("scene_type", "indoor"),
            text_in_image=aggregated.get("text_in_image"),
            objects_detected=aggregated.get("objects_detected", []),
            people_count=aggregated.get("people_count", 0),
            deepface_age=aggregated.get("deepface_age"),
            deepface_age_group=aggregated.get("deepface_age_group"),
            transcript=transcript.full_text or None,
            transcript_language=transcript.language or None,
            transcript_safety_flags=aggregated.get("transcript_safety_flags", []),
            has_audio=transcript.has_audio,
            embedding=embedding,
            video_duration_seconds=meta.get("duration", 0.0),
            video_fps=meta.get("fps", 0.0),
            video_resolution=meta.get("resolution", (0, 0)),
            total_frames_sampled=len(all_selected),
            frames_analyzed=len(frame_results),
            llm_calls_used=len(batches) * 2,  # 2 passes (safety + content) per batch
            needs_review=needs_review,
            processing_time_seconds=elapsed,
            frame_selection_breakdown={
                "sampled":              selection.total_frames_sampled,
                "after_phash":          selection.frames_after_phash,
                "after_histogram":      selection.frames_after_histogram,
                "after_msssim":         selection.frames_after_msssim,
                "after_optical_flow":   selection.frames_after_optical_flow,
                "after_shot_detection": selection.frames_after_shot_detection,
                "after_diversity":      selection.frames_after_diversity,
            },
            secondary_classifications=aggregated.get("secondary_classifications", {}),
        )

    def process_bytes(self, video_bytes: bytes,
                      caption: Optional[str] = None) -> VideoClassificationResult:
        """Process video from raw bytes. Writes to temp file, cleans up after."""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_bytes)
        tmp.close()
        path = tmp.name
        try:
            return self.process(path, caption)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
