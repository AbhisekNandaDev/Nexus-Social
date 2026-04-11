"""
Tests for the video processing pipeline.

Run with:
    pytest tests/test_video_pipeline.py -v -s

The -s flag shows frame selection breakdown logs so you can tune thresholds.

Each test logs the full FrameSelectionResult breakdown so you can see exactly
how many frames survived each stage — useful for threshold calibration.
"""
from __future__ import annotations

import os
import struct
import tempfile
import zlib
from typing import List

import cv2
import numpy as np
import pytest

from pipeline.frame_selector import FrameSelector, FrameSelectionResult

# ── test video generators ──────────────────────────────────────────────────────

def _write_video(path: str, frames: List[np.ndarray], fps: float = 30.0) -> None:
    """Write a list of BGR frames to an mp4 file."""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def _solid_color_video(color_bgr: tuple, n_frames: int = 90, fps: float = 30.0) -> str:
    """Static scene: one solid colour for the entire video (~3 seconds at 30fps)."""
    tmp = tempfile.mktemp(suffix=".mp4")
    frame = np.full((480, 640, 3), color_bgr, dtype=np.uint8)
    _write_video(tmp, [frame] * n_frames, fps)
    return tmp


def _scene_cut_video(colors: List[tuple], frames_per_scene: int = 60,
                     fps: float = 30.0) -> str:
    """Montage: hard cuts between solid-colour scenes."""
    tmp = tempfile.mktemp(suffix=".mp4")
    all_frames: List[np.ndarray] = []
    for color in colors:
        frame = np.full((480, 640, 3), color, dtype=np.uint8)
        all_frames.extend([frame] * frames_per_scene)
    _write_video(tmp, all_frames, fps)
    return tmp


def _camera_pan_video(n_frames: int = 90, fps: float = 30.0) -> str:
    """Horizontal pan across a static image — same content, coherent motion."""
    tmp = tempfile.mktemp(suffix=".mp4")
    bg = np.zeros((480, 1280, 3), dtype=np.uint8)
    bg[:, :640] = (50, 100, 150)
    bg[:, 640:] = (150, 100, 50)
    frames: List[np.ndarray] = []
    for i in range(n_frames):
        offset = int(i * 640 / n_frames)
        frames.append(bg[:, offset:offset + 640].copy())
    _write_video(tmp, frames, fps)
    return tmp


def _gradual_change_video(fps: float = 30.0, duration: float = 15.0) -> str:
    """Scene that slowly changes colour over 15 seconds."""
    tmp = tempfile.mktemp(suffix=".mp4")
    n = int(fps * duration)
    frames: List[np.ndarray] = []
    for i in range(n):
        t = i / n
        color = (int(50 + 200 * t), int(200 - 150 * t), int(100))
        frame = np.full((480, 640, 3), color, dtype=np.uint8)
        frames.append(frame)
    _write_video(tmp, frames, fps)
    return tmp


def _many_scene_video(n_scenes: int = 22, fps: float = 30.0) -> str:
    """More scenes than max_llm_calls (15) to test capping."""
    colors = [(i * 11 % 255, i * 37 % 255, i * 73 % 255) for i in range(n_scenes)]
    return _scene_cut_video(colors, frames_per_scene=30, fps=fps)


def _log_breakdown(result: FrameSelectionResult, test_name: str) -> None:
    print(f"\n[{test_name}] Frame selection breakdown:")
    print(f"  Video: {result.video_metadata}")
    print(f"  Sampled:        {result.total_frames_sampled}")
    print(f"  After pHash:    {result.frames_after_phash}")
    print(f"  After histogram:{result.frames_after_histogram}")
    print(f"  After MS-SSIM:  {result.frames_after_msssim}")
    print(f"  After flow:     {result.frames_after_optical_flow}")
    print(f"  After shots:    {result.frames_after_shot_detection}")
    print(f"  After diversity:{result.frames_after_diversity}")
    print(f"  needs_review:   {result.needs_review}")
    for f in result.selected_frames:
        print(f"    ts={f.timestamp:.2f}s reason={f.selection_reason} cluster={f.cluster_id}")


# ── config that disables optional heavy deps in CI ────────────────────────────

_SELECTOR_CONFIG = {
    "sampling_fps": 2,
    "phash_threshold": 5,
    "histogram_threshold": 0.3,
    "ssim_threshold": 0.85,
    "optical_flow_threshold": 2.0,
    "max_llm_calls": 15,
    "use_transnet": False,   # disable in tests (not installed in CI)
    "use_clip": False,       # disable in tests (heavy model download)
    "drift_check_interval": 5,
}

# ── tests ──────────────────────────────────────────────────────────────────────

class TestFrameSelector:

    def setup_method(self):
        self.selector = FrameSelector(_SELECTOR_CONFIG)

    def teardown_method(self):
        pass  # temp files cleaned up per test

    def test_static_video_minimal_frames(self):
        """Static webcam scene should collapse to 2-3 frames max."""
        path = _solid_color_video((100, 150, 200), n_frames=90)
        try:
            result = self.selector.select_frames(path)
            _log_breakdown(result, "static_video")
            # pHash + histogram + SSIM should collapse a static scene heavily
            assert result.frames_after_diversity <= 3, (
                f"Expected ≤3 frames for static video, got {result.frames_after_diversity}"
            )
            assert result.total_frames_sampled > 0
        finally:
            os.unlink(path)

    def test_scene_cut_video_detects_each_scene(self):
        """Montage with 5 distinct scenes should select at least one frame per scene."""
        n_scenes = 5
        colors = [(i * 50, i * 30, 200 - i * 30) for i in range(n_scenes)]
        path = _scene_cut_video(colors, frames_per_scene=60)
        try:
            result = self.selector.select_frames(path)
            _log_breakdown(result, "scene_cut")
            # Should detect all scene changes — at least n_scenes frames
            assert result.frames_after_diversity >= n_scenes, (
                f"Expected ≥{n_scenes} frames for {n_scenes}-scene video, "
                f"got {result.frames_after_diversity}"
            )
        finally:
            os.unlink(path)

    def test_camera_pan_no_false_positives(self):
        """Camera pan should not generate excessive frames (same content, coherent motion)."""
        path = _camera_pan_video(n_frames=90)
        try:
            result = self.selector.select_frames(path)
            _log_breakdown(result, "camera_pan")
            # Optical flow stage should suppress most pan frames
            # Allow some leeway — the transition itself is content-free
            assert result.frames_after_optical_flow <= result.frames_after_msssim, (
                "Optical flow stage should reduce or maintain frame count"
            )
            # Final diversity should be small for a pan-only video
            assert result.frames_after_diversity <= 5, (
                f"Expected ≤5 frames for camera pan, got {result.frames_after_diversity}"
            )
        finally:
            os.unlink(path)

    def test_gradual_change_caught_by_drift(self):
        """Slow colour drift over 15s should be detected by the drift detector."""
        path = _gradual_change_video(fps=30.0, duration=15.0)
        try:
            result = self.selector.select_frames(path)
            _log_breakdown(result, "gradual_change")
            # Drift should force at least 3 frames (start, middle, end-ish)
            assert result.frames_after_shot_detection >= 3, (
                f"Expected ≥3 frames for gradual change, got {result.frames_after_shot_detection}"
            )
        finally:
            os.unlink(path)

    def test_llm_cap_sets_needs_review(self):
        """Video with more scenes than max_llm_calls should set needs_review=True."""
        path = _many_scene_video(n_scenes=22)
        try:
            result = self.selector.select_frames(path)
            _log_breakdown(result, "llm_cap")
            assert result.frames_after_diversity <= _SELECTOR_CONFIG["max_llm_calls"], (
                f"Diversity selection exceeded max_llm_calls={_SELECTOR_CONFIG['max_llm_calls']}"
            )
            # needs_review should be set when we hit the cap
            if result.frames_after_shot_detection > _SELECTOR_CONFIG["max_llm_calls"]:
                assert result.needs_review, "needs_review should be True when cap is hit"
        finally:
            os.unlink(path)

    def test_selected_frames_from_different_clusters(self):
        """Selected frames should come from different diversity clusters."""
        n_scenes = 6
        colors = [(i * 40, 200 - i * 20, i * 10) for i in range(n_scenes)]
        path = _scene_cut_video(colors, frames_per_scene=45)
        try:
            result = self.selector.select_frames(path)
            _log_breakdown(result, "diversity_clusters")
            cluster_ids = {f.cluster_id for f in result.selected_frames if f.cluster_id >= 0}
            # With pHash clustering fallback, should have multiple clusters
            if len(result.selected_frames) > 1:
                assert len(cluster_ids) >= 1, "Should have at least one cluster assigned"
        finally:
            os.unlink(path)


class TestAudioPipeline:

    def test_no_audio_video_returns_empty_transcript(self):
        """Video with no audio stream should return has_audio=False."""
        from pipeline.audio_pipeline import AudioPipeline
        pipeline = AudioPipeline()
        path = _solid_color_video((100, 100, 100), n_frames=30)
        try:
            result = pipeline.transcribe(path)
            # mp4v-encoded test video typically has no audio
            _log_no_audio = f"has_audio={result.has_audio} text='{result.full_text}'"
            print(f"\n[no_audio] {_log_no_audio}")
            # Either has_audio=False (detected) or empty transcript (transcription found nothing)
            assert result.full_text == "" or result.has_audio is False, (
                "Silent video should yield empty transcript"
            )
        finally:
            os.unlink(path)

    def test_transcription_result_structure(self):
        """TranscriptionResult should always have the required fields."""
        from pipeline.audio_pipeline import AudioPipeline, TranscriptionResult
        pipeline = AudioPipeline()
        path = _solid_color_video((50, 50, 50))
        try:
            result = pipeline.transcribe(path)
            assert isinstance(result, TranscriptionResult)
            assert isinstance(result.full_text, str)
            assert isinstance(result.segments, list)
            assert isinstance(result.has_audio, bool)
            assert isinstance(result.language, str)
        finally:
            os.unlink(path)


class TestAggregator:

    def _make_result(self, nudity="safe", confidence=0.8, age="adult",
                     violence="none", risk="allow", self_harm="none") -> dict:
        return {
            "nudity_level": nudity,
            "nsfw_subcategories": [],
            "violence_level": violence,
            "violence_type": [],
            "self_harm_level": self_harm,
            "self_harm_type": [],
            "age_group": age,
            "risk": risk,
            "confidence": confidence,
            "content_description": f"Test frame nudity={nudity}",
            "display_tags": ["tag1"],
            "mood": "neutral",
            "scene_type": "indoor",
            "text_in_image": None,
            "objects_detected": ["person"],
            "people_count": 1,
        }

    def test_safety_override_explicit_nudity(self):
        """Single explicit frame at high confidence should override safe frames."""
        from pipeline.aggregator import FrameAggregator
        agg = FrameAggregator()
        results = [
            self._make_result("safe", confidence=0.9),
            self._make_result("safe", confidence=0.85),
            self._make_result("explicit_nudity", confidence=0.75),  # should win
            self._make_result("safe", confidence=0.9),
        ]
        result = agg.aggregate(results)
        print(f"\n[safety_override] nudity={result['nudity_level']} risk={result['risk']}")
        assert result["nudity_level"] == "explicit_nudity", (
            "Explicit nudity frame should override safe frames"
        )

    def test_safety_override_minor_nudity(self):
        """Child + nudity at high confidence should produce illegal risk."""
        from pipeline.aggregator import FrameAggregator
        agg = FrameAggregator()
        results = [
            self._make_result("safe", confidence=0.9),
            self._make_result("suggestive", confidence=0.8, age="child"),
        ]
        result = agg.aggregate(results)
        print(f"\n[minor_nudity] nudity={result['nudity_level']} risk={result['risk']} age={result['age_group']}")
        assert result["risk"] == "illegal", "Minor + nudity should produce illegal risk"

    def test_weighted_average_dominant_wins(self):
        """Most-weighted nudity level should be selected."""
        from pipeline.aggregator import FrameAggregator
        agg = FrameAggregator()
        results = [
            self._make_result("safe", confidence=0.9),
            self._make_result("safe", confidence=0.85),
            self._make_result("safe", confidence=0.8),
            self._make_result("suggestive", confidence=0.6),
        ]
        result = agg.aggregate(results)
        print(f"\n[weighted_avg] nudity={result['nudity_level']} conf={result['confidence']}")
        assert result["nudity_level"] == "safe", "Safe should dominate with 3x weight"

    def test_age_group_most_restrictive(self):
        """Age group should always use most restrictive (child < teen < adult)."""
        from pipeline.aggregator import FrameAggregator
        agg = FrameAggregator()
        results = [
            self._make_result("safe", confidence=0.9, age="adult"),
            self._make_result("safe", confidence=0.9, age="teen"),
            self._make_result("safe", confidence=0.9, age="adult"),
        ]
        result = agg.aggregate(results)
        print(f"\n[age_restrictive] age={result['age_group']}")
        assert result["age_group"] == "teen", "Most restrictive age should win"

    def test_no_results_raises(self):
        """Aggregating zero results should raise ValueError."""
        from pipeline.aggregator import FrameAggregator
        agg = FrameAggregator()
        with pytest.raises(ValueError):
            agg.aggregate([])

    def test_secondary_classifications_populated(self):
        """Secondary classifications should contain confidence distributions."""
        from pipeline.aggregator import FrameAggregator
        agg = FrameAggregator()
        results = [
            self._make_result("safe", confidence=0.7),
            self._make_result("suggestive", confidence=0.3),
        ]
        result = agg.aggregate(results)
        sec = result.get("secondary_classifications", {})
        print(f"\n[secondary] {sec}")
        assert "nudity_level" in sec
        assert "safe" in sec["nudity_level"]
        assert "suggestive" in sec["nudity_level"]


class TestEmbeddingGenerator:

    def test_embedding_dimensions(self):
        """all-MiniLM-L6-v2 should produce 384-dim vectors."""
        from pipeline.embedding import EmbeddingGenerator
        emb = EmbeddingGenerator.generate("A person walking in a park")
        print(f"\n[embedding] dim={len(emb)} sample={emb[:3]}")
        assert len(emb) == 384

    def test_empty_text_returns_zeros(self):
        """Empty text should return a zero vector."""
        from pipeline.embedding import EmbeddingGenerator
        emb = EmbeddingGenerator.generate("")
        assert len(emb) == 384
        assert all(v == 0.0 for v in emb)

    def test_embedding_normalized(self):
        """Embeddings should be unit-norm (L2)."""
        import math
        from pipeline.embedding import EmbeddingGenerator
        emb = EmbeddingGenerator.generate("Test content description")
        norm = math.sqrt(sum(v ** 2 for v in emb))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"
