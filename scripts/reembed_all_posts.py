"""Re-embed all classified posts using the new SigLIP visual encoder.

Usage:
    python scripts/reembed_all_posts.py

Reads each classified post's media file from disk, runs it through
EmbeddingGenerator, and updates the post_embeddings table.

For videos: extracts frames at stored timestamps via OpenCV, then calls
generate_for_video(). Falls back to thumbnail if no frame results exist.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(Path(_PROJECT_ROOT) / ".env")

import uuid

import cv2
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from pipeline.embedding import EmbeddingGenerator
from src.db.models.post import Post, PostEmbedding, PostFrameResult
from utils.logger import get_logger

logger = get_logger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/social_media_content",
).replace("postgresql+asyncpg://", "postgresql://")

BASE_DIR = Path(_PROJECT_ROOT)


def _extract_frame_at(video_path: str, timestamp: float) -> bytes | None:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
        ok, frame_bgr = cap.read()
        if not ok:
            return None
        _, buf = cv2.imencode(".jpg", frame_bgr)
        return buf.tobytes()
    except Exception:
        return None
    finally:
        cap.release()


def _embed_image_post(media_path: str) -> list[float] | None:
    try:
        with open(media_path, "rb") as f:
            image_bytes = f.read()
        return EmbeddingGenerator.generate(image_bytes)
    except Exception as exc:
        logger.warning("Image embed failed | path=%s error=%s", media_path, exc)
        return None


def _embed_video_post(
    media_path: str,
    frame_results: list[PostFrameResult],
    thumbnail_path: str | None,
) -> list[float] | None:
    if frame_results:
        frames = []
        for fr in frame_results:
            frame_bytes = _extract_frame_at(media_path, fr.timestamp_seconds)
            if frame_bytes:
                frames.append({
                    "frame_bytes": frame_bytes,
                    "selection_reason": fr.selection_reason or "",
                })
        if frames:
            return EmbeddingGenerator.generate_for_video(frames)

    # Fallback: embed thumbnail
    if thumbnail_path:
        thumb_full = str(BASE_DIR / thumbnail_path)
        if os.path.exists(thumb_full):
            return _embed_image_post(thumb_full)

    return None


def main() -> None:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    with Session(engine) as session:
        posts = session.scalars(
            select(Post).where(
                Post.status.in_(["published", "blocked"]),
                Post.classified_at.is_not(None),
            )
        ).all()

    total = len(posts)
    print(f"Found {total} classified posts to re-embed.")

    success = 0
    failed = 0
    t0 = time.monotonic()

    for i, post in enumerate(posts, 1):
        media_full = str(BASE_DIR / post.media_path) if post.media_path else None

        if not media_full or not os.path.exists(media_full):
            logger.warning("Missing media | post=%s path=%s", post.id, media_full)
            failed += 1
        else:
            if post.media_type == "image":
                embedding = _embed_image_post(media_full)
            else:
                with Session(engine) as session:
                    frame_results = session.scalars(
                        select(PostFrameResult)
                        .where(PostFrameResult.post_id == post.id)
                        .order_by(PostFrameResult.timestamp_seconds)
                    ).all()
                embedding = _embed_video_post(
                    media_full, list(frame_results), post.thumbnail_path
                )

            if embedding:
                with Session(engine) as session:
                    existing = session.scalar(
                        select(PostEmbedding).where(PostEmbedding.post_id == post.id)
                    )
                    if existing is None:
                        session.add(PostEmbedding(post_id=post.id, embedding=embedding))
                    else:
                        existing.embedding = embedding
                    session.commit()
                success += 1
            else:
                failed += 1

        if i % 50 == 0:
            elapsed = time.monotonic() - t0
            rate = i / elapsed
            remaining = (total - i) / rate if rate > 0 else 0
            print(
                f"  [{i}/{total}] {rate:.1f} posts/s — "
                f"ETA {remaining / 60:.1f} min | ok={success} fail={failed}"
            )

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.1f}s — success={success} failed={failed} total={total}")


if __name__ == "__main__":
    main()
