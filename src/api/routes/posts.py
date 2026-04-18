"""
POST /api/v1/posts          — upload image or video, queue classification
GET  /api/v1/posts/{id}     — poll status / fetch full result
POST /api/v1/posts/{id}/classify — trigger classification on an existing post
                                   (used for seeded content in status="uploaded")
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schema.posts import ClassifyPostResponse, PostDetailResponse, PostUploadResponse
from src.core.dependencies import get_current_user
from src.db.models.post import Post, PostEmbedding
from src.db.models.user import User
from src.db.session import get_db
from utils.logger import get_logger

logger = get_logger(__name__)

api_router = APIRouter()

# ── Media storage paths ────────────────────────────────────────────────────────
_BASE_DIR       = Path(__file__).resolve().parents[3]
_MEDIA_IMAGES   = _BASE_DIR / "media" / "images"
_MEDIA_VIDEOS   = _BASE_DIR / "media" / "videos"
_MEDIA_THUMBS   = _BASE_DIR / "media" / "thumbnails"

for _d in (_MEDIA_IMAGES, _MEDIA_VIDEOS, _MEDIA_THUMBS):
    _d.mkdir(parents=True, exist_ok=True)

# ── MIME / extension helpers ───────────────────────────────────────────────────
_VIDEO_MIME     = ("video/",)
_VIDEO_EXTS     = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
_IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_THUMBNAIL_SIZE = (400, 400)


def _is_video(content_type: str | None, filename: str | None) -> bool:
    if content_type:
        if any(content_type.lower().startswith(p) for p in _VIDEO_MIME):
            return True
    if filename:
        if os.path.splitext(filename)[1].lower() in _VIDEO_EXTS:
            return True
    return False


def _save_image(data: bytes, filename: str | None) -> tuple[str, str | None]:
    """Write image bytes to disk, generate thumbnail. Returns (media_path, thumb_path) relative to project root."""
    from PIL import Image
    import io

    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in _IMAGE_EXTS:
        ext = ".jpg"

    uid        = str(uuid.uuid4())
    img_path   = _MEDIA_IMAGES / f"{uid}{ext}"
    thumb_path = _MEDIA_THUMBS / f"{uid}.jpg"

    img_path.write_bytes(data)

    try:
        with Image.open(io.BytesIO(data)) as im:
            canvas = Image.new("RGB", _THUMBNAIL_SIZE, (0, 0, 0))
            im_rgb = im.convert("RGB")
            im_rgb.thumbnail(_THUMBNAIL_SIZE, Image.LANCZOS)
            offset = (
                (_THUMBNAIL_SIZE[0] - im_rgb.width) // 2,
                (_THUMBNAIL_SIZE[1] - im_rgb.height) // 2,
            )
            canvas.paste(im_rgb, offset)
            canvas.save(thumb_path, "JPEG", quality=85)
        thumb_rel = str(thumb_path.relative_to(_BASE_DIR))
    except Exception as exc:
        logger.warning("Thumbnail generation failed | error=%s", exc)
        thumb_rel = None

    return str(img_path.relative_to(_BASE_DIR)), thumb_rel


def _save_video(data: bytes, filename: str | None) -> tuple[str, str | None]:
    """Write video bytes to disk. Returns (media_path, None). Thumbnail extracted by pipeline."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in _VIDEO_EXTS:
        ext = ".mp4"

    uid      = str(uuid.uuid4())
    vid_path = _MEDIA_VIDEOS / f"{uid}{ext}"
    vid_path.write_bytes(data)

    return str(vid_path.relative_to(_BASE_DIR)), None


# ── Background classification task ─────────────────────────────────────────────

async def _run_classification(post_id: uuid.UUID, database_url: str) -> None:
    """
    Background task: load the post, run the pipeline, write results back.
    Runs in its own DB session (BackgroundTasks execute after response is sent).
    """
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from pipeline.image_pipeline import ImagePipeline
    from pipeline.video_pipeline import VideoPipeline
    from pipeline.embedding import EmbeddingGenerator

    engine  = create_async_engine(database_url, echo=False, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with Session() as db:
            post = await db.get(Post, post_id)
            if post is None:
                logger.error("classify bg: post %s not found", post_id)
                return

            abs_path = str(_BASE_DIR / post.media_path)
            logger.info("classify bg: starting | post=%s type=%s", post_id, post.media_type)

            result_fields: dict = {}

            # ── Image ─────────────────────────────────────────────────────────
            if post.media_type == "image":
                try:
                    pipeline = ImagePipeline(abs_path, "file")
                    r = await asyncio.to_thread(pipeline.classify)
                    result_fields = {
                        "nudity_level":             r.get("nudity_level"),
                        "nsfw_subcategories":        r.get("nsfw_subcategories", []),
                        "violence_level":            r.get("violence_level"),
                        "violence_type":             r.get("violence_type", []),
                        "self_harm_level":           r.get("self_harm_level"),
                        "self_harm_type":            r.get("self_harm_type", []),
                        "age_group":                 r.get("age_group"),
                        "risk":                      r.get("risk"),
                        "classification_confidence": r.get("confidence"),
                        "content_description":       r.get("content_description"),
                        "display_tags":              r.get("display_tags", []),
                        "mood":                      r.get("mood"),
                        "scene_type":                r.get("scene_type"),
                        "text_in_image":             r.get("text_in_image"),
                        "objects_detected":          r.get("objects_detected", []),
                        "people_count":              str(r.get("people_count", 0)),
                        "deepface_age":              r.get("deepface_age"),
                        "deepface_age_group":        r.get("deepface_age_group"),
                    }
                    # Generate embedding
                    emb_text = (result_fields.get("content_description") or "") + " " + (post.caption or "")
                    embedding = await asyncio.to_thread(EmbeddingGenerator.generate, emb_text.strip())
                except Exception as exc:
                    logger.error("Image classification failed | post=%s error=%s", post_id, exc, exc_info=True)
                    post.status = "error"
                    await db.commit()
                    return

            # ── Video ─────────────────────────────────────────────────────────
            elif post.media_type == "video":
                try:
                    pipeline = VideoPipeline()
                    r = await asyncio.to_thread(pipeline.process, abs_path, post.caption)
                    result_fields = {
                        "nudity_level":             r.nudity_level,
                        "nsfw_subcategories":        r.nsfw_subcategories,
                        "violence_level":            r.violence_level,
                        "violence_type":             r.violence_type,
                        "self_harm_level":           r.self_harm_level,
                        "self_harm_type":            r.self_harm_type,
                        "age_group":                 r.age_group,
                        "risk":                      r.risk,
                        "classification_confidence": r.classification_confidence,
                        "content_description":       r.content_description,
                        "display_tags":              r.display_tags,
                        "mood":                      r.mood,
                        "scene_type":                r.scene_type,
                        "text_in_image":             r.text_in_image,
                        "objects_detected":          r.objects_detected,
                        "people_count":              str(r.people_count),
                        "deepface_age":              r.deepface_age,
                        "deepface_age_group":        r.deepface_age_group,
                        "video_duration_seconds":    r.video_duration_seconds,
                        "frames_analyzed":           r.frames_analyzed,
                        "llm_calls_used":            r.llm_calls_used,
                        "needs_review":              r.needs_review,
                        "transcript":                r.transcript,
                        "transcript_language":       r.transcript_language,
                        "transcript_safety_flags":   r.transcript_safety_flags,
                        "secondary_classifications": r.secondary_classifications,
                    }
                    embedding = r.embedding

                    # Extract thumbnail from first frame if not already present
                    if not post.thumbnail_path:
                        import subprocess
                        thumb_uid  = str(uuid.uuid4())
                        thumb_path = _MEDIA_THUMBS / f"{thumb_uid}.jpg"
                        try:
                            subprocess.run(
                                ["ffmpeg", "-y", "-i", abs_path, "-vframes", "1",
                                 "-vf", "scale=400:400:force_original_aspect_ratio=decrease,"
                                        "pad=400:400:(ow-iw)/2:(oh-ih)/2",
                                 str(thumb_path)],
                                capture_output=True, timeout=30,
                            )
                            if thumb_path.exists():
                                result_fields["thumbnail_path"] = str(thumb_path.relative_to(_BASE_DIR))
                        except Exception:
                            pass
                except Exception as exc:
                    logger.error("Video classification failed | post=%s error=%s", post_id, exc, exc_info=True)
                    post.status = "error"
                    await db.commit()
                    return
            else:
                logger.error("classify bg: unknown media_type=%s post=%s", post.media_type, post_id)
                return

            # ── Persist classification results ─────────────────────────────────
            for field, value in result_fields.items():
                if hasattr(post, field):
                    setattr(post, field, value)

            new_status = "needs_review" if result_fields.get("needs_review") else "published"
            post.status       = new_status
            post.classified_at = datetime.now(timezone.utc)

            # ── Store embedding ────────────────────────────────────────────────
            if embedding:
                existing_emb = await db.scalar(
                    select(PostEmbedding).where(PostEmbedding.post_id == post_id)
                )
                if existing_emb is None:
                    db.add(PostEmbedding(post_id=post_id, embedding=embedding))

            await db.commit()
            logger.info(
                "classify bg: done | post=%s status=%s risk=%s",
                post_id, new_status, result_fields.get("risk"),
            )

    except Exception as exc:
        logger.error("classify bg: unhandled error | post=%s error=%s", post_id, exc, exc_info=True)
    finally:
        await engine.dispose()


# ── Route helpers ──────────────────────────────────────────────────────────────

def _get_database_url() -> str:
    from src.db.session import DATABASE_URL
    return DATABASE_URL


# ── POST / — upload ────────────────────────────────────────────────────────────

@api_router.post(
    "",
    response_model=PostUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload image or video — classification runs in the background",
)
async def upload_post(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    caption: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    is_vid = _is_video(file.content_type, file.filename)

    try:
        if is_vid:
            media_path, thumb_path = await asyncio.to_thread(_save_video, data, file.filename)
            media_type = "video"
        else:
            media_path, thumb_path = await asyncio.to_thread(_save_image, data, file.filename)
            media_type = "image"
    except Exception as exc:
        logger.error("File save failed | user=%s error=%s", current_user.id, exc)
        raise HTTPException(status_code=500, detail="Failed to save media") from exc

    post = Post(
        user_id=current_user.id,
        media_type=media_type,
        media_path=media_path,
        thumbnail_path=thumb_path,
        caption=caption,
        status="uploaded",
    )
    db.add(post)
    await db.commit()

    logger.info(
        "Post created | post=%s user=%s type=%s",
        post.id, current_user.id, media_type,
    )

    background_tasks.add_task(
        _run_classification, post.id, _get_database_url()
    )

    return PostUploadResponse(
        post_id=post.id,
        status=post.status,
        media_type=post.media_type,
        media_path=post.media_path,
        thumbnail_path=post.thumbnail_path,
        caption=post.caption,
    )


# ── GET /{post_id} — poll status / fetch result ────────────────────────────────

@api_router.get(
    "/{post_id}",
    response_model=PostDetailResponse,
    summary="Get post status and classification result",
)
async def get_post(
    post_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    post = await db.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")

    return PostDetailResponse(
        post_id=post.id,
        user_id=post.user_id,
        status=post.status,
        media_type=post.media_type,
        media_path=post.media_path,
        thumbnail_path=post.thumbnail_path,
        caption=post.caption,
        nudity_level=post.nudity_level,
        nsfw_subcategories=post.nsfw_subcategories,
        violence_level=post.violence_level,
        violence_type=post.violence_type,
        self_harm_level=post.self_harm_level,
        self_harm_type=post.self_harm_type,
        age_group=post.age_group,
        risk=post.risk,
        classification_confidence=post.classification_confidence,
        content_description=post.content_description,
        display_tags=post.display_tags,
        mood=post.mood,
        scene_type=post.scene_type,
        text_in_image=post.text_in_image,
        objects_detected=post.objects_detected,
        people_count=post.people_count,
        deepface_age=post.deepface_age,
        deepface_age_group=post.deepface_age_group,
        video_duration_seconds=post.video_duration_seconds,
        frames_analyzed=post.frames_analyzed,
        needs_review=post.needs_review,
        transcript=post.transcript,
        transcript_language=post.transcript_language,
        source_url=post.source_url,
        source_platform=post.source_platform,
        source_subreddit=post.source_subreddit,
        source_upvotes=post.source_upvotes,
        source_comments=post.source_comments,
        created_at=post.created_at,
        classified_at=post.classified_at,
    )


# ── POST /{post_id}/classify — trigger classification ─────────────────────────

@api_router.post(
    "/{post_id}/classify",
    response_model=ClassifyPostResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger classification for an uploaded post (e.g. seeded content)",
)
async def classify_post(
    post_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    post = await db.get(Post, post_id)
    if post is None:
        raise HTTPException(status_code=404, detail="Post not found")

    if post.status not in ("uploaded", "error"):
        raise HTTPException(
            status_code=409,
            detail=f"Post is already in status '{post.status}'. Only 'uploaded' or 'error' posts can be re-classified.",
        )

    media_abs = _BASE_DIR / post.media_path
    if not media_abs.exists():
        raise HTTPException(status_code=422, detail="Media file not found on disk")

    background_tasks.add_task(
        _run_classification, post.id, _get_database_url()
    )

    logger.info(
        "Classification queued | post=%s user=%s type=%s",
        post_id, current_user.id, post.media_type,
    )

    return ClassifyPostResponse(
        post_id=post_id,
        status="classifying",
        message="Classification started. Poll GET /posts/{post_id} for status.",
    )
