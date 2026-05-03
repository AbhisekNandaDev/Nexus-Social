from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional, Union

from pydantic import BaseModel


class PostUploadResponse(BaseModel):
    """Returned immediately after upload — classification runs in the background."""
    post_id: uuid.UUID
    status: str          # "uploaded" until classification finishes
    media_type: str      # "image" | "video"
    media_url: str
    thumbnail_url: Optional[str] = None
    caption: Optional[str] = None
    message: str = "Post uploaded. Classification queued."


class PostDetailResponse(BaseModel):
    """Full post detail — includes classification fields once status != 'uploaded'."""
    post_id: uuid.UUID
    user_id: uuid.UUID
    status: str
    media_type: str
    media_url: str
    thumbnail_url: Optional[str] = None
    caption: Optional[str] = None

    # Safety
    nudity_level: Optional[str] = None
    nsfw_subcategories: Optional[List[str]] = None
    violence_level: Optional[str] = None
    violence_type: Optional[List[str]] = None
    self_harm_level: Optional[str] = None
    self_harm_type: Optional[List[str]] = None
    age_group: Optional[str] = None
    risk: Optional[str] = None
    classification_confidence: Optional[float] = None

    # Content
    content_description: Optional[str] = None
    display_tags: Optional[List[str]] = None
    mood: Optional[str] = None
    scene_type: Optional[str] = None
    text_in_image: Optional[str] = None
    objects_detected: Optional[List[str]] = None
    people_count: Optional[Union[int, str]] = None

    # DeepFace
    deepface_age: Optional[int] = None
    deepface_age_group: Optional[str] = None

    # Video-specific
    video_duration_seconds: Optional[float] = None
    frames_analyzed: Optional[int] = None
    needs_review: bool = False
    transcript: Optional[str] = None
    transcript_language: Optional[str] = None

    # Source metadata (seeded content)
    source_url: Optional[str] = None
    source_platform: Optional[str] = None
    source_subreddit: Optional[str] = None
    source_upvotes: Optional[int] = None
    source_comments: Optional[int] = None

    # Timestamps
    created_at: datetime
    classified_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ClassifyPostResponse(BaseModel):
    """Returned from POST /{post_id}/classify."""
    post_id: uuid.UUID
    status: str
    message: str
