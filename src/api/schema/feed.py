from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class FeedPost(BaseModel):
    post_id: uuid.UUID
    user_id: uuid.UUID
    author_display_name: Optional[str] = None
    media_type: str
    media_url: str
    thumbnail_url: Optional[str] = None
    caption: Optional[str] = None
    display_tags: Optional[List[str]] = None
    mood: Optional[str] = None
    scene_type: Optional[str] = None
    nudity_level: Optional[str] = None
    source_platform: Optional[str] = None
    source_subreddit: Optional[str] = None
    created_at: datetime
    like_count: int = 0
    is_liked: bool = False

    class Config:
        from_attributes = True


class FeedResponse(BaseModel):
    posts: List[FeedPost]
    next_cursor: Optional[str] = None
    has_more: bool = False


class LikeResponse(BaseModel):
    post_id: uuid.UUID
    liked: bool
    like_count: int
