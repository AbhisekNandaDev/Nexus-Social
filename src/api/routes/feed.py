"""
Feed API.

GET  /api/v1/feed                   — personalized feed (Redis-backed, page-based)
GET  /api/v1/feed/explore           — recent published posts filtered by content prefs
GET  /api/v1/feed/following         — chronological posts from followed users
POST /api/v1/feed/{post_id}/like    — like a post  (also triggers taste update)
DELETE /api/v1/feed/{post_id}/like  — unlike a post
"""
from __future__ import annotations

import json
import uuid
from typing import List, Optional, Sequence

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schema.feed import FeedPost, FeedResponse, LikeResponse
from src.core.dependencies import get_current_user
from src.db.models.post import Post
from src.db.models.social import Follow, Like
from src.db.models.user import User, UserPreference
from src.db.redis import get_redis
from src.db.session import SessionLocal, get_db
from utils.logger import get_logger

logger = get_logger(__name__)

api_router = APIRouter()

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 50

_VIOLENCE_ORDER = ["none", "mild", "moderate", "severe", "extreme"]


# ── Content preference helpers ─────────────────────────────────────────────────

def _content_filters(prefs: Optional[UserPreference]) -> list:
    """Return SQLAlchemy WHERE clauses based on the user's content preferences."""
    filters: list = [
        Post.status == "published",
        or_(Post.risk.not_in(["block"]), Post.risk.is_(None)),
    ]

    if prefs is None:
        filters.append(or_(Post.nudity_level.in_(["safe", "contextual_exempt"]), Post.nudity_level.is_(None)))
        filters.append(or_(Post.violence_level == "none", Post.violence_level.is_(None)))
        filters.append(or_(Post.self_harm_level == "none", Post.self_harm_level.is_(None)))
        return filters

    allowed_nudity = {"safe", "contextual_exempt"}
    if prefs.nsfw_enabled:
        if prefs.suggestive_enabled:
            allowed_nudity.add("suggestive")
        if prefs.partial_nudity_enabled:
            allowed_nudity.add("partial_nudity")
        if prefs.explicit_enabled:
            allowed_nudity.add("explicit_nudity")
    filters.append(or_(Post.nudity_level.in_(allowed_nudity), Post.nudity_level.is_(None)))

    max_v = prefs.violence_max_level or "none"
    max_idx = _VIOLENCE_ORDER.index(max_v) if max_v in _VIOLENCE_ORDER else 0
    allowed_v = _VIOLENCE_ORDER[: max_idx + 1]
    filters.append(or_(Post.violence_level.in_(allowed_v), Post.violence_level.is_(None)))

    if not prefs.self_harm_visible:
        filters.append(or_(Post.self_harm_level == "none", Post.self_harm_level.is_(None)))

    return filters


def _post_filter(posts: list[Post], prefs: Optional[UserPreference]) -> list[Post]:
    """Apply content preference rules as an in-process post-filter.

    This runs *after* fetching from the DB so it doesn't need SQL predicates —
    the ordering from Redis is preserved, and only disallowed posts are dropped.
    Illegal content (risk='block') is always removed regardless of prefs.
    """
    filtered: list[Post] = []
    for post in posts:
        # Hard block — illegal content, no override
        if post.risk == "block":
            continue

        if prefs is None:
            if post.nudity_level not in (None, "safe", "contextual_exempt"):
                continue
            if post.violence_level not in (None, "none"):
                continue
            if post.self_harm_level not in (None, "none"):
                continue
            filtered.append(post)
            continue

        # Nudity check
        allowed_nudity = {"safe", "contextual_exempt"}
        if prefs.nsfw_enabled:
            if prefs.suggestive_enabled:
                allowed_nudity.add("suggestive")
            if prefs.partial_nudity_enabled:
                allowed_nudity.add("partial_nudity")
            if prefs.explicit_enabled:
                allowed_nudity.add("explicit_nudity")
        if post.nudity_level is not None and post.nudity_level not in allowed_nudity:
            continue

        # Violence check
        max_v = prefs.violence_max_level or "none"
        max_idx = _VIOLENCE_ORDER.index(max_v) if max_v in _VIOLENCE_ORDER else 0
        if (
            post.violence_level is not None
            and post.violence_level in _VIOLENCE_ORDER
            and _VIOLENCE_ORDER.index(post.violence_level) > max_idx
        ):
            continue

        # Self-harm check
        if not prefs.self_harm_visible and post.self_harm_level not in (None, "none"):
            continue

        filtered.append(post)

    return filtered


# ── Social data helpers ────────────────────────────────────────────────────────

def _to_url(base_url: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return base_url.rstrip("/") + "/" + path.lstrip("/")


async def _fetch_author_names(
    db: AsyncSession, posts: Sequence[Post]
) -> dict[uuid.UUID, Optional[str]]:
    if not posts:
        return {}
    user_ids = list({p.user_id for p in posts})
    rows = await db.execute(
        select(User.id, User.display_name).where(User.id.in_(user_ids))
    )
    return {r.id: r.display_name for r in rows}


async def _attach_social(
    db: AsyncSession,
    posts: Sequence[Post],
    current_user_id: uuid.UUID,
    author_names: dict[uuid.UUID, Optional[str]],
    base_url: str = "",
) -> List[FeedPost]:
    if not posts:
        return []
    post_ids = [p.id for p in posts]

    like_count_rows = await db.execute(
        select(Like.post_id, func.count(Like.id).label("cnt"))
        .where(Like.post_id.in_(post_ids))
        .group_by(Like.post_id)
    )
    like_counts: dict[uuid.UUID, int] = {r.post_id: r.cnt for r in like_count_rows}

    user_like_rows = await db.execute(
        select(Like.post_id)
        .where(Like.user_id == current_user_id, Like.post_id.in_(post_ids))
    )
    user_liked: set[uuid.UUID] = {r.post_id for r in user_like_rows}

    result: List[FeedPost] = []
    for post in posts:
        result.append(FeedPost(
            post_id=post.id,
            user_id=post.user_id,
            author_display_name=author_names.get(post.user_id),
            media_type=post.media_type,
            media_url=_to_url(base_url, post.media_path),
            thumbnail_url=_to_url(base_url, post.thumbnail_path),
            caption=post.caption,
            display_tags=post.display_tags,
            mood=post.mood,
            scene_type=post.scene_type,
            nudity_level=post.nudity_level,
            source_platform=post.source_platform,
            source_subreddit=post.source_subreddit,
            created_at=post.created_at,
            like_count=like_counts.get(post.id, 0),
            is_liked=post.id in user_liked,
        ))
    return result


async def _fetch_posts_by_ids(
    post_ids: list[uuid.UUID],
    db: AsyncSession,
) -> list[Post]:
    """Fetch Post objects by ID list, preserving the original order."""
    if not post_ids:
        return []
    rows = await db.execute(select(Post).where(Post.id.in_(post_ids)))
    post_map = {p.id: p for p in rows.scalars()}
    return [post_map[pid] for pid in post_ids if pid in post_map]


# ── GET /feed — personalized, Redis-backed ────────────────────────────────────

@api_router.get("", response_model=FeedResponse, summary="Personalized feed")
async def get_feed(
    request: Request,
    page: int = Query(default=0, ge=0, description="Zero-based page index"),
    limit: int = Query(default=_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    refresh: bool = Query(default=False, description="Invalidate cached feed (pull-to-refresh)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis=Depends(get_redis),
) -> FeedResponse:
    from services.recommendation.feed_builder import build_feed

    cache_key = f"feed:{current_user.id}"

    # Pull-to-refresh: invalidate on explicit signal only, not on every new post
    if refresh:
        await redis.delete(cache_key)

    raw = await redis.get(cache_key)
    if raw is not None:
        all_post_ids = [uuid.UUID(pid) for pid in json.loads(raw)]
    else:
        # Cache miss: build synchronously so this request gets a real response
        all_post_ids = await build_feed(current_user.id, db, redis)

    # Page slice from the cached ordered list
    offset = page * limit
    page_ids = all_post_ids[offset: offset + limit]
    has_more = (offset + limit) < len(all_post_ids)

    if not page_ids:
        return FeedResponse(posts=[], has_more=has_more)

    # Fetch posts preserving Redis-determined order
    posts = await _fetch_posts_by_ids(page_ids, db)

    # Post-filter: apply content preferences after fetch (preserves ranking order)
    prefs: Optional[UserPreference] = await db.scalar(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    posts = _post_filter(posts, prefs)

    author_names = await _fetch_author_names(db, posts)
    feed_posts = await _attach_social(
        db, posts, current_user.id, author_names, str(request.base_url)
    )

    return FeedResponse(posts=feed_posts, has_more=has_more)


# ── GET /feed/explore — recent/trending ───────────────────────────────────────

@api_router.get("/explore", response_model=FeedResponse, summary="Explore feed — recent posts")
async def get_explore_feed(
    request: Request,
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FeedResponse:
    import base64

    def _decode(c: Optional[str]) -> int:
        if not c:
            return 0
        try:
            return max(0, int(json.loads(base64.b64decode(c.encode()).decode()).get("o", 0)))
        except Exception:
            return 0

    def _encode(o: int) -> str:
        return base64.b64encode(json.dumps({"o": o}).encode()).decode()

    offset = _decode(cursor)

    prefs = await db.scalar(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    stmt = (
        select(Post)
        .where(and_(*_content_filters(prefs)))
        .order_by(Post.created_at.desc())
        .offset(offset)
        .limit(limit + 1)
    )
    rows = await db.execute(stmt)
    posts = list(rows.scalars())
    has_more = len(posts) > limit
    posts = posts[:limit]

    author_names = await _fetch_author_names(db, posts)
    feed_posts = await _attach_social(db, posts, current_user.id, author_names, str(request.base_url))

    next_cursor = _encode(offset + limit) if has_more else None
    return FeedResponse(posts=feed_posts, next_cursor=next_cursor, has_more=has_more)


# ── GET /feed/following ───────────────────────────────────────────────────────

@api_router.get("/following", response_model=FeedResponse, summary="Following feed — chronological")
async def get_following_feed(
    request: Request,
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FeedResponse:
    import base64

    def _decode(c: Optional[str]) -> int:
        if not c:
            return 0
        try:
            return max(0, int(json.loads(base64.b64decode(c.encode()).decode()).get("o", 0)))
        except Exception:
            return 0

    def _encode(o: int) -> str:
        return base64.b64encode(json.dumps({"o": o}).encode()).decode()

    offset = _decode(cursor)

    prefs = await db.scalar(
        select(UserPreference).where(UserPreference.user_id == current_user.id)
    )
    stmt = (
        select(Post)
        .join(Follow, Follow.following_id == Post.user_id)
        .where(Follow.follower_id == current_user.id)
        .where(and_(*_content_filters(prefs)))
        .order_by(Post.created_at.desc())
        .offset(offset)
        .limit(limit + 1)
    )
    rows = await db.execute(stmt)
    posts = list(rows.scalars())
    has_more = len(posts) > limit
    posts = posts[:limit]

    author_names = await _fetch_author_names(db, posts)
    feed_posts = await _attach_social(db, posts, current_user.id, author_names, str(request.base_url))

    next_cursor = _encode(offset + limit) if has_more else None
    return FeedResponse(posts=feed_posts, next_cursor=next_cursor, has_more=has_more)


# ── POST /feed/{post_id}/like ──────────────────────────────────────────────────

async def _run_taste_update(user_id: uuid.UUID, post_id: uuid.UUID) -> None:
    """Background task: update taste embedding in its own DB session."""
    from services.recommendation.taste_updater import update_taste_embedding
    async with SessionLocal() as db:
        await update_taste_embedding(user_id, post_id, db)


@api_router.post(
    "/{post_id}/like",
    response_model=LikeResponse,
    status_code=status.HTTP_200_OK,
    summary="Like a post",
)
async def like_post(
    post_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LikeResponse:
    post = await db.get(Post, post_id)
    if post is None or post.status not in ("published", "needs_review"):
        raise HTTPException(status_code=404, detail="Post not found")

    existing = await db.scalar(
        select(Like).where(Like.user_id == current_user.id, Like.post_id == post_id)
    )
    if existing is None:
        db.add(Like(user_id=current_user.id, post_id=post_id))
        await db.commit()
        # Async: update taste embedding without blocking the response
        background_tasks.add_task(_run_taste_update, current_user.id, post_id)
        logger.debug("Like added | user=%s post=%s", current_user.id, post_id)

    count = await db.scalar(
        select(func.count(Like.id)).where(Like.post_id == post_id)
    )
    return LikeResponse(post_id=post_id, liked=True, like_count=count or 0)


# ── DELETE /feed/{post_id}/like ────────────────────────────────────────────────

@api_router.delete(
    "/{post_id}/like",
    response_model=LikeResponse,
    status_code=status.HTTP_200_OK,
    summary="Unlike a post",
)
async def unlike_post(
    post_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> LikeResponse:
    existing = await db.scalar(
        select(Like).where(Like.user_id == current_user.id, Like.post_id == post_id)
    )
    if existing is not None:
        await db.delete(existing)
        await db.commit()
        logger.debug("Like removed | user=%s post=%s", current_user.id, post_id)

    count = await db.scalar(
        select(func.count(Like.id)).where(Like.post_id == post_id)
    )
    return LikeResponse(post_id=post_id, liked=False, like_count=count or 0)
