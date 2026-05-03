"""
Feed builder: pulls ANN candidates, scores them, applies diversity, writes to Redis.

Designed to run as a FastAPI BackgroundTask — not in the request path.
On a cold cache miss the caller invokes build_feed() synchronously and waits.

Flow:
  1. Fetch top 200 ANN candidates via pgvector approximate search.
  2. Batch-fetch social like counts for all 200 posts (single query).
  3. Score every candidate with the full four-signal scorer.
  4. Apply cluster-diversity filter.
  5. Serialise final post_id order to Redis (TTL 30 min).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from services.recommendation.diversity import ScoredPost, apply_diversity
from services.recommendation.scorer import (
    PostRecord,
    UserContext,
    _get_social_neighbors,
    compute_final_score,
)
from src.db.models.post import Post, PostEmbedding
from src.db.models.social import Like
from src.db.models.user import UserInterestProfile
from utils.logger import get_logger

logger = get_logger(__name__)

_FEED_TTL = 1800        # 30 minutes
_CANDIDATE_COUNT = 200


# ── ANN candidate fetch ───────────────────────────────────────────────────────

async def _fetch_ann_candidates(
    taste_embedding: Optional[list[float]],
    db: AsyncSession,
) -> list[PostRecord]:
    """Pull the top 200 posts closest to the user's taste in embedding space.

    Uses pgvector's <=> cosine-distance operator with the HNSW index.
    SET ivfflat.probes = 10 is included for forward-compatibility if the index
    is later switched to IVFFlat; it is a no-op against the current HNSW index.
    """
    # Probe configuration — harmless no-op when index type is HNSW
    await db.execute(text("SET LOCAL ivfflat.probes = 10"))

    if taste_embedding is not None:
        # ANN: order by cosine distance from the user's taste vector
        stmt = (
            select(
                PostEmbedding.post_id,
                PostEmbedding.cluster_id,
                PostEmbedding.embedding,
                Post.created_at,
                Post.source_upvotes,
            )
            .join(Post, Post.id == PostEmbedding.post_id)
            .where(Post.status == "published")
            .order_by(PostEmbedding.embedding.cosine_distance(taste_embedding))
            .limit(_CANDIDATE_COUNT)
        )
    else:
        # Cold-start fallback: most recent published posts
        stmt = (
            select(
                PostEmbedding.post_id,
                PostEmbedding.cluster_id,
                PostEmbedding.embedding,
                Post.created_at,
                Post.source_upvotes,
            )
            .join(Post, Post.id == PostEmbedding.post_id)
            .where(Post.status == "published")
            .order_by(Post.created_at.desc())
            .limit(_CANDIDATE_COUNT)
        )

    rows = await db.execute(stmt)
    return [
        PostRecord(
            post_id=row.post_id,
            cluster_id=row.cluster_id,
            embedding=row.embedding,
            created_at=row.created_at,
            like_count=0,           # filled in by _attach_like_counts
            source_upvotes=row.source_upvotes,
        )
        for row in rows
    ]


async def _attach_like_counts(
    posts: list[PostRecord],
    db: AsyncSession,
) -> None:
    """Populate like_count on each PostRecord in a single batch query."""
    if not posts:
        return
    post_ids = [p.post_id for p in posts]
    rows = await db.execute(
        select(Like.post_id, func.count(Like.id).label("cnt"))
        .where(Like.post_id.in_(post_ids))
        .group_by(Like.post_id)
    )
    counts = {r.post_id: r.cnt for r in rows}
    for post in posts:
        post.like_count = counts.get(post.post_id, 0)


async def _batch_social_likes(
    post_ids: list[uuid.UUID],
    neighbor_ids: set[uuid.UUID],
    db: AsyncSession,
) -> dict[uuid.UUID, int]:
    """Count likes from social-graph neighbors for all candidates in one query."""
    if not neighbor_ids or not post_ids:
        return {}
    rows = await db.execute(
        select(Like.post_id, func.count(Like.id).label("cnt"))
        .where(
            Like.post_id.in_(post_ids),
            Like.user_id.in_(neighbor_ids),
        )
        .group_by(Like.post_id)
    )
    return {r.post_id: r.cnt for r in rows}


# ── Main entry point ──────────────────────────────────────────────────────────

async def build_feed(
    user_id: uuid.UUID,
    db: AsyncSession,
    redis,
) -> list[uuid.UUID]:
    """Score, diversify, and cache a ranked post_id list for the user.

    Returns the ordered post_id list so the calling route can serve it
    immediately on a synchronous cache miss.
    """
    profile: Optional[UserInterestProfile] = await db.scalar(
        select(UserInterestProfile).where(UserInterestProfile.user_id == user_id)
    )

    taste = profile.taste_embedding if profile else None
    affinities = profile.cluster_affinities if profile else None
    total_likes = profile.total_likes if profile else 0

    user_ctx = UserContext(
        user_id=user_id,
        taste_embedding=taste,
        cluster_affinities=affinities,
        total_likes=total_likes,
    )

    # 1. ANN candidate fetch
    candidates = await _fetch_ann_candidates(taste, db)
    if not candidates:
        logger.warning("feed_builder: no candidates found for user=%s", user_id)
        return []

    # 2. Attach live like counts (batch)
    await _attach_like_counts(candidates, db)

    # 3. Pre-fetch social graph (cached) and batch social likes
    neighbors = await _get_social_neighbors(user_id, db, redis)
    post_ids = [c.post_id for c in candidates]
    social_counts = await _batch_social_likes(post_ids, neighbors, db)

    # 4. Score all candidates
    scored: list[ScoredPost] = []
    for post in candidates:
        s_likes = social_counts.get(post.post_id, 0)
        final = await compute_final_score(
            user_ctx, post, db, redis,
            neighbors=neighbors,
            social_likes=s_likes,
        )
        scored.append(ScoredPost(
            post_id=post.post_id,
            cluster_id=post.cluster_id,
            final_score=final,
        ))

    # 5. Diversity filter
    diverse = apply_diversity(scored)

    # 6. Write to Redis (TTL 30 min)
    ordered_ids = [str(p.post_id) for p in diverse]
    cache_key = f"feed:{user_id}"
    await redis.set(cache_key, json.dumps(ordered_ids), ex=_FEED_TTL)
    logger.info(
        "feed_builder: cached %d posts for user=%s", len(ordered_ids), user_id
    )
    return [p.post_id for p in diverse]
