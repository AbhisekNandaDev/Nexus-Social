"""
Recommendation scoring engine.

Combines four signals into a single final_score per candidate post:

  s1 = cluster_match_score   — how well the post's cluster fits user affinities
  s2 = cosine_similarity     — semantic distance in SigLIP embedding space
  s3 = social_graph_score    — engagement from users within 3 social hops
  s4 = recency_quality_score — freshness × engagement quality

Confidence scalar drives the cold→warm transition:

  confidence = min(total_likes / 50, 1.0)

  content_based  = 0.5·s1 + 0.5·s2   # pure content signals, no social needed
  collaborative  = 0.5·s2 + 0.5·s3   # blends semantic + social graph

  final_score = (1 - confidence) · content_based   # cold: ignore sparse social data
              + confidence       · collaborative    # warm: lean on social graph
              + 0.15             · s4               # recency always contributes

As the user accumulates likes (0→50), we smoothly shift from content-only to
collaborative scoring — avoiding noise from an empty social graph.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_GRAPH_CACHE_TTL = 300  # 5 min — match max social latency budget


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class PostRecord:
    post_id: uuid.UUID
    cluster_id: Optional[int]
    embedding: list[float]          # 768-dim, L2-normalised SigLIP vector
    created_at: datetime
    like_count: int                 # live count from the likes table
    source_upvotes: Optional[int] = None


@dataclass
class UserContext:
    user_id: uuid.UUID
    taste_embedding: Optional[list[float]]      # 768-dim, L2-normalised; None for cold users
    cluster_affinities: Optional[dict[str, float]]  # str(cluster_id) → affinity ∈ [0,1]
    total_likes: int


# ── Signal 1: Cluster affinity ────────────────────────────────────────────────

async def cluster_match_score(
    affinities: Optional[dict[str, float]],
    cluster_id: Optional[int],
) -> float:
    """Return the user's pre-computed affinity for the post's visual cluster.

    Returns 0.5 (neutral) when either side lacks data — this keeps cold-start
    posts competitive rather than penalising unclassified content.
    """
    if not affinities or cluster_id is None:
        return 0.5
    return float(affinities.get(str(cluster_id), 0.0))


# ── Signal 2: Semantic cosine similarity ─────────────────────────────────────

async def cosine_similarity_score(
    taste: Optional[list[float]],
    post_embedding: list[float],
) -> float:
    """Dot product on L2-normalised vectors equals cosine similarity.

    Clamped to [0, 1]: semantically opposite content yields negative dot products
    which we treat as zero rather than a negative reward.
    """
    if taste is None:
        return 0.0
    ta = np.asarray(taste, dtype=np.float32)
    pa = np.asarray(post_embedding, dtype=np.float32)
    return float(max(0.0, np.dot(ta, pa)))


# ── Signal 3: Social graph ────────────────────────────────────────────────────

async def _get_social_neighbors(
    user_id: uuid.UUID,
    db: AsyncSession,
    redis,
    hops: int = 3,
) -> set[uuid.UUID]:
    """Return UUIDs reachable within `hops` follow edges; cached 5 min in Redis."""
    cache_key = f"social:neighbors:{user_id}:{hops}"
    cached = await redis.get(cache_key)
    if cached:
        return {uuid.UUID(uid) for uid in json.loads(cached)}

    # Recursive CTE expands the follow graph level by level up to `hops`.
    # UNION (not UNION ALL) deduplicates cycles in dense follow graphs.
    sql = text("""
        WITH RECURSIVE neighbors AS (
            SELECT following_id AS uid, 1 AS depth
            FROM follows
            WHERE follower_id = :user_id

            UNION

            SELECT f.following_id AS uid, n.depth + 1
            FROM follows f
            INNER JOIN neighbors n ON f.follower_id = n.uid
            WHERE n.depth < :hops
        )
        SELECT DISTINCT uid FROM neighbors WHERE uid != :user_id
    """)
    rows = await db.execute(sql, {"user_id": user_id, "hops": hops})
    neighbor_ids: set[uuid.UUID] = {row.uid for row in rows}

    await redis.set(
        cache_key,
        json.dumps([str(uid) for uid in neighbor_ids]),
        ex=_GRAPH_CACHE_TTL,
    )
    return neighbor_ids


async def social_graph_score(
    user_id: uuid.UUID,
    post_id: uuid.UUID,
    db: AsyncSession,
    redis,
    hops: int = 3,
    neighbors: Optional[set[uuid.UUID]] = None,
    social_likes: Optional[int] = None,
) -> float:
    """Score how strongly the post is endorsed within the user's social graph.

    `social_likes` can be injected from a batch pre-computation in the feed
    builder to avoid 200 individual DB round-trips during a single build.
    `neighbors` similarly avoids re-fetching the graph per post.

    Normalises to [0, 1] — 5+ social likes yields a full score of 1.0.
    """
    if social_likes is not None:
        return min(social_likes / 5.0, 1.0)

    if neighbors is None:
        neighbors = await _get_social_neighbors(user_id, db, redis, hops)

    if not neighbors:
        return 0.0

    sql = text("""
        SELECT COUNT(*) FROM likes
        WHERE post_id = :post_id
          AND user_id = ANY(:neighbor_ids)
    """)
    result = await db.execute(
        sql,
        {"post_id": post_id, "neighbor_ids": [str(uid) for uid in neighbors]},
    )
    count = result.scalar() or 0
    return min(count / 5.0, 1.0)


# ── Signal 4: Recency × quality ───────────────────────────────────────────────

async def recency_quality_score(
    created_at: datetime,
    like_count: int,
    source_upvotes: Optional[int] = None,
) -> float:
    """Combine exponential freshness decay with a log-scaled engagement signal.

    Half-life: 48 h — a two-day-old post retains 50 % of its recency score.
    Engagement caps near 1.0 at 1 000 upvotes/likes (log scale prevents viral
    posts from drowning out everything else).
    """
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age_hours = max(0.0, (now - created_at).total_seconds() / 3600.0)

    recency = float(np.exp(-age_hours / 48.0))

    engagement = source_upvotes if source_upvotes is not None else like_count
    quality = float(np.log1p(engagement) / np.log1p(1000))
    quality = min(quality, 1.0)

    return 0.6 * recency + 0.4 * quality


# ── Final score combiner ──────────────────────────────────────────────────────

async def compute_final_score(
    user: UserContext,
    post: PostRecord,
    db: AsyncSession,
    redis,
    neighbors: Optional[set[uuid.UUID]] = None,
    social_likes: Optional[int] = None,
) -> float:
    """Combine all four signals using confidence-weighted blending.

    confidence ramps 0→1 as the user builds engagement history (saturates at
    50 likes). Cold users get pure content-based scores; warm users benefit
    from social graph signals that are only meaningful once populated.
    """
    s1 = await cluster_match_score(user.cluster_affinities, post.cluster_id)
    s2 = await cosine_similarity_score(user.taste_embedding, post.embedding)
    s3 = await social_graph_score(
        user.user_id, post.post_id, db, redis,
        neighbors=neighbors, social_likes=social_likes,
    )
    s4 = await recency_quality_score(post.created_at, post.like_count, post.source_upvotes)

    # confidence ramps 0→1 as the user builds engagement history (saturates at 50 likes)
    confidence = min(user.total_likes / 50.0, 1.0)

    content_based = 0.5 * s1 + 0.5 * s2
    collaborative = 0.5 * s2 + 0.5 * s3

    return (
        (1.0 - confidence) * content_based
        + confidence * collaborative
        + 0.15 * s4
    )
