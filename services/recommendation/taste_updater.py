"""
Incremental taste-embedding updater.

Called as a background task after every like event — never in the request path.
Uses a running weighted average so we never recompute from scratch:

    new_taste = (old_taste * (n-1) + post_embedding) / n

where n = total_likes after this like is counted.

After averaging, the vector is re-normalised to unit length so it stays in
cosine similarity space (SigLIP embeddings are L2-normalised at generation time).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.post import PostEmbedding
from src.db.models.user import UserInterestProfile
from utils.logger import get_logger

logger = get_logger(__name__)


async def update_taste_embedding(
    user_id: uuid.UUID,
    post_id: uuid.UUID,
    db: AsyncSession,
) -> None:
    """Fetch the liked post's embedding and incrementally update the user profile.

    Idempotent-safe: if the post has no embedding or the profile row is missing,
    the function returns silently rather than raising.
    """
    profile: UserInterestProfile | None = await db.scalar(
        select(UserInterestProfile).where(UserInterestProfile.user_id == user_id)
    )
    if profile is None:
        logger.debug("taste_updater: no profile for user=%s, skipping", user_id)
        return

    post_emb_row: PostEmbedding | None = await db.scalar(
        select(PostEmbedding).where(PostEmbedding.post_id == post_id)
    )
    if post_emb_row is None or post_emb_row.embedding is None:
        logger.debug("taste_updater: no embedding for post=%s, skipping", post_id)
        return

    post_vec = np.asarray(post_emb_row.embedding, dtype=np.float64)

    # Increment first so n reflects the post we're incorporating
    profile.total_likes += 1
    n = profile.total_likes

    if profile.taste_embedding is None or n == 1:
        # First like: seed with the post's embedding directly
        new_vec = post_vec
    else:
        old_vec = np.asarray(profile.taste_embedding, dtype=np.float64)
        # Weighted running average: old history contributes (n-1) weight, new like contributes 1
        new_vec = (old_vec * (n - 1) + post_vec) / n

    # Re-normalise to unit sphere so cosine dot products remain valid
    norm = np.linalg.norm(new_vec)
    if norm > 0:
        new_vec = new_vec / norm

    profile.taste_embedding = new_vec.tolist()
    profile.updated_at = datetime.now(timezone.utc)

    await db.commit()
    logger.debug("taste_updater: updated taste for user=%s (n=%d)", user_id, n)
