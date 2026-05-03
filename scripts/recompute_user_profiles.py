"""Rebuild taste embeddings for all users with likes using new SigLIP embeddings.

Usage:
    python scripts/recompute_user_profiles.py

For each user with ≥1 like:
  - Loads liked posts' embeddings
  - Weighted average with exponential 14-day time-decay
  - Computes cluster_affinities (top 10 clusters by like count)
  - Computes top_display_tags (top 10 tags from liked posts)
  - Upserts user_interest_profiles
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(Path(_PROJECT_ROOT) / ".env")

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from src.db.models.post import Post, PostEmbedding
from src.db.models.social import Like
from src.db.models.user import User, UserInterestProfile

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/social_media_content",
).replace("postgresql+asyncpg://", "postgresql://")

HALF_LIFE_DAYS = 14.0


def _time_weight(liked_at: datetime | None, now: datetime) -> float:
    if liked_at is None:
        return 1.0
    age_days = (now - liked_at.replace(tzinfo=timezone.utc if liked_at.tzinfo is None else liked_at.tzinfo)).total_seconds() / 86400
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def main() -> None:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    now = datetime.now(timezone.utc)

    with Session(engine) as session:
        # All users who have at least one like
        liked_user_ids = session.scalars(
            select(Like.user_id).distinct()
        ).all()

    print(f"Found {len(liked_user_ids)} users with likes.")
    updated = 0

    for user_id in liked_user_ids:
        with Session(engine) as session:
            likes = session.scalars(
                select(Like).where(Like.user_id == user_id)
            ).all()

            if not likes:
                continue

            embeddings: list[list[float]] = []
            weights: list[float] = []
            tag_counter: Counter = Counter()
            cluster_counter: Counter = Counter()

            for like in likes:
                post = session.get(Post, like.post_id)
                if post is None:
                    continue

                pe = session.scalar(
                    select(PostEmbedding).where(PostEmbedding.post_id == like.post_id)
                )
                if pe is None or pe.embedding is None:
                    continue

                w = _time_weight(getattr(like, "created_at", None), now)
                embeddings.append(pe.embedding)
                weights.append(w)

                for tag in (post.display_tags or []):
                    tag_counter[tag] += 1

                if pe.cluster_id is not None:
                    cluster_counter[pe.cluster_id] += 1

            if not embeddings:
                continue

            # Weighted average taste embedding
            arr = np.array(embeddings, dtype=np.float32)
            w_arr = np.array(weights, dtype=np.float32)
            taste = (arr * w_arr[:, np.newaxis]).sum(axis=0)
            norm = np.linalg.norm(taste)
            if norm > 0:
                taste = taste / norm

            # Cluster affinities: top 10 as percentage dict
            total_likes = sum(cluster_counter.values())
            cluster_affinities = {
                str(cid): round(cnt / total_likes, 4)
                for cid, cnt in cluster_counter.most_common(10)
            }

            top_tags = [tag for tag, _ in tag_counter.most_common(10)]

            # Upsert UserInterestProfile
            profile = session.scalar(
                select(UserInterestProfile).where(UserInterestProfile.user_id == user_id)
            )
            if profile is None:
                profile = UserInterestProfile(user_id=user_id)
                session.add(profile)

            profile.taste_embedding    = taste.tolist()
            profile.cluster_affinities = cluster_affinities
            profile.top_display_tags   = top_tags
            profile.total_likes        = len(likes)
            profile.updated_at         = now
            session.commit()
            updated += 1

    print(f"Updated {updated} user interest profiles. Done.")


if __name__ == "__main__":
    main()
