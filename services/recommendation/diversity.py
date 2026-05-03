"""
Diversity filter applied to scored candidates before writing to Redis.

Algorithm:
  1. Sort candidates descending by final_score.
  2. Walk the sorted list; each time a cluster repeats, multiply that post's
     score by 0.7^repeat_count (compounding penalty per additional repeat).
  3. Hard cap: once a cluster has contributed 5 posts to this page, all further
     posts from that cluster are dropped entirely.
  4. Re-sort after penalties so the final ordering reflects adjusted scores.

The compounding 0.7× penalty progressively deprioritises over-represented
clusters while still allowing the best posts from any cluster to surface.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

_REPEAT_PENALTY = 0.7
_MAX_PER_CLUSTER = 5    # hard cap per page (20 posts)


@dataclass
class ScoredPost:
    post_id: uuid.UUID
    cluster_id: Optional[int]
    final_score: float


def apply_diversity(
    posts: list[ScoredPost],
    page_size: int = 20,
) -> list[ScoredPost]:
    """Return a re-ranked list with cluster diversity enforced.

    `page_size` is informational — the hard cap is applied globally over the
    full candidate list so that paginating the result still respects the cap.
    """
    posts = sorted(posts, key=lambda p: p.final_score, reverse=True)

    # cluster_id → how many times it has appeared so far
    cluster_seen: dict[Optional[int], int] = {}
    penalised: list[ScoredPost] = []

    for post in posts:
        count = cluster_seen.get(post.cluster_id, 0)

        if count >= _MAX_PER_CLUSTER:
            continue  # hard cap — skip entirely

        if count > 0:
            # 0.7^1 for the 2nd occurrence, 0.7^2 for the 3rd, etc.
            adjusted = post.final_score * (_REPEAT_PENALTY ** count)
            post = ScoredPost(
                post_id=post.post_id,
                cluster_id=post.cluster_id,
                final_score=adjusted,
            )

        cluster_seen[post.cluster_id] = count + 1
        penalised.append(post)

    return sorted(penalised, key=lambda p: p.final_score, reverse=True)
