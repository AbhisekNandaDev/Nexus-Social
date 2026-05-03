"""Run K-means on 768-dim SigLIP embeddings and store cluster centroids.

Usage:
    python scripts/recompute_clusters.py

K = min(50, post_count // 10), minimum 5.
Uses MiniBatchKMeans for speed. Stores centroids in cluster_centroids and
updates each post_embedding with cluster_id + cluster_distance.
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
from sklearn.cluster import MiniBatchKMeans
from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.orm import Session

from src.db.models.cluster import ClusterCentroid
from src.db.models.post import Post, PostEmbedding

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/social_media_content",
).replace("postgresql+asyncpg://", "postgresql://")


def main() -> None:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    # ── Load all non-null embeddings ──────────────────────────────────────────
    with Session(engine) as session:
        rows = session.execute(
            select(PostEmbedding.id, PostEmbedding.post_id, PostEmbedding.embedding)
            .where(PostEmbedding.embedding.is_not(None))
        ).all()

    if not rows:
        print("No embeddings found — run reembed_all_posts.py first.")
        return

    emb_ids    = [r.id      for r in rows]
    post_ids   = [r.post_id for r in rows]
    embeddings = np.array([r.embedding for r in rows], dtype=np.float32)

    n = len(embeddings)
    k = max(5, min(50, n // 10))
    print(f"Clustering {n} embeddings into K={k} clusters …")

    # ── K-means ───────────────────────────────────────────────────────────────
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_  # (K, 768)

    # L2-normalize centroids for cosine-similarity alignment
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    centroids = centroids / norms

    # Distances from each point to its centroid
    distances = np.linalg.norm(embeddings - centroids[labels], axis=1)

    # ── Gather representative tags per cluster ────────────────────────────────
    with Session(engine) as session:
        post_tags: dict = {}
        for pid in post_ids:
            post = session.get(Post, pid)
            post_tags[pid] = post.display_tags or [] if post else []

    cluster_tag_counters: dict[int, Counter] = {c: Counter() for c in range(k)}
    for i, label in enumerate(labels):
        for tag in post_tags.get(post_ids[i], []):
            cluster_tag_counters[int(label)][tag] += 1

    # ── Write cluster centroids ───────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    post_counts = Counter(labels.tolist())

    with Session(engine) as session:
        session.execute(delete(ClusterCentroid))
        for c in range(k):
            top_tags = [tag for tag, _ in cluster_tag_counters[c].most_common(5)]
            session.add(ClusterCentroid(
                id=c,
                centroid=centroids[c].tolist(),
                post_count=post_counts[c],
                representative_tags=top_tags,
                updated_at=now,
            ))
        session.commit()

    # ── Update post_embeddings with cluster assignments ───────────────────────
    with Session(engine) as session:
        for i, emb_id in enumerate(emb_ids):
            pe = session.get(PostEmbedding, emb_id)
            if pe:
                pe.cluster_id       = int(labels[i])
                pe.cluster_distance = float(distances[i])
        session.commit()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'Cluster':>7}  {'Posts':>6}  Top tags")
    print("-" * 55)
    for c in range(k):
        tags = ", ".join(t for t, _ in cluster_tag_counters[c].most_common(5)) or "—"
        print(f"  {c:>5}  {post_counts[c]:>6}  {tags}")

    print(f"\nStored {k} centroids. Done.")


if __name__ == "__main__":
    main()
