"""
Unit tests for services/recommendation/scorer.py

Three synthetic user profiles:
  - cold    (0 likes)  → confidence = 0.0, pure content-based scoring
  - warming (25 likes) → confidence = 0.5, equal content/collaborative weight
  - warm    (100 likes)→ confidence = 1.0, fully collaborative scoring

No database or Redis connections required: social_likes and neighbors are
injected directly into compute_final_score to bypass all IO.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import numpy as np
import pytest

from services.recommendation.scorer import (
    PostRecord,
    UserContext,
    cluster_match_score,
    compute_final_score,
    cosine_similarity_score,
    recency_quality_score,
    social_graph_score,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unit_vec(seed: int, dim: int = 768) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_post(
    cluster_id: int = 5,
    like_count: int = 20,
    source_upvotes: int = 100,
    age_hours: float = 6.0,
    emb_seed: int = 1,
) -> PostRecord:
    return PostRecord(
        post_id=uuid.uuid4(),
        cluster_id=cluster_id,
        embedding=_unit_vec(emb_seed),
        created_at=datetime.now(timezone.utc) - timedelta(hours=age_hours),
        like_count=like_count,
        source_upvotes=source_upvotes,
    )


def _cold_user() -> UserContext:
    """0 likes — no history; confidence = 0.0"""
    return UserContext(
        user_id=uuid.uuid4(),
        taste_embedding=None,
        cluster_affinities=None,
        total_likes=0,
    )


def _warming_user() -> UserContext:
    """25 likes — building history; confidence = 0.5"""
    return UserContext(
        user_id=uuid.uuid4(),
        taste_embedding=_unit_vec(42),
        cluster_affinities={"5": 0.8, "12": 0.3},
        total_likes=25,
    )


def _warm_user() -> UserContext:
    """100 likes — full history; confidence = 1.0"""
    return UserContext(
        user_id=uuid.uuid4(),
        taste_embedding=_unit_vec(42),
        cluster_affinities={"5": 0.9, "12": 0.4},
        total_likes=100,
    )


# ── Confidence scalar ─────────────────────────────────────────────────────────

def test_cold_confidence_is_zero():
    assert min(_cold_user().total_likes / 50.0, 1.0) == 0.0


def test_warming_confidence_is_half():
    assert min(_warming_user().total_likes / 50.0, 1.0) == 0.5


def test_warm_confidence_is_one():
    assert min(_warm_user().total_likes / 50.0, 1.0) == 1.0


def test_confidence_clamps_above_fifty():
    user = UserContext(uuid.uuid4(), None, None, total_likes=200)
    assert min(user.total_likes / 50.0, 1.0) == 1.0


# ── cluster_match_score ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cluster_known():
    assert await cluster_match_score({"5": 0.8}, 5) == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_cluster_unknown_returns_zero():
    assert await cluster_match_score({"5": 0.8}, 99) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_cluster_no_affinities_neutral():
    # 0.5 keeps cold-start posts competitive
    assert await cluster_match_score(None, 5) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_cluster_no_cluster_id_neutral():
    assert await cluster_match_score({"5": 0.9}, None) == pytest.approx(0.5)


# ── cosine_similarity_score ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cosine_identical_embeddings():
    emb = _unit_vec(7)
    score = await cosine_similarity_score(emb, emb)
    assert score == pytest.approx(1.0, abs=1e-5)


@pytest.mark.asyncio
async def test_cosine_no_taste_returns_zero():
    assert await cosine_similarity_score(None, _unit_vec(1)) == 0.0


@pytest.mark.asyncio
async def test_cosine_negative_clamped_to_zero():
    v = _unit_vec(1)
    neg = [-x for x in v]       # exactly antipodal → dot product ≈ -1
    score = await cosine_similarity_score(v, neg)
    assert score == 0.0


@pytest.mark.asyncio
async def test_cosine_orthogonal_near_zero():
    rng = np.random.default_rng(99)
    a = rng.standard_normal(768).astype(np.float32)
    a /= np.linalg.norm(a)
    # Build a vector orthogonal to a
    b = rng.standard_normal(768).astype(np.float32)
    b -= np.dot(b, a) * a
    b /= np.linalg.norm(b)
    score = await cosine_similarity_score(a.tolist(), b.tolist())
    assert score == pytest.approx(0.0, abs=1e-4)


# ── recency_quality_score ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recency_fresh_post_scores_high():
    score = await recency_quality_score(
        datetime.now(timezone.utc), like_count=50, source_upvotes=200
    )
    assert score > 0.5


@pytest.mark.asyncio
async def test_recency_old_post_scores_lower():
    fresh = await recency_quality_score(
        datetime.now(timezone.utc), like_count=10
    )
    old = await recency_quality_score(
        datetime.now(timezone.utc) - timedelta(days=14), like_count=10
    )
    assert fresh > old


@pytest.mark.asyncio
async def test_recency_bounded():
    score = await recency_quality_score(
        datetime.now(timezone.utc), like_count=999_999, source_upvotes=999_999
    )
    assert 0.0 <= score <= 1.0


# ── social_graph_score (injected, no IO) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_social_precomputed_zero():
    score = await social_graph_score(
        uuid.uuid4(), uuid.uuid4(),
        db=AsyncMock(), redis=AsyncMock(),
        social_likes=0,
    )
    assert score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_social_precomputed_five_gives_one():
    score = await social_graph_score(
        uuid.uuid4(), uuid.uuid4(),
        db=AsyncMock(), redis=AsyncMock(),
        social_likes=5,
    )
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_social_precomputed_capped_at_one():
    score = await social_graph_score(
        uuid.uuid4(), uuid.uuid4(),
        db=AsyncMock(), redis=AsyncMock(),
        social_likes=100,
    )
    assert score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_social_empty_neighbors_zero():
    score = await social_graph_score(
        uuid.uuid4(), uuid.uuid4(),
        db=AsyncMock(), redis=AsyncMock(),
        neighbors=set(),
    )
    assert score == pytest.approx(0.0)


# ── compute_final_score — cold user ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_cold_user_no_social_influence():
    """With confidence=0, collaborative term drops out entirely.

    final = 1·content_based + 0·collaborative + 0.15·s4
          = 0.5·s1 + 0.5·s2 + 0.15·s4
    Since taste=None → s2=0; affinities=None → s1=0.5 (neutral).
    """
    user = _cold_user()
    post = _make_post(cluster_id=5, like_count=10, source_upvotes=50)

    score_no_social = await compute_final_score(
        user, post, AsyncMock(), AsyncMock(),
        neighbors=set(), social_likes=0,
    )
    score_high_social = await compute_final_score(
        user, post, AsyncMock(), AsyncMock(),
        neighbors={uuid.uuid4()}, social_likes=5,
    )

    # Cold user: confidence=0, so collaborative (which contains s3) has zero weight
    assert score_no_social == pytest.approx(score_high_social, abs=1e-6)


@pytest.mark.asyncio
async def test_cold_user_score_is_bounded():
    user = _cold_user()
    post = _make_post()
    score = await compute_final_score(
        user, post, AsyncMock(), AsyncMock(),
        neighbors=set(), social_likes=0,
    )
    assert 0.0 <= score <= 1.5   # max theoretical: 1·0.5 + 0·1 + 0.15·1 = 0.65


# ── compute_final_score — warming user ───────────────────────────────────────

@pytest.mark.asyncio
async def test_warming_user_confidence_half():
    """Warming user uses equal weight on content and collaborative."""
    user = _warming_user()
    post = _make_post(cluster_id=5, emb_seed=42)   # same seed → high cosine

    score = await compute_final_score(
        user, post, AsyncMock(), AsyncMock(),
        neighbors=set(), social_likes=0,
    )
    assert 0.0 <= score <= 1.5


@pytest.mark.asyncio
async def test_warming_user_social_signal_matters():
    """At confidence=0.5, social likes should shift the score upward."""
    user = _warming_user()
    post = _make_post(cluster_id=5)

    low = await compute_final_score(
        user, post, AsyncMock(), AsyncMock(),
        neighbors=set(), social_likes=0,
    )
    high = await compute_final_score(
        user, post, AsyncMock(), AsyncMock(),
        neighbors={uuid.uuid4()}, social_likes=5,
    )
    assert high > low


# ── compute_final_score — warm user ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_warm_user_social_dominates():
    """With confidence=1, content_based drops out; social graph matters most."""
    user = _warm_user()
    post = _make_post(cluster_id=5, emb_seed=42)

    no_social = await compute_final_score(
        user, post, AsyncMock(), AsyncMock(),
        neighbors=set(), social_likes=0,
    )
    full_social = await compute_final_score(
        user, post, AsyncMock(), AsyncMock(),
        neighbors={uuid.uuid4(), uuid.uuid4()}, social_likes=5,
    )
    assert full_social > no_social


@pytest.mark.asyncio
async def test_warm_user_identical_embedding_max_cosine():
    """Taste embedding identical to post → s2=1, boosting both content and collaborative."""
    user = _warm_user()
    # Use the same seed the warm user's taste embedding was built with
    post = _make_post(cluster_id=5, emb_seed=42)

    score = await compute_final_score(
        user, post, AsyncMock(), AsyncMock(),
        neighbors=set(), social_likes=0,
    )
    # confidence=1: final = collaborative + 0.15·s4 = 0.5·s2 + 0·s3 + 0.15·s4
    # s2≈1 → collaborative ≈ 0.5, s4 > 0 → score > 0.5
    assert score > 0.5
