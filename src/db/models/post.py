import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base


class Post(Base):
    __tablename__ = "posts"

    # ── Core fields ───────────────────────────────────────────────────────────
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    media_type: Mapped[str] = mapped_column(String(10))           # "image" | "video"
    media_path: Mapped[str] = mapped_column(String(500))
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="uploaded")

    # ── Safety classification ─────────────────────────────────────────────────
    nudity_level: Mapped[str | None] = mapped_column(String(30), nullable=True)
    nsfw_subcategories: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    violence_level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    violence_type: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    self_harm_level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    self_harm_type: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    age_group: Mapped[str | None] = mapped_column(String(10), nullable=True)
    risk: Mapped[str | None] = mapped_column(String(10), nullable=True)
    classification_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Content understanding ─────────────────────────────────────────────────
    content_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    display_tags: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    mood: Mapped[str | None] = mapped_column(String(20), nullable=True)
    scene_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    text_in_image: Mapped[str | None] = mapped_column(Text, nullable=True)
    objects_detected: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    people_count: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # ── DeepFace output ───────────────────────────────────────────────────────
    deepface_age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deepface_age_group: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # ── Video-specific fields ─────────────────────────────────────────────────
    video_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    frames_analyzed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_calls_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript_language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    transcript_safety_flags: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    secondary_classifications: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # ── Source metadata (seeded content) ──────────────────────────────────────
    source_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    source_platform: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_subreddit: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_upvotes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_comments: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    classified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="posts")  # noqa: F821
    embedding: Mapped["PostEmbedding"] = relationship(back_populates="post", uselist=False)
    frame_results: Mapped[list["PostFrameResult"]] = relationship(back_populates="post")
    likes: Mapped[list["Like"]] = relationship(back_populates="post")  # noqa: F821

    __table_args__ = (
        Index("ix_posts_user_id", "user_id"),
        Index("ix_posts_status", "status"),
        Index("ix_posts_nudity_level", "nudity_level"),
        Index("ix_posts_risk", "risk"),
        Index("ix_posts_created_at", "created_at"),
    )


class PostEmbedding(Base):
    __tablename__ = "post_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("posts.id"), unique=True, nullable=False
    )
    embedding: Mapped[list] = mapped_column(Vector(768), nullable=False)
    cluster_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cluster_centroids.id"), nullable=True
    )
    cluster_distance: Mapped[float | None] = mapped_column(Float, nullable=True)

    post: Mapped["Post"] = relationship(back_populates="embedding")
    cluster: Mapped["ClusterCentroid"] = relationship(back_populates="post_embeddings")  # noqa: F821

    __table_args__ = (
        Index("ix_post_embeddings_cluster_id", "cluster_id"),
        # HNSW index for fast cosine similarity search
        Index(
            "ix_post_embeddings_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )


class PostFrameResult(Base):
    """Per-frame classification results for video posts."""

    __tablename__ = "post_frame_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("posts.id"), nullable=False
    )
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    selection_reason: Mapped[str | None] = mapped_column(String(30), nullable=True)
    classification: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    post: Mapped["Post"] = relationship(back_populates="frame_results")

    __table_args__ = (Index("ix_post_frame_results_post_id", "post_id"),)
