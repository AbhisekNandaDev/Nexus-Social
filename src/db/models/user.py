import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    hashed_password: Mapped[str | None] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    avatar_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    source_platform: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source_username: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    posts: Mapped[list["Post"]] = relationship(back_populates="user")  # noqa: F821
    preference: Mapped["UserPreference"] = relationship(back_populates="user", uselist=False)
    interest_profile: Mapped["UserInterestProfile"] = relationship(back_populates="user", uselist=False)
    following: Mapped[list["Follow"]] = relationship(  # noqa: F821
        foreign_keys="Follow.follower_id", back_populates="follower"
    )
    followers: Mapped[list["Follow"]] = relationship(  # noqa: F821
        foreign_keys="Follow.following_id", back_populates="following_user"
    )
    likes: Mapped[list["Like"]] = relationship(back_populates="user")  # noqa: F821

    __table_args__ = (Index("ix_users_email", "email"),)


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), unique=True
    )
    nsfw_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    suggestive_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    partial_nudity_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    explicit_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    suggestive_subcategories: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    partial_subcategories: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    explicit_subcategories: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    violence_max_level: Mapped[str] = mapped_column(String(20), default="none")
    self_harm_visible: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped["User"] = relationship(back_populates="preference")


class UserInterestProfile(Base):
    __tablename__ = "user_interest_profiles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), unique=True
    )
    taste_embedding: Mapped[list | None] = mapped_column(Vector(384), nullable=True)
    cluster_affinities: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    top_display_tags: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    total_likes: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="interest_profile")
