from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.base import Base


class ClusterCentroid(Base):
    __tablename__ = "cluster_centroids"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # 0 to K-1
    centroid: Mapped[list] = mapped_column(Vector(768), nullable=False)
    post_count: Mapped[int] = mapped_column(Integer, default=0)
    representative_tags: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    post_embeddings: Mapped[list["PostEmbedding"]] = relationship(back_populates="cluster")  # noqa: F821
