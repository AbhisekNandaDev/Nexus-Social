"""Migrate embeddings: Ollama text (nomic-embed-text) → SigLIP visual (768-dim).

Nulls all existing embeddings (computed from text descriptions — invalid for
the new visual encoder) and resets cluster data so recompute scripts start
clean. Column types are already vector(768) from the initial schema; if the
DB was created from an older schema at vector(384) the ALTER TABLE statements
will upgrade them.

Revision ID: 001
Revises:
Create Date: 2026-04-22
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # ── 1. Drop HNSW index (must be dropped before altering column type) ───────
    conn.execute(text("DROP INDEX IF EXISTS ix_post_embeddings_hnsw"))

    # ── 2. Detect current vector dimension for post_embeddings.embedding ───────
    row = conn.execute(text("""
        SELECT atttypmod
        FROM   pg_attribute
        WHERE  attrelid = 'post_embeddings'::regclass
          AND  attname  = 'embedding'
          AND  attnum   > 0
    """)).fetchone()

    current_dim = (row[0] - 1) if row and row[0] and row[0] > 0 else None

    # ── 3. Alter column types if still at old dimension ────────────────────────
    if current_dim is not None and current_dim != 768:
        # NULL out first so the USING cast doesn't fail on mismatched dims
        conn.execute(text("UPDATE post_embeddings          SET embedding       = NULL"))
        conn.execute(text("UPDATE user_interest_profiles   SET taste_embedding = NULL"))
        conn.execute(text("UPDATE cluster_centroids        SET centroid        = NULL"))

        conn.execute(text(
            "ALTER TABLE post_embeddings        "
            "ALTER COLUMN embedding       TYPE vector(768)"
        ))
        conn.execute(text(
            "ALTER TABLE user_interest_profiles "
            "ALTER COLUMN taste_embedding TYPE vector(768)"
        ))
        conn.execute(text(
            "ALTER TABLE cluster_centroids      "
            "ALTER COLUMN centroid        TYPE vector(768)"
        ))
    else:
        # Columns already vector(768) — just zero out stale text-based values
        conn.execute(text("UPDATE post_embeddings SET embedding = NULL"))
        conn.execute(text("UPDATE user_interest_profiles SET taste_embedding = NULL"))

    # ── 4. Reset cluster assignments on post_embeddings ───────────────────────
    conn.execute(text(
        "UPDATE post_embeddings "
        "SET cluster_id = NULL, cluster_distance = NULL"
    ))

    # ── 5. Clear cluster centroids (will be recomputed) ───────────────────────
    conn.execute(text("DELETE FROM cluster_centroids"))

    # ── 6. Reset user interest profiles ───────────────────────────────────────
    conn.execute(text(
        "UPDATE user_interest_profiles "
        "SET cluster_affinities = '{}'"
    ))

    # ── 7. Recreate HNSW index for cosine similarity search ───────────────────
    conn.execute(text("""
        CREATE INDEX ix_post_embeddings_hnsw
        ON post_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """))


def downgrade() -> None:
    # Downgrade is destructive — embeddings cannot be recovered.
    # Drop and recreate index only; data loss is accepted.
    conn = op.get_bind()
    conn.execute(text("DROP INDEX IF EXISTS ix_post_embeddings_hnsw"))
    conn.execute(text("UPDATE post_embeddings SET embedding = NULL, cluster_id = NULL, cluster_distance = NULL"))
    conn.execute(text("DELETE FROM cluster_centroids"))
    conn.execute(text("UPDATE user_interest_profiles SET taste_embedding = NULL, cluster_affinities = '{}'"))
    conn.execute(text("""
        CREATE INDEX ix_post_embeddings_hnsw
        ON post_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """))
