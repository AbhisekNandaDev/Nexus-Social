import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/social_media_content",
)

# Convert plain postgresql:// URLs (from docker-compose env) to the asyncpg scheme
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

# Pool tuning — configurable via env vars so load tests can raise limits without
# a code change. Keep total (pool_size + max_overflow) <= Postgres max_connections.
# Default Postgres max_connections = 100; we cap at 50 to leave headroom for
# migrations, admin queries, and future services.
_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "20"))
_MAX_OVERFLOW = int(os.environ.get("DB_MAX_OVERFLOW", "30"))
_POOL_TIMEOUT = int(os.environ.get("DB_POOL_TIMEOUT", "10"))  # fail fast, not after 30s

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=_POOL_SIZE,
    max_overflow=_MAX_OVERFLOW,
    pool_timeout=_POOL_TIMEOUT,
    pool_recycle=3600,  # recycle idle connections after 1h to avoid stale sockets
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""
    async with SessionLocal() as session:
        yield session
