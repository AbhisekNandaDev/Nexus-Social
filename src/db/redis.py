import redis.asyncio as aioredis
from redis.asyncio import Redis

from src.core.config import REDIS_URL

# Module-level pool — initialised at startup, shared across all requests.
_redis: Redis | None = None


async def init_redis() -> None:
    global _redis
    _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    await _redis.ping()


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


async def get_redis() -> Redis:
    """FastAPI dependency that returns the shared Redis client."""
    return _redis
