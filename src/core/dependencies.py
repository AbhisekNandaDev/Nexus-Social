import uuid

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import decode_access_token
from src.db.models.user import User
from src.db.redis import get_redis
from src.db.session import get_db

# tokenUrl points to the login endpoint so Swagger UI's "Authorize" button works.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    redis: Redis = Depends(get_redis),
    db: AsyncSession = Depends(get_db),
) -> User:
    payload = decode_access_token(token)
    jti: str = payload["jti"]
    user_id: str = payload["sub"]

    # Fast revocation check — O(1) Redis lookup before touching Postgres.
    if await redis.exists(f"blacklist:jti:{jti}"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await db.get(User, uuid.UUID(user_id))
    if not user or not user.is_active or user.is_synthetic:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user
