import asyncio
import hashlib
import secrets
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import HTTPException, status
from jose import JWTError, jwt

from src.core.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    REFRESH_TOKEN_EXPIRE_DAYS,
)

# bcrypt is CPU-bound and synchronous — running it directly in an async handler
# blocks the entire event loop for ~200-400ms per call.
# This pool offloads bcrypt to threads, keeping the event loop free.
# max_workers is intentionally small: bcrypt is CPU-saturating so more threads
# than cores provides no throughput gain and causes thrashing.
_bcrypt_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bcrypt")

# Pre-hashed dummy used in the constant-time login rejection path.
# Computed once at startup — never on the hot path.
_DUMMY_HASH: str = bcrypt.hashpw(
    b"__dummy_password_never_matches__",
    bcrypt.gensalt(rounds=12),
).decode("utf-8")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


async def hash_password_async(password: str) -> str:
    """Non-blocking bcrypt hash — runs in the thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_bcrypt_pool, hash_password, password)


async def verify_password_async(plain: str, hashed: str) -> bool:
    """Non-blocking bcrypt verify — runs in the thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_bcrypt_pool, verify_password, plain, hashed)


def create_access_token(user_id: str) -> tuple[str, str]:
    """Returns (encoded_jwt, jti)."""
    jti = str(uuid.uuid4())
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": user_id, "jti": jti, "exp": expire, "type": "access"}
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token, jti


def create_refresh_token() -> tuple[str, str]:
    """Returns (raw_token, sha256_hash). Store the hash; send the raw token to the client."""
    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, token_hash


def decode_access_token(token: str) -> dict:
    """Decodes and validates an access token. Raises 401 on any failure."""
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise credentials_exc
        return payload
    except JWTError:
        raise credentials_exc


def refresh_token_expires_at() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)


def refresh_token_redis_ttl() -> int:
    """TTL in seconds for storing a refresh token hash in Redis."""
    return REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60


def dummy_hash() -> str:
    """Returns the pre-computed dummy hash for constant-time login rejection."""
    return _DUMMY_HASH
