import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schema.auth import (
    LoginRequest,
    LoginResponse,
    LogoutRequest,
    MessageResponse,
    RefreshRequest,
    RefreshResponse,
    RegisterRequest,
    RegisterResponse,
    UserResponse,
)
from src.core.dependencies import get_current_user, oauth2_scheme
from src.core.security import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    dummy_hash,
    hash_password_async,
    refresh_token_expires_at,
    refresh_token_redis_ttl,
    verify_password_async,
)
from src.db.models.auth import RefreshToken
from src.db.models.user import User, UserInterestProfile, UserPreference
from src.db.redis import get_redis
from src.db.session import get_db

api_router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _issue_tokens(
    user: User,
    db: AsyncSession,
    redis: Redis,
    device_hint: str | None = None,
) -> tuple[str, str]:
    """Create a new access + refresh token pair, persist the refresh token in
    both Postgres (source of truth) and Redis (fast lookup), and return
    (access_token, raw_refresh_token)."""
    access_token, _jti = create_access_token(str(user.id))
    raw_refresh, token_hash = create_refresh_token()

    db_token = RefreshToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=refresh_token_expires_at(),
        device_hint=device_hint,
    )
    db.add(db_token)
    await db.commit()

    await redis.set(
        f"refresh:{token_hash}",
        str(user.id),
        ex=refresh_token_redis_ttl(),
    )

    return access_token, raw_refresh


# ── Register ───────────────────────────────────────────────────────────────────

@api_router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    # Reject duplicate emails
    existing = await db.scalar(select(User).where(User.email == body.email))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(
        email=body.email,
        hashed_password=await hash_password_async(body.password),
        display_name=body.display_name,
    )
    db.add(user)
    await db.flush()  # get user.id before creating related rows

    db.add(UserPreference(user_id=user.id))
    db.add(UserInterestProfile(user_id=user.id))
    await db.commit()
    await db.refresh(user)

    device_hint = request.headers.get("User-Agent", "")[:255]
    access_token, raw_refresh = await _issue_tokens(user, db, redis, device_hint)

    return RegisterResponse(
        user_id=user.id,
        email=user.email,
        display_name=user.display_name,
        access_token=access_token,
        refresh_token=raw_refresh,
    )


# ── Login ──────────────────────────────────────────────────────────────────────

@api_router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    user = await db.scalar(select(User).where(User.email == body.email))

    # Constant-time rejection: always run bcrypt even when user is missing,
    # to prevent timing-based email enumeration.
    stored_hash = user.hashed_password if (user and not user.is_synthetic) else dummy_hash()

    if not await verify_password_async(body.password, stored_hash) or not user or user.is_synthetic:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    device_hint = request.headers.get("User-Agent", "")[:255]
    access_token, raw_refresh = await _issue_tokens(user, db, redis, device_hint)

    return LoginResponse(
        user_id=user.id,
        access_token=access_token,
        refresh_token=raw_refresh,
    )


# ── Refresh ────────────────────────────────────────────────────────────────────

@api_router.post("/refresh", response_model=RefreshResponse)
async def refresh(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()

    # 1. Fast path: Redis tells us if the token is still live.
    user_id_str = await redis.get(f"refresh:{token_hash}")

    # 2. Postgres is the source of truth regardless — validate and revoke atomically.
    db_token = await db.scalar(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.revoked.is_(False),
            RefreshToken.expires_at > datetime.now(timezone.utc),
        )
    )
    if not db_token:
        # If Redis had the token but DB rejected it, something is wrong — could be replay.
        if user_id_str:
            await redis.delete(f"refresh:{token_hash}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # 3. Rotate: revoke old token.
    db_token.revoked = True
    await db.flush()
    await redis.delete(f"refresh:{token_hash}")

    # 4. Issue new pair.
    user = await db.get(User, db_token.user_id)
    if not user or not user.is_active:
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    access_token, raw_refresh = await _issue_tokens(user, db, redis, db_token.device_hint)

    return RefreshResponse(access_token=access_token)


# ── Logout ─────────────────────────────────────────────────────────────────────

@api_router.post("/logout", response_model=MessageResponse)
async def logout(
    body: LogoutRequest,
    token: str = Depends(oauth2_scheme),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    # 1. Blacklist the current access token so it can't be used after logout.
    #    TTL = remaining lifetime of the token (no need to store dead tokens forever).
    payload = decode_access_token(token)
    jti: str = payload["jti"]
    exp: int = payload["exp"]
    remaining_ttl = max(int(exp - datetime.now(timezone.utc).timestamp()), 1)
    await redis.set(f"blacklist:jti:{jti}", "1", ex=remaining_ttl)

    # 2. Revoke the refresh token.
    token_hash = hashlib.sha256(body.refresh_token.encode()).hexdigest()
    db_token = await db.scalar(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.user_id == current_user.id,
            RefreshToken.revoked.is_(False),
        )
    )
    if db_token:
        db_token.revoked = True
        await db.commit()
        await redis.delete(f"refresh:{token_hash}")

    return MessageResponse(message="Logged out successfully")


# ── Me ─────────────────────────────────────────────────────────────────────────

@api_router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        is_synthetic=current_user.is_synthetic,
        created_at=current_user.created_at,
    )
