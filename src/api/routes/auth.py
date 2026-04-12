import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
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
from src.core.dependencies import bearer_scheme, get_current_user
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
from src.db.session import SessionLocal, get_db

api_router = APIRouter()


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _issue_tokens(
    user_id,
    redis: Redis,
    device_hint: str | None = None,
) -> tuple[str, str]:
    """Issue a new access + refresh token pair.

    Uses its own short-lived DB session so the connection is held only for the
    INSERT (< 5ms) and released immediately — not for the full request lifetime.
    The caller must NOT hold a DB connection when calling this function.
    """
    access_token, _jti = create_access_token(str(user_id))
    raw_refresh, token_hash = create_refresh_token()

    async with SessionLocal() as db:
        db.add(RefreshToken(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=refresh_token_expires_at(),
            device_hint=device_hint,
        ))
        await db.commit()

    await redis.set(
        f"refresh:{token_hash}",
        str(user_id),
        ex=refresh_token_redis_ttl(),
    )
    return access_token, raw_refresh


# ── Register ───────────────────────────────────────────────────────────────────

@api_router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    request: Request,
    redis: Redis = Depends(get_redis),
):
    # ── Phase 1: uniqueness check — hold connection ~1ms then release ──────────
    async with SessionLocal() as db:
        existing = await db.scalar(select(User).where(User.email == body.email))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    # ── Phase 2: bcrypt hash — CPU-bound, NO connection held (~200-400ms) ──────
    hashed = await hash_password_async(body.password)

    # ── Phase 3: write user rows — hold connection ~3ms then release ───────────
    async with SessionLocal() as db:
        user = User(
            email=body.email,
            hashed_password=hashed,
            display_name=body.display_name,
        )
        db.add(user)
        await db.flush()
        db.add(UserPreference(user_id=user.id))
        db.add(UserInterestProfile(user_id=user.id))
        await db.commit()
        user_id = user.id
        user_email = user.email
        user_display_name = user.display_name

    # ── Phase 4: issue tokens — own session, hold ~2ms then release ────────────
    device_hint = request.headers.get("User-Agent", "")[:255]
    access_token, raw_refresh = await _issue_tokens(user_id, redis, device_hint)

    return RegisterResponse(
        user_id=user_id,
        email=user_email,
        display_name=user_display_name,
        access_token=access_token,
        refresh_token=raw_refresh,
    )


# ── Login ──────────────────────────────────────────────────────────────────────

@api_router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    request: Request,
    redis: Redis = Depends(get_redis),
):
    # ── Phase 1: read user — hold connection ~1ms then release ─────────────────
    async with SessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == body.email))
        # Extract plain values before closing session
        if user and not user.is_synthetic and user.hashed_password:
            stored_hash = user.hashed_password
            user_id = user.id
            is_active = user.is_active
            valid_user = True
        else:
            stored_hash = None
            user_id = None
            is_active = False
            valid_user = False

    # ── Phase 2: bcrypt verify — CPU-bound, NO connection held (~200-400ms) ────
    # Always run bcrypt (even for unknown emails) to prevent timing-based
    # email enumeration attacks.
    password_ok = await verify_password_async(body.password, stored_hash or dummy_hash())

    if not password_ok or not valid_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    # ── Phase 3: issue tokens — own session, hold ~2ms then release ────────────
    device_hint = request.headers.get("User-Agent", "")[:255]
    access_token, raw_refresh = await _issue_tokens(user_id, redis, device_hint)

    return LoginResponse(
        user_id=user_id,
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

    # Fast path: Redis tells us if the token is still live.
    user_id_str = await redis.get(f"refresh:{token_hash}")

    # Postgres is the source of truth — validate and revoke atomically.
    db_token = await db.scalar(
        select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.revoked.is_(False),
            RefreshToken.expires_at > datetime.now(timezone.utc),
        )
    )
    if not db_token:
        if user_id_str:
            await redis.delete(f"refresh:{token_hash}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    # Rotate: revoke old token.
    db_token.revoked = True
    token_user_id = db_token.user_id
    device_hint = db_token.device_hint
    await db.flush()
    await redis.delete(f"refresh:{token_hash}")

    # Verify user is still active.
    user = await db.get(User, token_user_id)
    if not user or not user.is_active:
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    await db.commit()

    # Issue new pair via own session.
    access_token, raw_refresh = await _issue_tokens(token_user_id, redis, device_hint)

    return RefreshResponse(access_token=access_token)


# ── Logout ─────────────────────────────────────────────────────────────────────

@api_router.post("/logout", response_model=MessageResponse)
async def logout(
    body: LogoutRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
):
    # Blacklist the access token for its remaining lifetime.
    token = credentials.credentials
    payload = decode_access_token(token)
    jti: str = payload["jti"]
    exp: int = payload["exp"]
    remaining_ttl = max(int(exp - datetime.now(timezone.utc).timestamp()), 1)
    await redis.set(f"blacklist:jti:{jti}", "1", ex=remaining_ttl)

    # Revoke the refresh token.
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
