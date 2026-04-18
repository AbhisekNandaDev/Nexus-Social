import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schema.onboarding import (
    SaveInterestsRequest,
    SaveInterestsResponse,
    SuggestedUser,
    SuggestionsResponse,
)
from src.core.dependencies import get_current_user
from src.db.models.user import User, UserInterestProfile
from src.db.session import get_db

api_router = APIRouter()

_SUGGESTIONS_LIMIT = 20
_CANDIDATE_LIMIT = 50


# ── POST /interests ────────────────────────────────────────────────────────────

@api_router.post("/interests", response_model=SaveInterestsResponse)
async def save_interests(
    body: SaveInterestsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    profile = await db.scalar(
        select(UserInterestProfile).where(UserInterestProfile.user_id == current_user.id)
    )
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interest profile not found",
        )

    profile.top_display_tags = body.interests
    await db.commit()

    return SaveInterestsResponse(
        message="Interests saved",
        interests_count=len(body.interests),
    )


# ── GET /suggestions ───────────────────────────────────────────────────────────

@api_router.get("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Load the user's stored interest tags.
    profile = await db.scalar(
        select(UserInterestProfile).where(UserInterestProfile.user_id == current_user.id)
    )
    tags: list[str] = profile.top_display_tags if (profile and profile.top_display_tags) else []

    if not tags:
        return SuggestionsResponse(suggestions=[])

    # Find candidate users whose published posts overlap with the interest tags.
    # Uses jsonb_array_elements_text to unnest the JSONB display_tags array so
    # we can do a simple equality check against the interest tag list.
    candidates_sql = text("""
        WITH matching AS (
            SELECT
                p.user_id,
                COUNT(DISTINCT p.id)                                  AS match_count,
                (
                    SELECT p2.thumbnail_path
                    FROM   posts p2
                    WHERE  p2.user_id = p.user_id
                      AND  p2.status  = 'published'
                      AND  p2.thumbnail_path IS NOT NULL
                    ORDER BY p2.created_at DESC
                    LIMIT  1
                )                                                      AS sample_thumbnail
            FROM  posts p,
                  jsonb_array_elements_text(p.display_tags) AS tag
            WHERE p.status  = 'published'
              AND p.user_id != :current_user_id
              AND tag        = ANY(:tags)
            GROUP BY p.user_id
            ORDER BY match_count DESC
            LIMIT  :candidate_limit
        )
        SELECT
            u.id                                                       AS user_id,
            u.display_name,
            u.avatar_path,
            m.sample_thumbnail,
            (
                SELECT COUNT(*)
                FROM   posts
                WHERE  user_id = u.id AND status = 'published'
            )                                                          AS posts_count,
            (
                SELECT COUNT(*)
                FROM   follows
                WHERE  following_id = u.id
            )                                                          AS followers_count,
            (
                SELECT COALESCE(
                    array_agg(DISTINCT tag2 ORDER BY tag2),
                    ARRAY[]::text[]
                )
                FROM   posts p3,
                       jsonb_array_elements_text(p3.display_tags) AS tag2
                WHERE  p3.user_id = u.id
                  AND  p3.status  = 'published'
                  AND  tag2       = ANY(:tags)
            )                                                          AS top_tags
        FROM  matching m
        JOIN  users u ON u.id = m.user_id
        WHERE u.is_active = true
          AND u.id NOT IN (
              SELECT following_id
              FROM   follows
              WHERE  follower_id = :current_user_id
          )
        ORDER BY m.match_count DESC
        LIMIT :suggestions_limit
    """)

    rows = (
        await db.execute(
            candidates_sql,
            {
                "current_user_id": current_user.id,
                "tags": tags,
                "candidate_limit": _CANDIDATE_LIMIT,
                "suggestions_limit": _SUGGESTIONS_LIMIT,
            },
        )
    ).fetchall()

    suggestions = [
        SuggestedUser(
            user_id=uuid.UUID(str(row.user_id)),
            display_name=row.display_name,
            avatar_path=row.avatar_path,
            posts_count=int(row.posts_count),
            followers_count=int(row.followers_count),
            top_tags=list(row.top_tags) if row.top_tags else [],
            sample_post_thumbnail=row.sample_thumbnail,
        )
        for row in rows
    ]

    return SuggestionsResponse(suggestions=suggestions)
