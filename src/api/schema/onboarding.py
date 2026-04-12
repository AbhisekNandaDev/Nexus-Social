import uuid

from pydantic import BaseModel, field_validator


class SaveInterestsRequest(BaseModel):
    interests: list[str]

    @field_validator("interests")
    @classmethod
    def min_three_interests(cls, v: list[str]) -> list[str]:
        cleaned = [i.strip().lower() for i in v if i.strip()]
        if len(cleaned) < 3:
            raise ValueError("At least 3 interests are required")
        return cleaned


class SaveInterestsResponse(BaseModel):
    message: str
    interests_count: int


class SuggestedUser(BaseModel):
    user_id: uuid.UUID
    display_name: str | None
    avatar_path: str | None
    posts_count: int
    followers_count: int
    top_tags: list[str]
    sample_post_thumbnail: str | None


class SuggestionsResponse(BaseModel):
    suggestions: list[SuggestedUser]
