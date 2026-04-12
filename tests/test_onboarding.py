"""
Tests for the onboarding API endpoints.

  POST /api/v1/onboarding/interests
  GET  /api/v1/onboarding/suggestions

Run with:
    pytest tests/test_onboarding.py -v
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from src.api.main import app
from src.core.dependencies import get_current_user, get_db
from src.db.models.user import User, UserInterestProfile


# ── shared test fixtures ───────────────────────────────────────────────────────

_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _make_user(user_id: uuid.UUID = _USER_ID) -> User:
    u = User()
    u.id = user_id
    u.email = "test@example.com"
    u.display_name = "Test User"
    u.is_active = True
    u.is_synthetic = False
    return u


def _make_profile(tags: list[str] | None = None) -> UserInterestProfile:
    p = UserInterestProfile()
    p.id = uuid.uuid4()
    p.user_id = _USER_ID
    p.top_display_tags = tags
    return p


_UNSET = object()


def _make_suggestion_row(
    user_id: uuid.UUID | None = None,
    display_name: str = "Creator",
    avatar_path: str | None = "/media/avatars/x.jpg",
    sample_thumbnail: str | None = "/media/thumbnails/x.jpg",
    posts_count: int = 10,
    followers_count: int = 500,
    top_tags: list[str] | None = _UNSET,  # type: ignore[assignment]
) -> MagicMock:
    row = MagicMock()
    row.user_id = user_id or uuid.uuid4()
    row.display_name = display_name
    row.avatar_path = avatar_path
    row.sample_thumbnail = sample_thumbnail
    row.posts_count = posts_count
    row.followers_count = followers_count
    # Use _UNSET sentinel so callers can explicitly pass None to test null handling.
    row.top_tags = ["yoga"] if top_tags is _UNSET else top_tags
    return row


def _make_db(scalar_return=None, execute_rows: list | None = None) -> AsyncMock:
    """Build an async DB session mock."""
    db = AsyncMock()
    db.scalar.return_value = scalar_return
    mock_result = MagicMock()
    mock_result.fetchall.return_value = execute_rows or []
    db.execute.return_value = mock_result
    db.commit = AsyncMock()
    return db


def _override_current_user():
    return _make_user()


def _build_client(db: AsyncMock) -> TestClient:
    async def _get_db_override():
        yield db

    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_db] = _get_db_override
    return TestClient(app)


def _clear_overrides():
    app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# POST /api/v1/onboarding/interests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveInterests:

    def teardown_method(self):
        _clear_overrides()

    # ── happy path ─────────────────────────────────────────────────────────────

    def test_saves_three_interests_successfully(self):
        db = _make_db(scalar_return=_make_profile())
        client = _build_client(db)

        resp = client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["yoga", "food", "travel"]},
        )

        assert resp.status_code == 200
        assert resp.json() == {"message": "Interests saved", "interests_count": 3}

    def test_saves_more_than_three_interests(self):
        db = _make_db(scalar_return=_make_profile())
        client = _build_client(db)

        resp = client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["yoga", "food", "travel", "photography", "fitness"]},
        )

        assert resp.status_code == 200
        assert resp.json()["interests_count"] == 5

    def test_persists_tags_on_profile(self):
        profile = _make_profile()
        db = _make_db(scalar_return=profile)
        client = _build_client(db)

        client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["yoga", "food", "travel"]},
        )

        assert profile.top_display_tags == ["yoga", "food", "travel"]
        db.commit.assert_called_once()

    def test_overwrites_existing_tags(self):
        """A second call replaces previous interests, not appends."""
        profile = _make_profile(tags=["old_tag1", "old_tag2", "old_tag3"])
        db = _make_db(scalar_return=profile)
        client = _build_client(db)

        client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["yoga", "food", "travel"]},
        )

        assert profile.top_display_tags == ["yoga", "food", "travel"]

    # ── normalisation ──────────────────────────────────────────────────────────

    def test_strips_whitespace_and_lowercases_interests(self):
        profile = _make_profile()
        db = _make_db(scalar_return=profile)
        client = _build_client(db)

        client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["  Yoga  ", "FOOD", " Travel "]},
        )

        assert profile.top_display_tags == ["yoga", "food", "travel"]

    def test_whitespace_only_entries_are_discarded(self):
        """Entries that reduce to '' after strip() are dropped before the min-3 check."""
        db = _make_db(scalar_return=_make_profile())
        client = _build_client(db)

        # "  " → stripped to "" → discarded; only 2 real interests remain → 422
        resp = client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["yoga", "  ", "food"]},
        )

        assert resp.status_code == 422

    def test_interests_count_reflects_cleaned_list(self):
        profile = _make_profile()
        db = _make_db(scalar_return=profile)
        client = _build_client(db)

        resp = client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["  Yoga  ", "FOOD", "   travel   "]},
        )

        assert resp.status_code == 200
        assert resp.json()["interests_count"] == 3

    # ── validation errors ──────────────────────────────────────────────────────

    def test_rejects_fewer_than_three_interests(self):
        db = _make_db(scalar_return=_make_profile())
        client = _build_client(db)

        resp = client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["yoga", "food"]},
        )

        assert resp.status_code == 422

    def test_rejects_one_interest(self):
        db = _make_db(scalar_return=_make_profile())
        client = _build_client(db)

        resp = client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["yoga"]},
        )

        assert resp.status_code == 422

    def test_rejects_empty_list(self):
        db = _make_db(scalar_return=_make_profile())
        client = _build_client(db)

        resp = client.post(
            "/api/v1/onboarding/interests",
            json={"interests": []},
        )

        assert resp.status_code == 422

    def test_rejects_missing_interests_field(self):
        db = _make_db(scalar_return=_make_profile())
        client = _build_client(db)

        resp = client.post("/api/v1/onboarding/interests", json={})

        assert resp.status_code == 422

    # ── error cases ────────────────────────────────────────────────────────────

    def test_returns_404_when_profile_not_found(self):
        db = _make_db(scalar_return=None)
        client = _build_client(db)

        resp = client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["yoga", "food", "travel"]},
        )

        assert resp.status_code == 404

    def test_does_not_commit_when_profile_missing(self):
        db = _make_db(scalar_return=None)
        client = _build_client(db)

        client.post(
            "/api/v1/onboarding/interests",
            json={"interests": ["yoga", "food", "travel"]},
        )

        db.commit.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/onboarding/suggestions
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetSuggestions:

    def teardown_method(self):
        _clear_overrides()

    # ── empty / short-circuit ──────────────────────────────────────────────────

    def test_returns_empty_when_profile_not_found(self):
        db = _make_db(scalar_return=None)
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        assert resp.json() == {"suggestions": []}

    def test_returns_empty_when_tags_are_none(self):
        db = _make_db(scalar_return=_make_profile(tags=None))
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        assert resp.json() == {"suggestions": []}

    def test_returns_empty_when_tags_list_is_empty(self):
        db = _make_db(scalar_return=_make_profile(tags=[]))
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        assert resp.json() == {"suggestions": []}

    def test_db_query_not_executed_when_no_tags(self):
        """If there are no tags we should short-circuit before hitting the DB."""
        db = _make_db(scalar_return=_make_profile(tags=None))
        client = _build_client(db)

        client.get("/api/v1/onboarding/suggestions")

        db.execute.assert_not_called()

    def test_returns_empty_when_no_matching_users_in_db(self):
        db = _make_db(scalar_return=_make_profile(tags=["yoga", "fitness"]), execute_rows=[])
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        assert resp.json() == {"suggestions": []}

    # ── happy path ─────────────────────────────────────────────────────────────

    def test_returns_single_suggestion_with_correct_shape(self):
        other_id = uuid.uuid4()
        row = _make_suggestion_row(
            user_id=other_id,
            display_name="YogaWithAditi",
            avatar_path="/media/avatars/aditi.jpg",
            sample_thumbnail="/media/thumbnails/post1.jpg",
            posts_count=120,
            followers_count=45000,
            top_tags=["yoga", "fitness"],
        )
        db = _make_db(
            scalar_return=_make_profile(tags=["yoga", "fitness"]),
            execute_rows=[row],
        )
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        suggestions = resp.json()["suggestions"]
        assert len(suggestions) == 1

        s = suggestions[0]
        assert s["user_id"] == str(other_id)
        assert s["display_name"] == "YogaWithAditi"
        assert s["avatar_path"] == "/media/avatars/aditi.jpg"
        assert s["sample_post_thumbnail"] == "/media/thumbnails/post1.jpg"
        assert s["posts_count"] == 120
        assert s["followers_count"] == 45000
        assert s["top_tags"] == ["yoga", "fitness"]

    def test_returns_multiple_suggestions(self):
        rows = [
            _make_suggestion_row(display_name=f"Creator{i}", top_tags=["yoga"])
            for i in range(5)
        ]
        db = _make_db(
            scalar_return=_make_profile(tags=["yoga", "food"]),
            execute_rows=rows,
        )
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        assert len(resp.json()["suggestions"]) == 5

    def test_db_query_executed_when_tags_present(self):
        db = _make_db(
            scalar_return=_make_profile(tags=["yoga"]),
            execute_rows=[],
        )
        client = _build_client(db)

        client.get("/api/v1/onboarding/suggestions")

        db.execute.assert_called_once()

    # ── edge cases in row data ─────────────────────────────────────────────────

    def test_null_avatar_path_is_allowed(self):
        row = _make_suggestion_row(avatar_path=None)
        db = _make_db(
            scalar_return=_make_profile(tags=["yoga"]),
            execute_rows=[row],
        )
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        assert resp.json()["suggestions"][0]["avatar_path"] is None

    def test_null_sample_thumbnail_is_allowed(self):
        row = _make_suggestion_row(sample_thumbnail=None)
        db = _make_db(
            scalar_return=_make_profile(tags=["yoga"]),
            execute_rows=[row],
        )
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        assert resp.json()["suggestions"][0]["sample_post_thumbnail"] is None

    def test_empty_top_tags_list_in_row(self):
        row = _make_suggestion_row(top_tags=[])
        db = _make_db(
            scalar_return=_make_profile(tags=["yoga"]),
            execute_rows=[row],
        )
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        assert resp.json()["suggestions"][0]["top_tags"] == []

    def test_none_top_tags_in_row_coerced_to_empty_list(self):
        row = _make_suggestion_row(top_tags=None)
        db = _make_db(
            scalar_return=_make_profile(tags=["yoga"]),
            execute_rows=[row],
        )
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        assert resp.status_code == 200
        assert resp.json()["suggestions"][0]["top_tags"] == []

    def test_suggestion_user_ids_are_valid_uuids(self):
        rows = [_make_suggestion_row(user_id=uuid.uuid4()) for _ in range(3)]
        db = _make_db(
            scalar_return=_make_profile(tags=["yoga", "food"]),
            execute_rows=rows,
        )
        client = _build_client(db)

        resp = client.get("/api/v1/onboarding/suggestions")

        for s in resp.json()["suggestions"]:
            uuid.UUID(s["user_id"])  # raises ValueError if invalid
