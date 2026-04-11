# Social Media Content Platform — API Endpoints

> **Total: 52 endpoints across 12 domains**
> All endpoints return JSON. Auth endpoints are public; all others require JWT Bearer token unless noted.

---

## 1. Authentication (5 endpoints)

### `POST /api/v1/auth/register` — Public
Create new account. Auto-creates default user_preferences (all filters OFF) and empty user_interest_profile.

**Request:**
```json
{
  "email": "user@example.com",
  "display_name": "Sarah Chen",
  "password": "securepass123",
  "confirm_password": "securepass123"
}
```

**Response (201):**
```json
{
  "user_id": "uuid",
  "email": "user@example.com",
  "display_name": "Sarah Chen",
  "access_token": "jwt_token",
  "refresh_token": "jwt_refresh_token",
  "token_type": "bearer"
}
```

**Errors:** 409 email already exists, 422 validation (password mismatch, weak password, invalid email)

---

### `POST /api/v1/auth/login` — Public
Authenticate and return tokens.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "securepass123"
}
```

**Response (200):**
```json
{
  "user_id": "uuid",
  "access_token": "jwt_token",
  "refresh_token": "jwt_refresh_token",
  "token_type": "bearer"
}
```

**Errors:** 401 invalid credentials, 403 account disabled

---

### `POST /api/v1/auth/refresh` — Public
Refresh expired access token using refresh token.

**Request:**
```json
{
  "refresh_token": "jwt_refresh_token"
}
```

**Response (200):**
```json
{
  "access_token": "new_jwt_token",
  "token_type": "bearer"
}
```

**Errors:** 401 invalid/expired refresh token

---

### `POST /api/v1/auth/logout` — JWT Required
Invalidate refresh token (add to Redis blocklist).

**Request:**
```json
{
  "refresh_token": "jwt_refresh_token"
}
```

**Response (200):**
```json
{
  "message": "Logged out successfully"
}
```

---

### `POST /api/v1/auth/forgot-password` — Public
Send password reset email (implementation placeholder).

**Request:**
```json
{
  "email": "user@example.com"
}
```

**Response (200):**
```json
{
  "message": "If this email exists, a reset link has been sent"
}
```

---

## 2. User Profile (5 endpoints)

### `GET /api/v1/users/me` — JWT Required
Get current user's profile with stats.

**Response (200):**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "display_name": "Sarah Chen",
  "avatar_path": "/media/avatars/uuid.jpg",
  "bio": "Yoga instructor & traveler",
  "is_synthetic": false,
  "created_at": "2026-04-01T10:00:00Z",
  "stats": {
    "posts_count": 42,
    "followers_count": 1234,
    "following_count": 567
  },
  "interests": ["yoga", "travel", "cooking"]
}
```

---

### `PUT /api/v1/users/me` — JWT Required
Update profile fields.

**Request:**
```json
{
  "display_name": "Sarah C.",
  "bio": "Yoga instructor & world traveler 🌍"
}
```

**Response (200):** Updated user object

---

### `GET /api/v1/users/{user_id}` — JWT Required
Get another user's public profile.

**Response (200):**
```json
{
  "id": "uuid",
  "display_name": "Marco Rossi",
  "avatar_path": "/media/avatars/uuid.jpg",
  "bio": "Italian food lover",
  "stats": {
    "posts_count": 89,
    "followers_count": 5678,
    "following_count": 234
  },
  "interests": ["cooking", "pasta", "italian"],
  "is_following": true,
  "is_followed_by": false
}
```

**Errors:** 404 user not found

---

### `POST /api/v1/users/me/avatar` — JWT Required
Upload profile picture (multipart/form-data).

**Request:** `multipart/form-data` with `avatar` file field (JPEG/PNG, max 5MB)

**Response (200):**
```json
{
  "avatar_path": "/media/avatars/uuid.jpg"
}
```

---

### `GET /api/v1/users/me/interests` — JWT Required
Get user's top display tags and cluster affinity summary.

**Response (200):**
```json
{
  "top_tags": ["yoga", "travel", "cooking", "photography"],
  "cluster_affinities": [
    {"cluster_id": 7, "weight": 0.82, "representative_tags": ["yoga", "meditation", "wellness"]},
    {"cluster_id": 12, "weight": 0.45, "representative_tags": ["travel", "landscape", "adventure"]}
  ],
  "total_likes": 67,
  "confidence_level": 0.95
}
```

---

## 3. Onboarding (2 endpoints)

### `POST /api/v1/onboarding/interests` — JWT Required
Save initial interest selections from onboarding screen. Seeds the user_interest_profile.

**Request:**
```json
{
  "interests": ["fitness", "food", "travel", "photography"]
}
```

**Response (200):**
```json
{
  "message": "Interests saved",
  "interests_count": 4
}
```

**Validation:** Minimum 3 interests required.

---

### `GET /api/v1/onboarding/suggestions` — JWT Required
Get suggested users to follow based on selected interests. Returns users whose posts cluster around the selected interest areas.

**Response (200):**
```json
{
  "suggestions": [
    {
      "user_id": "uuid",
      "display_name": "YogaWithAditi",
      "avatar_path": "/media/avatars/uuid.jpg",
      "posts_count": 120,
      "followers_count": 45000,
      "top_tags": ["yoga", "meditation", "wellness"],
      "sample_post_thumbnail": "/media/thumbnails/uuid.jpg"
    }
  ]
}
```

---

## 4. Posts (8 endpoints)

### `POST /api/v1/posts/upload` — JWT Required
Upload image or video with optional caption. Images are classified synchronously (2-5s). Videos are queued for async processing.

**Request:** `multipart/form-data`
- `media` (file): image or video file
- `caption` (string, optional): post caption

**Response for images (201):**
```json
{
  "post_id": "uuid",
  "status": "published",
  "media_type": "image",
  "classification": {
    "nudity_level": "safe",
    "violence_level": "none",
    "self_harm_level": "none",
    "risk": "allow",
    "display_tags": ["yoga", "beach", "fitness"],
    "mood": "peaceful",
    "confidence": 0.92
  }
}
```

**Response for videos (202):**
```json
{
  "post_id": "uuid",
  "status": "processing",
  "media_type": "video",
  "message": "Video is being analyzed. You'll be notified when it's live."
}
```

**Errors:** 413 file too large, 415 unsupported media type, 422 validation

---

### `GET /api/v1/posts/{post_id}` — JWT Required
Get single post with full metadata, comments preview, and like status.

**Response (200):**
```json
{
  "id": "uuid",
  "user": {
    "id": "uuid",
    "display_name": "Sarah Chen",
    "avatar_path": "/media/avatars/uuid.jpg"
  },
  "media_type": "image",
  "media_url": "/media/images/uuid.jpg",
  "thumbnail_url": "/media/thumbnails/uuid.jpg",
  "caption": "Morning yoga session on the beach",
  "display_tags": ["yoga", "beach", "fitness"],
  "mood": "peaceful",
  "scene_type": "outdoor",
  "status": "published",
  "likes_count": 1247,
  "comments_count": 45,
  "is_liked": true,
  "is_bookmarked": false,
  "created_at": "2026-04-10T08:00:00Z",
  "recent_comments": [
    {
      "id": "uuid",
      "user": {"display_name": "Alex", "avatar_path": "..."},
      "text": "Beautiful spot!",
      "created_at": "2026-04-10T09:00:00Z"
    }
  ]
}
```

**Errors:** 404 not found, 403 post is blocked/deleted and requester is not the author

---

### `DELETE /api/v1/posts/{post_id}` — JWT Required
Soft delete own post (status → deleted). Only the author can delete.

**Response (200):**
```json
{
  "message": "Post deleted"
}
```

**Errors:** 403 not the author, 404 not found

---

### `GET /api/v1/posts/{post_id}/status` — JWT Required
Get processing status of a post. Used by the Post Status screen to poll during video analysis.

**Response (200):**
```json
{
  "post_id": "uuid",
  "status": "processing",
  "media_type": "video",
  "progress": {
    "uploaded": true,
    "frames_extracted": true,
    "frames_analyzed": 8,
    "total_frames_selected": 12,
    "transcript_complete": true,
    "aggregation_complete": false
  }
}
```

---

### `GET /api/v1/posts/user/{user_id}` — JWT Required
Get a user's posts grid for the profile screen. Returns thumbnails in a paginated grid.

**Query params:** `?page=1&page_size=30`

**Response (200):**
```json
{
  "posts": [
    {
      "id": "uuid",
      "thumbnail_url": "/media/thumbnails/uuid.jpg",
      "media_type": "image",
      "likes_count": 1247,
      "status": "published"
    }
  ],
  "total": 42,
  "page": 1,
  "page_size": 30
}
```

---

### `GET /api/v1/posts/me/pending` — JWT Required
Get current user's posts that are still processing or blocked.

**Response (200):**
```json
{
  "pending": [
    {
      "id": "uuid",
      "thumbnail_url": "/media/thumbnails/uuid.jpg",
      "media_type": "video",
      "status": "processing",
      "created_at": "2026-04-10T10:00:00Z"
    }
  ],
  "blocked": [
    {
      "id": "uuid",
      "thumbnail_url": "/media/thumbnails/uuid.jpg",
      "media_type": "image",
      "status": "blocked",
      "risk": "illegal",
      "block_reason": "Content classified as containing nudity with detected minor"
    }
  ]
}
```

---

### `GET /api/v1/posts/me/liked` — JWT Required
Get posts the current user has liked (for profile "Liked" tab).

**Query params:** `?page=1&page_size=30`

**Response (200):** Same structure as posts grid.

---

### `GET /api/v1/posts/{post_id}/classification` — JWT Required
Get full 19-field classification detail. Useful for debugging and admin review.

**Response (200):**
```json
{
  "post_id": "uuid",
  "nudity_level": "safe",
  "nsfw_subcategories": [],
  "violence_level": "none",
  "violence_type": [],
  "self_harm_level": "none",
  "self_harm_type": [],
  "age_group": "adult",
  "risk": "allow",
  "classification_confidence": 0.92,
  "content_description": "Woman doing yoga tree pose on sandy beach at sunrise...",
  "display_tags": ["yoga", "beach", "fitness", "sunrise"],
  "mood": "peaceful",
  "scene_type": "outdoor",
  "text_in_image": null,
  "objects_detected": ["person", "ocean", "sand"],
  "people_count": "1",
  "deepface_age": 28,
  "deepface_age_group": "adult",
  "video_specific": {
    "duration_seconds": null,
    "frames_analyzed": null,
    "llm_calls_used": null,
    "needs_review": false,
    "transcript": null,
    "secondary_classifications": null
  }
}
```

---

## 5. Likes (3 endpoints)

### `POST /api/v1/posts/{post_id}/like` — JWT Required
Like a post. Triggers async update of user's taste_embedding and cluster_affinities.

**Response (201):**
```json
{
  "message": "Post liked",
  "likes_count": 1248
}
```

**Errors:** 409 already liked, 404 post not found

---

### `DELETE /api/v1/posts/{post_id}/like` — JWT Required
Unlike a post.

**Response (200):**
```json
{
  "message": "Post unliked",
  "likes_count": 1247
}
```

**Errors:** 404 not liked / post not found

---

### `GET /api/v1/posts/{post_id}/likes` — JWT Required
Get paginated list of users who liked a post.

**Query params:** `?page=1&page_size=20`

**Response (200):**
```json
{
  "likes": [
    {
      "user_id": "uuid",
      "display_name": "Alex Rivera",
      "avatar_path": "/media/avatars/uuid.jpg",
      "is_following": false
    }
  ],
  "total": 1247,
  "page": 1
}
```

---

## 6. Comments (5 endpoints)

### `GET /api/v1/posts/{post_id}/comments` — JWT Required
Get comments for a post with nested replies (1 level deep).

**Query params:** `?page=1&page_size=20`

**Response (200):**
```json
{
  "comments": [
    {
      "id": "uuid",
      "user": {
        "id": "uuid",
        "display_name": "Marco Rossi",
        "avatar_path": "/media/avatars/uuid.jpg"
      },
      "text": "Beautiful spot! Where is this?",
      "likes_count": 12,
      "is_liked": false,
      "created_at": "2026-04-10T09:00:00Z",
      "replies": [
        {
          "id": "uuid",
          "user": {"id": "uuid", "display_name": "Sarah Chen", "avatar_path": "..."},
          "text": "Thanks! It's Goa, India 🇮🇳",
          "likes_count": 5,
          "is_liked": true,
          "created_at": "2026-04-10T09:15:00Z"
        }
      ]
    }
  ],
  "total": 45,
  "page": 1
}
```

---

### `POST /api/v1/posts/{post_id}/comments` — JWT Required
Add a top-level comment to a post. Triggers notification to post author.

**Request:**
```json
{
  "text": "Beautiful spot! Where is this?"
}
```

**Response (201):** Created comment object.

**Validation:** text max 1000 chars, not empty.

---

### `POST /api/v1/comments/{comment_id}/reply` — JWT Required
Reply to a comment (1 level nesting only — replying to a reply becomes a top-level reply to the parent). Triggers notification to comment author.

**Request:**
```json
{
  "text": "Thanks! It's Goa, India 🇮🇳"
}
```

**Response (201):** Created reply object.

---

### `DELETE /api/v1/comments/{comment_id}` — JWT Required
Delete own comment (hard delete — removes text, keeps placeholder "Comment deleted").

**Response (200):**
```json
{
  "message": "Comment deleted"
}
```

**Errors:** 403 not the author

---

### `POST /api/v1/comments/{comment_id}/like` — JWT Required
Like/unlike a comment (toggle behavior).

**Response (200):**
```json
{
  "is_liked": true,
  "likes_count": 13
}
```

---

## 7. Feed and Discovery (4 endpoints)

### `GET /api/v1/feed` — JWT Required
The main personalized feed. Runs the full recommendation pipeline: candidate retrieval → embedding similarity + social graph + cluster affinity scoring → content preference filter → paginated response.

**Query params:** `?cursor=last_post_score&page_size=20`

**Response (200):**
```json
{
  "posts": [
    {
      "id": "uuid",
      "user": {
        "id": "uuid",
        "display_name": "Sarah Chen",
        "avatar_path": "/media/avatars/uuid.jpg"
      },
      "media_type": "image",
      "media_url": "/media/images/uuid.jpg",
      "thumbnail_url": "/media/thumbnails/uuid.jpg",
      "caption": "Morning yoga session on the beach",
      "display_tags": ["yoga", "beach", "fitness"],
      "likes_count": 1247,
      "comments_count": 45,
      "is_liked": false,
      "time_ago": "2h ago",
      "created_at": "2026-04-10T08:00:00Z"
    }
  ],
  "next_cursor": "0.7823",
  "has_more": true
}
```

**Empty feed response:** Returns `suggestions` array of users to follow instead of empty posts array.

---

### `GET /api/v1/explore` — JWT Required
Explore/discover grid. Shows posts from creators the user doesn't follow, biased toward user's cluster affinities but with diversity enforcement.

**Query params:** `?cursor=xxx&page_size=30&category=fitness` (category is optional filter from trending tags)

**Response (200):**
```json
{
  "posts": [
    {
      "id": "uuid",
      "thumbnail_url": "/media/thumbnails/uuid.jpg",
      "media_type": "video",
      "likes_count": 3421,
      "display_tags": ["gym", "deadlift"]
    }
  ],
  "next_cursor": "xxx",
  "has_more": true
}
```

---

### `GET /api/v1/explore/trending` — JWT Required
Get trending tags and topics. Derived from cluster representative_tags weighted by recent engagement volume.

**Response (200):**
```json
{
  "trending": [
    {"tag": "sunset", "post_count": 234, "trend": "rising"},
    {"tag": "recipe", "post_count": 189, "trend": "stable"},
    {"tag": "workout", "post_count": 167, "trend": "rising"}
  ],
  "categories": [
    {"label": "Fitness", "cluster_ids": [7, 14, 22], "post_count": 890},
    {"label": "Food", "cluster_ids": [3, 12], "post_count": 654}
  ]
}
```

---

### `GET /api/v1/search` — JWT Required
Search posts by tags, captions, or usernames.

**Query params:** `?q=yoga&type=posts|users|tags&page=1&page_size=20`

**Response (200) — type=posts:**
```json
{
  "results": [
    {
      "id": "uuid",
      "thumbnail_url": "/media/thumbnails/uuid.jpg",
      "media_type": "image",
      "caption": "Morning yoga on the beach",
      "display_tags": ["yoga", "beach"],
      "likes_count": 1247,
      "user": {"display_name": "Sarah Chen", "avatar_path": "..."}
    }
  ],
  "total": 156,
  "page": 1
}
```

**Response (200) — type=users:**
```json
{
  "results": [
    {
      "id": "uuid",
      "display_name": "YogaWithAditi",
      "avatar_path": "...",
      "followers_count": 45000,
      "is_following": false
    }
  ],
  "total": 12,
  "page": 1
}
```

---

## 8. Social Graph (5 endpoints)

### `POST /api/v1/social/follow/{user_id}` — JWT Required
Follow a user. Invalidates the current user's feed cache in Redis. Creates a notification for the followed user.

**Response (201):**
```json
{
  "message": "Now following Marco Rossi",
  "followers_count": 5679
}
```

**Errors:** 409 already following, 400 cannot follow yourself, 404 user not found

---

### `DELETE /api/v1/social/unfollow/{user_id}` — JWT Required
Unfollow a user. Invalidates feed cache.

**Response (200):**
```json
{
  "message": "Unfollowed Marco Rossi",
  "followers_count": 5678
}
```

---

### `GET /api/v1/social/{user_id}/followers` — JWT Required
Get paginated followers list.

**Query params:** `?page=1&page_size=20`

**Response (200):**
```json
{
  "followers": [
    {
      "user_id": "uuid",
      "display_name": "Alex Rivera",
      "avatar_path": "...",
      "is_following": true,
      "is_followed_by": true
    }
  ],
  "total": 1234,
  "page": 1
}
```

---

### `GET /api/v1/social/{user_id}/following` — JWT Required
Get paginated following list. Same response structure as followers.

---

### `GET /api/v1/social/suggestions` — JWT Required
Get follow suggestions. Combines: taste-neighbours (users with similar cluster affinities), popular creators in user's top clusters, and mutual connections (followed by people you follow).

**Response (200):**
```json
{
  "suggestions": [
    {
      "user_id": "uuid",
      "display_name": "YogaWithAditi",
      "avatar_path": "...",
      "followers_count": 45000,
      "top_tags": ["yoga", "meditation"],
      "reason": "Similar interests",
      "mutual_followers": 3
    }
  ]
}
```

---

## 9. Content Preferences (3 endpoints)

### `GET /api/v1/preferences` — JWT Required
Get current user's full content filter settings.

**Response (200):**
```json
{
  "nsfw_enabled": false,
  "suggestive_enabled": false,
  "partial_nudity_enabled": false,
  "explicit_enabled": false,
  "suggestive_subcategories": [],
  "partial_subcategories": [],
  "explicit_subcategories": [],
  "violence_max_level": "none",
  "self_harm_visible": false
}
```

---

### `PUT /api/v1/preferences` — JWT Required
Update content filter settings. Invalidates cached feed in Redis so next feed request applies the new filters.

**Request:**
```json
{
  "nsfw_enabled": true,
  "suggestive_enabled": true,
  "partial_nudity_enabled": false,
  "explicit_enabled": false,
  "suggestive_subcategories": ["swimwear", "cleavage"],
  "partial_subcategories": [],
  "explicit_subcategories": [],
  "violence_max_level": "mild",
  "self_harm_visible": false
}
```

**Response (200):** Updated preferences object.

**Validation:**
- If `nsfw_enabled` is false, all tier toggles and subcategories are ignored/reset
- If a tier is disabled, its subcategories are ignored/reset
- Subcategories must be from the allowed set for each tier
- `violence_max_level` must be one of: none, mild, moderate, graphic, extreme

---

### `POST /api/v1/preferences/reset` — JWT Required
Reset all filters to defaults (everything OFF). Invalidates feed cache.

**Response (200):**
```json
{
  "message": "Preferences reset to defaults"
}
```

---

## 10. Notifications (3 endpoints)

### `GET /api/v1/notifications` — JWT Required
Get paginated notification list.

**Query params:** `?page=1&page_size=20&type=all|likes|comments|follows`

**Response (200):**
```json
{
  "notifications": [
    {
      "id": "uuid",
      "type": "like",
      "is_read": false,
      "actor": {
        "user_id": "uuid",
        "display_name": "Alex Rivera",
        "avatar_path": "..."
      },
      "post": {
        "post_id": "uuid",
        "thumbnail_url": "/media/thumbnails/uuid.jpg"
      },
      "message": "liked your post",
      "created_at": "2026-04-10T09:00:00Z"
    },
    {
      "id": "uuid",
      "type": "follow",
      "is_read": false,
      "actor": {
        "user_id": "uuid",
        "display_name": "Priya Sharma",
        "avatar_path": "..."
      },
      "post": null,
      "message": "started following you",
      "created_at": "2026-04-10T08:30:00Z"
    },
    {
      "id": "uuid",
      "type": "post_published",
      "is_read": true,
      "actor": null,
      "post": {
        "post_id": "uuid",
        "thumbnail_url": "/media/thumbnails/uuid.jpg"
      },
      "message": "Your post is now live!",
      "created_at": "2026-04-10T08:00:00Z"
    }
  ],
  "total": 89,
  "unread_count": 5,
  "page": 1
}
```

---

### `POST /api/v1/notifications/read` — JWT Required
Mark notifications as read.

**Request:**
```json
{
  "notification_ids": ["uuid1", "uuid2", "uuid3"]
}
```

**Response (200):**
```json
{
  "marked_read": 3
}
```

---

### `GET /api/v1/notifications/unread-count` — JWT Required
Get unread notification count for the tab bar badge.

**Response (200):**
```json
{
  "unread_count": 5
}
```

---

## 11. Media Serving (3 endpoints)

### `GET /media/images/{filename}` — Public
Serve image file. Static file serving via FastAPI `StaticFiles` mount.

**Response:** Image bytes with appropriate `Content-Type` header (image/jpeg, image/png, etc.)

---

### `GET /media/videos/{filename}` — Public
Serve video file with HTTP range request support for streaming/seeking.

**Response:** Video bytes with `Content-Type: video/mp4`, `Accept-Ranges: bytes`, `Content-Range` headers.

**Implementation note:** Use FastAPI `StreamingResponse` with range request parsing for efficient video seeking without loading entire file into memory.

---

### `GET /media/thumbnails/{filename}` — Public
Serve thumbnail image. Static file serving.

---

## 12. Reporting and Safety (3 endpoints)

### `POST /api/v1/posts/{post_id}/report` — JWT Required
Report a post for policy violation.

**Request:**
```json
{
  "reason": "nudity",
  "details": "This post contains explicit content that wasn't properly classified"
}
```

**Allowed reasons:** spam, nudity, violence, harassment, self_harm, misinformation, copyright, other

**Response (201):**
```json
{
  "report_id": "uuid",
  "message": "Report submitted. We'll review this within 24 hours."
}
```

**Errors:** 409 already reported this post

---

### `POST /api/v1/users/{user_id}/report` — JWT Required
Report a user.

**Request:**
```json
{
  "reason": "harassment",
  "details": "Sending threatening messages"
}
```

**Response (201):**
```json
{
  "report_id": "uuid",
  "message": "Report submitted"
}
```

---

### `POST /api/v1/users/{user_id}/block` — JWT Required
Block a user. All their content is hidden from your feed, they can't see your posts, follow/like interactions are removed.

**Response (200):**
```json
{
  "message": "User blocked",
  "blocked_user_id": "uuid"
}
```

**Errors:** 400 cannot block yourself

---

## 13. Health and System (3 endpoints)

### `GET /` — Public
Health ping.

**Response (200):**
```json
{
  "status": "ok"
}
```

---

### `GET /health` — Public
Detailed health check including dependency connectivity.

**Response (200):**
```json
{
  "status": "healthy",
  "dependencies": {
    "database": "connected",
    "redis": "connected",
    "ollama": "connected",
    "media_storage": "writable"
  },
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

---

### `GET /api/v1/stats` — JWT Required
Platform statistics (for admin/dashboard use).

**Response (200):**
```json
{
  "total_users": 1234,
  "total_posts": 5678,
  "total_published": 5432,
  "total_blocked": 23,
  "total_processing": 5,
  "total_clusters": 50,
  "posts_today": 89,
  "signups_today": 12
}
```

---

## Additional Database Tables Required

The following tables need to be added to the schema to support all endpoints:

### `comments`
| Column | Type | Notes |
|---|---|---|
| id | UUID | PK |
| post_id | UUID | FK → posts.id, indexed |
| user_id | UUID | FK → users.id |
| parent_id | UUID | FK → comments.id, nullable (for replies) |
| text | TEXT | max 1000 chars |
| likes_count | INTEGER | default 0, denormalized counter |
| is_deleted | BOOLEAN | default False (soft delete) |
| created_at | TIMESTAMPTZ | |

### `comment_likes`
| Column | Type | Notes |
|---|---|---|
| id | UUID | PK |
| comment_id | UUID | FK → comments.id |
| user_id | UUID | FK → users.id |
| created_at | TIMESTAMPTZ | |

UNIQUE constraint on (comment_id, user_id).

### `notifications`
| Column | Type | Notes |
|---|---|---|
| id | UUID | PK |
| user_id | UUID | FK → users.id, indexed (recipient) |
| type | VARCHAR(20) | like, comment, follow, reply, post_published, post_blocked |
| actor_id | UUID | FK → users.id, nullable (system notifications have no actor) |
| post_id | UUID | FK → posts.id, nullable |
| comment_id | UUID | FK → comments.id, nullable |
| is_read | BOOLEAN | default False, indexed |
| created_at | TIMESTAMPTZ | indexed DESC |

### `reports`
| Column | Type | Notes |
|---|---|---|
| id | UUID | PK |
| reporter_id | UUID | FK → users.id |
| post_id | UUID | FK → posts.id, nullable |
| reported_user_id | UUID | FK → users.id, nullable |
| reason | VARCHAR(20) | spam, nudity, violence, harassment, self_harm, misinformation, copyright, other |
| details | TEXT | nullable, free-text explanation |
| status | VARCHAR(20) | pending, reviewed, resolved, dismissed |
| created_at | TIMESTAMPTZ | |

### `blocks`
| Column | Type | Notes |
|---|---|---|
| id | UUID | PK |
| blocker_id | UUID | FK → users.id |
| blocked_id | UUID | FK → users.id |
| created_at | TIMESTAMPTZ | |

UNIQUE constraint on (blocker_id, blocked_id).

---

## API-wide Conventions

### Authentication
All JWT-protected endpoints require header: `Authorization: Bearer <access_token>`
Unauthorized requests return `401 {"detail": "Not authenticated"}`
Expired tokens return `401 {"detail": "Token expired"}`

### Pagination
- List endpoints use cursor-based pagination for feeds: `?cursor=xxx&page_size=20`
- Grid/list endpoints use offset pagination: `?page=1&page_size=20`
- Response always includes `has_more` or `total` count

### Error Format
All errors follow consistent format:
```json
{
  "detail": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "field": "optional_field_name"
}
```

### Rate Limiting
- Auth endpoints: 5 requests/minute per IP
- Upload endpoint: 10 requests/hour per user
- Like/comment/follow: 60 requests/minute per user
- Feed/explore: 30 requests/minute per user
- Search: 20 requests/minute per user

### CORS
Allow origins: configured per environment (localhost:3000 for dev, production domain for prod)