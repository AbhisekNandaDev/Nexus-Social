# Database Schema

PostgreSQL 16 + pgvector. All tables are created automatically on application startup via SQLAlchemy `create_all`. The `vector` extension is enabled at startup with `CREATE EXTENSION IF NOT EXISTS vector`.

---

## Table of Contents

- [users](#users)
- [user_preferences](#user_preferences)
- [user_interest_profiles](#user_interest_profiles)
- [posts](#posts)
- [post_embeddings](#post_embeddings)
- [post_frame_results](#post_frame_results)
- [cluster_centroids](#cluster_centroids)
- [follows](#follows)
- [likes](#likes)
- [Relationships](#relationships)
- [Indexes](#indexes)
- [Notes](#notes)

---

## users

Stores both real accounts and synthetic users created during Reddit seeding or test persona generation. Synthetic users are data containers for the recommendation engine — they have no password or email.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default uuid4 | |
| `email` | VARCHAR(255) | UNIQUE, INDEX, nullable | Null for synthetic users |
| `hashed_password` | VARCHAR(255) | nullable | Null for synthetic users |
| `display_name` | VARCHAR(100) | nullable | |
| `avatar_path` | VARCHAR(500) | nullable | Relative path to stored file |
| `is_synthetic` | BOOLEAN | default `False` | `True` for seeded/test users |
| `source_platform` | VARCHAR(50) | nullable | e.g. `"reddit"` |
| `source_username` | VARCHAR(100) | nullable | Original username on source platform |
| `is_active` | BOOLEAN | default `True` | |
| `created_at` | TIMESTAMPTZ | default `now()` | |

---

## user_preferences

Content filter settings per user. Controls which nudity levels, subcategories, and violence levels are visible in a user's feed.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default uuid4 | |
| `user_id` | UUID | FK → users.id, UNIQUE | One row per user |
| `nsfw_enabled` | BOOLEAN | default `False` | |
| `suggestive_enabled` | BOOLEAN | default `False` | |
| `partial_nudity_enabled` | BOOLEAN | default `False` | |
| `explicit_enabled` | BOOLEAN | default `False` | |
| `suggestive_subcategories` | JSONB | nullable | Allowed subcategory list e.g. `["cleavage"]` |
| `partial_subcategories` | JSONB | nullable | Allowed subcategory list |
| `explicit_subcategories` | JSONB | nullable | Allowed subcategory list |
| `violence_max_level` | VARCHAR(20) | default `"none"` | `none / mild / moderate / graphic / extreme` |
| `self_harm_visible` | BOOLEAN | default `False` | |

---

## user_interest_profiles

Stores the learned taste profile for each user, used by the recommendation engine. Updated incrementally as the user likes posts.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default uuid4 | |
| `user_id` | UUID | FK → users.id, UNIQUE | One row per user |
| `taste_embedding` | VECTOR(384) | nullable | Weighted average of liked post embeddings |
| `cluster_affinities` | JSONB | nullable | e.g. `{"7": 0.82, "12": 0.45}` |
| `top_display_tags` | JSONB | nullable | e.g. `["yoga", "cooking"]` — used in UI |
| `total_likes` | INTEGER | default `0` | Used for progressive confidence calculation |
| `updated_at` | TIMESTAMPTZ | nullable | Timestamp of last profile recalculation |

---

## posts

Central table. Stores both uploaded media metadata and the full output of the content analysis pipeline (safety classification, content understanding, DeepFace, and video-specific fields).

### Core fields

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default uuid4 | |
| `user_id` | UUID | FK → users.id, INDEX | |
| `media_type` | VARCHAR(10) | | `"image"` or `"video"` |
| `media_path` | VARCHAR(500) | | Relative path to stored file |
| `thumbnail_path` | VARCHAR(500) | nullable | |
| `caption` | TEXT | nullable | Reddit title for seeded posts |
| `status` | VARCHAR(20) | INDEX, default `"uploaded"` | `uploaded / processing / published / blocked / deleted` |

### Safety classification (set by content analysis pipeline)

| Column | Type | Notes |
|---|---|---|
| `nudity_level` | VARCHAR(30) | `safe / suggestive / partial_nudity / explicit_nudity / sexual_activity` |
| `nsfw_subcategories` | JSONB | e.g. `["cleavage", "swimwear"]` |
| `violence_level` | VARCHAR(20) | `none / mild / moderate / graphic / extreme` |
| `violence_type` | JSONB | e.g. `["fighting", "weapons"]` |
| `self_harm_level` | VARCHAR(20) | `none / implied / depicted / instructional` |
| `self_harm_type` | JSONB | e.g. `["cutting", "substance_abuse"]` |
| `age_group` | VARCHAR(10) | `child / teen / adult / unknown` |
| `risk` | VARCHAR(10) | `allow / restrict / nsfw / block / illegal` |
| `classification_confidence` | FLOAT | `0.0 – 1.0` |

### Content understanding (set by content analysis pipeline)

| Column | Type | Notes |
|---|---|---|
| `content_description` | TEXT | 2-3 sentence free-text description |
| `display_tags` | JSONB | e.g. `["yoga", "beach", "fitness"]` |
| `mood` | VARCHAR(20) | `happy / sad / peaceful / energetic / romantic / dark / neutral / humorous / inspirational` |
| `scene_type` | VARCHAR(20) | `indoor / outdoor / studio / urban / nature / underwater / aerial` |
| `text_in_image` | TEXT | Visible text detected in image, or null |
| `objects_detected` | JSONB | e.g. `["car", "dog", "guitar"]` |
| `people_count` | VARCHAR(10) | `"0" / "1" / "2" / "group"` |

### DeepFace output

| Column | Type | Notes |
|---|---|---|
| `deepface_age` | INTEGER | Estimated numeric age, nullable |
| `deepface_age_group` | VARCHAR(10) | `child / teen / adult / unknown`, nullable |

### Video-specific fields

| Column | Type | Notes |
|---|---|---|
| `video_duration_seconds` | FLOAT | nullable |
| `frames_analyzed` | INTEGER | Frames sent to LLM, nullable |
| `llm_calls_used` | INTEGER | Total LLM calls consumed, nullable |
| `needs_review` | BOOLEAN | `True` if LLM call cap was hit, default `False` |
| `transcript` | TEXT | Full Whisper transcription, nullable |
| `transcript_language` | VARCHAR(10) | ISO-639-1 code e.g. `"en"`, nullable |
| `transcript_safety_flags` | JSONB | Flagged dangerous segments, nullable |
| `secondary_classifications` | JSONB | Weighted avg breakdown per frame, nullable |

### Source metadata (seeded content)

| Column | Type | Notes |
|---|---|---|
| `source_url` | VARCHAR(1000) | Original Reddit post URL, nullable |
| `source_platform` | VARCHAR(50) | e.g. `"reddit"`, nullable |
| `source_subreddit` | VARCHAR(100) | nullable |
| `source_upvotes` | INTEGER | Reddit upvote count, nullable |
| `source_comments` | INTEGER | Reddit comment count, nullable |

### Timestamps

| Column | Type | Notes |
|---|---|---|
| `created_at` | TIMESTAMPTZ | default `now()`, DESC index for feed queries |
| `classified_at` | TIMESTAMPTZ | Set when pipeline completes, nullable |

---

## post_embeddings

One row per post. Stores the 384-dimensional content embedding (all-MiniLM / nomic-embed-text) used for similarity search and cluster assignment.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default uuid4 | |
| `post_id` | UUID | FK → posts.id, UNIQUE | One embedding per post |
| `embedding` | VECTOR(384) | HNSW INDEX | Content description embedding |
| `cluster_id` | INTEGER | FK → cluster_centroids.id, INDEX, nullable | Null until clustering runs |
| `cluster_distance` | FLOAT | nullable | Distance to assigned centroid |

**HNSW index:**
```sql
CREATE INDEX ON post_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

---

## post_frame_results

Per-frame classification results for video posts. Used for debugging, admin review, and potential re-aggregation without re-running the pipeline.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default uuid4 | |
| `post_id` | UUID | FK → posts.id, INDEX | |
| `frame_index` | INTEGER | | Original frame number in video |
| `timestamp_seconds` | FLOAT | | Seconds into video |
| `selection_reason` | VARCHAR(30) | nullable | `first_frame / scene_change / diversity_centroid / drift / last_frame` |
| `classification` | JSONB | nullable | Full 19-field classification for this frame |
| `cluster_id` | INTEGER | nullable | CLIP diversity cluster assignment |

---

## cluster_centroids

KMeans cluster centroids computed over the post embedding space. Used by the recommendation engine to score user affinity per topic cluster.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | INTEGER | PK | 0 to K-1 |
| `centroid` | VECTOR(384) | | Mean embedding of all posts in cluster |
| `post_count` | INTEGER | | Number of posts assigned to this cluster |
| `representative_tags` | JSONB | nullable | Most common `display_tags` in cluster |
| `updated_at` | TIMESTAMPTZ | nullable | Timestamp of last recomputation |

---

## follows

Directed follow graph between users.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default uuid4 | |
| `follower_id` | UUID | FK → users.id, INDEX | The user who follows |
| `following_id` | UUID | FK → users.id, INDEX | The user being followed |
| `created_at` | TIMESTAMPTZ | default `now()` | |

**Constraints:**
- `UNIQUE(follower_id, following_id)` — no duplicate follows
- `CHECK(follower_id != following_id)` — no self-follows

---

## likes

Records which user liked which post. Also drives the recommendation engine — each like updates `user_interest_profiles.taste_embedding`.

| Column | Type | Constraints | Notes |
|---|---|---|---|
| `id` | UUID | PK, default uuid4 | |
| `user_id` | UUID | FK → users.id, INDEX | |
| `post_id` | UUID | FK → posts.id, INDEX | |
| `created_at` | TIMESTAMPTZ | default `now()` | |

**Constraints:**
- `UNIQUE(user_id, post_id)` — a user can only like a post once

---

## Relationships

```
users ──< posts                     (one user → many posts)
users ──< likes                     (one user → many likes)
users ──< follows (as follower)     (one user → many outgoing follows)
users ──< follows (as following)    (one user → many incoming follows)
users ──1 user_preferences          (one user → one preference row)
users ──1 user_interest_profiles    (one user → one interest profile)

posts ──1 post_embeddings           (one post → one embedding)
posts ──< post_frame_results        (one post → many frame results, videos only)
posts ──< likes                     (one post → many likes)

post_embeddings >── cluster_centroids   (many embeddings → one centroid)
cluster_centroids ──< post_embeddings   (one centroid → many embeddings)
```

---

## Indexes

| Table | Column(s) | Type | Purpose |
|---|---|---|---|
| `users` | `email` | BTREE | Login lookup |
| `posts` | `user_id` | BTREE | Feed queries by user |
| `posts` | `status` | BTREE | Filter by pipeline status |
| `posts` | `nudity_level` | BTREE | Content filter queries |
| `posts` | `risk` | BTREE | Safety dashboard queries |
| `posts` | `created_at` | BTREE (DESC) | Chronological feed |
| `post_embeddings` | `embedding` | HNSW cosine | Similarity search |
| `post_embeddings` | `cluster_id` | BTREE | Cluster membership lookup |
| `post_frame_results` | `post_id` | BTREE | Per-video frame lookup |
| `follows` | `follower_id` | BTREE | "Who am I following?" |
| `follows` | `following_id` | BTREE | "Who follows me?" |
| `likes` | `user_id` | BTREE | User's liked posts |
| `likes` | `post_id` | BTREE | Post's like count |

---

## Notes

**pgvector dimension — 384**
All `VECTOR(384)` columns assume the `all-MiniLM-L6-v2` embedding model (or `nomic-embed-text` via Ollama configured to 384 dimensions). If you switch to a different embedding model, update the dimension in [src/db/models/post.py](../src/db/models/post.py) and [src/db/models/user.py](../src/db/models/user.py) and recreate the affected tables.

**Synthetic users**
Reddit seeding creates one `User` row per unique Reddit poster with `is_synthetic = True`. These rows have no `email` or `hashed_password` and cannot log in. They exist purely so posts have a valid `user_id` FK and the recommendation engine has author signal.

**Status lifecycle**
```
uploaded → processing → published
                     └→ blocked
                     └→ deleted
```
The content analysis pipeline transitions a post from `uploaded` → `processing` → `published` (or `blocked` if `risk = illegal/block`).

**Cluster recomputation**
`cluster_centroids` is populated by an offline KMeans job that runs over all `post_embeddings`. After each run, `post_embeddings.cluster_id` and `cluster_distance` are updated in bulk. `user_interest_profiles.cluster_affinities` is then recomputed from the user's like history against the new centroids.
