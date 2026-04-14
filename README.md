# Social Media Content Platform

A full-stack social media backend that combines automated content moderation, semantic recommendation, and a Reddit-based data pipeline. Posts are classified by a local LLM pipeline (Ollama + DeepFace + Faster-Whisper) and stored with 768-dimensional embeddings for vector-based feed ranking — all on-device with no external API dependencies.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Classification Output](#classification-output)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Option A — Docker (recommended)](#option-a--docker-recommended)
  - [Option B — Local Development](#option-b--local-development)
- [Configuration](#configuration)
- [API Reference](#api-reference)
  - [Health](#get-health)
  - [Auth](#auth)
  - [Onboarding](#onboarding)
  - [Posts](#posts)
  - [Image Classification (standalone)](#image-classification-standalone)
- [Seeding](#seeding)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Technical Notes](#technical-notes)

---

## Features

- **Image moderation** — Single-pass LLM analysis with nudity, violence, self-harm, and age classification
- **Video moderation** — Full 7-stage pipeline: intelligent frame selection, parallel audio transcription, batched LLM classification, and result aggregation
- **7-stage frame selection cascade** — pHash deduplication → color histogram → MS-SSIM → optical flow → shot boundary detection → diversity selection; cheap filters run first to minimise LLM calls
- **Dual age detection** — DeepFace biological age estimation runs in parallel with LLM-based age group classification for cross-validation
- **Audio-aware analysis** — Faster-Whisper transcribes spoken content; dangerous keyword hits trigger targeted frame re-analysis
- **Semantic embeddings** — Content descriptions are embedded via `nomic-embed-text` (768-dim) and stored in pgvector for cosine-similarity feed ranking
- **User preference filtering** — Per-user NSFW/violence/self-harm gates enforced at feed time
- **Reddit data pipeline** — Concurrent producer-consumer seeder with Redis job queue; downloads, classifies, and stores 5k+ posts across 49 subreddits with resumability
- **Fully local** — LLM inference, transcription, and face analysis all run on-device; no external API calls required
- **Structured JSON responses** — All outputs are validated Pydantic models covering 20+ fields across safety and content dimensions

---

## Architecture

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     Reddit Seeder (scripts/)                  │
  │  asyncpraw producer → aiohttp downloads → Redis job queue    │
  │  pipeline workers (N) → ImagePipeline / VideoPipeline        │
  └──────────────────────────────────┬───────────────────────────┘
                                     │ seeds media/ + posts table
                                     ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                     FastAPI Application                       │
  │   Auth · Onboarding · Posts · Image Classification           │
  └──────┬──────────────────────────┬────────────────────────────┘
         │ POST /api/v1/posts        │ POST /api/v1/posts/{id}/classify
         ▼                          ▼
  ┌─────────────┐         ┌──────────────────┐
  │  Save media │         │ Background Task  │
  │  Create Post│         │ (FastAPI BG)     │
  │  status=    │         └────────┬─────────┘
  │  "uploaded" │                  │
  └─────────────┘       ┌─────────┴──────────┐
                         │                    │
                   ┌─────▼──────┐      ┌──────▼──────┐
                   │   Image    │      │    Video    │
                   │  Pipeline  │      │  Pipeline   │
                   └─────┬──────┘      └──────┬──────┘
                         │                    │
              ┌──────────┘    ┌───────────────┼───────────────┐
              │               │               │               │
       ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
       │  Ollama LLM │ │ Frame       │ │  Whisper   │ │  Ollama LLM │
       │  (qwen3.5)  │ │ Selector    │ │  (Audio)   │ │  (batched)  │
       └──────┬──────┘ │ (7 stages)  │ └─────┬──────┘ └──────┬──────┘
              │        └──────┬──────┘        │               │
       ┌──────▼──────┐        └───────────────┴───────┐       │
       │  DeepFace   │                                │       │
       │  Age Detect │                         ┌──────▼───────▼──────┐
       └─────────────┘                         │     Aggregator       │
                                               └──────────┬───────────┘
                                                          │
                                               ┌──────────▼───────────┐
                                               │  nomic-embed-text    │
                                               │  768-dim embedding   │
                                               └──────────┬───────────┘
                                                          │
                                               ┌──────────▼───────────┐
                                               │  PostgreSQL 16       │
                                               │  + pgvector          │
                                               │  status → "published"│
                                               └──────────────────────┘
```

---

## Classification Output

Every classified post stores a structured result covering:

### Safety Fields

| Field | Values |
|---|---|
| `nudity_level` | `safe` · `suggestive` · `partial_nudity` · `explicit_nudity` · `sexual_activity` |
| `nsfw_subcategories` | 12-category list (e.g. `cleavage`, `genitals_visible`) |
| `violence_level` | `none` · `mild` · `moderate` · `graphic` · `extreme` |
| `violence_type` | `fighting` · `weapons` · `blood_gore` · `animal_cruelty` · `domestic_violence` · `war_conflict` |
| `self_harm_level` | `none` · `implied` · `depicted` · `instructional` |
| `self_harm_type` | `cutting` · `substance_abuse` · `suicide_reference` · `eating_disorder` · `dangerous_challenge` |
| `age_group` | `child (<13)` · `teen (13-17)` · `adult (18+)` · `unknown` |
| `risk` | `allow` · `restrict` · `nsfw` · `block` · `illegal` |
| `confidence` | `0.0 – 1.0` |

### Content Fields

| Field | Description |
|---|---|
| `content_description` | 2-3 sentence plain-English summary |
| `display_tags` | 3-5 short labels (e.g. `beach`, `sunset`, `couple`) |
| `mood` | `happy` · `sad` · `angry` · `peaceful` · `energetic` · `romantic` · `dark` · `neutral` · `humorous` · `inspirational` |
| `scene_type` | `indoor` · `outdoor` · `studio` · `urban` · `nature` · `underwater` · `aerial` |
| `text_in_image` | Visible text extracted from image, or `null` |
| `objects_detected` | List of prominent objects |
| `people_count` | `0` · `1` · `2` · `"group"` |

### Age Detection (dual)

| Field | Description |
|---|---|
| `deepface_age` | Raw estimated age from DeepFace |
| `deepface_age_group` | Categorical group derived from DeepFace result |

### Video Metadata

| Field | Description |
|---|---|
| `video_duration_seconds` | Duration in seconds |
| `frames_analyzed` | Frames actually sent to LLM |
| `needs_review` | `true` if LLM call cap was hit |
| `transcript` | Full audio transcription |
| `transcript_language` | ISO-639-1 language code (e.g. `"en"`) |

---

## Prerequisites

| Dependency | Notes |
|---|---|
| Docker + Docker Compose | For the recommended Docker setup |
| Ollama | Install from [ollama.com](https://ollama.com) — runs on the host |
| Python 3.11+ | For local development only |
| ffmpeg | Required for video processing (included in Docker image) |
| Redis | Included in docker-compose; or `brew install redis` locally |

Pull the required Ollama models before starting:

```bash
ollama pull qwen3.5:9b          # LLM for classification
ollama pull nomic-embed-text    # 768-dim embedding model
```

---

## Installation

### Option A — Docker (recommended)

This starts the FastAPI app, PostgreSQL + pgvector, and Redis.

**1. Clone the repository**

```bash
git clone <repo-url>
cd Social_Media_Content
```

**2. Create `.env`**

```bash
cp .env.example .env   # or create manually — see Configuration section
```

**3. Start services**

```bash
docker compose up --build
```

The first build downloads Python dependencies, DeepFace weights (~200 MB), and Whisper models (~150 MB). Subsequent starts are fast.

**4. Verify**

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

API docs are available at `http://localhost:8000/docs`.

**Services started:**

| Service | Port | Description |
|---|---|---|
| `app` | `8000` | FastAPI moderation API |
| `db` | `5433` | PostgreSQL 16 + pgvector |
| `redis` | `6379` | Redis (job queue + caching) |

> **Ollama** runs on the host machine. The app container reaches it via `host.docker.internal:11434` (Docker Desktop on macOS/Windows). On Linux, set `OLLAMA_HOST` to your host LAN IP in `docker-compose.yml`.

> **DB port is 5433** (not 5432) to avoid conflicts with a locally-installed PostgreSQL on the host.

**Stopping services**

```bash
docker compose down          # Stop containers, keep database data
docker compose down -v       # Stop containers AND delete all volumes (data lost)
```

---

### Option B — Local Development

**1. Clone and create a virtual environment**

```bash
git clone <repo-url>
cd Social_Media_Content
python3.11 -m venv env
source env/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Install system dependencies**

```bash
# macOS
brew install ffmpeg redis

# Ubuntu / Debian
sudo apt-get install ffmpeg libgl1 libglib2.0-0 redis-server
```

**4. Configure environment**

Create a `.env` file at the project root (see [Configuration](#configuration)).

**5. Start supporting services**

```bash
# PostgreSQL + Redis via Docker (recommended even for local dev):
docker compose up db redis

# Or start Redis separately:
redis-server
```

**6. Start Ollama**

```bash
ollama serve
```

**7. Start the API server**

```bash
uvicorn src.api.main:app --reload
```

Server is available at `http://127.0.0.1:8000`.

---

## Configuration

### `.env` file

Create a `.env` file in the project root. The app loads it automatically on startup.

```env
# Reddit API credentials (for seeding scripts)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password
REDDIT_USER_AGENT=YourApp/1.0 by u/your_username

# Database (asyncpg driver required)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/social_media_content

# Redis
REDIS_URL=redis://localhost:6379
```

### `config/config.yml`

All pipeline tuning lives here. No restart needed in development (`--reload` picks up changes).

```yaml
image_classification:
  provider: "ollama"
  model: "qwen3.5:9b"       # Any vision-capable Ollama model
  think: false               # Set true to enable chain-of-thought reasoning

transcription:
  enabled: true
  model: "base"              # tiny | base | small | medium | large-v3
  device: "cpu"              # cpu | cuda
  language: null             # null = auto-detect, or "en", "es", "hi", etc.

video_processing:
  sampling_fps: 2            # Frames sampled per second from video
  phash_threshold: 5         # Hamming distance for near-duplicate removal
  histogram_threshold: 0.3   # Color histogram distance gate
  ssim_threshold: 0.85       # MS-SSIM minimum for frame retention
  optical_flow_threshold: 2.0
  max_llm_calls: 15          # Hard cap on LLM calls per video
  batch_size: 3              # Frames sent per LLM call (1-3)
  use_transnet: true
  use_clip: true

embedding:
  model: "nomic-embed-text"
  dim: 768

audio:
  whisper_model: "base"
  whisper_device: "cpu"
  whisper_compute_type: "int8"
  max_transcript_length: 5000
```

**Environment variables** (override via shell or docker-compose):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama server URL |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@db:5433/social_media_content` | PostgreSQL async connection string |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |

---

## API Reference

### `GET /health`

Health check.

```json
{"status": "ok"}
```

---

### Auth

#### `POST /api/v1/auth/register`

Register a new user with email and password.

**Request** — `application/json`

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response** — `201 Created`

```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

#### `POST /api/v1/auth/login`

Authenticate and receive tokens.

**Request** — `application/json`

```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response** — `200 OK`

```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

#### `POST /api/v1/auth/refresh`

Exchange a refresh token for a new access token.

#### `POST /api/v1/auth/logout`

Revoke the current refresh token.

---

### Onboarding

#### `POST /api/v1/onboarding/preferences`

Set content preferences for the authenticated user (NSFW gates, violence tolerance, etc.).

**Request** — `application/json` (requires `Authorization: Bearer <token>`)

```json
{
  "nsfw_enabled": false,
  "suggestive_enabled": true,
  "violence_max_level": "mild",
  "self_harm_visible": false
}
```

**Response** — `200 OK` — updated preference object.

---

### Posts

#### `POST /api/v1/posts`

Upload a new image or video post. Classification runs asynchronously in the background.

**Request** — `multipart/form-data` (requires `Authorization: Bearer <token>`)

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | Image (`jpg`, `png`, `webp`) or video (`mp4`, `mov`, `avi`, `mkv`, `webm`) |
| `caption` | string | no | Optional post caption |

**Response** — `202 Accepted`

```json
{
  "post_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "uploaded",
  "message": "Post uploaded. Classification is running in the background."
}
```

Classification updates the post `status` to `published` (or `needs_review` / `error`) when complete.

---

#### `GET /api/v1/posts/{post_id}`

Retrieve a post with its full classification result.

**Response** — `200 OK`

```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "user_id": "...",
  "media_type": "image",
  "media_path": "media/images/3fa85f64.jpg",
  "caption": "Sunset hike",
  "status": "published",
  "nudity_level": "safe",
  "violence_level": "none",
  "self_harm_level": "none",
  "age_group": "adult",
  "risk": "allow",
  "classification_confidence": 0.95,
  "content_description": "A person standing on a mountain trail at sunset...",
  "display_tags": ["hiking", "sunset", "mountain", "outdoor"],
  "mood": "peaceful",
  "scene_type": "outdoor",
  "deepface_age": 29,
  "deepface_age_group": "adult",
  "created_at": "2026-04-15T10:00:00Z",
  "classified_at": "2026-04-15T10:00:42Z"
}
```

Returns `404` if the post does not exist, `202` with `status: "uploaded"` if classification is still running.

---

#### `POST /api/v1/posts/{post_id}/classify`

Trigger (or re-trigger) classification for an existing post. Useful for posts seeded directly into the database with `status="uploaded"` or posts that previously errored.

**Response** — `202 Accepted`

```json
{
  "post_id": "3fa85f64-...",
  "message": "Classification started in background."
}
```

Returns `404` if post not found, `400` if post is already `published`.

---

### Image Classification (standalone)

Classify a file without creating a post in the database. Useful for testing the pipeline directly.

#### `POST /api/v1/image_classification/image-classification`

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `image` | file | Image (`jpg`, `png`, `webp`, `gif`) or video (`mp4`, `mov`, `avi`, `mkv`, `webm`) |

**Response** — `200 OK` — full classification JSON (same fields as Post detail above).

#### `POST /api/v1/image_classification/image-classification-hybrid`

Age-focused classification using the biological maturity prompt.

**Response**

```json
{
  "maturity": "adult",
  "confidence": 0.95,
  "observations": ["Subject appears to be over 18 based on facial structure and body proportions."]
}
```

---

## Seeding

Two scripts populate the database with real Reddit content for development and testing.

### `scripts/seed_from_reddit.py`

Lightweight seeder — downloads media from Reddit and stores raw posts without running the classification pipeline.

```bash
python scripts/seed_from_reddit.py \
  --subreddits fitness,yoga,photography \
  --limit 50 \
  --skip-videos
```

| Flag | Default | Description |
|---|---|---|
| `--subreddits` | all 49 | Comma-separated subreddit list |
| `--limit` | 100 | Max posts per subreddit |
| `--skip-videos` | false | Skip video posts |

Progress is saved to `scripts/seed_progress.json` — re-running resumes where it left off.

---

### `scripts/seed_and_classify.py`

Full pipeline seeder. Creates 100 synthetic users with real names, downloads media concurrently via aiohttp, and classifies each post in parallel pipeline workers backed by a Redis job queue.

```bash
python scripts/seed_and_classify.py --workers 4
```

| Flag | Default | Description |
|---|---|---|
| `--workers` | `4` | Number of parallel pipeline workers |
| `--subreddits` | all 49 | Comma-separated subreddit list |
| `--limit` | 5000 | Total target post count |
| `--skip-videos` | false | Skip video posts |
| `--reset-redis` | false | Flush Redis state and start fresh |
| `--only-classify` | false | Skip downloading; classify posts already in DB |

**How it works:**

```
asyncpraw producer
  → fetch submissions (49 subreddits)
  → aiohttp concurrent downloads (25 simultaneous, Semaphore-gated)
  → save to media/images/ or media/videos/
  → create Post row (status="uploaded")
  → RPUSH to Redis seed:queue
                 ↓
pipeline workers (N, concurrent)
  → BLPOP from seed:queue
  → asyncio.to_thread(ImagePipeline / VideoPipeline)
  → commit classification fields (status → "published")
  → commit embedding (non-fatal — won't block publish on Ollama hiccup)
```

**Back-pressure:** The producer pauses when the queue depth exceeds 200 to prevent unbounded memory growth.

**Resumability:** Reddit submission IDs are tracked in Redis `seed:seen` SET. Re-running the script skips already-downloaded posts.

**Media output:**

```
media/
├── images/      # {uuid}.jpg
├── videos/      # {uuid}.mp4
└── thumbnails/  # {uuid}.jpg  (video thumbnails, extracted via ffmpeg)
```

---

## Running Tests

```bash
# Activate environment
source env/bin/activate

# Run video pipeline tests (synthetic video generators, no test fixtures needed)
pytest tests/test_video_pipeline.py -v -s
```

The `-s` flag prints frame selection cascade breakdowns — useful for tuning threshold values in `config.yml`.

---

## Project Structure

```
Social_Media_Content/
├── config/
│   └── config.yml                        # All pipeline configuration
├── scripts/
│   ├── seed_from_reddit.py               # Lightweight Reddit downloader
│   └── seed_and_classify.py              # Full pipeline seeder with Redis queue
├── src/
│   ├── api/
│   │   ├── main.py                       # FastAPI app, startup, middleware
│   │   ├── routes/
│   │   │   ├── auth.py                   # Register, login, refresh, logout
│   │   │   ├── onboarding.py             # User preferences setup
│   │   │   ├── posts.py                  # Upload, retrieve, classify posts
│   │   │   └── image_classification.py   # Standalone classification endpoint
│   │   └── schema/
│   │       ├── posts.py                  # Post request/response models
│   │       └── image_classification.py   # Classification response models
│   └── db/
│       ├── base.py                       # SQLAlchemy declarative base
│       ├── session.py                    # Async engine + session factory
│       ├── redis.py                      # Redis connection lifecycle
│       └── models/
│           ├── user.py                   # User, UserPreference, UserInterestProfile
│           ├── post.py                   # Post, PostEmbedding, PostFrameResult
│           └── cluster.py               # ClusterCentroid (for K-means feed ranking)
├── pipeline/
│   ├── image_pipeline.py                 # Single-image classification
│   ├── video_pipeline.py                 # 7-stage video orchestration
│   ├── frame_selector.py                 # 7-stage frame filtering cascade
│   ├── aggregator.py                     # Multi-frame result aggregation
│   ├── embedding.py                      # nomic-embed-text embedding generation
│   └── audio_pipeline.py                # Audio extraction + Whisper transcription
├── utils/
│   ├── ollama_llm_provider.py            # Ollama chat wrapper and JSON extraction
│   ├── image_prompts.py                  # LLM prompt variants
│   ├── predict_age.py                    # DeepFace age analyzer
│   ├── logger.py                         # Rotating file logger setup
│   └── common_functions.py              # YAML config loader
├── tests/
│   └── test_video_pipeline.py
├── media/                                # Downloaded media (git-ignored)
│   ├── images/
│   ├── videos/
│   └── thumbnails/
├── logs/                                 # Runtime logs (auto-created, rotated at 10 MB)
├── .env                                  # Local secrets (git-ignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── setup.py
```

---

## Technical Notes

**Why local Ollama instead of a cloud LLM?**
All inference runs on-device, eliminating per-token costs, network latency, and data-privacy concerns for sensitive content moderation workloads.

**Why two DB commits per post?**
Classification fields (`nudity_level`, `risk`, `display_tags`, etc.) and the embedding are committed in separate transactions. If Ollama is temporarily unavailable during embedding generation, the post still transitions to `published` — it just won't have a vector until the next classify call. This keeps the pipeline fault-tolerant.

**Why two-pass LLM classification for video?**
Safety fields are checked first. If a frame is classified as `illegal` or `block`, the content (description, tags, mood) pass is skipped — saving tokens and reducing latency on clearly violating content.

**Why is `torch` commented out in requirements?**
PyTorch and TensorFlow both install custom memory allocators that conflict when loaded in the same process. DeepFace requires TensorFlow; CLIP and TransNetV2 require PyTorch. The current setup uses Ollama's embedding API instead of `open-clip-torch` to avoid this conflict. Re-enabling CLIP requires running it in a subprocess or separate service.

**Frame selection order matters**
The 7-stage cascade is ordered cheapest-first: pHash and histogram comparisons are O(1) per frame; SSIM and optical flow are significantly more expensive. Most duplicate frames are eliminated before the costly stages run.

**Why asyncio.to_thread() in the pipeline workers?**
The image/video pipeline code is synchronous (OpenCV, DeepFace, Whisper). Running it directly in an async worker would block the event loop. `asyncio.to_thread()` offloads it to a thread pool so the async producer loop and multiple workers can run concurrently without stalling.

**Vector dimensions: 768**
`nomic-embed-text` produces 768-dimensional vectors. All pgvector columns (`post_embeddings.embedding`, `user_interest_profiles.taste_embedding`, `cluster_centroids.centroid`) use `Vector(768)`. The HNSW index on `post_embeddings` is configured with `m=16, ef_construction=64` for fast cosine similarity search.
