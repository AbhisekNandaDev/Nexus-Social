# Social Media Content Moderation API

A FastAPI-based service for automated safety and content moderation of images and videos. It combines a local LLM (Ollama), facial age detection (DeepFace), and audio transcription (Faster-Whisper) to produce structured, multi-dimensional classification results — entirely on-device with no external API dependencies.

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
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Technical Notes](#technical-notes)

---

## Features

- **Image moderation** — Single-pass LLM analysis with nudity, violence, self-harm, and age classification
- **Video moderation** — Full 5-stage pipeline: intelligent frame selection, parallel audio transcription, batched LLM classification, and result aggregation
- **7-stage frame selection cascade** — pHash deduplication → color histogram → MS-SSIM → optical flow → shot boundary detection → diversity selection; cheap filters run first to minimise LLM calls
- **Dual age detection** — DeepFace biological age estimation runs in parallel with LLM-based age group classification for cross-validation
- **Audio-aware analysis** — Faster-Whisper transcribes spoken content; dangerous keyword hits trigger targeted frame re-analysis
- **Semantic embeddings** — Content descriptions are embedded via Ollama (`nomic-embed-text`) and stored pgvector-ready for downstream similarity search
- **Fully local** — LLM inference, transcription, and face analysis all run on-device; no external API calls required
- **Structured JSON responses** — All outputs are validated Pydantic models covering 20+ fields across safety and content dimensions

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │            FastAPI Application           │
                        │         POST /image-classification       │
                        └──────────────────┬──────────────────────┘
                                           │
                          ┌────────────────┴────────────────┐
                          │                                  │
                    ┌─────▼──────┐                   ┌──────▼──────┐
                    │   Image    │                    │    Video    │
                    │  Pipeline  │                    │  Pipeline   │
                    └─────┬──────┘                   └──────┬──────┘
                          │                                  │
               ┌──────────┘               ┌─────────────────┼──────────────────┐
               │                          │                  │                  │
        ┌──────▼──────┐           ┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
        │  Ollama LLM │           │ Frame        │  │ Audio        │  │  Ollama LLM  │
        │  (qwen3.5)  │           │ Selector     │  │ Pipeline     │  │  (batched)   │
        └──────┬──────┘           │ (7 stages)   │  │ (Whisper)    │  └──────┬───────┘
               │                  └───────┬──────┘  └───────┬──────┘         │
        ┌──────▼──────┐                   └────────┬─────────┘        ┌──────▼───────┐
        │  DeepFace   │                            │                   │  Aggregator  │
        │  Age Detect │                     ┌──────▼──────┐           └──────┬───────┘
        └─────────────┘                     │  Embedding  │                  │
                                            │  Generator  │           ┌──────▼───────┐
                                            └─────────────┘           │  PostgreSQL  │
                                                                       │  + pgvector  │
                                                                       └──────────────┘
```

---

## Classification Output

Every request returns a structured JSON response covering:

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
| `is_video` | `true` / `false` |
| `video_duration` | Duration in seconds |
| `video_fps` | Frames per second of source file |
| `frames_sampled` | Frames remaining after cheap filters |
| `frames_classified` | Frames actually sent to LLM |
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
| ffmpeg | Required for video audio extraction (included in Docker image) |

Pull the required Ollama model before starting:

```bash
ollama pull qwen3.5:9b          # LLM for classification
ollama pull nomic-embed-text    # Embedding model
```

---

## Installation

### Option A — Docker (recommended)

This starts the FastAPI app and a PostgreSQL + pgvector database.

**1. Clone the repository**

```bash
git clone <repo-url>
cd Social_Media_Content
```

**2. Start services**

```bash
docker compose up --build
```

The first build downloads Python dependencies, DeepFace weights (~200 MB), and Whisper models (~150 MB). Subsequent starts are fast.

**3. Verify**

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

API docs are available at `http://localhost:8000/docs`.

**Services started:**

| Service | Port | Description |
|---|---|---|
| `app` | `8000` | FastAPI moderation API |
| `db` | `5432` | PostgreSQL 16 + pgvector |

> **Ollama** runs on the host machine. The app container reaches it via `host.docker.internal:11434` (Docker Desktop on macOS/Windows). On Linux, set `OLLAMA_HOST` to your host LAN IP in `docker-compose.yml`.

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
pip install -e .
# or
pip install -r requirements.txt
```

**3. Install system dependencies**

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg libgl1 libglib2.0-0
```

**4. Set environment variables**

```bash
export OLLAMA_HOST=http://localhost:11434
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/social_media_content
```

**5. Start Ollama**

```bash
ollama serve
```

**6. (Optional) Start PostgreSQL with pgvector**

```bash
docker compose up db
```

**7. Start the API server**

```bash
uvicorn src.api.main:app --reload
```

Server is available at `http://127.0.0.1:8000`.

---

## Configuration

All settings live in [config/config.yml](config/config.yml). No restart needed in development (`--reload` picks up changes).

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
  use_transnet: true         # Use TransNetV2 for shot boundary detection
  use_clip: true             # Use CLIP for diversity-based frame selection

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
| `DATABASE_URL` | `postgresql://postgres:postgres@db:5432/social_media_content` | PostgreSQL connection string |

---

## API Reference

### `GET /health`

Health check.

```json
{"status": "ok"}
```

---

### `POST /api/v1/image_classification/image-classification`

Classify an image or video file for safety and content.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `image` | file | Image (`jpg`, `png`, `webp`, `gif`) or video (`mp4`, `mov`, `avi`, `mkv`, `webm`) |

**Response** — `200 OK`

```json
{
  "nudity_level": "safe",
  "nsfw_subcategories": [],
  "violence_level": "none",
  "violence_type": null,
  "self_harm_level": "none",
  "self_harm_type": null,
  "age_group": "adult",
  "risk": "allow",
  "confidence": 0.97,
  "content_description": "Two adults walking along a beach at sunset.",
  "display_tags": ["beach", "sunset", "couple", "outdoor"],
  "mood": "peaceful",
  "scene_type": "outdoor",
  "text_in_image": null,
  "objects_detected": ["person", "ocean", "sand"],
  "people_count": 2,
  "deepface_age": 28,
  "deepface_age_group": "adult",
  "is_video": false
}
```

---

### `POST /api/v1/image_classification/image-classification-hybrid`

Age-focused classification using the biological maturity prompt.

**Request** — `multipart/form-data` — same `image` field.

**Response**

```json
{
  "maturity": "adult",
  "confidence": 0.95,
  "observations": ["Subject appears to be over 18 based on facial structure and body proportions."]
}
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
│   └── config.yml                  # All configuration (model, thresholds, etc.)
├── src/
│   └── api/
│       ├── main.py                 # FastAPI app, middleware, health endpoint
│       ├── routes/
│       │   └── image_classification.py   # Endpoint handlers
│       └── schema/
│           └── image_classification.py   # Pydantic request/response models
├── pipeline/
│   ├── image_pipeline.py           # Single-image and batch classification
│   ├── video_pipeline.py           # 5-stage video orchestration
│   ├── frame_selector.py           # 7-stage frame filtering cascade
│   ├── aggregator.py               # Multi-frame result aggregation
│   ├── embedding.py                # Ollama embedding generation
│   └── audio_pipeline.py           # Audio extraction and Whisper transcription
├── utils/
│   ├── ollama_llm_provider.py      # Ollama chat wrapper and JSON extraction
│   ├── image_prompts.py            # LLM prompt variants (4 types)
│   ├── predict_age.py              # DeepFace age analyzer
│   ├── logger.py                   # Rotating file logger setup
│   └── common_functions.py         # YAML config loader
├── tests/
│   └── test_video_pipeline.py      # Video pipeline tests with synthetic videos
├── data/                           # Sample images for manual testing
├── logs/                           # Runtime logs (auto-created, rotated at 10 MB)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── setup.py
```

---

## Technical Notes

**Why local Ollama instead of a cloud LLM?**
All inference runs on-device, eliminating per-token costs, network latency, and data-privacy concerns for sensitive content moderation workloads.

**Why two-pass LLM classification for video?**
Safety fields are checked first. If a frame is classified as `illegal` or `block`, the content (description, tags, mood) pass is skipped — saving tokens and reducing latency on clearly violating content.

**Why is `torch` commented out in requirements?**
PyTorch and TensorFlow both install custom memory allocators that conflict when loaded in the same process. DeepFace requires TensorFlow; CLIP and TransNetV2 require PyTorch. The current setup uses Ollama's embedding API instead of `open-clip-torch` to avoid this conflict. Re-enabling CLIP requires running it in a subprocess or separate service.

**Frame selection order matters**
The 7-stage cascade is ordered cheapest-first: pHash and histogram comparisons are O(1) per frame; SSIM and optical flow are significantly more expensive. Most duplicate frames are eliminated before the costly stages run.

**pgvector readiness**
Embeddings are generated as 768-dimensional float vectors compatible with `pgvector`. The database schema and storage layer are ready to be wired up — the infrastructure (PostgreSQL + pgvector container, `DATABASE_URL` env var) is already in place.
