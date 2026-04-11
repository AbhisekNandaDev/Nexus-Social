# Social Media Content Moderation — Project Summary

## Overview

A **FastAPI-based image moderation service** that classifies social media images for nudity level, subject age group, and risk level. It combines a local LLM (Ollama) with computer-vision-based age detection (DeepFace) to make moderation decisions.

---

## Project Structure

```
Social_Media_Content/
├── config/
│   ├── __init__.py
│   └── config.yml               # Model config and classification class definitions
├── pipeline/
│   ├── __init__.py
│   └── image_pipeline.py        # Standalone pipeline: DeepFace + LLM age prediction
├── src/
│   └── api/
│       ├── main.py              # FastAPI app entry point + request logging middleware
│       ├── routes/
│       │   └── image_classification.py  # POST endpoints
│       └── schema/
│           └── image_classification.py  # Pydantic response models
├── utils/
│   ├── __init__.py
│   ├── common_functions.py      # YAML config loader
│   ├── image_prompts.py         # LLM prompt builders
│   ├── logger.py                # Centralised logging setup
│   ├── ollama_llm_provider.py   # Ollama chat client
│   └── predict_age.py           # DeepFace age estimator
├── logs/
│   └── app.log                  # Rotating log file (auto-created at runtime)
├── data/                        # Sample test images
├── dummy_test/
│   └── test_age.py
├── requirements.txt
└── setup.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| LLM provider | Ollama (local) |
| Vision model | `ministral-3:3b` (configurable) |
| Age detection | DeepFace + OpenCV + tf-keras |
| Data validation | Pydantic v2 |
| Configuration | PyYAML |
| Logging | Python `logging` + `RotatingFileHandler` |

---

## API Endpoints

### `GET /`
Health ping — returns `{"Hello": "World"}`.

### `GET /health`
Health check — returns `{"status": "ok"}`.

### `POST /api/v1/image_classification/image-classification`
Full moderation classification using the LLM.

**Request:** `multipart/form-data` with an `image` file field.

**Response:**
```json
{
  "nudity_level": "safe | suggestive | partial_nudity | explicit_nudity | sexual_activity",
  "age_group": "child | teen | adult | unknown",
  "risk": "allow | restrict | nsfw | illegal",
  "reason": "short explanation"
}
```

### `POST /api/v1/image_classification/image-classification-hybrid`
Age-focused classification using the LLM biological maturity prompt.

**Request:** `multipart/form-data` with an `image` file field.

**Response:**
```json
{
  "maturity": "child | teen | adult | unknown",
  "confidence": 0.0,
  "observations": ["list of visual features"]
}
```

---

## Classification Logic

### Nudity Levels

| Value | Description |
|---|---|
| `safe` | No sexual content, fully clothed |
| `suggestive` | Provocative but no nudity (cleavage, tight clothes, poses) |
| `partial_nudity` | Skin exposed but private parts covered (sheer, thong, etc.) |
| `explicit_nudity` | Visible private parts (nipples, genitals, bare buttocks) |
| `sexual_activity` | Intercourse, oral sex, masturbation, ejaculation |

### Risk Matrix

| Nudity Level | Child | Teen | Adult |
|---|---|---|---|
| safe | allow | allow | allow |
| suggestive | block | restrict | allow |
| partial_nudity | illegal | illegal | nsfw |
| explicit_nudity | illegal | illegal | nsfw |
| sexual_activity | illegal | illegal | nsfw |

### Age Groups

| Value | Range |
|---|---|
| `child` | Under 13 |
| `teen` | 13–17 |
| `adult` | 18+ |
| `unknown` | Cannot determine |

---

## Modules

### `utils/logger.py`
Centralised logging setup. Call once at startup; all other modules call `get_logger(__name__)`.

- **Console handler** — structured output during development
- **Rotating file handler** — `logs/app.log`, 10 MB per file, 5 backups
- **Format:** `YYYY-MM-DD HH:MM:SS | LEVEL | module | func:line | message`
- Suppresses noisy third-party loggers (`httpx`, `httpcore`, `uvicorn.access`)

```python
from utils.logger import setup_logging, get_logger

setup_logging()               # call once at app entry point
logger = get_logger(__name__) # in every module
```

To enable verbose (DEBUG) output:
```python
setup_logging(level="DEBUG")
```

### `utils/ollama_llm_provider.py`
Thin wrapper around `ollama.Client`. Sends a structured message list (system + user + image bytes) to the configured model and extracts JSON from the response via regex.

### `utils/predict_age.py`
Accepts raw image bytes or a numpy array. Decodes bytes → OpenCV BGR image → calls `DeepFace.analyze(actions=['age'])` with `enforce_detection=False`.

### `utils/image_prompts.py`
Builds the `messages` list for Ollama:
- `get_image_classification_prompt(image_bytes)` — full moderation prompt
- `get_age_prediction_prompt(image_bytes)` — biological maturity prompt

### `pipeline/image_pipeline.py`
Standalone pipeline that runs both detection methods in sequence.

```python
pipeline = ImagePipeline("path/to/image.jpg", "file")
deepface_age, llm_response = pipeline.run()

# or from bytes:
pipeline = ImagePipeline(image_bytes, "bytes")
```

### `config/config.yml`
```yaml
image_classification:
  provider: "ollama"
  model: "ministral-3:3b"
```

---

## Running the Server

```bash
# 1. Activate virtual environment
source env/bin/activate

# 2. Install package (required for src.* imports)
pip install -e .

# 3. Start server
uvicorn src.api.main:app --reload
```

Server runs at `http://127.0.0.1:8000`.
Interactive API docs: `http://127.0.0.1:8000/docs`

### Common flags

| Flag | Purpose |
|---|---|
| `--reload` | Auto-restart on code changes (dev only) |
| `--host 0.0.0.0` | Expose on all interfaces |
| `--port 8080` | Change port (default 8000) |
| `--log-level warning` | Suppress uvicorn access logs |

---

## Dependencies (`requirements.txt`)

```
ollama           # Ollama Python client
PyYAML           # YAML config parsing
fastapi          # Web framework
uvicorn          # ASGI server
pydantic         # Request/response validation
python-multipart # File upload support
deepface         # Face analysis and age estimation
tf-keras         # TensorFlow Keras (DeepFace backend)
```

---

## Log Output Example

```
2026-03-12 10:30:01 | INFO     | src.api.main | on_startup:32 | Application startup complete
2026-03-12 10:30:05 | INFO     | src.api.main | log_requests:17 | Request  POST /api/v1/image_classification/image-classification
2026-03-12 10:30:05 | INFO     | src.api.routes.image_classification | image_classification:22 | Image classification request | filename=photo.jpg content_type=image/jpeg
2026-03-12 10:30:05 | INFO     | utils.ollama_llm_provider | __init__:14 | OllamaProvider initialised | model=ministral-3:3b
2026-03-12 10:30:07 | INFO     | utils.ollama_llm_provider | get_response:27 | Parsed JSON from LLM response | result={...}
2026-03-12 10:30:07 | INFO     | src.api.routes.image_classification | image_classification:37 | Classification complete | filename=photo.jpg nudity=safe age_group=adult risk=allow
2026-03-12 10:30:07 | INFO     | src.api.main | log_requests:21 | Response POST /api/v1/image_classification/image-classification | status=200 | 2134.5ms
```

Logs are written to both the console and `logs/app.log` (auto-created).
