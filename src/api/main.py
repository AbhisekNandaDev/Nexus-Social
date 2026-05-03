import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env before any module that reads os.environ (session.py, config.py, redis.py)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from src.api.routes import image_classification
from src.api.routes import auth as auth_routes
from src.api.routes import feed as feed_routes
from src.api.routes import onboarding as onboarding_routes
from src.api.routes import posts as posts_routes
from src.db.base import Base
from src.db.session import engine
from src.db.redis import init_redis, close_redis
import src.db.models  # noqa: F401 — registers all models with Base.metadata
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = FastAPI()

_MEDIA_DIR = Path(__file__).resolve().parents[2] / "media"
_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(_MEDIA_DIR)), name="media")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_classification.api_router, prefix="/api/v1/image_classification", tags=["Image Classification"])
app.include_router(auth_routes.api_router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(onboarding_routes.api_router, prefix="/api/v1/onboarding", tags=["Onboarding"])
app.include_router(posts_routes.api_router, prefix="/api/v1/posts", tags=["Posts"])
app.include_router(feed_routes.api_router, prefix="/api/v1/feed", tags=["Feed"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("Request  %s %s", request.method, request.url.path)
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Response %s %s | status=%d | %.1fms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        # Enable pgvector extension (idempotent)
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create all tables that don't already exist
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database ready — pgvector extension enabled, all tables created")

    await init_redis()
    logger.info("Redis ready")

    logger.info("Application startup complete")


@app.on_event("shutdown")
async def on_shutdown():
    await engine.dispose()
    await close_redis()
    logger.info("Application shutdown")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health_check():
    return {"status": "ok"}