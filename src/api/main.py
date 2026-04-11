import time
from fastapi import FastAPI, Request
from sqlalchemy import text

from src.api.routes import image_classification
from src.db.base import Base
from src.db.session import engine
import src.db.models  # noqa: F401 — registers all models with Base.metadata
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

app = FastAPI()

app.include_router(image_classification.api_router,prefix="/api/v1/image_classification",tags=["Image Classification"])


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
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def on_shutdown():
    await engine.dispose()
    logger.info("Application shutdown")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health_check():
    return {"status": "ok"}