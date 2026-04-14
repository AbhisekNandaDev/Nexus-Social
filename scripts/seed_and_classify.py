"""
seed_and_classify.py — Full pipeline seeder.

Pulls ~5,000 posts from Reddit, downloads media concurrently, runs them
through the ImagePipeline / VideoPipeline, and stores everything to
PostgreSQL. Redis is used for:
  • Deduplication (seen post IDs survive restarts)
  • Pipeline job queue (producer ↔ consumer decoupled)
  • Atomic progress counters
  • Per-subreddit rate-limit tokens

Usage:
    python scripts/seed_and_classify.py
    python scripts/seed_and_classify.py --subreddits EarthPorn,yoga --limit 50
    python scripts/seed_and_classify.py --skip-videos --workers 6
    python scripts/seed_and_classify.py --only-classify      # skip download, drain queue
    python scripts/seed_and_classify.py --reset-redis        # wipe Redis state and restart

Flags:
    --subreddits  Comma-separated list (default: full 49-subreddit catalogue)
    --limit       Posts per subreddit (default: per-catalogue target)
    --skip-videos Images only — faster for testing
    --workers     Parallel pipeline workers (default: 4)
    --only-classify  Drain existing Redis queue without new downloads
    --reset-redis    Clear all seed:* keys before starting
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path as _Path

# Ensure local source tree takes precedence over any installed egg/package
_PROJECT_ROOT = str(_Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import uuid
from pathlib import Path

import aiohttp
import asyncpraw
import redis.asyncio as aioredis
from dotenv import load_dotenv
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent.parent / ".env")

BASE_DIR         = Path(__file__).resolve().parent.parent
MEDIA_IMAGES     = BASE_DIR / "media" / "images"
MEDIA_VIDEOS     = BASE_DIR / "media" / "videos"
MEDIA_THUMBNAILS = BASE_DIR / "media" / "thumbnails"

for _d in (MEDIA_IMAGES, MEDIA_VIDEOS, MEDIA_THUMBNAILS):
    _d.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("seeder")

# ── Redis keys ─────────────────────────────────────────────────────────────────
RK_SEEN     = "seed:seen"          # SET  — reddit post IDs already downloaded
RK_QUEUE    = "seed:queue"         # LIST — pipeline job queue
RK_STATS    = "seed:stats"         # HASH — progress counters
RK_ERRORS   = "seed:errors"        # LIST — error messages (capped at 500)
RK_USERS    = "seed:users"         # LIST — pool of user_id strings
MAX_ERRORS  = 500

# ── 100 Real-name synthetic users ─────────────────────────────────────────────
REAL_NAMES = [
    "Arjun Sharma",      "Priya Nair",        "Mohammed Al-Farsi",  "Sofia Andersson",
    "James Okafor",      "Yuki Tanaka",        "Isabella Rossi",     "Carlos Mendez",
    "Amara Diallo",      "Noah Williams",      "Zara Ahmed",         "Liam Chen",
    "Fatima Hassan",     "Ethan Brooks",       "Mei Lin",            "Oliver Schmidt",
    "Aisha Patel",       "Lucas Fernandez",    "Hana Yamamoto",      "Mateo Garcia",
    "Selin Yilmaz",      "Kai Nakamura",       "Elena Petrov",       "Daniel Osei",
    "Layla Abdullah",    "Marcus Johnson",      "Yuna Kim",           "Rafael Souza",
    "Ingrid Larsson",    "Adebayo Olu",        "Camille Dubois",     "Ravi Krishnan",
    "Nadia Kowalski",    "Samuel Mensah",       "Rin Tanaka",         "Valentina Reyes",
    "Tariq Mahmoud",     "Grace Adeyemi",       "Hugo Bernard",       "Ananya Iyer",
    "Tyler Watson",      "Nur Hidayah",         "Ben Fischer",        "Chioma Eze",
    "Leila Nazari",      "Jayden Park",         "Emma Johansson",     "Kofi Asante",
    "Paula Jimenez",     "Mia Novak",           "Hamza Khalid",       "Ivy Zhou",
    "Leo Marques",       "Sara Lindqvist",      "Usman Ibrahim",      "Chloe Beaumont",
    "Akira Hayashi",     "Dani Ruiz",           "Fatou Diop",         "Ryan Murphy",
    "Manon Leclerc",     "Ibrahima Balde",      "Noa Levi",           "Aiden Clarke",
    "Yara El-Amin",      "Sven Eriksson",       "Tina Nguyen",        "Felix Hartmann",
    "Abena Mensah",      "Luca Bianchi",        "Shreya Kapoor",      "Omar El-Rashid",
    "Alicia Moreno",     "Dmitri Volkov",       "Hina Ito",           "Kwame Asiedu",
    "Lia Santos",        "Nour Khalil",         "Jack Thompson",      "Mei-Ling Wu",
    "Amira Toure",       "Diego Romero",        "Petra Horak",        "Kiran Reddy",
    "Bintou Coulibaly",  "Theo Martin",         "Yuki Suzuki",        "Ana Florea",
    "Seun Adeyinka",     "Clara Hoffman",       "Bilal Chaudhry",     "Zoe Papadopoulos",
    "Max Richter",       "Adaeze Obi",          "Leon Dubois",        "Sana Mir",
    "Elias Nguyen",      "Riya Malhotra",       "Kenji Matsumoto",    "Amara Jallow",
]

# ── Subreddit catalogue ────────────────────────────────────────────────────────
SUBREDDIT_CATALOGUE = [
    ("fitness",             "fitness", 100), ("yoga",                "fitness", 100),
    ("crossfit",            "fitness", 100), ("bodybuilding",        "fitness", 100),
    ("running",             "fitness", 100), ("FoodPorn",            "food",    100),
    ("cooking",             "food",    100), ("Baking",              "food",    100),
    ("streetfood",          "food",    100), ("GifRecipes",          "food",    100),
    ("travel",              "travel",  100), ("EarthPorn",           "nature",  100),
    ("hiking",              "travel",  100), ("CampingandHiking",    "travel",  100),
    ("NatureIsFuckingLit",  "nature",  100), ("CityPorn",            "travel",  100),
    ("Art",                 "art",     100), ("photography",         "art",     100),
    ("DigitalArt",          "art",     100), ("itookapicture",       "art",     100),
    ("streetwear",          "fashion", 100), ("MakeupAddiction",     "fashion", 100),
    ("malefashionadvice",   "fashion", 100), ("battlestations",      "tech",    100),
    ("pcmasterrace",        "tech",    100), ("gadgets",             "tech",    100),
    ("aww",                 "pets",    100), ("cats",                "pets",    100),
    ("dogs",                "pets",    100), ("Aquariums",           "pets",    100),
    ("memes",               "memes",   100), ("dankmemes",           "memes",   100),
    ("funny",               "memes",   100), ("sports",              "sports",  100),
    ("soccer",              "sports",  100), ("nba",                 "sports",  100),
    ("Cricket",             "sports",  100), ("gaming",              "gaming",  100),
    ("GamePhysics",         "gaming",  100), ("retrogaming",         "gaming",  100),
    ("carporn",             "auto",    100), ("motorcycles",         "auto",    100),
    ("CozyPlaces",          "home",    100), ("RoomPorn",            "home",    100),
    ("gardening",           "home",    100), ("guitar",              "music",   100),
    ("MusicBattlestations", "music",   100), ("GoneMild",            "nsfw",    100),
    ("lingerie",            "nsfw",    100),
]

# ── Media config ───────────────────────────────────────────────────────────────
IMAGE_HOSTS      = ("i.redd.it", "i.imgur.com", "preview.redd.it")
IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
VIDEO_MAX_SECS   = 120
VIDEO_MAX_MB     = 100
THUMBNAIL_SIZE   = (400, 400)
MIN_IMAGE_DIM    = 200
DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=45)
DL_CONCURRENCY   = 25    # max simultaneous HTTP downloads
MAX_QUEUE_SIZE   = 200   # back-pressure: pause producer when queue is large


# ── Helpers — URL resolution ───────────────────────────────────────────────────

def _image_url(submission) -> str | None:
    from urllib.parse import urlparse
    url  = getattr(submission, "url", "") or ""
    host = urlparse(url).netloc.lower()
    ext  = Path(urlparse(url).path).suffix.lower()
    if any(host.endswith(h) for h in IMAGE_HOSTS) and ext in IMAGE_EXTS:
        return url
    if "imgur.com" in host and not ext:
        return url + ".jpg"
    preview = getattr(submission, "preview", None)
    if preview:
        images = preview.get("images", [])
        if images:
            src = images[0].get("source", {})
            u = src.get("url", "").replace("&amp;", "&")
            if u:
                return u
    return None


def _video_info(submission) -> tuple[str, int] | None:
    """Returns (fallback_url, duration) or None."""
    if not (getattr(submission, "is_video", False) and getattr(submission, "media", None)):
        return None
    rv = (submission.media or {}).get("reddit_video", {})
    duration = rv.get("duration", 9999)
    url = rv.get("fallback_url")
    if url and duration <= VIDEO_MAX_SECS:
        return url, duration
    return None


# ── Helpers — disk I/O (sync, run in thread) ──────────────────────────────────

def _save_image_sync(data: bytes) -> tuple[str, str | None] | None:
    uid        = str(uuid.uuid4())
    img_path   = MEDIA_IMAGES / f"{uid}.jpg"
    thumb_path = MEDIA_THUMBNAILS / f"{uid}.jpg"
    try:
        import io
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                return None
            rgb = im.convert("RGB")
            rgb.save(img_path, "JPEG", quality=90)
            # Thumbnail
            canvas = Image.new("RGB", THUMBNAIL_SIZE, (0, 0, 0))
            rgb.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            offset = (
                (THUMBNAIL_SIZE[0] - rgb.width) // 2,
                (THUMBNAIL_SIZE[1] - rgb.height) // 2,
            )
            canvas.paste(rgb, offset)
            canvas.save(thumb_path, "JPEG", quality=85)
    except Exception as exc:
        log.debug("Image save error: %s", exc)
        return None
    rel = lambda p: str(p.relative_to(BASE_DIR))
    return rel(img_path), rel(thumb_path)


def _save_video_sync(video_data: bytes, audio_data: bytes | None) -> tuple[str, str | None] | None:
    uid      = str(uuid.uuid4())
    vid_raw  = MEDIA_VIDEOS / f"{uid}_v.mp4"
    aud_raw  = MEDIA_VIDEOS / f"{uid}_a.mp4"
    merged   = MEDIA_VIDEOS / f"{uid}.mp4"
    thumb    = MEDIA_THUMBNAILS / f"{uid}.jpg"

    vid_raw.write_bytes(video_data)
    has_audio = audio_data is not None
    if has_audio:
        aud_raw.write_bytes(audio_data)

    out = vid_raw
    try:
        cmd = ["ffmpeg", "-y", "-i", str(vid_raw)]
        if has_audio:
            cmd += ["-i", str(aud_raw), "-c:v", "copy", "-c:a", "aac", str(merged)]
        else:
            cmd += ["-c:v", "copy", str(merged)]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode == 0:
            vid_raw.unlink(missing_ok=True)
            if has_audio:
                aud_raw.unlink(missing_ok=True)
            out = merged
    except Exception:
        pass

    # First-frame thumbnail
    thumb_rel = None
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out), "-vframes", "1",
             "-vf", f"scale=400:400:force_original_aspect_ratio=decrease,"
                    "pad=400:400:(ow-iw)/2:(oh-ih)/2", str(thumb)],
            capture_output=True, timeout=30,
        )
        if thumb.exists():
            thumb_rel = str(thumb.relative_to(BASE_DIR))
    except Exception:
        pass

    return str(out.relative_to(BASE_DIR)), thumb_rel


# ── Redis state wrapper ────────────────────────────────────────────────────────

class RedisState:
    def __init__(self, redis: aioredis.Redis):
        self.r = redis

    async def is_seen(self, reddit_id: str) -> bool:
        return bool(await self.r.sismember(RK_SEEN, reddit_id))

    async def mark_seen(self, reddit_id: str) -> None:
        await self.r.sadd(RK_SEEN, reddit_id)

    async def seen_count(self) -> int:
        return await self.r.scard(RK_SEEN)

    async def push_job(self, job: dict) -> None:
        await self.r.rpush(RK_QUEUE, json.dumps(job))

    async def pop_job(self, timeout: int = 3) -> dict | None:
        result = await self.r.blpop(RK_QUEUE, timeout=timeout)
        if result:
            return json.loads(result[1])
        return None

    async def queue_len(self) -> int:
        return await self.r.llen(RK_QUEUE)

    async def incr(self, field: str, by: int = 1) -> None:
        await self.r.hincrby(RK_STATS, field, by)

    async def get_stats(self) -> dict:
        raw = await self.r.hgetall(RK_STATS)
        return {k: int(v) for k, v in raw.items()}

    async def log_error(self, msg: str) -> None:
        await self.r.lpush(RK_ERRORS, msg)
        await self.r.ltrim(RK_ERRORS, 0, MAX_ERRORS - 1)

    async def reset(self) -> None:
        await self.r.delete(RK_SEEN, RK_QUEUE, RK_STATS, RK_ERRORS, RK_USERS)
        log.info("Redis seed keys cleared")

    async def cache_users(self, user_ids: list[str]) -> None:
        await self.r.delete(RK_USERS)
        await self.r.rpush(RK_USERS, *user_ids)

    async def random_user_id(self) -> str | None:
        """Return a random user_id from the cached pool."""
        import random
        count = await self.r.llen(RK_USERS)
        if not count:
            return None
        idx = random.randint(0, count - 1)
        return await self.r.lindex(RK_USERS, idx)


# ── User creation ─────────────────────────────────────────────────────────────

async def ensure_real_users(db: AsyncSession, state: RedisState) -> list[str]:
    """Create 100 synthetic real-name users if they don't exist. Cache IDs in Redis."""
    from src.db.models.user import User

    # Check if already cached
    cached_count = await state.r.llen(RK_USERS)
    if cached_count >= len(REAL_NAMES):
        ids = await state.r.lrange(RK_USERS, 0, -1)
        log.info("Users loaded from Redis cache (%d)", len(ids))
        return ids

    log.info("Creating/loading %d real-name synthetic users...", len(REAL_NAMES))
    user_ids: list[str] = []

    for name in REAL_NAMES:
        username = name.lower().replace(" ", "_")
        user = await db.scalar(
            select(User).where(
                User.source_platform == "seed",
                User.source_username == username,
            )
        )
        if user is None:
            user = User(
                display_name=name,
                is_synthetic=True,
                source_platform="seed",
                source_username=username,
            )
            db.add(user)
            await db.flush()
        user_ids.append(str(user.id))

    await db.commit()
    await state.cache_users(user_ids)
    log.info("Users ready — %d in pool", len(user_ids))
    return user_ids


# ── Async HTTP download ────────────────────────────────────────────────────────

async def _fetch(session: aiohttp.ClientSession, url: str, max_mb: float = 110) -> bytes | None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; seeder/1.0)"}
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                return None
            data = b""
            async for chunk in resp.content.iter_chunked(1 << 20):
                data += chunk
                if len(data) > max_mb * 1024 * 1024:
                    return None
            return data
    except Exception as exc:
        log.debug("Fetch error: %s — %s", url, exc)
        return None


# ── Producer: Reddit → download → DB → queue ─────────────────────────────────

async def producer(
    reddit: asyncpraw.Reddit,
    db: AsyncSession,
    state: RedisState,
    catalogue: list,
    override_limit: int | None,
    skip_videos: bool,
    total_target: int,
    done_event: asyncio.Event,
) -> None:
    semaphore = asyncio.Semaphore(DL_CONCURRENCY)

    async def process_submission(session, submission, sub_name, user_id):
        reddit_id = submission.id
        if await state.is_seen(reddit_id):
            return

        img_url  = _image_url(submission) if not submission.is_self else None
        vid_info = _video_info(submission) if not skip_videos else None

        if not img_url and not vid_info:
            return

        # ── Download ──────────────────────────────────────────────────────────
        media_path = thumb_path = None
        media_type = None

        async with semaphore:
            if img_url:
                data = await _fetch(session, img_url)
                if data is None:
                    await state.incr("errors")
                    await state.log_error(f"img_fetch_fail:{reddit_id}:{img_url}")
                    return
                result = await asyncio.to_thread(_save_image_sync, data)
                if result is None:
                    await state.incr("errors")
                    return
                media_path, thumb_path = result
                media_type = "image"

            elif vid_info:
                fallback_url, _ = vid_info
                video_data = await _fetch(session, fallback_url, max_mb=VIDEO_MAX_MB)
                if video_data is None:
                    await state.incr("errors")
                    return
                # Try audio stream
                base = fallback_url.split("?")[0]
                audio_url = "/".join(base.split("/")[:-1]) + "/DASH_audio.mp4"
                audio_data = await _fetch(session, audio_url, max_mb=20)
                result = await asyncio.to_thread(
                    _save_video_sync, video_data, audio_data
                )
                if result is None:
                    await state.incr("errors")
                    return
                media_path, thumb_path = result
                media_type = "video"

        # ── Create Post row ───────────────────────────────────────────────────
        from src.db.models.post import Post
        post = Post(
            user_id=uuid.UUID(user_id),
            media_type=media_type,
            media_path=media_path,
            thumbnail_path=thumb_path,
            caption=submission.title[:2000] if submission.title else None,
            status="uploaded",
            source_url=f"https://reddit.com{submission.permalink}",
            source_platform="reddit",
            source_subreddit=sub_name,
            source_upvotes=submission.score,
            source_comments=submission.num_comments,
        )
        db.add(post)
        await db.commit()

        # ── Mark seen + push pipeline job ─────────────────────────────────────
        await state.mark_seen(reddit_id)
        await state.incr("downloaded")
        await state.incr(media_type + "s")

        job = {
            "post_id":    str(post.id),
            "media_type": media_type,
            "media_path": media_path,
            "caption":    submission.title,
        }
        await state.push_job(job)

        downloaded = (await state.get_stats()).get("downloaded", 0)
        log.info("↓ %d/%d  r/%s  [%s]  %s",
                 downloaded, total_target, sub_name, media_type,
                 Path(media_path).name)

    # ── Main producer loop ────────────────────────────────────────────────────
    connector = aiohttp.TCPConnector(limit=DL_CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(timeout=DOWNLOAD_TIMEOUT, connector=connector) as session:
        for sub_name, _cat, default_limit in catalogue:
            limit = override_limit if override_limit is not None else default_limit

            try:
                subreddit = await reddit.subreddit(sub_name)
                submissions = []
                async for s in subreddit.top(time_filter="year", limit=min(limit * 4, 500)):
                    submissions.append(s)
                log.info("r/%s — fetched %d submissions", sub_name, len(submissions))
            except Exception as exc:
                log.error("r/%s fetch failed: %s", sub_name, exc)
                continue

            stored_this_sub = 0
            tasks = []

            for submission in submissions:
                if stored_this_sub >= limit:
                    break

                # Back-pressure: don't get too far ahead of the pipeline workers
                while await state.queue_len() > MAX_QUEUE_SIZE:
                    await asyncio.sleep(2)

                user_id = await state.random_user_id()
                if not user_id:
                    continue

                task = asyncio.create_task(
                    process_submission(session, submission, sub_name, user_id)
                )
                tasks.append(task)
                stored_this_sub += 1

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            stats = await state.get_stats()
            log.info("r/%s complete | total=%s images=%s videos=%s errors=%s queue=%d",
                     sub_name,
                     stats.get("downloaded", 0),
                     stats.get("images", 0),
                     stats.get("videos", 0),
                     stats.get("errors", 0),
                     await state.queue_len())

            await asyncio.sleep(1)  # polite gap between subreddits

    log.info("Producer finished — signalling pipeline workers to drain and exit")
    done_event.set()


# ── Consumer: queue → pipeline → DB update ───────────────────────────────────

async def pipeline_worker(
    worker_id: int,
    state: RedisState,
    database_url: str,
    done_event: asyncio.Event,
) -> None:
    from pipeline.image_pipeline import ImagePipeline
    from pipeline.video_pipeline import VideoPipeline
    from pipeline.embedding import EmbeddingGenerator
    from src.db.models.post import Post, PostEmbedding
    from datetime import datetime, timezone

    engine  = create_async_engine(database_url, echo=False, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    log.info("Pipeline worker #%d started", worker_id)
    consecutive_empty = 0

    while True:
        job = await state.pop_job(timeout=3)

        if job is None:
            if done_event.is_set() and await state.queue_len() == 0:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    log.info("Worker #%d: queue empty + producer done — exiting", worker_id)
                    break
            await asyncio.sleep(1)
            continue

        consecutive_empty = 0
        post_id    = uuid.UUID(job["post_id"])
        media_type = job["media_type"]
        media_path = str(BASE_DIR / job["media_path"])
        caption    = job.get("caption") or ""

        try:
            if media_type == "image":
                result_raw = await asyncio.to_thread(
                    lambda: ImagePipeline(media_path, "file").classify()
                )
                fields = {
                    "nudity_level":             result_raw.get("nudity_level"),
                    "nsfw_subcategories":        result_raw.get("nsfw_subcategories", []),
                    "violence_level":            result_raw.get("violence_level"),
                    "violence_type":             result_raw.get("violence_type", []),
                    "self_harm_level":           result_raw.get("self_harm_level"),
                    "self_harm_type":            result_raw.get("self_harm_type", []),
                    "age_group":                 result_raw.get("age_group"),
                    "risk":                      result_raw.get("risk"),
                    "classification_confidence": result_raw.get("confidence"),
                    "content_description":       result_raw.get("content_description"),
                    "display_tags":              result_raw.get("display_tags", []),
                    "mood":                      result_raw.get("mood"),
                    "scene_type":                result_raw.get("scene_type"),
                    "text_in_image":             result_raw.get("text_in_image"),
                    "objects_detected":          result_raw.get("objects_detected", []),
                    "people_count":              str(result_raw.get("people_count", 0)),
                    "deepface_age":              result_raw.get("deepface_age"),
                    "deepface_age_group":        result_raw.get("deepface_age_group"),
                }
                emb_text = (fields.get("content_description") or "") + " " + caption
                embedding = await asyncio.to_thread(
                    EmbeddingGenerator.generate, emb_text.strip()
                )

            else:  # video
                vp = VideoPipeline()
                r  = await asyncio.to_thread(vp.process, media_path, caption)
                fields = {
                    "nudity_level":             r.nudity_level,
                    "nsfw_subcategories":        r.nsfw_subcategories,
                    "violence_level":            r.violence_level,
                    "violence_type":             r.violence_type,
                    "self_harm_level":           r.self_harm_level,
                    "self_harm_type":            r.self_harm_type,
                    "age_group":                 r.age_group,
                    "risk":                      r.risk,
                    "classification_confidence": r.classification_confidence,
                    "content_description":       r.content_description,
                    "display_tags":              r.display_tags,
                    "mood":                      r.mood,
                    "scene_type":                r.scene_type,
                    "text_in_image":             r.text_in_image,
                    "objects_detected":          r.objects_detected,
                    "people_count":              str(r.people_count),
                    "deepface_age":              r.deepface_age,
                    "deepface_age_group":        r.deepface_age_group,
                    "video_duration_seconds":    r.video_duration_seconds,
                    "frames_analyzed":           r.frames_analyzed,
                    "llm_calls_used":            r.llm_calls_used,
                    "needs_review":              r.needs_review,
                    "transcript":                r.transcript,
                    "transcript_language":       r.transcript_language,
                    "transcript_safety_flags":   r.transcript_safety_flags,
                    "secondary_classifications": r.secondary_classifications,
                }
                embedding = r.embedding

        except Exception as exc:
            log.error("Worker #%d pipeline error | post=%s error=%s", worker_id, post_id, exc)
            await state.incr("pipeline_errors")
            await state.log_error(f"pipeline:{post_id}:{exc}")
            # Mark post as error
            async with Session() as db:
                post = await db.get(Post, post_id)
                if post:
                    post.status = "error"
                    await db.commit()
            continue

        # ── Write classification fields + status (always committed) ──────────
        try:
            async with Session() as db:
                post = await db.get(Post, post_id)
                if post is None:
                    continue
                for field, value in fields.items():
                    if hasattr(post, field):
                        setattr(post, field, value)
                post.status        = "needs_review" if fields.get("needs_review") else "published"
                post.classified_at = datetime.now(timezone.utc)
                await db.commit()
        except Exception as exc:
            log.error("Worker #%d classification commit error | post=%s error=%s", worker_id, post_id, exc)
            await state.incr("pipeline_errors")
            continue

        # ── Store embedding separately (non-fatal if dim mismatch) ───────────
        if embedding:
            try:
                async with Session() as db:
                    existing = await db.scalar(
                        select(PostEmbedding).where(PostEmbedding.post_id == post_id)
                    )
                    if existing is None:
                        db.add(PostEmbedding(post_id=post_id, embedding=embedding))
                        await db.commit()
            except Exception as exc:
                log.debug("Embedding store skipped | post=%s reason=%s", post_id, exc)

        await state.incr("classified")
        classified = (await state.get_stats()).get("classified", 0)
        log.info("✓ #%d classified %d | post=%s risk=%s",
                 worker_id, classified, post_id, fields.get("risk"))

    await engine.dispose()
    log.info("Pipeline worker #%d done", worker_id)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    # ── Env / credentials ─────────────────────────────────────────────────────
    client_id     = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent    = os.environ.get("REDDIT_USER_AGENT", "social-media-seeder/1.0")
    reddit_user   = os.environ.get("REDDIT_USERNAME")
    reddit_pass   = os.environ.get("REDDIT_PASSWORD")
    redis_url     = os.environ.get("REDIS_URL", "redis://localhost:6379")
    database_url  = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5433/social_media_content",
    )
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    if not client_id or not client_secret:
        log.error("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET missing in .env")
        sys.exit(1)

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_client = aioredis.from_url(redis_url, decode_responses=True)
    try:
        await redis_client.ping()
        log.info("Redis OK — %s", redis_url)
    except Exception as exc:
        log.error("Redis connection failed: %s", exc)
        sys.exit(1)
    state = RedisState(redis_client)

    if args.reset_redis:
        await state.reset()

    # ── DB ────────────────────────────────────────────────────────────────────
    engine  = create_async_engine(database_url, echo=False, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    from sqlalchemy import text as sa_text
    try:
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        log.info("DB OK — %s", database_url.split("@")[-1])
    except Exception as exc:
        log.error("DB connection failed: %s", exc)
        sys.exit(1)

    # ── Build catalogue ───────────────────────────────────────────────────────
    catalogue = SUBREDDIT_CATALOGUE
    if args.subreddits:
        requested = {s.strip().lower() for s in args.subreddits.split(",")}
        catalogue  = [(s, c, t) for s, c, t in catalogue if s.lower() in requested]
        if not catalogue:
            log.error("No subreddits matched")
            sys.exit(1)

    override_limit = args.limit
    total_target   = sum(override_limit or t for _, _, t in catalogue)

    log.info(
        "Seeder starting — %d subreddits · ~%d posts · %d pipeline workers%s",
        len(catalogue), total_target, args.workers,
        " · images only" if args.skip_videos else "",
    )

    # ── Create users ──────────────────────────────────────────────────────────
    async with Session() as db:
        await ensure_real_users(db, state)

    # ── Run producer + pipeline workers concurrently ──────────────────────────
    done_event = asyncio.Event()

    async with asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=reddit_user,
        password=reddit_pass,
        ratelimit_seconds=300,
    ) as reddit:
        async with Session() as db:
            workers = [
                asyncio.create_task(
                    pipeline_worker(i + 1, state, database_url, done_event)
                )
                for i in range(args.workers)
            ]

            if not args.only_classify:
                await producer(reddit, db, state, catalogue, override_limit,
                               args.skip_videos, total_target, done_event)
            else:
                log.info("--only-classify: skipping download, draining existing queue")
                done_event.set()

            await asyncio.gather(*workers)

    # ── Final summary (read stats before closing Redis) ───────────────────────
    raw_stats = await redis_client.hgetall(RK_STATS)
    stats = {k: int(v) for k, v in raw_stats.items()}

    await redis_client.aclose()
    await engine.dispose()

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Downloaded         : {stats.get('downloaded', 0)}")
    print(f"    Images           : {stats.get('images', 0)}")
    print(f"    Videos           : {stats.get('videos', 0)}")
    print(f"  Classified         : {stats.get('classified', 0)}")
    print(f"  Download errors    : {stats.get('errors', 0)}")
    print(f"  Pipeline errors    : {stats.get('pipeline_errors', 0)}")
    print(f"  Media location     : {BASE_DIR}/media/")
    print("=" * 65)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed + classify social media DB from Reddit")
    p.add_argument("--subreddits",     default=None, help="Comma-separated subreddit names")
    p.add_argument("--limit",          type=int, default=None, help="Posts per subreddit")
    p.add_argument("--skip-videos",    action="store_true", help="Images only")
    p.add_argument("--workers",        type=int, default=4, help="Parallel pipeline workers (default: 4)")
    p.add_argument("--only-classify",  action="store_true", help="Drain queue only, skip downloads")
    p.add_argument("--reset-redis",    action="store_true", help="Clear Redis seed keys before starting")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
