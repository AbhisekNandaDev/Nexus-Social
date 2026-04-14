"""
seed_from_reddit.py — Download ~5,000 image/video posts from Reddit and store
them in PostgreSQL with synthetic users.

Usage:
    python scripts/seed_from_reddit.py
    python scripts/seed_from_reddit.py --subreddits fitness,yoga --limit 50
    python scripts/seed_from_reddit.py --skip-videos
    python scripts/seed_from_reddit.py --subreddits memes --limit 20

Requires: asyncpraw, requests, Pillow, python-dotenv, sqlalchemy[asyncio], asyncpg
FFmpeg must be installed and on PATH for video merging.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import asyncpraw
import requests
from dotenv import load_dotenv
from PIL import Image
from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_IMAGES    = BASE_DIR / "media" / "images"
MEDIA_VIDEOS    = BASE_DIR / "media" / "videos"
MEDIA_THUMBNAILS = BASE_DIR / "media" / "thumbnails"
PROGRESS_FILE   = BASE_DIR / "scripts" / "seed_progress.json"

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

# ── Subreddit catalogue ────────────────────────────────────────────────────────
# (subreddit_name, category, target_posts)
SUBREDDIT_CATALOGUE = [
    # Fitness (500)
    ("fitness",             "fitness", 100),
    ("yoga",                "fitness", 100),
    ("crossfit",            "fitness", 100),
    ("bodybuilding",        "fitness", 100),
    ("running",             "fitness", 100),
    # Food (500)
    ("FoodPorn",            "food",    100),
    ("cooking",             "food",    100),
    ("Baking",              "food",    100),
    ("streetfood",          "food",    100),
    ("GifRecipes",          "food",    100),
    # Travel & Nature (600)
    ("travel",              "travel",  100),
    ("EarthPorn",           "nature",  100),
    ("hiking",              "travel",  100),
    ("CampingandHiking",    "travel",  100),
    ("NatureIsFuckingLit",  "nature",  100),
    ("CityPorn",            "travel",  100),
    # Art (400)
    ("Art",                 "art",     100),
    ("photography",         "art",     100),
    ("DigitalArt",          "art",     100),
    ("itookapicture",       "art",     100),
    # Fashion (300)
    ("streetwear",          "fashion", 100),
    ("MakeupAddiction",     "fashion", 100),
    ("malefashionadvice",   "fashion", 100),
    # Tech (300)
    ("battlestations",      "tech",    100),
    ("pcmasterrace",        "tech",    100),
    ("gadgets",             "tech",    100),
    # Pets (400)
    ("aww",                 "pets",    100),
    ("cats",                "pets",    100),
    ("dogs",                "pets",    100),
    ("Aquariums",           "pets",    100),
    # Memes (300)
    ("memes",               "memes",   100),
    ("dankmemes",           "memes",   100),
    ("funny",               "memes",   100),
    # Sports (400)
    ("sports",              "sports",  100),
    ("soccer",              "sports",  100),
    ("nba",                 "sports",  100),
    ("Cricket",             "sports",  100),
    # Gaming (300)
    ("gaming",              "gaming",  100),
    ("GamePhysics",         "gaming",  100),
    ("retrogaming",         "gaming",  100),
    # Auto (200)
    ("carporn",             "auto",    100),
    ("motorcycles",         "auto",    100),
    # Home (300)
    ("CozyPlaces",          "home",    100),
    ("RoomPorn",            "home",    100),
    ("gardening",           "home",    100),
    # Music (200)
    ("guitar",              "music",   100),
    ("MusicBattlestations", "music",   100),
    # NSFW test (200)
    ("GoneMild",            "nsfw",    100),
    ("lingerie",            "nsfw",    100),
]

# ── Config ─────────────────────────────────────────────────────────────────────
IMAGE_HOSTS      = ("i.redd.it", "i.imgur.com", "preview.redd.it")
IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
VIDEO_MAX_SECS   = 120
VIDEO_MAX_MB     = 100
THUMBNAIL_SIZE   = (400, 400)
MIN_IMAGE_DIM    = 200
DOWNLOAD_TIMEOUT = 30


# ── Progress tracker ───────────────────────────────────────────────────────────

class ProgressTracker:
    """Persist downloaded reddit_post_ids so the run is resumable."""

    def __init__(self, path: Path):
        self.path = path
        self.seen: set[str] = set()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.seen = set(data.get("downloaded", []))
                log.info("Resume: %d posts already downloaded", len(self.seen))
            except Exception:
                self.seen = set()

    def save(self):
        self.path.write_text(json.dumps({"downloaded": list(self.seen)}, indent=2))

    def mark(self, reddit_id: str):
        self.seen.add(reddit_id)

    def already_done(self, reddit_id: str) -> bool:
        return reddit_id in self.seen


# ── Blocking media helpers (run via asyncio.to_thread) ────────────────────────

def _http_get(url: str, stream: bool = False) -> requests.Response | None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; seeder/1.0)"}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=DOWNLOAD_TIMEOUT, stream=stream)
            if r.status_code == 200:
                return r
            return None
        except requests.RequestException as exc:
            log.debug("HTTP error (attempt %d): %s", attempt + 1, exc)
            time.sleep(1)
    return None


def _download_image_sync(url: str) -> tuple[str, str] | None:
    from urllib.parse import urlparse
    ext = Path(urlparse(url).path).suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = ".jpg"

    uid        = str(uuid.uuid4())
    img_path   = MEDIA_IMAGES / f"{uid}{ext}"
    thumb_path = MEDIA_THUMBNAILS / f"{uid}.jpg"

    r = _http_get(url)
    if r is None:
        return None
    img_path.write_bytes(r.content)

    try:
        with Image.open(img_path) as im:
            w, h = im.size
            if w < MIN_IMAGE_DIM or h < MIN_IMAGE_DIM:
                img_path.unlink(missing_ok=True)
                return None
            canvas = Image.new("RGB", THUMBNAIL_SIZE, (0, 0, 0))
            im_rgb = im.convert("RGB")
            im_rgb.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
            offset = (
                (THUMBNAIL_SIZE[0] - im_rgb.width) // 2,
                (THUMBNAIL_SIZE[1] - im_rgb.height) // 2,
            )
            canvas.paste(im_rgb, offset)
            canvas.save(thumb_path, "JPEG", quality=85)
    except Exception as exc:
        log.debug("Pillow error: %s", exc)
        img_path.unlink(missing_ok=True)
        return None

    rel = lambda p: str(p.relative_to(BASE_DIR))
    return rel(img_path), rel(thumb_path)


def _ffmpeg_ok() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _download_video_sync(
    fallback_url: str, reddit_id: str
) -> tuple[str, str] | None:
    uid        = str(uuid.uuid4())
    vid_raw    = MEDIA_VIDEOS / f"{uid}_video.mp4"
    aud_raw    = MEDIA_VIDEOS / f"{uid}_audio.mp4"
    merged     = MEDIA_VIDEOS / f"{uid}.mp4"
    thumb_path = MEDIA_THUMBNAILS / f"{uid}.jpg"

    # Download video stream
    rv = _http_get(fallback_url, stream=True)
    if rv is None:
        return None

    video_bytes = b""
    for chunk in rv.iter_content(chunk_size=1 << 20):
        video_bytes += chunk
        if len(video_bytes) > VIDEO_MAX_MB * 1024 * 1024:
            log.debug("Oversized video, skipping %s", reddit_id)
            return None
    vid_raw.write_bytes(video_bytes)

    # Audio stream (same base path, different segment)
    base_url  = fallback_url.split("?")[0]
    audio_url = "/".join(base_url.split("/")[:-1]) + "/DASH_audio.mp4"
    ra = _http_get(audio_url)
    has_audio = ra is not None
    if has_audio:
        aud_raw.write_bytes(ra.content)

    # Merge with FFmpeg
    out_path = vid_raw
    if _ffmpeg_ok():
        try:
            cmd = ["ffmpeg", "-y", "-i", str(vid_raw)]
            if has_audio:
                cmd += ["-i", str(aud_raw)]
            cmd += ["-c:v", "copy", "-c:a", "aac", str(merged)] if has_audio else ["-c:v", "copy", str(merged)]
            res = subprocess.run(cmd, capture_output=True, timeout=120)
            if res.returncode == 0:
                vid_raw.unlink(missing_ok=True)
                if has_audio:
                    aud_raw.unlink(missing_ok=True)
                out_path = merged
            else:
                log.debug("FFmpeg error: %s", res.stderr.decode()[:200])
        except subprocess.TimeoutExpired:
            log.debug("FFmpeg timeout for %s", reddit_id)

    # Thumbnail from first frame
    thumb_ok = False
    if _ffmpeg_ok() and out_path.exists():
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(out_path),
                    "-vframes", "1",
                    "-vf",
                    f"scale={THUMBNAIL_SIZE[0]}:{THUMBNAIL_SIZE[1]}:"
                    "force_original_aspect_ratio=decrease,"
                    f"pad={THUMBNAIL_SIZE[0]}:{THUMBNAIL_SIZE[1]}:(ow-iw)/2:(oh-ih)/2",
                    str(thumb_path),
                ],
                capture_output=True,
                timeout=30,
            )
            thumb_ok = thumb_path.exists()
        except Exception:
            pass

    rel = lambda p: str(p.relative_to(BASE_DIR))
    return rel(out_path), (rel(thumb_path) if thumb_ok else None)


# ── URL helpers ────────────────────────────────────────────────────────────────

def _image_url(submission) -> str | None:
    """Return a direct image URL from a submission, or None."""
    from urllib.parse import urlparse

    url  = getattr(submission, "url", "") or ""
    host = urlparse(url).netloc.lower()
    ext  = Path(urlparse(url).path).suffix.lower()

    if any(host.endswith(h) for h in IMAGE_HOSTS) and ext in IMAGE_EXTS:
        return url

    # imgur link without extension
    if "imgur.com" in host and not ext:
        return url + ".jpg"

    # preview.redd.it fallback (gallery / crosspost)
    preview = getattr(submission, "preview", None)
    if preview:
        images = preview.get("images", [])
        if images:
            src = images[0].get("source", {})
            purl = src.get("url", "").replace("&amp;", "&")
            if purl:
                return purl

    return None


def _is_reddit_video(submission) -> bool:
    return bool(
        getattr(submission, "is_video", False)
        and getattr(submission, "media", None)
        and "reddit_video" in (submission.media or {})
    )


def _video_fallback_url(submission) -> str | None:
    rv = (submission.media or {}).get("reddit_video", {})
    duration = rv.get("duration", 9999)
    if duration > VIDEO_MAX_SECS:
        return None
    return rv.get("fallback_url") or None


# ── DB helpers ─────────────────────────────────────────────────────────────────

async def get_or_create_user(
    db: AsyncSession, reddit_username: str, user_cache: dict
) -> str:
    if reddit_username in user_cache:
        return user_cache[reddit_username]

    from src.db.models.user import User

    user = await db.scalar(
        select(User).where(
            User.source_platform == "reddit",
            User.source_username == reddit_username,
        )
    )
    if user is None:
        user = User(
            display_name=reddit_username,
            is_synthetic=True,
            source_platform="reddit",
            source_username=reddit_username,
        )
        db.add(user)
        await db.flush()

    user_cache[reddit_username] = str(user.id)
    return user_cache[reddit_username]


async def insert_post(
    db: AsyncSession,
    *,
    user_id: str,
    media_type: str,
    media_path: str,
    thumbnail_path: str | None,
    caption: str,
    source_url: str,
    subreddit: str,
    score: int,
    num_comments: int,
) -> None:
    from src.db.models.post import Post

    db.add(Post(
        user_id=uuid.UUID(user_id),
        media_type=media_type,
        media_path=media_path,
        thumbnail_path=thumbnail_path,
        caption=caption[:2000] if caption else None,
        status="uploaded",
        source_url=source_url,
        source_platform="reddit",
        source_subreddit=subreddit,
        source_upvotes=score,
        source_comments=num_comments,
    ))


# ── Subreddit seeder ───────────────────────────────────────────────────────────

async def seed_subreddit(
    reddit: asyncpraw.Reddit,
    db: AsyncSession,
    sub_name: str,
    limit: int,
    skip_videos: bool,
    progress: ProgressTracker,
    user_cache: dict,
    counters: dict,
) -> int:
    try:
        subreddit = await reddit.subreddit(sub_name)
        # Fetch more than needed to account for non-media posts
        fetch_limit = min(limit * 4, 500)
        submissions = []
        async for s in subreddit.top(time_filter="year", limit=fetch_limit):
            submissions.append(s)
    except Exception as exc:
        log.error("Failed to fetch r/%s: %s", sub_name, exc)
        return 0

    log.info("r/%s — fetched %d submissions from Reddit", sub_name, len(submissions))
    posts_stored = 0

    for submission in submissions:
        if posts_stored >= limit:
            break
        if progress.already_done(submission.id):
            continue

        # Classify submission type
        img_url   = _image_url(submission) if not submission.is_self else None
        vid_url   = _video_fallback_url(submission) if (not skip_videos and _is_reddit_video(submission)) else None

        if not img_url and not vid_url:
            continue

        # Resolve author
        try:
            author_name = submission.author.name if submission.author else "deleted"
        except Exception:
            author_name = "deleted"

        try:
            uid = await get_or_create_user(db, author_name, user_cache)
        except Exception as exc:
            log.error("DB user error (%s): %s", submission.id, exc)
            counters["errors"] += 1
            continue

        # Download media in a thread (blocking I/O)
        result     = None
        media_type = None
        try:
            if vid_url:
                result     = await asyncio.to_thread(_download_video_sync, vid_url, submission.id)
                media_type = "video"
            elif img_url:
                result     = await asyncio.to_thread(_download_image_sync, img_url)
                media_type = "image"
        except Exception as exc:
            log.error("Download error (%s): %s", submission.id, exc)
            counters["errors"] += 1
            continue

        if result is None:
            counters["errors"] += 1
            continue

        media_path, thumbnail_path = result

        try:
            await insert_post(
                db,
                user_id=uid,
                media_type=media_type,
                media_path=media_path,
                thumbnail_path=thumbnail_path,
                caption=submission.title,
                source_url=f"https://reddit.com{submission.permalink}",
                subreddit=sub_name,
                score=submission.score,
                num_comments=submission.num_comments,
            )
            await db.commit()

            progress.mark(submission.id)
            posts_stored    += 1
            counters["total"] += 1
            counters[media_type + "s"] += 1

            if posts_stored % 10 == 0:
                progress.save()
                log.info(
                    "Downloaded %d — r/%s in progress (%d posts) | img=%d vid=%d err=%d",
                    counters["total"], sub_name, posts_stored,
                    counters["images"], counters["videos"], counters["errors"],
                )
        except Exception as exc:
            log.error("DB error for %s: %s", submission.id, exc)
            await db.rollback()
            counters["errors"] += 1

    return posts_stored


# ── Entry point ────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    client_id     = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent    = os.environ.get("REDDIT_USER_AGENT", "social-media-seeder/1.0")
    username      = os.environ.get("REDDIT_USERNAME")
    password      = os.environ.get("REDDIT_PASSWORD")

    if not client_id or not client_secret:
        log.error("REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in .env")
        sys.exit(1)

    # Build subreddit list
    catalogue = SUBREDDIT_CATALOGUE
    if args.subreddits:
        requested = {s.strip().lower() for s in args.subreddits.split(",")}
        catalogue  = [(s, c, t) for s, c, t in catalogue if s.lower() in requested]
        if not catalogue:
            log.error("None of the requested subreddits matched the catalogue.")
            sys.exit(1)

    override_limit = args.limit
    total_target   = sum(override_limit or t for _, _, t in catalogue)

    # DB engine
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/social_media_content",
    )
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine  = create_async_engine(database_url, echo=False, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    # ── Fail fast: verify DB is reachable before touching Reddit ────────────
    from sqlalchemy import text as sa_text
    try:
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        log.info("DB connection OK — %s", database_url.split("@")[-1])
    except Exception as exc:
        log.error("Cannot connect to database: %s", exc)
        log.error("DATABASE_URL in use: %s", database_url)
        log.error("Make sure Docker is running: docker compose up -d db")
        await engine.dispose()
        sys.exit(1)

    progress   = ProgressTracker(PROGRESS_FILE)
    user_cache: dict[str, str] = {}
    counters   = {"total": 0, "images": 0, "videos": 0, "errors": 0}

    log.info(
        "Starting seeder — %d subreddits, target ~%d posts%s",
        len(catalogue), total_target,
        " (images only)" if args.skip_videos else "",
    )

    async with asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password,
        ratelimit_seconds=300,
    ) as reddit:
        async with Session() as db:
            for idx, (sub_name, _cat, default_limit) in enumerate(catalogue):
                limit  = override_limit if override_limit is not None else default_limit

                log.info(
                    "[%d/%d] r/%s — fetching up to %d posts",
                    idx + 1, len(catalogue), sub_name, limit,
                )

                stored = await seed_subreddit(
                    reddit=reddit,
                    db=db,
                    sub_name=sub_name,
                    limit=limit,
                    skip_videos=args.skip_videos,
                    progress=progress,
                    user_cache=user_cache,
                    counters=counters,
                )

                log.info(
                    "Downloaded %d/%d — r/%s complete (%d posts)",
                    counters["total"], total_target, sub_name, stored,
                )
                progress.save()

                if idx < len(catalogue) - 1:
                    await asyncio.sleep(1)  # polite delay between subreddits

    await engine.dispose()

    print("\n" + "=" * 60)
    print("  SEEDING COMPLETE")
    print("=" * 60)
    print(f"  Total downloaded   : {counters['total']}")
    print(f"  Images             : {counters['images']}")
    print(f"  Videos             : {counters['videos']}")
    print(f"  Failed / skipped   : {counters['errors']}")
    print(f"  Unique Reddit users: {len(user_cache)}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed social media DB from Reddit")
    p.add_argument(
        "--subreddits", default=None,
        help="Comma-separated subreddit names (default: all in catalogue)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Posts per subreddit (default: per-subreddit value in catalogue)",
    )
    p.add_argument(
        "--skip-videos", action="store_true",
        help="Images only — faster for testing",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
