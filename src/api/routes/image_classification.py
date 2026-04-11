import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.ollama_llm_provider import OllamaProvider
from utils.common_functions import load_config
from utils.predict_age import predict_age
from src.api.schema.image_classification import ImageClassificationResponse, AgeGroup
from utils.image_prompts import get_image_classification_prompt, get_age_prediction_prompt
from pipeline.video_pipeline import VideoPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

api_router = APIRouter()

_base_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_base_dir)))
_config = load_config(config_path=os.path.join(_project_dir, "config", "config.yml"))

llm_client = OllamaProvider(
    _config["image_classification"]["model"],
    think=_config["image_classification"].get("think", False),
)

# MIME types that indicate a video upload
_VIDEO_MIME_PREFIXES = ("video/",)
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def _is_video(content_type: str | None, filename: str | None) -> bool:
    """Return True when the upload is a video file."""
    if content_type:
        for prefix in _VIDEO_MIME_PREFIXES:
            if content_type.lower().startswith(prefix):
                return True
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in _VIDEO_EXTENSIONS:
            return True
    return False


def _derive_age_group(age: float) -> AgeGroup:
    if age < 13:
        return AgeGroup.child
    if age < 18:
        return AgeGroup.teen
    return AgeGroup.adult


# ── unified endpoint ──────────────────────────────────────────────────────────

@api_router.post("/image-classification", response_model=ImageClassificationResponse)
async def image_classification(image: UploadFile = File(...)):
    """Classify an image **or** a video.

    For images:  runs LLM classification + DeepFace age detection.
    For videos:  runs the frame-sampling pipeline (2 fps, pixel-diff + SSIM
                 filters, drift detector, weighted-average aggregation) then
                 returns the same response shape with additional video metadata.
    """
    logger.info(
        "Classification request | filename=%s content_type=%s",
        image.filename,
        image.content_type,
    )
    media_bytes: bytes = await image.read()
    logger.debug("Media loaded | size=%d bytes", len(media_bytes))

    if _is_video(image.content_type, image.filename):
        return await _classify_video(media_bytes, image.filename)
    return await _classify_image(media_bytes, image.filename)


# ── image path ────────────────────────────────────────────────────────────────

async def _classify_image(image_bytes: bytes, filename: str) -> ImageClassificationResponse:
    message_request = get_image_classification_prompt(image_bytes)
    try:
        llm_response = llm_client.get_response(image_bytes, message_request)
    except Exception as exc:
        logger.error("LLM call failed | filename=%s error=%s", filename, exc, exc_info=True)
        raise HTTPException(status_code=502, detail="LLM provider error") from exc

    if not isinstance(llm_response, dict):
        logger.warning(
            "Unexpected LLM response type | type=%s response=%s",
            type(llm_response).__name__,
            llm_response,
        )
        raise HTTPException(status_code=422, detail="Could not parse model response")

    # DeepFace age detection (non-fatal)
    deepface_age = None
    deepface_age_group = None
    try:
        deepface_age = predict_age(image_bytes)
        deepface_age_group = _derive_age_group(deepface_age)
        logger.info(
            "DeepFace complete | age=%s age_group=%s", deepface_age, deepface_age_group
        )
    except Exception as exc:
        logger.warning(
            "DeepFace failed (non-fatal) | filename=%s error=%s", filename, exc
        )

    llm_response["deepface_age"] = deepface_age
    llm_response["deepface_age_group"] = deepface_age_group.value if deepface_age_group else None

    logger.info(
        "Image classification complete | filename=%s nudity=%s violence=%s risk=%s confidence=%s",
        filename,
        llm_response.get("nudity_level"),
        llm_response.get("violence_level"),
        llm_response.get("risk"),
        llm_response.get("confidence"),
    )
    return ImageClassificationResponse(**llm_response)


# ── video path ────────────────────────────────────────────────────────────────

async def _classify_video(video_bytes: bytes, filename: str) -> ImageClassificationResponse:
    pipeline = VideoPipeline()
    try:
        result = pipeline.process_bytes(video_bytes)
    except ValueError as exc:
        logger.error("Video pipeline error | filename=%s error=%s", filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Video pipeline failed | filename=%s error=%s", filename, exc, exc_info=True
        )
        raise HTTPException(status_code=502, detail="Video processing error") from exc

    logger.info(
        "Video classification complete | filename=%s nudity=%s risk=%s "
        "duration=%.1fs classified=%d needs_review=%s",
        filename,
        result.nudity_level,
        result.risk,
        result.video_duration_seconds,
        result.frames_analyzed,
        result.needs_review,
    )
    return ImageClassificationResponse(
        nudity_level=result.nudity_level,
        nsfw_subcategories=result.nsfw_subcategories,
        violence_level=result.violence_level,
        violence_type=result.violence_type,
        self_harm_level=result.self_harm_level,
        self_harm_type=result.self_harm_type,
        age_group=result.age_group,
        risk=result.risk,
        confidence=result.classification_confidence,
        content_description=result.content_description,
        display_tags=result.display_tags,
        mood=result.mood,
        scene_type=result.scene_type,
        text_in_image=result.text_in_image,
        objects_detected=result.objects_detected,
        people_count=result.people_count,
        deepface_age=result.deepface_age,
        deepface_age_group=result.deepface_age_group,
        is_video=True,
        video_duration=result.video_duration_seconds,
        video_fps=result.video_fps,
        frames_sampled=result.total_frames_sampled,
        frames_classified=result.frames_analyzed,
        needs_review=result.needs_review,
        transcript=result.transcript,
        transcript_language=result.transcript_language,
    )


# ── hybrid (age-only) endpoint — unchanged ────────────────────────────────────

@api_router.post("/image-classification-hybrid")
async def image_classification_hybrid(image: UploadFile = File(...)):
    logger.info(
        "Hybrid classification request | filename=%s content_type=%s",
        image.filename,
        image.content_type,
    )
    image_bytes = await image.read()
    logger.debug("Image loaded | size=%d bytes", len(image_bytes))

    message_request = get_age_prediction_prompt(image_bytes)
    try:
        llm_response = llm_client.get_response(image_bytes, message_request)
    except Exception as exc:
        logger.error(
            "LLM call failed | filename=%s error=%s", image.filename, exc, exc_info=True
        )
        raise HTTPException(status_code=502, detail="LLM provider error") from exc

    logger.info(
        "Hybrid classification complete | filename=%s response=%s",
        image.filename,
        llm_response,
    )
    return llm_response
