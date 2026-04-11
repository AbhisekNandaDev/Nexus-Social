import sys
import os
from typing import List, Optional
# Add project root to Python path
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ollama_llm_provider import OllamaProvider
from utils.common_functions import load_config
from utils.predict_age import predict_age
from utils.image_prompts import (
    get_image_classification_prompt,
    get_age_prediction_prompt,
    get_safety_prompt,
    get_content_prompt,
)
from utils.logger import setup_logging, get_logger

config = load_config(config_path="config/config.yml")
logger = get_logger(__name__)


def _derive_age_group(age: float) -> str:
    if age < 13:
        return "child"
    if age < 18:
        return "teen"
    return "adult"


class ImagePipeline:
    def __init__(self, image, image_type):
        self.image = image
        self.image_type = image_type
        logger.debug("ImagePipeline created | image_type=%s", image_type)

    def _load_bytes(self) -> bytes:
        if self.image_type == "file":
            logger.debug("Reading image from file | path=%s", self.image)
            with open(self.image, "rb") as f:
                return f.read()
        return self.image

    def predict_age(self):
        logger.info("Running DeepFace age prediction | image_type=%s", self.image_type)
        return predict_age(self._load_bytes())

    def predict_age_llm(self, model_name=config["image_classification"]["model"]):
        logger.info("Running LLM age prediction | model=%s image_type=%s", model_name, self.image_type)
        image_bytes = self._load_bytes()
        message_request = get_age_prediction_prompt(image_bytes)
        llm_client = OllamaProvider(model_name)
        llm_response = llm_client.get_response(image_bytes, message_request)
        logger.info("LLM age prediction complete | result=%s", llm_response)
        return llm_response

    def classify(
        self,
        model_name=config["image_classification"]["model"],
        llm_client: OllamaProvider = None,  # pass a shared instance to avoid recreating per frame
        deepface: bool = True,              # set False for video frames (aggregator discards it anyway)
    ) -> dict:
        """Run LLM classification and optionally DeepFace, return merged result dict."""
        logger.info("Running full image classification | model=%s image_type=%s deepface=%s",
                    model_name, self.image_type, deepface)
        image_bytes = self._load_bytes()

        # LLM full classification — reuse caller-supplied client if provided
        if llm_client is None:
            llm_client = OllamaProvider(model_name)
        message_request = get_image_classification_prompt(image_bytes)
        llm_response = llm_client.get_response(image_bytes, message_request)
        if not isinstance(llm_response, dict):
            logger.warning("LLM returned non-dict response | type=%s", type(llm_response).__name__)
            llm_response = {}

        # DeepFace age — skipped for video frames (saves ~2s/frame, result unused by aggregator)
        deepface_age = None
        deepface_age_group = None
        if deepface:
            try:
                deepface_age = predict_age(image_bytes)
                deepface_age_group = _derive_age_group(deepface_age)
                logger.info("DeepFace complete | age=%s age_group=%s", deepface_age, deepface_age_group)
            except Exception as exc:
                logger.warning("DeepFace failed (non-fatal) | error=%s", exc)

        llm_response["deepface_age"] = deepface_age
        llm_response["deepface_age_group"] = deepface_age_group

        logger.info("Classification complete | result=%s", llm_response)
        return llm_response

    def run(self):
        logger.info("Pipeline run started")
        x = self.predict_age()
        y = self.predict_age_llm()
        logger.info("Pipeline run complete | deepface_age=%s llm_response=%s", x, y)
        return x, y

    @staticmethod
    def classify_batch(
        frames: List[bytes],
        llm_client: OllamaProvider,
        timestamps: Optional[List[float]] = None,
        transcript_context: Optional[str] = None,
    ) -> dict:
        """
        Two-pass batch classification for 1-3 video frames sent together.

        Pass 1 (safety): nudity / violence / self-harm / age / risk.
          - All frames sent in one message so the LLM has cross-frame context.
          - Short-circuits if risk == block | illegal (skips content pass).
        Pass 2 (content): description / tags / mood / scene / objects / text.
          - Focused prompt, fewer tokens, faster inference.

        Returns a merged dict with all fields expected by FrameAggregator.
        """
        # ── Pass 1: safety ────────────────────────────────────────────────────
        safety_msgs = get_safety_prompt(frames, timestamps=timestamps,
                                        transcript_ctx=transcript_context)
        safety_raw = llm_client.get_response(b"", safety_msgs)
        if not isinstance(safety_raw, dict):
            logger.warning("Safety pass returned non-dict | type=%s", type(safety_raw).__name__)
            safety_raw = {}

        risk = safety_raw.get("risk", "allow")
        logger.debug("Safety pass done | risk=%s confidence=%s", risk, safety_raw.get("confidence"))

        # ── Pass 2: content (skip if blocked) ─────────────────────────────────
        if risk in ("block", "illegal"):
            logger.info("Short-circuit: risk=%s — skipping content pass", risk)
            content_raw: dict = {}
        else:
            content_msgs = get_content_prompt(frames)
            content_raw = llm_client.get_response(b"", content_msgs)
            if not isinstance(content_raw, dict):
                content_raw = {}
            logger.debug("Content pass done | tags=%s", content_raw.get("display_tags"))

        return {
            # Content fields (safety takes precedence for any overlapping keys)
            "content_description": content_raw.get("content_description", ""),
            "display_tags":        content_raw.get("display_tags", []),
            "mood":                content_raw.get("mood", "neutral"),
            "scene_type":          content_raw.get("scene_type", "indoor"),
            "text_in_image":       content_raw.get("text_in_image"),
            "objects_detected":    content_raw.get("objects_detected", []),
            "people_count":        content_raw.get("people_count", 0),
            # Safety fields (override content where keys overlap)
            **safety_raw,
            # DeepFace not run for video batches
            "deepface_age":        None,
            "deepface_age_group":  None,
        }


if __name__ == "__main__":
    setup_logging()
    with open("data/image copy.png", "rb") as image_file:
        image_bytes = image_file.read()
    image_pipeline = ImagePipeline(image_bytes, "bytes")
    import json
    print(json.dumps(image_pipeline.classify(), indent=2))
