import os
from typing import List, Optional
from utils.common_functions import load_config

_base_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_base_dir)
config = load_config(os.path.join(_project_dir, "config", "config.yml"))


# ── single-image combined prompt (used by ImagePipeline for still images) ─────

def get_image_classification_prompt(image_data: bytes):
    system_prompt = """You are a strict social media safety and content classifier.

Analyze the image and return a single JSON object with ALL of the following fields.

--- SAFETY CLASSIFICATION ---

nudity_level (pick ONE):
  safe              - Fully clothed, no sexual content
  suggestive        - Provocative but no nudity (tight clothes, cleavage, bedroom poses)
  partial_nudity    - Skin exposure with private parts covered (sheer, hand-bra, thong, underboob)
  explicit_nudity   - Visible nipples, genitals, anus, or bare buttocks
  sexual_activity   - Intercourse, oral sex, masturbation, groping, ejaculation

nsfw_subcategories (list, empty [] if nudity_level is safe):
  cleavage | tight_clothing | provocative_pose | swimwear | sheer_clothing |
  thong | topless_back | side_nudity | full_frontal | bare_buttocks |
  nipples_visible | genitals_visible

violence_level (pick ONE):
  none | mild | moderate | graphic | extreme

violence_type (list, empty [] if violence_level is none):
  fighting | weapons | blood_gore | animal_cruelty | domestic_violence | war_conflict

self_harm_level (pick ONE):
  none | implied | depicted | instructional

self_harm_type (list, empty [] if self_harm_level is none):
  cutting | substance_abuse | suicide_reference | eating_disorder | dangerous_challenge

age_group (pick ONE — estimate from visible people; use unknown if no people):
  child (under 13) | teen (13-17) | adult (18+) | unknown

risk (pick ONE — apply strictest rule across all detected issues):
  allow    - Safe for all audiences
  restrict - Teen-appropriate, not for children
  nsfw     - Adult content, legal
  block    - Harmful but not illegal (extreme violence, self-harm)
  illegal  - Child sexual abuse, content illegal in most jurisdictions

  Rules:
  - safe + adult/unknown -> allow
  - suggestive + child -> illegal; + teen -> restrict; + adult -> allow
  - partial_nudity + child/teen -> illegal; + adult -> nsfw
  - explicit_nudity/sexual_activity + child/teen -> illegal; + adult -> nsfw
  - violence graphic/extreme -> block
  - self_harm depicted/instructional -> block
  - Minor + ANY nudity -> illegal (overrides everything)

confidence (float 0.0-1.0): your overall certainty in this classification

--- CONTENT UNDERSTANDING ---

content_description: 2-3 sentence plain-English description of what is in the image

display_tags: list of 3-5 short human-readable tags (e.g. ["beach", "sunset", "couple"])

mood (pick ONE):
  happy | sad | angry | peaceful | energetic | romantic | dark | neutral | humorous | inspirational

scene_type (pick ONE):
  indoor | outdoor | studio | urban | nature | underwater | aerial

text_in_image: any readable text visible in the image (memes, signs, captions), or null if none

objects_detected: list of prominent objects (e.g. ["guitar", "microphone", "crowd"])

people_count: 0, 1, 2, or "group" (3 or more people)

Return ONLY a valid JSON object. No markdown, no explanation, no extra text."""

    user_prompt = """Classify this image. Return ONLY the following JSON structure with no extra text:
{
  "nudity_level": "safe | suggestive | partial_nudity | explicit_nudity | sexual_activity",
  "nsfw_subcategories": [],
  "violence_level": "none | mild | moderate | graphic | extreme",
  "violence_type": [],
  "self_harm_level": "none | implied | depicted | instructional",
  "self_harm_type": [],
  "age_group": "child | teen | adult | unknown",
  "risk": "allow | restrict | nsfw | block | illegal",
  "confidence": 0.0,
  "content_description": "2-3 sentence description",
  "display_tags": [],
  "mood": "happy | sad | angry | peaceful | energetic | romantic | dark | neutral | humorous | inspirational",
  "scene_type": "indoor | outdoor | studio | urban | nature | underwater | aerial",
  "text_in_image": null,
  "objects_detected": [],
  "people_count": 0
}"""

    message_request = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt, "images": [image_data]},
    ]
    return message_request


# ── two-pass prompts for video batch classification ────────────────────────────

def get_safety_prompt(
    images: List[bytes],
    timestamps: Optional[List[float]] = None,
    transcript_ctx: Optional[str] = None,
) -> list:
    """
    Pass 1 — safety fields only, for a batch of 1-3 video frames.

    Sending all frames together gives the model cross-frame context:
    a person reaching for something ambiguous in frame 1 may be reaching
    for a weapon in frame 2. Classify the MOST CONCERNING level seen
    across any frame in the batch.
    """
    n = len(images)
    ts_clause = ""
    if timestamps:
        ts_str = ", ".join(f"{t:.1f}s" for t in timestamps[:n])  # type: ignore[index]
        ts_clause = f" (video timestamps: {ts_str})"

    transcript_clause = ""
    if transcript_ctx and transcript_ctx.strip():
        transcript_clause = (
            f"\n\nAudio context for these frames:\n\"{transcript_ctx.strip()}\"\n"
            "Use this audio context to resolve visual ambiguity. "
            "A weapon that looks like a prop but the speaker says 'I'll kill you' — classify as violent."
        )

    system_prompt = f"""You are a strict social media safety classifier analysing {n} video frame{'s' if n > 1 else ''}{ts_clause}.

Your task: identify the MOST CONCERNING safety signal present in ANY of the frames.
If frame 1 is safe but frame 2 shows violence — classify as violent.
Use cross-frame context: an action building across frames should be judged on its outcome.{transcript_clause}

--- SAFETY FIELDS ---

nudity_level (pick ONE — worst across all frames):
  safe | suggestive | partial_nudity | explicit_nudity | sexual_activity

nsfw_subcategories (list all found across all frames):
  cleavage | tight_clothing | provocative_pose | swimwear | sheer_clothing |
  thong | topless_back | side_nudity | full_frontal | bare_buttocks |
  nipples_visible | genitals_visible

violence_level (pick ONE — worst across all frames):
  none | mild | moderate | graphic | extreme

violence_type (list all found):
  fighting | weapons | blood_gore | animal_cruelty | domestic_violence | war_conflict

self_harm_level (pick ONE — worst across all frames):
  none | implied | depicted | instructional

self_harm_type (list all found):
  cutting | substance_abuse | suicide_reference | eating_disorder | dangerous_challenge

age_group (pick ONE — most restrictive across all visible people):
  child | teen | adult | unknown

risk (apply strictest rule):
  allow | restrict | nsfw | block | illegal

confidence (float 0.0-1.0): certainty across all frames

Return ONLY valid JSON. No markdown, no explanation."""

    user_prompt = """Classify the safety of these video frames. Return ONLY:
{
  "nudity_level": "safe",
  "nsfw_subcategories": [],
  "violence_level": "none",
  "violence_type": [],
  "self_harm_level": "none",
  "self_harm_type": [],
  "age_group": "unknown",
  "risk": "allow",
  "confidence": 0.0
}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt, "images": images},
    ]


def get_content_prompt(images: List[bytes]) -> list:
    """
    Pass 2 — content/metadata fields only, for a batch of 1-3 video frames.
    Only called when safety pass did not return block/illegal.
    Shorter prompt = faster inference for this descriptive pass.
    """
    n = len(images)
    system_prompt = f"""You are a social media content analyst describing {n} video frame{'s' if n > 1 else ''}.

Provide a concise description of the overall content shown across all frames,
as if describing a short video clip. Focus on what is happening, not safety.

Return ONLY valid JSON. No markdown, no explanation."""

    user_prompt = """Describe the content of these video frames. Return ONLY:
{
  "content_description": "2-3 sentence description of what is happening across the frames",
  "display_tags": ["tag1", "tag2", "tag3"],
  "mood": "happy | sad | angry | peaceful | energetic | romantic | dark | neutral | humorous | inspirational",
  "scene_type": "indoor | outdoor | studio | urban | nature | underwater | aerial",
  "text_in_image": null,
  "objects_detected": ["obj1", "obj2"],
  "people_count": 0
}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt, "images": images},
    ]


# ── age prediction (unchanged) ─────────────────────────────────────────────────

def get_age_prediction_prompt(image_data: bytes):
    system_prompt = """You are a biological age classifier used for safety moderation.

Your task is NOT to guess an exact age.
Your task is to evaluate physical maturity based on facial and body development.

Carefully analyze:
- face shape and bone structure
- skin texture
- facial hair
- jawline definition
- eye proportions relative to face
- forehead size
- body proportions if visible
- presence of puberty or pre-puberty traits

Classify the person into ONE category:

child:
pre-puberty features, round face, small jaw, large eyes relative to face

teen:
adolescent development, partial maturity, smooth skin, weak jawline, little or no facial hair

adult:
fully developed facial structure, defined jaw, mature skin, adult bone proportions

Return STRICT JSON ONLY:

{
  "maturity": "child | teen | adult | unknown",
  "confidence": 0.0-1.0,
  "observations": ["list the visual features that led to the decision"]
}"""
    user_prompt = """Analyze the physical maturity of the person in this image."""
    message_request = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt, "images": [image_data]},
    ]
    return message_request
