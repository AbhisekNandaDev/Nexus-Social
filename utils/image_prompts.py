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

risk (pick ONE — derive strictly from the rules below):
  allow    - Safe for all audiences
  restrict - Teen-appropriate, not for children
  nsfw     - Adult content, legal
  block    - Harmful but not illegal (extreme violence, self-harm)
  illegal  - Child sexual abuse, content illegal in most jurisdictions

  Rules (apply in order, highest priority first):
  - Minor (child/teen) + ANY nudity → illegal (overrides everything)
  - explicit_nudity/sexual_activity + adult → nsfw
  - partial_nudity + adult → nsfw
  - suggestive + child → illegal; + teen → restrict; + adult → allow
  - violence graphic/extreme → block
  - self_harm depicted/instructional → block
  - safe + adult/unknown → allow

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

people_count (pick ONE string): "0" | "1" | "2" | "group" (3 or more people)

IMPORTANT: Output the JSON object immediately. Do not write any text before or after it. Do not explain your reasoning."""

    user_prompt = """Classify this image. Return ONLY this JSON with every field filled in:
{
  "nudity_level": "safe",
  "nsfw_subcategories": [],
  "violence_level": "none",
  "violence_type": [],
  "self_harm_level": "none",
  "self_harm_type": [],
  "age_group": "unknown",
  "risk": "allow",
  "confidence": 0.0,
  "content_description": "",
  "display_tags": [],
  "mood": "neutral",
  "scene_type": "indoor",
  "text_in_image": null,
  "objects_detected": [],
  "people_count": "0"
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
    frame_ref = "this video frame" if n == 1 else f"these {n} video frames"

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

    system_prompt = f"""You are a strict social media safety classifier analysing {frame_ref}{ts_clause}.

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

risk (pick ONE — apply rules in order, highest priority first):
  allow | restrict | nsfw | block | illegal

  Rules:
  - Minor (child/teen) + ANY nudity → illegal (overrides everything)
  - explicit_nudity/sexual_activity + adult → nsfw
  - partial_nudity + adult → nsfw
  - suggestive + teen → restrict; + child → illegal
  - violence graphic/extreme → block
  - self_harm depicted/instructional → block
  - Otherwise → allow

confidence (float 0.0-1.0): certainty across all frames

IMPORTANT: Output the JSON object immediately. Do not write any text before or after it."""

    user_prompt = f"""Classify the safety of {frame_ref}. Return ONLY:
{{
  "nudity_level": "",
  "nsfw_subcategories": [],
  "violence_level": "",
  "violence_type": [],
  "self_harm_level": "",
  "self_harm_type": [],
  "age_group": "",
  "risk": "",
  "confidence": 0.0
}}"""

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
    frame_ref = "this video frame" if n == 1 else f"these {n} video frames"

    system_prompt = f"""You are a social media content analyst. The content in {frame_ref} has already passed safety review.
Your job is purely descriptive — describe what is happening visually, not whether it is safe.

Provide a concise description of the overall content shown across all frames,
as if describing a short video clip. Focus on what is happening, not safety.

For mood and scene_type: if frames disagree, choose the value that represents the majority or the final frame.

people_count (pick ONE string): "0" | "1" | "2" | "group" (3 or more people)

IMPORTANT: Output the JSON object immediately. Do not write any text before or after it."""

    user_prompt = f"""Describe the content of {frame_ref}. Return ONLY:
{{
  "content_description": "",
  "display_tags": [],
  "mood": "",
  "scene_type": "",
  "text_in_image": null,
  "objects_detected": [],
  "people_count": "0"
}}"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt, "images": images},
    ]


# ── age prediction ─────────────────────────────────────────────────────────────

def get_age_prediction_prompt(image_data: bytes):
    system_prompt = """You are a biological age classifier used for safety moderation.

If no person is visible in the image, return immediately:
{"maturity": "no_person", "confidence": 1.0, "observations": ["no visible person"]}

Otherwise, your task is NOT to guess an exact age.
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

IMPORTANT: Output the JSON object immediately. Do not write any text before or after it."""

    user_prompt = """Analyze the physical maturity of the person in this image. Return ONLY:
{
  "maturity": "",
  "confidence": 0.0,
  "observations": []
}"""

    message_request = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt, "images": [image_data]},
    ]
    return message_request
