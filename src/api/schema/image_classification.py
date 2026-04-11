from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List, Union


class NudityLevel(str, Enum):
    safe = "safe"
    suggestive = "suggestive"
    partial_nudity = "partial_nudity"
    explicit_nudity = "explicit_nudity"
    sexual_activity = "sexual_activity"


class NsfwSubcategory(str, Enum):
    cleavage = "cleavage"
    tight_clothing = "tight_clothing"
    provocative_pose = "provocative_pose"
    swimwear = "swimwear"
    sheer_clothing = "sheer_clothing"
    thong = "thong"
    topless_back = "topless_back"
    side_nudity = "side_nudity"
    full_frontal = "full_frontal"
    bare_buttocks = "bare_buttocks"
    nipples_visible = "nipples_visible"
    genitals_visible = "genitals_visible"


class ViolenceLevel(str, Enum):
    none = "none"
    mild = "mild"
    moderate = "moderate"
    graphic = "graphic"
    extreme = "extreme"


class ViolenceType(str, Enum):
    fighting = "fighting"
    weapons = "weapons"
    blood_gore = "blood_gore"
    animal_cruelty = "animal_cruelty"
    domestic_violence = "domestic_violence"
    war_conflict = "war_conflict"


class SelfHarmLevel(str, Enum):
    none = "none"
    implied = "implied"
    depicted = "depicted"
    instructional = "instructional"


class SelfHarmType(str, Enum):
    cutting = "cutting"
    substance_abuse = "substance_abuse"
    suicide_reference = "suicide_reference"
    eating_disorder = "eating_disorder"
    dangerous_challenge = "dangerous_challenge"


class AgeGroup(str, Enum):
    child = "child"
    teen = "teen"
    adult = "adult"
    unknown = "unknown"


class Risk(str, Enum):
    allow = "allow"
    restrict = "restrict"
    nsfw = "nsfw"
    block = "block"
    illegal = "illegal"


class Mood(str, Enum):
    happy = "happy"
    sad = "sad"
    angry = "angry"
    peaceful = "peaceful"
    energetic = "energetic"
    romantic = "romantic"
    dark = "dark"
    neutral = "neutral"
    humorous = "humorous"
    inspirational = "inspirational"


class SceneType(str, Enum):
    indoor = "indoor"
    outdoor = "outdoor"
    studio = "studio"
    urban = "urban"
    nature = "nature"
    underwater = "underwater"
    aerial = "aerial"


class ImageClassificationResponse(BaseModel):
    # Safety classification
    nudity_level: NudityLevel
    nsfw_subcategories: List[NsfwSubcategory] = Field(default_factory=list)
    violence_level: ViolenceLevel
    violence_type: List[ViolenceType] = Field(default_factory=list)
    self_harm_level: SelfHarmLevel
    self_harm_type: List[SelfHarmType] = Field(default_factory=list)
    age_group: AgeGroup
    risk: Risk
    confidence: float = Field(ge=0.0, le=1.0)

    # Content understanding
    content_description: str
    display_tags: List[str] = Field(default_factory=list)
    mood: Mood
    scene_type: SceneType
    text_in_image: Optional[str] = None
    objects_detected: List[str] = Field(default_factory=list)
    people_count: Union[int, str]  # 0, 1, 2, or "group"

    # DeepFace output (populated separately, not from LLM; None for video)
    deepface_age: Optional[float] = None
    deepface_age_group: Optional[AgeGroup] = None

    # Video metadata (None / False for image inputs)
    is_video: bool = False
    video_duration: Optional[float] = None      # seconds
    video_fps: Optional[float] = None
    frames_sampled: Optional[int] = None        # frames evaluated by cheap filters
    frames_classified: Optional[int] = None     # frames actually sent to LLM
    needs_review: bool = False                  # True when LLM cap (15) was hit

    # Transcription (populated for video when transcription is enabled)
    transcript: Optional[str] = None
    transcript_language: Optional[str] = None   # ISO-639-1 code, e.g. "en"


class ImageClassificationRequest(BaseModel):
    image_path: str
