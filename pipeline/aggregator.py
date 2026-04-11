"""Aggregates per-frame LLM classifications into a single video-level result.

Priority order:
    1. Safety override   — single high-confidence explicit/illegal frame wins
    2. Transcript check  — dangerous audio upgrades risk level
    3. Weighted average  — confidence-weighted aggregation of all other fields
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_RISK_ORDER:      List[str] = ["allow", "restrict", "nsfw", "block", "illegal"]
_AGE_ORDER:       List[str] = ["unknown", "adult", "teen", "child"]   # highest = most restrictive
_NUDITY_ORDER:    List[str] = ["safe", "suggestive", "partial_nudity", "explicit_nudity", "sexual_activity"]
_VIOLENCE_ORDER:  List[str] = ["none", "mild", "moderate", "graphic", "extreme"]
_SELF_HARM_ORDER: List[str] = ["none", "implied", "depicted", "instructional"]
_OVERRIDE_NUDITY = frozenset({"explicit_nudity", "sexual_activity"})
_OVERRIDE_CONFIDENCE = 0.7


def _most_restrictive_age(*ages: str) -> str:
    best_idx = 0
    best = "unknown"
    for age in ages:
        idx = _AGE_ORDER.index(age) if age in _AGE_ORDER else 0
        if idx > best_idx:
            best_idx = idx
            best = age
    return best


def _most_restrictive_field(values: List[str], order: List[str], default: str) -> str:
    """Return the value with the highest index in the severity order list."""
    best_idx = 0
    for v in values:
        idx = order.index(v) if v in order else 0
        best_idx = max(best_idx, idx)
    return order[best_idx] if values else default


def _dominant(scores: Dict[str, float]) -> str:
    return max(scores, key=lambda k: scores[k])


def _dedup_ordered(lst: List) -> List:
    seen: set = set()
    return [x for x in lst if x not in seen and not seen.add(x)]  # type: ignore[func-returns-value]


def _coerce_text(value) -> str:
    """Normalise text_in_image to str — LLM sometimes returns a list."""
    if isinstance(value, list):
        return " | ".join(str(v) for v in value if v)
    return str(value) if value else ""


class FrameAggregator:
    def __init__(self, safety_override_confidence: float = _OVERRIDE_CONFIDENCE,
                 ollama_client=None):
        self.safety_override_confidence = safety_override_confidence
        self._ollama = ollama_client  # OllamaProvider instance, optional

    def aggregate(self, frame_results: List[dict], transcript=None) -> dict:
        """
        Combine per-frame classifications and optional transcript into one result.

        transcript: TranscriptionResult or None
        """
        if not frame_results:
            raise ValueError("No frame results to aggregate")

        # Step 1: safety override
        override = self._find_override(frame_results)
        if override is not None:
            result = self._build_override_result(override, frame_results)
        else:
            result = self._weighted_average(frame_results)

        # Step 2: transcript safety
        flags: List[dict] = []
        if transcript and getattr(transcript, "full_text", ""):
            result, flags = self._check_transcript(transcript.full_text, result)

        result["transcript"] = getattr(transcript, "full_text", None) or None
        result["transcript_language"] = getattr(transcript, "language", None) or None
        result["transcript_safety_flags"] = flags
        result["has_audio"] = getattr(transcript, "has_audio", False)
        result["frames_analyzed"] = len(frame_results)
        return result

    # ── safety override ───────────────────────────────────────────────────────

    def _find_override(self, results: List[dict]) -> Optional[dict]:
        best: Optional[dict] = None
        best_level = -1

        for r in results:
            nudity = r.get("nudity_level", "safe")
            conf = float(r.get("confidence", 0.0))
            age = r.get("age_group", "unknown")
            violence = r.get("violence_level", "none")
            self_harm = r.get("self_harm_level", "none")
            risk = r.get("risk", "allow")
            thr = self.safety_override_confidence

            level = -1
            if nudity in _OVERRIDE_NUDITY and conf >= thr:
                level = max(level, _RISK_ORDER.index("block"))
            if age in ("child", "teen") and nudity != "safe" and conf >= thr:
                level = max(level, _RISK_ORDER.index("illegal"))
            if violence == "extreme" and conf >= thr:
                level = max(level, _RISK_ORDER.index("block"))
            if self_harm == "instructional" and conf >= thr:
                level = max(level, _RISK_ORDER.index("block"))
            if risk == "illegal" and conf >= thr:
                level = max(level, _RISK_ORDER.index("illegal"))

            if level > best_level:
                best_level = level
                best = r
                logger.info("Safety override candidate | nudity=%s age=%s violence=%s "
                            "self_harm=%s risk=%s conf=%.2f ts=%s",
                            nudity, age, violence, self_harm, risk, conf,
                            r.get("timestamp", "?"))

        return best if best_level >= 0 else None

    def _build_override_result(self, override: dict, all_results: List[dict]) -> dict:
        all_nsfw, all_vtypes, all_shtypes, all_tags, all_obj, texts = [], [], [], [], [], []
        for r in all_results:
            all_nsfw.extend(r.get("nsfw_subcategories", []))
            all_vtypes.extend(r.get("violence_type", []))
            all_shtypes.extend(r.get("self_harm_type", []))
            all_tags.extend(r.get("display_tags", []))
            all_obj.extend(r.get("objects_detected", []))
            t = _coerce_text(r.get("text_in_image"))
            if t:
                texts.append(t)

        result = dict(override)
        result.update({
            "nsfw_subcategories": _dedup_ordered(all_nsfw),
            "violence_type": _dedup_ordered(all_vtypes),
            "self_harm_type": _dedup_ordered(all_shtypes),
            "display_tags": _dedup_ordered(all_tags)[:5],
            "objects_detected": _dedup_ordered(all_obj)[:10],
            "text_in_image": " | ".join(dict.fromkeys(texts)) if texts else None,
            "deepface_age": None,
            "deepface_age_group": None,
            "secondary_classifications": self._compute_secondary(all_results),
        })
        return result

    # ── weighted average ──────────────────────────────────────────────────────

    def _weighted_average(self, results: List[dict]) -> dict:
        nudity_w: Dict[str, float] = {}
        violence_w: Dict[str, float] = {}
        self_harm_w: Dict[str, float] = {}
        risk_w: Dict[str, float] = {}
        mood_w: Dict[str, float] = {}
        scene_w: Dict[str, float] = {}

        all_nsfw, all_vtypes, all_shtypes, all_tags, all_obj = [], [], [], [], []
        descriptions, texts, people_counts, age_groups = [], [], [], []
        total_conf = 0.0

        all_nudity:    List[str] = []
        all_violence:  List[str] = []
        all_self_harm: List[str] = []

        for r in results:
            c = float(r.get("confidence", 0.5))
            total_conf += c

            # Collect raw safety values for max-based aggregation
            all_nudity.append(r.get("nudity_level", "safe"))
            all_violence.append(r.get("violence_level", "none"))
            all_self_harm.append(r.get("self_harm_level", "none"))

            # Risk still uses weighted approach (informed by all safety dimensions)
            val = r.get("risk", "allow")
            risk_w[val] = risk_w.get(val, 0.0) + c

            for val2, store in [(r.get("mood"), mood_w), (r.get("scene_type"), scene_w)]:
                if val2:
                    store[val2] = store.get(val2, 0.0) + c

            all_nsfw.extend(r.get("nsfw_subcategories", []))
            all_vtypes.extend(r.get("violence_type", []))
            all_shtypes.extend(r.get("self_harm_type", []))
            all_tags.extend(r.get("display_tags", []))
            all_obj.extend(r.get("objects_detected", []))
            if r.get("content_description"):
                descriptions.append(r["content_description"])
            t = _coerce_text(r.get("text_in_image"))
            if t:
                texts.append(t)
            if r.get("people_count") is not None:
                people_counts.append(r["people_count"])
            if r.get("age_group"):
                age_groups.append(r["age_group"])

        # Safety fields: most restrictive across any frame.
        # A single explicit frame among 14 safe ones IS the problem — don't average it away.
        final_nudity    = _most_restrictive_field(all_nudity,    _NUDITY_ORDER,    "safe")
        final_violence  = _most_restrictive_field(all_violence,  _VIOLENCE_ORDER,  "none")
        final_self_harm = _most_restrictive_field(all_self_harm, _SELF_HARM_ORDER, "none")

        # Age: most restrictive (youngest) across all frames
        final_age = _most_restrictive_age(*age_groups) if age_groups else "unknown"

        # Risk: worst tier with ≥ 20% weight — but always at least as severe as
        # what the max safety levels imply, so a single bad frame can't be diluted.
        dominant_risk = "allow"
        for tier in reversed(_RISK_ORDER):
            if risk_w.get(tier, 0.0) / max(total_conf, 1e-9) >= 0.20:
                dominant_risk = tier
                break
        # Floor: derive minimum risk from final safety levels
        if final_nudity in ("explicit_nudity", "sexual_activity"):
            dominant_risk = _RISK_ORDER[max(_RISK_ORDER.index(dominant_risk),
                                            _RISK_ORDER.index("nsfw"))]
        if final_violence in ("graphic", "extreme") or final_self_harm in ("depicted", "instructional"):
            dominant_risk = _RISK_ORDER[max(_RISK_ORDER.index(dominant_risk),
                                            _RISK_ORDER.index("block"))]
        if final_age in ("child", "teen") and final_nudity != "safe":
            dominant_risk = "illegal"

        # People count: maximum across frames
        max_people = 0
        for p in people_counts:
            try:
                v = 999 if str(p).lower() == "group" else int(p)
                max_people = max(max_people, v)
            except (ValueError, TypeError):
                pass
        final_people: object = "group" if max_people >= 999 else max_people

        return {
            "nudity_level":    final_nudity,
            "nsfw_subcategories": _dedup_ordered(all_nsfw),
            "violence_level":  final_violence,
            "violence_type":   _dedup_ordered(all_vtypes),
            "self_harm_level": final_self_harm,
            "self_harm_type":  _dedup_ordered(all_shtypes),
            "age_group":       final_age,
            "risk":            dominant_risk,
            "confidence":      round(total_conf / len(results), 3),
            "content_description": descriptions[0] if descriptions else "",
            "display_tags":    _dedup_ordered(all_tags)[:5],
            "mood":            _dominant(mood_w) if mood_w else "neutral",
            "scene_type":      _dominant(scene_w) if scene_w else "indoor",
            "text_in_image":   " | ".join(dict.fromkeys(texts)) if texts else None,
            "objects_detected": _dedup_ordered(all_obj)[:10],
            "people_count":    final_people,
            "deepface_age":    None,
            "deepface_age_group": None,
            "secondary_classifications": self._compute_secondary(results),
        }

    def _compute_secondary(self, results: List[dict]) -> dict:
        fields = ["nudity_level", "violence_level", "self_harm_level", "risk"]
        total = max(sum(float(r.get("confidence", 0.5)) for r in results), 1e-9)
        secondary: Dict[str, Dict[str, float]] = {}
        for field in fields:
            dist: Dict[str, float] = {}
            for r in results:
                v = r.get(field)
                c = float(r.get("confidence", 0.5))
                if v:
                    dist[v] = round(dist.get(v, 0.0) + c / total, 3)
            secondary[field] = dist
        return secondary

    # ── transcript safety ─────────────────────────────────────────────────────

    def _check_transcript(self, text: str, result: dict):
        if self._ollama is None:
            return result, []

        try:
            prompt = [{
                "role": "user",
                "content": (
                    "Analyze this video transcript for safety concerns. Return JSON:\n"
                    "{\n"
                    '  "has_self_harm_content": bool,\n'
                    '  "has_violence_content": bool,\n'
                    '  "has_hate_speech": bool,\n'
                    '  "has_dangerous_instructions": bool,\n'
                    '  "flagged_segments": [{"text": "...", "reason": "..."}],\n'
                    '  "overall_safety": "safe" | "concerning" | "dangerous"\n'
                    "}\n"
                    "Only flag genuinely dangerous content, not casual mentions.\n\n"
                    f"Transcript:\n{text[:5000]}"
                ),
            }]
            resp = self._ollama.client.chat(
                model=self._ollama.model_name,
                messages=prompt,
                think=self._ollama.think,
            )
            raw = resp["message"]["content"]
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                return result, []

            analysis = json.loads(m.group())
            flags = analysis.get("flagged_segments", [])
            overall = analysis.get("overall_safety", "safe")

            if overall == "dangerous":
                result["risk"] = "block"
                result["needs_review"] = True
                logger.info("Transcript: dangerous | flags=%d", len(flags))
            elif overall == "concerning" and result.get("risk") == "allow":
                result["risk"] = "restrict"
                result["needs_review"] = True
                logger.info("Transcript: concerning — upgraded risk to restrict")

            return result, flags
        except Exception as exc:
            logger.warning("Transcript safety check failed | error=%s", exc)
            return result, []
