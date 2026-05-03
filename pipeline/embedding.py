"""Content embedding via SigLIP vision encoder (google/siglip-base-patch16-224).

Encodes raw image bytes directly — no LLM text description needed.
Output: 768-dim L2-normalized vector.

First use: model downloads automatically (~1.2 GB) to HuggingFace cache.
Device priority: MPS (Mac) → CUDA → CPU.
"""
from __future__ import annotations

import io
import os
from typing import List

import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)

_base_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_base_dir)

try:
    from utils.common_functions import load_config as _load_config
    _cfg = _load_config(os.path.join(_project_dir, "config", "config.yml"))
    _MODEL_NAME: str = _cfg.get("embedding", {}).get("model", "google/siglip-base-patch16-224")
    _EMBEDDING_DIM: int = int(_cfg.get("embedding", {}).get("dim", 768))
except Exception:
    _MODEL_NAME = "google/siglip-base-patch16-224"
    _EMBEDDING_DIM = 768

# Weights for video frame aggregation by selection reason
_REASON_WEIGHTS: dict[str, float] = {
    "scene_change":       1.5,
    "diversity_centroid": 1.5,
    "first_frame":        1.0,
    "last_frame":         1.0,
    "drift":              1.0,
    "safety_flag":        0.5,
}


class EmbeddingGenerator:
    _model = None
    _processor = None
    _device: str = "cpu"

    @classmethod
    def load_model(cls):
        if cls._model is not None:
            return cls._model, cls._processor

        import torch
        from transformers import AutoModel, SiglipImageProcessor

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        logger.info("Loading SigLIP | model=%s device=%s", _MODEL_NAME, device)
        # SiglipImageProcessor only — avoids sentencepiece (text tokenizer not needed)
        cls._processor = SiglipImageProcessor.from_pretrained(_MODEL_NAME)
        cls._model = AutoModel.from_pretrained(_MODEL_NAME).to(device)
        cls._model.eval()
        cls._device = device
        logger.info("SigLIP ready | device=%s dim=%d", device, _EMBEDDING_DIM)
        return cls._model, cls._processor

    @classmethod
    def generate(cls, image_bytes: bytes) -> List[float]:
        """Encode raw image bytes via SigLIP. Returns 768-dim L2-normalized vector."""
        import torch

        model, processor = cls.load_model()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(cls._device)
            with torch.no_grad():
                vision_out = model.vision_model(**inputs)
                pooled = vision_out.pooler_output  # (1, 768)
            vec = pooled.squeeze(0).float().cpu().numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        except Exception as exc:
            logger.warning("SigLIP embedding failed | error=%s — returning zeros", exc)
            return [0.0] * _EMBEDDING_DIM

    @classmethod
    def generate_for_video(cls, frames: List[dict]) -> List[float]:
        """Weighted-average embedding across multiple video frames.

        frames: [{"frame_bytes": bytes, "selection_reason": str}, ...]
        Returns 768-dim L2-normalized vector.
        """
        if not frames:
            return [0.0] * _EMBEDDING_DIM

        embeddings: List[List[float]] = []
        weights: List[float] = []
        for frame in frames:
            frame_bytes = frame.get("frame_bytes")
            reason = frame.get("selection_reason", "")
            if not frame_bytes:
                continue
            emb = cls.generate(frame_bytes)
            embeddings.append(emb)
            weights.append(_REASON_WEIGHTS.get(reason, 1.0))

        if not embeddings:
            return [0.0] * _EMBEDDING_DIM

        arr = np.array(embeddings, dtype=np.float32)     # (N, 768)
        w = np.array(weights, dtype=np.float32)          # (N,)
        weighted = (arr * w[:, np.newaxis]).sum(axis=0)  # (768,)
        norm = np.linalg.norm(weighted)
        if norm > 0:
            weighted = weighted / norm
        return weighted.tolist()
