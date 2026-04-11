"""Content embedding generation via Ollama embedding models.

Uses Ollama (already running for LLM) to avoid loading PyTorch into the same
process as TensorFlow (DeepFace) and ONNX Runtime (faster-whisper) — all three
bring their own native allocators which conflict and cause SIGABRT crashes.

Default model: nomic-embed-text (768-dim, pull with: ollama pull nomic-embed-text)
"""
from __future__ import annotations

import os
from typing import List

import ollama

from utils.logger import get_logger

logger = get_logger(__name__)

_base_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_base_dir)

try:
    from utils.common_functions import load_config as _load_config
    _cfg = _load_config(os.path.join(_project_dir, "config", "config.yml"))
    _EMBEDDING_MODEL: str = _cfg.get("embedding", {}).get("model", "nomic-embed-text")
    _EMBEDDING_DIM: int = int(_cfg.get("embedding", {}).get("dim", 768))
except Exception:
    _EMBEDDING_MODEL = "nomic-embed-text"
    _EMBEDDING_DIM = 768


class EmbeddingGenerator:
    _client: ollama.Client = None  # type: ignore[assignment]

    @classmethod
    def _get_client(cls) -> ollama.Client:
        if cls._client is None:
            cls._client = ollama.Client()
        return cls._client

    @classmethod
    def generate(cls, text: str, model: str = _EMBEDDING_MODEL) -> List[float]:
        """Encode text via Ollama embedding model. Returns a flat float list.

        First use:  ollama pull nomic-embed-text
        Dimension:  768 (nomic-embed-text) — update config embedding.dim if you
                    switch models (e.g. all-minilm → 384, mxbai-embed-large → 1024).
        """
        if not text.strip():
            return [0.0] * _EMBEDDING_DIM
        try:
            client = cls._get_client()
            response = client.embeddings(model=model, prompt=text)
            embedding: List[float] = response["embedding"]
            logger.debug("Embedding generated | model=%s dim=%d", model, len(embedding))
            return embedding
        except Exception as exc:
            logger.warning("Embedding failed | model=%s error=%s — returning zeros", model, exc)
            return [0.0] * _EMBEDDING_DIM
