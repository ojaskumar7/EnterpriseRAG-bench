"""BGE embedding provider backed by sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from app.embeddings.base import DEFAULT_QUERY_INSTRUCTION

DEFAULT_MODEL_NAME = "BAAI/bge-large-en-v1.5"
DEFAULT_DIMENSION = 1024


class BGEEmbeddingProvider:
    """Generate normalized BGE embeddings for documents and queries."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dimension: int = DEFAULT_DIMENSION,
        device: str = "auto",
        batch_size: int = 32,
        dtype: str = "float32",
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        model: Any | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        normalized_dtype = dtype.strip().lower()
        if normalized_dtype not in {"float32", "float16"}:
            raise ValueError("dtype must be one of: float32, float16")

        self.model_name = model_name
        self.dimension = dimension
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.dtype = normalized_dtype
        self.query_instruction = query_instruction
        self._model = model
        if model is not None:
            self._validate_dimension(model)
            self._apply_dtype(model)

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = device.strip().lower()
        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return normalized

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._validate_dimension(self._model)
            self._apply_dtype(self._model)
        return self._model

    def _apply_dtype(self, model: Any) -> None:
        if self.dtype == "float16":
            half = getattr(model, "half", None)
            if not callable(half):
                raise ValueError("model does not support float16 conversion")
            half()

    def _validate_dimension(self, model: Any) -> None:
        get_dimension = getattr(model, "get_embedding_dimension", None)
        detected = (
            get_dimension() if callable(get_dimension) else model.get_sentence_embedding_dimension()
        )
        if detected != self.dimension:
            raise ValueError(
                f"model dimension {detected} does not match configured dimension {self.dimension}"
            )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([f"{self.query_instruction}{text}"])[0]
