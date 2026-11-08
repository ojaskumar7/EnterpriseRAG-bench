"""Deterministic embedding provider for tests and dry runs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence


class FakeEmbeddingProvider:
    """Produce deterministic unit vectors without loading a model."""

    def __init__(self, dimension: int = 1024) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.model_name = "fake-embedding"
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[index % len(digest)] / 255.0 + 1e-6 for index in range(self.dimension)]
        norm = math.sqrt(sum(value * value for value in raw))
        return [value / norm for value in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
