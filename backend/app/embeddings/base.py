"""Provider-agnostic embedding interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingProvider(Protocol):
    """Minimal interface required by the loader and retriever."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed document texts without a query instruction."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query with the provider-specific query instruction."""
