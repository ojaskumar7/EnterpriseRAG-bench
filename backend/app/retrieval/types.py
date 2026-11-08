"""Shared retrieval result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a retrieval strategy."""

    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    chunk_index: int
    content: str
    score: float
    metadata: dict[str, Any]
