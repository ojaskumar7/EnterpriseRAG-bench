"""Structured context produced from ranked retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBlock:
    citation_id: int
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    chunk_index: int
    content: str
    score: float
    token_count: int


@dataclass(frozen=True)
class BuiltContext:
    query: str
    blocks: tuple[ContextBlock, ...]
    token_count: int
    token_budget: int
    candidate_count: int
    omitted_count: int
    selected_doc_count: int
    source_type_counts: dict[str, int]
