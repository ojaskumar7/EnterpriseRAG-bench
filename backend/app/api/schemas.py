"""API request and response schemas."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    knowledge_base_id: UUID
    top_k: int = Field(default=50, gt=0, le=100)
    token_budget: int = Field(default=4000, gt=0, le=32_000)
    max_chunks_per_doc: int = Field(default=3, gt=0, le=20)
    source_types: list[str] | None = None


class CitationSource(BaseModel):
    citation_id: int
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    citation_ids: list[int]
    grounding_decision: str
    grounding_reason: str
    sources: list[CitationSource]
    latency_ms: float
