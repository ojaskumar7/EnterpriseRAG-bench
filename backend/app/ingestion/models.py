"""Validated document and chunk models for ingestion."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SourceType = Literal[
    "slack",
    "gmail",
    "linear",
    "google_drive",
    "hubspot",
    "fireflies",
    "github",
    "jira",
    "confluence",
]

DOC_ID_PATTERN = re.compile(r"^dsid_[0-9a-f]{32}$")


class DocumentModel(BaseModel):
    """Base model with shared validation and immutability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str

    @field_validator("doc_id")
    @classmethod
    def validate_doc_id(cls, value: str) -> str:
        if DOC_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("doc_id must match ^dsid_[0-9a-f]{32}$")
        return value


class RawDocument(DocumentModel):
    """A document as loaded from a source, before normalization."""

    source_type: SourceType
    title: str
    content: str
    source_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(DocumentModel):
    """A document after deterministic text normalization."""

    source_type: SourceType
    title: str
    content: str
    raw_content: str
    content_hash: str
    char_count: int
    is_empty: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(DocumentModel):
    """A token-bounded chunk derived from a normalized document."""

    chunk_id: str
    chunk_index: int
    content: str
    token_count: int
    start_char: int
    end_char: int
    content_hash: str
    source_type: SourceType
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)
