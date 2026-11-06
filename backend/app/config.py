"""Configuration for ingestion runs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ingestion.models import SourceType


class IngestionConfig(BaseModel):
    """Validated settings for one ingestion run."""

    model_config = ConfigDict(frozen=True)

    raw_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed")
    chunk_size: int = 400
    chunk_overlap: int = 64
    min_chunk_tokens: int = 50
    tokenizer_name: str = "BAAI/bge-large-en-v1.5"
    source_types: tuple[SourceType, ...] = (
        "confluence",
        "jira",
        "github",
        "hubspot",
    )
    max_docs: int | None = None
    max_uncompressed_bytes: int = Field(default=5 * 1024 * 1024, gt=0)

    @model_validator(mode="after")
    def validate_settings(self) -> IngestionConfig:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if not 0 <= self.min_chunk_tokens <= self.chunk_size:
            raise ValueError("min_chunk_tokens must be in [0, chunk_size]")
        if not self.tokenizer_name:
            raise ValueError("tokenizer_name must not be empty")
        if not self.source_types:
            raise ValueError("source_types must not be empty")
        if self.max_docs is not None and self.max_docs <= 0:
            raise ValueError("max_docs must be positive")
        return self
