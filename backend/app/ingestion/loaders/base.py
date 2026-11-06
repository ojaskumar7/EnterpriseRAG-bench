"""Shared loader protocol and issue reporting."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, get_args

from pydantic import BaseModel, ConfigDict

from app.ingestion.models import RawDocument, SourceType

SOURCE_TYPES = frozenset(get_args(SourceType))


class LoaderIssue(BaseModel):
    """A skipped or rejected source item and its reason."""

    model_config = ConfigDict(frozen=True)

    source_path: str
    reason: str


class DocumentLoader(Protocol):
    """Protocol implemented by document loaders."""

    issues: list[LoaderIssue]

    def iter_documents(self) -> Iterator[RawDocument]:
        """Yield validated raw documents."""
        ...
