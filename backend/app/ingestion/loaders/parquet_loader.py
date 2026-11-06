"""Loader for the EnterpriseRAG-Bench parquet schema."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from app.ingestion.loaders.base import SOURCE_TYPES, LoaderIssue
from app.ingestion.models import DOC_ID_PATTERN, RawDocument

REQUIRED_COLUMNS = frozenset({"doc_id", "source_type", "title", "content"})


class ParquetDocumentLoader:
    """Read documents from a parquet file in configurable batches."""

    def __init__(self, path: Path, batch_size: int = 1000) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.path = Path(path)
        self.batch_size = batch_size
        self.issues: list[LoaderIssue] = []

    def iter_batches(self) -> Iterator[list[RawDocument]]:
        """Yield batches of valid documents, recording rejected rows."""

        self.issues.clear()
        parquet_file = pq.ParquetFile(self.path)
        available = set(parquet_file.schema_arrow.names)
        missing = REQUIRED_COLUMNS - available
        if missing:
            raise ValueError(f"{self.path}: missing required columns: {', '.join(sorted(missing))}")

        row_index = 0
        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size,
            columns=sorted(REQUIRED_COLUMNS),
        ):
            documents: list[RawDocument] = []
            for row in batch.to_pylist():
                document = self._row_to_document(row, row_index)
                row_index += 1
                if document is not None:
                    documents.append(document)
            yield documents

    def iter_documents(self) -> Iterator[RawDocument]:
        """Yield valid documents across all batches."""

        for batch in self.iter_batches():
            yield from batch

    def _row_to_document(
        self,
        row: dict[str, object],
        row_index: int,
    ) -> RawDocument | None:
        source_path = f"{self.path.name}:{row_index}"
        doc_id = row.get("doc_id")
        if not isinstance(doc_id, str) or DOC_ID_PATTERN.fullmatch(doc_id) is None:
            self._skip(source_path, "invalid doc_id")
            return None

        source_type = row.get("source_type")
        if not isinstance(source_type, str) or source_type not in SOURCE_TYPES:
            self._skip(source_path, f"unknown source type: {source_type}")
            return None

        title = row.get("title")
        content = row.get("content")
        if not isinstance(title, str) or not isinstance(content, str):
            self._skip(source_path, "invalid title or content")
            return None

        return RawDocument(
            doc_id=doc_id,
            source_type=source_type,
            title=title,
            content=content,
            source_path=source_path,
            metadata={"row_index": row_index},
        )

    def _skip(self, source_path: str, reason: str) -> None:
        self.issues.append(LoaderIssue(source_path=source_path, reason=reason))
