"""End-to-end ingestion pipeline for normalized, chunked documents."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import IngestionConfig
from app.ingestion.chunking.token_chunker import TokenChunker
from app.ingestion.loaders.base import DocumentLoader, LoaderIssue
from app.ingestion.loaders.zip_loader import ZipDocumentLoader
from app.ingestion.models import NormalizedDocument, RawDocument
from app.ingestion.preprocessing.normalize import normalize_document

Normalizer = Callable[[RawDocument], NormalizedDocument]


class DocumentFailure(BaseModel):
    """A document that raised during normalization or chunking."""

    model_config = ConfigDict(frozen=True)

    source_path: str
    doc_id: str
    stage: Literal["normalize", "chunk"]
    error: str


class IngestionResult(BaseModel):
    """Counts and diagnostics from one ingestion run.

    Loaded documents partition into chunked, skipped, duplicate, and failed.
    Loader issues and filtered documents are tracked separately.
    """

    model_config = ConfigDict(frozen=True)

    config: IngestionConfig
    documents_loaded: int = 0
    documents_chunked: int = 0
    documents_skipped: int = 0
    documents_empty: int = 0
    documents_filtered: int = 0
    documents_duplicate: int = 0
    documents_failed: int = 0
    chunks_written: int = 0
    loader_issues: int = 0
    documents_by_source: dict[str, int] = Field(default_factory=dict)
    chunks_by_source: dict[str, int] = Field(default_factory=dict)
    issues: list[LoaderIssue] = Field(default_factory=list)
    failures: list[DocumentFailure] = Field(default_factory=list)


@dataclass
class _RunStats:
    documents_loaded: int = 0
    documents_chunked: int = 0
    documents_skipped: int = 0
    documents_empty: int = 0
    documents_filtered: int = 0
    documents_duplicate: int = 0
    documents_failed: int = 0
    chunks_written: int = 0
    documents_by_source: dict[str, int] = field(default_factory=dict)
    chunks_by_source: dict[str, int] = field(default_factory=dict)
    issues: list[LoaderIssue] = field(default_factory=list)
    failures: list[DocumentFailure] = field(default_factory=list)


def run_ingestion(
    config: IngestionConfig,
    *,
    loader: DocumentLoader | None = None,
    chunker: TokenChunker | None = None,
    normalizer: Normalizer = normalize_document,
) -> IngestionResult:
    """Load, normalize, chunk, and persist documents for one configuration."""

    loaders = [loader] if loader is not None else _discover_loaders(config)
    if chunker is None:
        chunker = TokenChunker(
            tokenizer_name=config.tokenizer_name,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_tokens=config.min_chunk_tokens,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = config.output_dir / "chunks.jsonl"
    stats_path = config.output_dir / "stats.json"
    temp_chunks_path = chunks_path.with_name(f".{chunks_path.name}.tmp")
    temp_stats_path = stats_path.with_name(f".{stats_path.name}.tmp")

    stats = _RunStats(
        documents_by_source=dict.fromkeys(config.source_types, 0),
        chunks_by_source=dict.fromkeys(config.source_types, 0),
    )
    seen_documents: set[tuple[str, str]] = set()

    try:
        with temp_chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
            stop = False
            for current_loader in loaders:
                if config.max_docs is not None and stats.documents_loaded >= config.max_docs:
                    break

                for raw in current_loader.iter_documents():
                    if raw.source_type not in config.source_types:
                        stats.documents_filtered += 1
                        continue
                    if config.max_docs is not None and stats.documents_loaded >= config.max_docs:
                        stop = True
                        break

                    stats.documents_loaded += 1
                    stats.documents_by_source[raw.source_type] += 1

                    try:
                        document = normalizer(raw)
                    except Exception as exc:
                        stats.documents_failed += 1
                        stats.failures.append(
                            DocumentFailure(
                                source_path=raw.source_path,
                                doc_id=raw.doc_id,
                                stage="normalize",
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue

                    document_key = (document.doc_id, document.content_hash)
                    if document_key in seen_documents:
                        stats.documents_duplicate += 1
                        continue
                    seen_documents.add(document_key)

                    if document.is_empty:
                        stats.documents_skipped += 1
                        stats.documents_empty += 1
                        continue

                    try:
                        chunks = chunker.chunk(document)
                    except Exception as exc:
                        stats.documents_failed += 1
                        stats.failures.append(
                            DocumentFailure(
                                source_path=raw.source_path,
                                doc_id=raw.doc_id,
                                stage="chunk",
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue

                    if not chunks:
                        stats.documents_skipped += 1
                        continue

                    for chunk in chunks:
                        handle.write(
                            json.dumps(
                                chunk.model_dump(mode="json"),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                        handle.write("\n")

                    stats.documents_chunked += 1
                    stats.chunks_written += len(chunks)
                    stats.chunks_by_source[document.source_type] += len(chunks)

                stats.issues.extend(current_loader.issues)
                if stop:
                    break

        result = IngestionResult(
            config=config,
            documents_loaded=stats.documents_loaded,
            documents_chunked=stats.documents_chunked,
            documents_skipped=stats.documents_skipped,
            documents_empty=stats.documents_empty,
            documents_filtered=stats.documents_filtered,
            documents_duplicate=stats.documents_duplicate,
            documents_failed=stats.documents_failed,
            chunks_written=stats.chunks_written,
            loader_issues=len(stats.issues),
            documents_by_source=stats.documents_by_source,
            chunks_by_source=stats.chunks_by_source,
            issues=stats.issues,
            failures=stats.failures,
        )

        temp_stats_path.write_text(
            json.dumps(
                result.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temp_chunks_path.replace(chunks_path)
        temp_stats_path.replace(stats_path)
    finally:
        temp_chunks_path.unlink(missing_ok=True)
        temp_stats_path.unlink(missing_ok=True)

    return result


def _discover_loaders(config: IngestionConfig) -> list[ZipDocumentLoader]:
    slices_dir = config.raw_dir / "slices"
    if not slices_dir.is_dir():
        raise FileNotFoundError(f"raw slices directory not found: {slices_dir}")

    zip_paths = sorted(slices_dir.glob("*.zip"))
    selected = [
        path
        for path in zip_paths
        if any(path.name.startswith(f"{source}_") for source in config.source_types)
    ]
    if not selected:
        raise FileNotFoundError(f"no zip slices found for sources in {slices_dir}")

    return [
        ZipDocumentLoader(
            path,
            max_uncompressed_bytes=config.max_uncompressed_bytes,
        )
        for path in selected
    ]
