"""Idempotent embedding and database loading pipeline."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories import (
    ChunkUpsert,
    DocumentUpsert,
    get_embedded_chunk_ids,
    upsert_chunks,
    upsert_documents,
)
from app.embeddings.base import EmbeddingProvider


@dataclass
class LoadStats:
    """Counts produced by one loader run."""

    documents_seen: int = 0
    chunks_seen: int = 0
    chunks_written: int = 0
    chunks_embedded: int = 0
    chunks_skipped: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _read_batches(
    chunks_path: Path,
    batch_size: int,
    limit: int | None,
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    seen = 0
    with chunks_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                batch.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            seen += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if limit is not None and seen >= limit:
                break
    if batch:
        yield batch


def _document_key(chunk: dict[str, Any]) -> tuple[str, str, str]:
    try:
        return (
            chunk["source_type"],
            chunk["doc_id"],
            chunk["document_content_hash"],
        )
    except KeyError as exc:
        raise ValueError(
            "chunk is missing document_content_hash; regenerate the Block 1 corpus"
        ) from exc


async def load_chunks(
    session_factory: async_sessionmaker[AsyncSession],
    chunks_path: Path,
    provider: EmbeddingProvider,
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    batch_size: int = 256,
    limit: int | None = None,
    progress: Callable[[LoadStats], None] | None = None,
    progress_every: int = 10_000,
) -> LoadStats:
    """Embed and load a chunk JSONL file, skipping already embedded chunks."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    stats = LoadStats()
    seen_documents: set[tuple[str, str, str]] = set()
    last_progress = 0

    async with session_factory() as session:
        for batch in _read_batches(chunks_path, batch_size, limit):
            stats.chunks_seen += len(batch)

            document_rows: dict[tuple[str, str, str], DocumentUpsert] = {}
            for chunk in batch:
                key = _document_key(chunk)
                seen_documents.add(key)
                document_rows.setdefault(
                    key,
                    DocumentUpsert(
                        tenant_id=tenant_id,
                        knowledge_base_id=knowledge_base_id,
                        doc_id=chunk["doc_id"],
                        source_type=chunk["source_type"],
                        title=chunk.get("title") or "",
                        content_hash=key[2],
                        metadata=chunk.get("metadata") or {},
                    ),
                )

            document_ids = await upsert_documents(session, list(document_rows.values()))
            stats.documents_seen = len(seen_documents)

            batch_chunk_ids = [chunk["chunk_id"] for chunk in batch]
            embedded_ids = await get_embedded_chunk_ids(
                session,
                batch_chunk_ids,
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            )
            new_chunks = [chunk for chunk in batch if chunk["chunk_id"] not in embedded_ids]
            stats.chunks_skipped += len(batch) - len(new_chunks)

            if new_chunks:
                embeddings = await asyncio.to_thread(
                    provider.embed_documents,
                    [chunk["content"] for chunk in new_chunks],
                )
                chunk_rows = [
                    ChunkUpsert(
                        tenant_id=tenant_id,
                        knowledge_base_id=knowledge_base_id,
                        document_id=document_ids[_document_key(chunk)],
                        doc_id=chunk["doc_id"],
                        chunk_id=chunk["chunk_id"],
                        chunk_index=chunk["chunk_index"],
                        content=chunk["content"],
                        token_count=chunk["token_count"],
                        start_char=chunk["start_char"],
                        end_char=chunk["end_char"],
                        content_hash=chunk["content_hash"],
                        source_type=chunk["source_type"],
                        title=chunk.get("title") or "",
                        metadata=chunk.get("metadata") or {},
                        embedding=embedding,
                        embedding_model=provider.model_name,
                    )
                    for chunk, embedding in zip(new_chunks, embeddings, strict=True)
                ]
                stats.chunks_written += await upsert_chunks(session, chunk_rows)
                stats.chunks_embedded += len(new_chunks)

            await session.commit()

            if progress is not None and stats.chunks_seen - last_progress >= progress_every:
                last_progress = stats.chunks_seen
                progress(stats)

    if progress is not None:
        progress(stats)
    return stats
