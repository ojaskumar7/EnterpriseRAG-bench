"""Database operations used by ingestion and retrieval."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChunkRecord, Document


@dataclass(frozen=True)
class DocumentUpsert:
    tenant_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    doc_id: str
    source_type: str
    title: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkUpsert:
    tenant_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    document_id: uuid.UUID
    doc_id: str
    chunk_id: str
    chunk_index: int
    content: str
    token_count: int
    start_char: int
    end_char: int
    content_hash: str
    source_type: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    embedding_model: str | None = None


@dataclass(frozen=True)
class ChunkSearchResult:
    chunk: ChunkRecord
    distance: float
    score: float


@dataclass(frozen=True)
class LexicalSearchResult:
    chunk: ChunkRecord
    rank: float


async def upsert_documents(
    session: AsyncSession,
    documents: Sequence[DocumentUpsert],
) -> dict[tuple[str, str, str], uuid.UUID]:
    """Upsert documents and return IDs keyed by document identity."""

    if not documents:
        return {}

    values = [
        {
            "tenant_id": document.tenant_id,
            "knowledge_base_id": document.knowledge_base_id,
            "doc_id": document.doc_id,
            "source_type": document.source_type,
            "title": document.title,
            "content_hash": document.content_hash,
            "doc_metadata": document.metadata,
        }
        for document in documents
    ]
    statement = pg_insert(Document).values(values)
    statement = statement.on_conflict_do_update(
        constraint="uq_documents_identity",
        set_={
            "title": statement.excluded.title,
            "doc_metadata": statement.excluded.doc_metadata,
        },
    ).returning(
        Document.doc_id,
        Document.source_type,
        Document.content_hash,
        Document.id,
    )

    rows = (await session.execute(statement)).all()
    return {(row.source_type, row.doc_id, row.content_hash): row.id for row in rows}


async def upsert_chunks(session: AsyncSession, chunks: Sequence[ChunkUpsert]) -> int:
    """Insert or update a batch of chunks."""

    if not chunks:
        return 0

    values = [
        {
            "tenant_id": chunk.tenant_id,
            "knowledge_base_id": chunk.knowledge_base_id,
            "document_id": chunk.document_id,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "content_hash": chunk.content_hash,
            "source_type": chunk.source_type,
            "title": chunk.title,
            "doc_metadata": chunk.metadata,
            "embedding": chunk.embedding,
            "embedding_model": chunk.embedding_model,
        }
        for chunk in chunks
    ]
    statement = pg_insert(ChunkRecord).values(values)
    statement = statement.on_conflict_do_update(
        constraint="uq_chunks_identity",
        set_={
            "document_id": statement.excluded.document_id,
            "doc_id": statement.excluded.doc_id,
            "chunk_index": statement.excluded.chunk_index,
            "content": statement.excluded.content,
            "token_count": statement.excluded.token_count,
            "start_char": statement.excluded.start_char,
            "end_char": statement.excluded.end_char,
            "content_hash": statement.excluded.content_hash,
            "source_type": statement.excluded.source_type,
            "title": statement.excluded.title,
            "doc_metadata": statement.excluded.doc_metadata,
            "embedding": statement.excluded.embedding,
            "embedding_model": statement.excluded.embedding_model,
        },
    )
    await session.execute(statement)
    return len(chunks)


async def get_embedded_chunk_ids(
    session: AsyncSession,
    chunk_ids: Iterable[str],
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
) -> set[str]:
    """Return the subset of chunk IDs that already have an embedding."""

    ids = list(chunk_ids)
    if not ids:
        return set()

    statement = select(ChunkRecord.chunk_id).where(
        ChunkRecord.tenant_id == tenant_id,
        ChunkRecord.knowledge_base_id == knowledge_base_id,
        ChunkRecord.chunk_id.in_(ids),
        ChunkRecord.embedding.is_not(None),
    )
    rows = (await session.execute(statement)).scalars().all()
    return set(rows)


async def search_chunks(
    session: AsyncSession,
    query_embedding: Sequence[float],
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    top_k: int = 10,
    source_types: Sequence[str] | None = None,
) -> list[ChunkSearchResult]:
    """Return chunks ordered by cosine distance."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    distance = ChunkRecord.embedding.cosine_distance(query_embedding).label("distance")
    statement = (
        select(ChunkRecord, distance)
        .where(
            ChunkRecord.tenant_id == tenant_id,
            ChunkRecord.knowledge_base_id == knowledge_base_id,
            ChunkRecord.embedding.is_not(None),
        )
        .order_by(distance)
        .limit(top_k)
    )
    if source_types:
        statement = statement.where(ChunkRecord.source_type.in_(list(source_types)))

    rows = (await session.execute(statement)).all()
    return [
        ChunkSearchResult(
            chunk=row[0],
            distance=float(row[1]),
            score=1.0 - float(row[1]),
        )
        for row in rows
    ]


async def search_chunks_lexical(
    session: AsyncSession,
    query: str,
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    top_k: int = 10,
    source_types: Sequence[str] | None = None,
) -> list[LexicalSearchResult]:
    """Return chunks ranked by PostgreSQL full-text search."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not query.strip():
        return []

    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(ChunkRecord.search_vector, tsquery, 32).label("rank")
    statement = (
        select(ChunkRecord, rank)
        .where(
            ChunkRecord.tenant_id == tenant_id,
            ChunkRecord.knowledge_base_id == knowledge_base_id,
            ChunkRecord.search_vector.op("@@")(tsquery),
        )
        .order_by(rank.desc(), ChunkRecord.chunk_id.asc())
        .limit(top_k)
    )
    if source_types:
        statement = statement.where(ChunkRecord.source_type.in_(list(source_types)))

    rows = (await session.execute(statement)).all()
    return [LexicalSearchResult(chunk=row[0], rank=float(row[1])) for row in rows]


async def count_rows(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID | None = None,
    knowledge_base_id: uuid.UUID | None = None,
) -> dict[str, int]:
    """Return document, chunk, and embedded-chunk counts."""

    document_statement = select(func.count()).select_from(Document)
    chunk_statement = select(func.count()).select_from(ChunkRecord)
    embedded_statement = (
        select(func.count()).select_from(ChunkRecord).where(ChunkRecord.embedding.is_not(None))
    )
    if tenant_id is not None:
        document_statement = document_statement.where(Document.tenant_id == tenant_id)
        chunk_statement = chunk_statement.where(ChunkRecord.tenant_id == tenant_id)
        embedded_statement = embedded_statement.where(ChunkRecord.tenant_id == tenant_id)
    if knowledge_base_id is not None:
        document_statement = document_statement.where(
            Document.knowledge_base_id == knowledge_base_id
        )
        chunk_statement = chunk_statement.where(ChunkRecord.knowledge_base_id == knowledge_base_id)
        embedded_statement = embedded_statement.where(
            ChunkRecord.knowledge_base_id == knowledge_base_id
        )

    document_count = await session.scalar(document_statement)
    chunk_count = await session.scalar(chunk_statement)
    embedded_count = await session.scalar(embedded_statement)
    return {
        "documents": int(document_count or 0),
        "chunks": int(chunk_count or 0),
        "embedded_chunks": int(embedded_count or 0),
    }
