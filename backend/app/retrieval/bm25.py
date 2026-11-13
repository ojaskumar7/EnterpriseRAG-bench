"""In-process BM25 retrieval over a versioned index."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories import get_chunks_by_ids
from app.retrieval.bm25_index import (
    BM25CatalogEntry,
    BM25IndexManager,
    BM25IndexSnapshot,
    BM25IndexStale,
    rebuild_command,
)
from app.retrieval.types import RetrievedChunk


class BM25Retriever:
    def __init__(
        self,
        manager: BM25IndexManager,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
    ) -> None:
        self.manager = manager
        self.session_factory = session_factory
        self.tenant_id = tenant_id
        self.knowledge_base_id = knowledge_base_id

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        tenant_id = str(self.tenant_id)
        knowledge_base_id = str(self.knowledge_base_id)
        snapshot: BM25IndexSnapshot = await asyncio.to_thread(
            self.manager.load,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )
        tokens = await asyncio.to_thread(snapshot.tokenize, query)
        if not tokens:
            return []

        allowed = set(source_types) if source_types else None
        candidate_k = top_k
        if allowed is not None:
            candidate_k = min(max(top_k * 4, top_k), snapshot.row_count)

        selected: list[tuple[BM25CatalogEntry, float]] = []
        while True:
            indices, scores = await asyncio.to_thread(snapshot.search, tokens, candidate_k)
            selected = [
                (snapshot.catalog[index], score)
                for index, score in zip(indices, scores, strict=True)
                if allowed is None or snapshot.catalog[index].source_type in allowed
            ]
            if len(selected) >= top_k or candidate_k >= snapshot.row_count:
                break
            next_k = min(snapshot.row_count, candidate_k * 2)
            if next_k == candidate_k:
                break
            candidate_k = next_k

        selected = selected[:top_k]
        if not selected:
            return []

        chunk_ids = [entry.chunk_id for entry, _ in selected]
        async with self.session_factory() as session:
            rows = await get_chunks_by_ids(
                session,
                chunk_ids,
                tenant_id=self.tenant_id,
                knowledge_base_id=self.knowledge_base_id,
                source_types=source_types,
            )

        by_id = {row.chunk_id: row for row in rows}
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in by_id]
        if missing:
            raise BM25IndexStale(
                f"BM25 index references {len(missing)} missing chunk(s) for tenant "
                f"{tenant_id} and KB {knowledge_base_id}; rebuild with: "
                f"{rebuild_command(tenant_id, knowledge_base_id)}"
            )

        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                doc_id=row.doc_id,
                source_type=row.source_type,
                title=row.title,
                chunk_index=row.chunk_index,
                content=row.content,
                score=score,
                metadata=row.doc_metadata,
            )
            for entry, score in selected
            for row in [by_id[entry.chunk_id]]
        ]
