"""Lexical retrieval over PostgreSQL full-text search."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories import search_chunks_lexical
from app.retrieval.types import RetrievedChunk


class LexicalRetriever:
    """Run a PostgreSQL full-text query and return ranked chunks."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
    ) -> None:
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
        """Retrieve chunks using only user-provided inputs."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        async with self.session_factory() as session:
            results = await search_chunks_lexical(
                session,
                query,
                tenant_id=self.tenant_id,
                knowledge_base_id=self.knowledge_base_id,
                top_k=top_k,
                source_types=source_types,
            )

        return [
            RetrievedChunk(
                chunk_id=result.chunk.chunk_id,
                doc_id=result.chunk.doc_id,
                source_type=result.chunk.source_type,
                title=result.chunk.title,
                chunk_index=result.chunk.chunk_index,
                content=result.chunk.content,
                score=result.rank,
                metadata=result.chunk.doc_metadata,
            )
            for result in results
        ]
