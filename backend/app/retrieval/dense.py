"""Dense retrieval over pgvector."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories import search_chunks
from app.embeddings.base import EmbeddingProvider
from app.retrieval.types import RetrievedChunk


class DenseRetriever:
    """Embed a query and retrieve the nearest chunks."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
    ) -> None:
        self.provider = provider
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
        """Retrieve chunks for a query using only user-provided inputs."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        query_embedding = await asyncio.to_thread(self.provider.embed_query, query)
        async with self.session_factory() as session:
            results = await search_chunks(
                session,
                query_embedding,
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
                score=result.score,
                metadata=result.chunk.doc_metadata,
            )
            for result in results
        ]
