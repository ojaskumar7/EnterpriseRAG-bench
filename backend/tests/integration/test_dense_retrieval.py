"""Integration tests for dense retrieval over pgvector."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.repositories import ChunkUpsert, DocumentUpsert, upsert_chunks, upsert_documents
from app.retrieval.dense import DenseRetriever
from app.settings import Settings

pytestmark = pytest.mark.integration


class FixedProvider:
    model_name = "fixed-embedding"
    dimension = 1024

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [_unit_vector(0) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return _unit_vector(0)


def _unit_vector(index: int, dimension: int = 1024) -> list[float]:
    vector = [0.0] * dimension
    vector[index] = 1.0
    return vector


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(Settings().database_url)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1 FROM documents LIMIT 1"))
            await connection.execute(text("SELECT 1 FROM chunks LIMIT 1"))
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"PostgreSQL schema unavailable: {exc}")

    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def test_retriever_returns_cosine_order(session_factory) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    document = DocumentUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        doc_id=f"dsid_{uuid.uuid4().hex}",
        source_type="github",
        title="Repository",
        content_hash="a" * 64,
    )

    async with session_factory() as session:
        document_ids = await upsert_documents(session, [document])
        document_id = document_ids[(document.source_type, document.doc_id, document.content_hash)]
        chunks = [
            ChunkUpsert(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                doc_id=document.doc_id,
                chunk_id=f"{document.doc_id}:{index}:{index}",
                chunk_index=index,
                content=f"chunk {index}",
                token_count=2,
                start_char=0,
                end_char=1,
                content_hash=str(index) * 64,
                source_type="github",
                title="Repository",
                embedding=_unit_vector(index),
                embedding_model="fixed-embedding",
            )
            for index in (0, 1)
        ]
        await upsert_chunks(session, chunks)
        await session.commit()

    retriever = DenseRetriever(
        FixedProvider(),
        session_factory,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    )
    results = await retriever.retrieve("query", top_k=2)

    assert [result.chunk_index for result in results] == [0, 1]
    assert results[0].score > results[1].score
    assert results[0].doc_id == document.doc_id
    assert results[0].content == "chunk 0"
