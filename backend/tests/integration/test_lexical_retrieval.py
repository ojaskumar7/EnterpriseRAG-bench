"""Integration tests for PostgreSQL lexical retrieval."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.repositories import (
    ChunkUpsert,
    DocumentUpsert,
    search_chunks_lexical,
    upsert_chunks,
    upsert_documents,
)
from app.settings import Settings

pytestmark = pytest.mark.integration


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


async def _seed_chunk(
    session,
    *,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
    doc_id: str,
    content: str,
    title: str,
    source_type: str,
) -> None:
    document = DocumentUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        doc_id=doc_id,
        source_type=source_type,
        title=title,
        content_hash="a" * 64,
    )
    document_id = (await upsert_documents(session, [document]))[
        (document.source_type, document.doc_id, document.content_hash)
    ]
    await upsert_chunks(
        session,
        [
            ChunkUpsert(
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                doc_id=doc_id,
                chunk_id=f"{doc_id}:0:abc",
                chunk_index=0,
                content=content,
                token_count=len(content.split()),
                start_char=0,
                end_char=len(content),
                content_hash="b" * 64,
                source_type=source_type,
                title=title,
                embedding=None,
                embedding_model=None,
            )
        ],
    )
    await session.commit()


async def test_lexical_search_orders_by_rank(session_factory) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    async with session_factory() as session:
        await _seed_chunk(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            doc_id="doc-alpha",
            content="alpha beta gamma",
            title="Alpha guide",
            source_type="confluence",
        )
        await _seed_chunk(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            doc_id="doc-delta",
            content="delta epsilon",
            title="Delta guide",
            source_type="confluence",
        )

    async with session_factory() as session:
        results = await search_chunks_lexical(
            session,
            "alpha",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            top_k=5,
        )

    assert [result.chunk.doc_id for result in results] == ["doc-alpha"]
    assert results[0].rank > 0


async def test_lexical_search_honors_source_filter(session_factory) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    async with session_factory() as session:
        await _seed_chunk(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            doc_id="doc-confluence",
            content="rotation policy",
            title="Rotation",
            source_type="confluence",
        )
        await _seed_chunk(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            doc_id="doc-github",
            content="rotation policy",
            title="Rotation",
            source_type="github",
        )

    async with session_factory() as session:
        results = await search_chunks_lexical(
            session,
            "rotation",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            top_k=5,
            source_types=["github"],
        )

    assert [result.chunk.doc_id for result in results] == ["doc-github"]
