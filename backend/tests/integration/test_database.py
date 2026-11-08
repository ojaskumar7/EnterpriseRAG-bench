"""Integration tests for the pgvector schema and repositories."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.repositories import (
    ChunkUpsert,
    DocumentUpsert,
    count_rows,
    get_embedded_chunk_ids,
    search_chunks,
    upsert_chunks,
    upsert_documents,
)
from app.settings import Settings

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(Settings().database_url)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1 FROM documents LIMIT 1"))
            await connection.execute(text("SELECT 1 FROM chunks LIMIT 1"))
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"PostgreSQL schema unavailable: {exc}")

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as current:
        yield current
    await engine.dispose()


def _unit_vector(index: int, dimension: int = 1024) -> list[float]:
    vector = [0.0] * dimension
    vector[index] = 1.0
    return vector


async def test_extension_and_hnsw_index_exist(session) -> None:
    extension = await session.scalar(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    index = await session.scalar(
        text(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'chunks' AND indexname = 'ix_chunks_embedding_hnsw'"
        )
    )

    assert extension == "vector"
    assert index == "ix_chunks_embedding_hnsw"


async def test_document_upsert_is_content_addressed(session) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    document = DocumentUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        doc_id=f"dsid_{uuid.uuid4().hex}",
        source_type="confluence",
        title="Title",
        content_hash="a" * 64,
        metadata={"origin": "test"},
    )

    first = await upsert_documents(session, [document])
    second = await upsert_documents(session, [document])
    await session.commit()

    key = (document.source_type, document.doc_id, document.content_hash)
    assert first[key] == second[key]

    changed = DocumentUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        doc_id=document.doc_id,
        source_type=document.source_type,
        title="Changed",
        content_hash="b" * 64,
    )
    changed_ids = await upsert_documents(session, [changed])
    await session.commit()

    assert changed_ids[(changed.source_type, changed.doc_id, changed.content_hash)] != first[key]


async def test_chunk_upsert_and_embedding_skip(session) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    document = DocumentUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        doc_id=f"dsid_{uuid.uuid4().hex}",
        source_type="github",
        title="Repository",
        content_hash="c" * 64,
    )
    document_ids = await upsert_documents(session, [document])
    document_id = document_ids[(document.source_type, document.doc_id, document.content_hash)]
    chunk = ChunkUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        doc_id=document.doc_id,
        chunk_id=f"{document.doc_id}:0:abc",
        chunk_index=0,
        content="first chunk",
        token_count=2,
        start_char=0,
        end_char=11,
        content_hash="d" * 64,
        source_type="github",
        title="Repository",
        embedding=_unit_vector(0),
        embedding_model="fake-embedding",
    )

    assert await upsert_chunks(session, [chunk]) == 1
    await session.commit()
    assert await get_embedded_chunk_ids(
        session,
        [chunk.chunk_id],
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    ) == {chunk.chunk_id}

    assert await upsert_chunks(session, [chunk]) == 1
    await session.commit()
    counts = await count_rows(
        session,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    )
    assert counts == {"documents": 1, "chunks": 1, "embedded_chunks": 1}


async def test_search_orders_by_cosine_distance(session) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    document = DocumentUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        doc_id=f"dsid_{uuid.uuid4().hex}",
        source_type="jira",
        title="Ticket",
        content_hash="e" * 64,
    )
    document_id = (await upsert_documents(session, [document]))[
        (document.source_type, document.doc_id, document.content_hash)
    ]

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
            source_type="jira",
            title="Ticket",
            embedding=_unit_vector(index),
            embedding_model="fake-embedding",
        )
        for index in (0, 1)
    ]
    await upsert_chunks(session, chunks)
    await session.commit()

    results = await search_chunks(
        session,
        _unit_vector(0),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        top_k=2,
    )

    assert [result.chunk.chunk_index for result in results] == [0, 1]
    assert results[0].score > results[1].score


async def test_search_honors_source_filter(session) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    document = DocumentUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        doc_id=f"dsid_{uuid.uuid4().hex}",
        source_type="hubspot",
        title="Article",
        content_hash="f" * 64,
    )
    document_id = (await upsert_documents(session, [document]))[
        (document.source_type, document.doc_id, document.content_hash)
    ]
    chunk = ChunkUpsert(
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        doc_id=document.doc_id,
        chunk_id=f"{document.doc_id}:0:abc",
        chunk_index=0,
        content="filtered chunk",
        token_count=2,
        start_char=0,
        end_char=1,
        content_hash="g" * 64,
        source_type="hubspot",
        title="Article",
        embedding=_unit_vector(0),
        embedding_model="fake-embedding",
    )
    await upsert_chunks(session, [chunk])
    await session.commit()

    matching = await search_chunks(
        session,
        _unit_vector(0),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        source_types=["hubspot"],
    )
    missing = await search_chunks(
        session,
        _unit_vector(0),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        source_types=["github"],
    )

    assert [result.chunk.chunk_id for result in matching] == [chunk.chunk_id]
    assert missing == []


async def test_lexical_search_schema_exists(session) -> None:
    generated = await session.scalar(
        text(
            "SELECT is_generated FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'search_vector'"
        )
    )
    index = await session.scalar(
        text(
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename = 'chunks' AND indexname = 'ix_chunks_search_vector'"
        )
    )

    assert generated == "ALWAYS"
    assert index == "ix_chunks_search_vector"
