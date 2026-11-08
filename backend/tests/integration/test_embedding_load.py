"""Integration tests for the idempotent embedding loader."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.repositories import count_rows
from app.embeddings.fake import FakeEmbeddingProvider
from app.embeddings.loader import load_chunks
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


def _chunk(
    doc_id: str,
    document_hash: str,
    index: int,
    content: str,
    source_type: str,
    title: str,
) -> dict[str, object]:
    return {
        "chunk_id": f"{doc_id}:{index}:{index:012d}",
        "doc_id": doc_id,
        "chunk_index": index,
        "content": content,
        "token_count": len(content.split()),
        "start_char": 0,
        "end_char": len(content),
        "content_hash": f"{index:064d}",
        "document_content_hash": document_hash,
        "source_type": source_type,
        "title": title,
        "metadata": {"origin": "test"},
    }


def _write_chunks(path: Path, chunks: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")


async def test_loader_writes_embeddings_and_is_idempotent(
    session_factory,
    tmp_path: Path,
) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    doc_one = f"dsid_{uuid.uuid4().hex}"
    doc_two = f"dsid_{uuid.uuid4().hex}"
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            _chunk(doc_one, "a" * 64, 0, "alpha beta", "confluence", "Doc One"),
            _chunk(doc_one, "a" * 64, 1, "gamma delta", "confluence", "Doc One"),
            _chunk(doc_two, "b" * 64, 0, "epsilon zeta", "github", "Doc Two"),
        ],
    )

    first = await load_chunks(
        session_factory,
        chunks_path,
        FakeEmbeddingProvider(dimension=1024),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        batch_size=2,
    )
    assert first.as_dict() == {
        "documents_seen": 2,
        "chunks_seen": 3,
        "chunks_written": 3,
        "chunks_embedded": 3,
        "chunks_skipped": 0,
    }

    async with session_factory() as session:
        counts = await count_rows(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )
        dimensions = await session.scalar(
            text(
                "SELECT count(*) FROM chunks "
                "WHERE tenant_id = :tenant_id AND vector_dims(embedding) = 1024"
            ),
            {"tenant_id": tenant_id},
        )
    assert counts == {"documents": 2, "chunks": 3, "embedded_chunks": 3}
    assert dimensions == 3

    second = await load_chunks(
        session_factory,
        chunks_path,
        FakeEmbeddingProvider(dimension=1024),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        batch_size=2,
    )
    assert second.as_dict() == {
        "documents_seen": 2,
        "chunks_seen": 3,
        "chunks_written": 0,
        "chunks_embedded": 0,
        "chunks_skipped": 3,
    }

    async with session_factory() as session:
        counts = await count_rows(
            session,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )
    assert counts == {"documents": 2, "chunks": 3, "embedded_chunks": 3}


async def test_loader_limit_limits_chunks(session_factory, tmp_path: Path) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    doc_id = f"dsid_{uuid.uuid4().hex}"
    chunks_path = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks_path,
        [
            _chunk(doc_id, "c" * 64, 0, "one", "hubspot", "Doc"),
            _chunk(doc_id, "c" * 64, 1, "two", "hubspot", "Doc"),
        ],
    )

    stats = await load_chunks(
        session_factory,
        chunks_path,
        FakeEmbeddingProvider(dimension=1024),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        limit=1,
    )

    assert stats.chunks_seen == 1
    assert stats.chunks_embedded == 1
