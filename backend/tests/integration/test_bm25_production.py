import subprocess
import sys
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.db.repositories import (
    ChunkUpsert,
    DocumentUpsert,
    upsert_chunks,
    upsert_documents,
)
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.bm25_builder import build_bm25_index, load_bm25_corpus
from app.retrieval.bm25_index import BM25IndexManager, BM25IndexStale
from app.settings import Settings


@pytest_asyncio.fixture
async def session_factory():
    engine = create_async_engine(Settings().database_url)
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1 FROM chunks LIMIT 1"))
    except Exception as exc:
        await engine.dispose()
        pytest.skip(f"PostgreSQL schema unavailable: {exc}")

    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.mark.integration
async def test_retriever_raises_when_chunk_source_type_changes(
    session_factory,
    tmp_path,
) -> None:
    tenant_id = uuid.uuid4()
    knowledge_base_id = uuid.uuid4()
    doc_id = f"dsid_{uuid.uuid4().hex}"
    chunk_id = f"{doc_id}:0:0"

    async with session_factory() as session:
        document = DocumentUpsert(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            doc_id=doc_id,
            source_type="confluence",
            title="Needle",
            content_hash="a" * 64,
        )
        document_ids = await upsert_documents(session, [document])
        chunk = ChunkUpsert(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_ids[("confluence", doc_id, "a" * 64)],
            doc_id=doc_id,
            chunk_id=chunk_id,
            chunk_index=0,
            content="needle alpha",
            token_count=2,
            start_char=0,
            end_char=12,
            content_hash="b" * 64,
            source_type="confluence",
            title="Needle",
        )
        assert await upsert_chunks(session, [chunk]) == 1
        await session.commit()

    try:
        async with session_factory() as session:
            corpus = await load_bm25_corpus(
                session,
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
            )
        build_bm25_index(
            corpus=corpus,
            index_root=tmp_path,
            tenant_id=str(tenant_id),
            knowledge_base_id=str(knowledge_base_id),
        )
        retriever = BM25Retriever(
            BM25IndexManager(tmp_path),
            session_factory,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )

        assert [result.chunk_id for result in await retriever.retrieve("needle", top_k=1)] == [
            chunk_id
        ]

        async with session_factory() as session:
            await session.execute(
                text(
                    "UPDATE chunks SET source_type = 'github' "
                    "WHERE tenant_id = :tenant_id AND knowledge_base_id = :knowledge_base_id "
                    "AND chunk_id = :chunk_id"
                ),
                {
                    "tenant_id": tenant_id,
                    "knowledge_base_id": knowledge_base_id,
                    "chunk_id": chunk_id,
                },
            )
            await session.commit()

        with pytest.raises(BM25IndexStale, match="metadata"):
            await retriever.retrieve("needle", top_k=1)
    finally:
        async with session_factory() as session:
            await session.execute(
                text(
                    "DELETE FROM chunks WHERE tenant_id = :tenant_id "
                    "AND knowledge_base_id = :knowledge_base_id"
                ),
                {"tenant_id": tenant_id, "knowledge_base_id": knowledge_base_id},
            )
            await session.execute(
                text(
                    "DELETE FROM documents WHERE tenant_id = :tenant_id "
                    "AND knowledge_base_id = :knowledge_base_id"
                ),
                {"tenant_id": tenant_id, "knowledge_base_id": knowledge_base_id},
            )
            await session.commit()


@pytest.mark.integration
def test_build_cli_check_detects_missing_index(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_bm25_index.py",
            "--tenant-id",
            str(uuid.uuid4()),
            "--knowledge-base-id",
            str(uuid.uuid4()),
            "--index-root",
            str(tmp_path),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "rebuild" in result.stderr.lower()
