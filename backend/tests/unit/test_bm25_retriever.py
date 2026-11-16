from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.bm25_index import (
    BM25CatalogEntry,
    BM25IndexSnapshot,
    BM25IndexStale,
    BM25Manifest,
)


class FakeSnapshot:
    def __init__(self) -> None:
        self.manifest = BM25Manifest(
            schema_version=1,
            version="v1",
            tenant_id="tenant",
            knowledge_base_id="kb",
            created_at="2026-11-12T08:30:00+00:00",
            document_count=20,
            row_count=20,
            corpus_fingerprint="f" * 64,
            catalog_sha256="c" * 64,
            k1=1.5,
            b=0.75,
            method="lucene",
            stopwords="english",
            stemmer="english",
            bm25s_version="0.3.16",
            pystemmer_version="3.1.0",
        )
        self.catalog = tuple(
            BM25CatalogEntry(
                row_index=index,
                chunk_id=f"chunk-{index}",
                doc_id=f"doc-{index}",
                source_type="pdf" if index == 19 else "html",
                title=f"Title {index}",
                chunk_index=index,
                content_hash=f"hash-{index}",
            )
            for index in range(20)
        )

    @property
    def row_count(self) -> int:
        return len(self.catalog)

    def tokenize(self, query: str) -> list[str]:
        return ["needle"]

    def search(self, tokens: Sequence[str], k: int) -> tuple[list[int], list[float]]:
        indices = list(range(min(k, self.row_count)))
        return indices, [1.0 for _ in indices]


class FakeManager:
    def __init__(self, snapshot: BM25IndexSnapshot) -> None:
        self.snapshot = snapshot

    def load(self, *, tenant_id: str, knowledge_base_id: str) -> BM25IndexSnapshot:
        return self.snapshot


class ZeroScoreSnapshot(FakeSnapshot):
    def __init__(self) -> None:
        super().__init__()
        self.search_calls = 0

    def tokenize(self, query: str) -> list[str]:
        del query
        return ["zzzznotinvocabulary"]

    def search(self, tokens: Sequence[str], k: int) -> tuple[list[int], list[float]]:
        del tokens
        self.search_calls += 1
        indices = list(range(min(k, self.row_count)))
        return indices, [0.0 for _ in indices]


class PartialSnapshot(FakeSnapshot):
    def __init__(self) -> None:
        super().__init__()
        self.search_calls = 0

    def search(self, tokens: Sequence[str], k: int) -> tuple[list[int], list[float]]:
        del tokens, k
        self.search_calls += 1
        return [0, 1, 2], [2.0, 1.5, 1.0]


class FakeChunk:
    def __init__(self, chunk_id: str, source_type: str) -> None:
        index = int(chunk_id.rsplit("-", 1)[1])
        self.chunk_id = chunk_id
        self.doc_id = f"doc-{index}"
        self.source_type = source_type
        self.title = f"Title {index}"
        self.chunk_index = index
        self.content = "needle"
        self.content_hash = f"hash-{index}"
        self.doc_metadata = {}


class FakeSession:
    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def _session_factory() -> FakeSession:
    return FakeSession()


async def test_retriever_overfetches_for_source_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = FakeSnapshot()

    async def fake_get_chunks_by_ids(
        session: object,
        chunk_ids: Sequence[str],
        **kwargs: object,
    ) -> list[FakeChunk]:
        del session, kwargs
        return [
            FakeChunk(chunk_id, "pdf" if chunk_id == "chunk-19" else "html")
            for chunk_id in chunk_ids
        ]

    monkeypatch.setattr("app.retrieval.bm25.get_chunks_by_ids", fake_get_chunks_by_ids)

    retriever = BM25Retriever(
        FakeManager(snapshot),
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )
    results = await retriever.retrieve("needle", top_k=2, source_types=["pdf"])

    assert [result.chunk_id for result in results] == ["chunk-19"]


async def test_retriever_returns_empty_for_out_of_vocabulary_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_get_chunks_by_ids(*args: object, **kwargs: object) -> list[FakeChunk]:
        raise AssertionError("OOV query must not hydrate chunks")

    monkeypatch.setattr("app.retrieval.bm25.get_chunks_by_ids", fail_get_chunks_by_ids)
    retriever = BM25Retriever(
        FakeManager(ZeroScoreSnapshot()),
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    assert await retriever.retrieve("zzzznotinvocabulary", top_k=2) == []


async def test_retriever_returns_empty_for_out_of_vocabulary_query_with_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_get_chunks_by_ids(*args: object, **kwargs: object) -> list[FakeChunk]:
        raise AssertionError("OOV query must not hydrate chunks")

    monkeypatch.setattr("app.retrieval.bm25.get_chunks_by_ids", fail_get_chunks_by_ids)
    snapshot = ZeroScoreSnapshot()
    retriever = BM25Retriever(
        FakeManager(snapshot),
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    assert await retriever.retrieve("zzzznotinvocabulary", top_k=2, source_types=["pdf"]) == []
    assert snapshot.search_calls == 1


async def test_retriever_rejects_stale_catalog_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = FakeSnapshot()

    async def fake_get_chunks_by_ids(
        session: object,
        chunk_ids: Sequence[str],
        **kwargs: object,
    ) -> list[FakeChunk]:
        del session, kwargs
        return [FakeChunk(chunk_id, "jira") for chunk_id in chunk_ids]

    monkeypatch.setattr("app.retrieval.bm25.get_chunks_by_ids", fake_get_chunks_by_ids)
    retriever = BM25Retriever(
        FakeManager(snapshot),
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    with pytest.raises(BM25IndexStale, match="metadata"):
        await retriever.retrieve("needle", top_k=2)


async def test_retriever_rejects_missing_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = FakeSnapshot()

    async def fake_get_chunks_by_ids(*args: object, **kwargs: object) -> list[FakeChunk]:
        return []

    monkeypatch.setattr("app.retrieval.bm25.get_chunks_by_ids", fake_get_chunks_by_ids)

    retriever = BM25Retriever(
        FakeManager(snapshot),
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    with pytest.raises(BM25IndexStale, match="rebuild"):
        await retriever.retrieve("needle", top_k=2)


async def test_retriever_returns_empty_for_blank_query() -> None:
    retriever = BM25Retriever(
        FakeManager(FakeSnapshot()),
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    assert await retriever.retrieve("   ", top_k=2) == []


async def test_retriever_stops_when_positive_hits_are_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = PartialSnapshot()

    async def fake_get_chunks_by_ids(
        session: object,
        chunk_ids: Sequence[str],
        **kwargs: object,
    ) -> list[FakeChunk]:
        del session, kwargs
        return [FakeChunk(chunk_id, "html") for chunk_id in chunk_ids]

    monkeypatch.setattr("app.retrieval.bm25.get_chunks_by_ids", fake_get_chunks_by_ids)
    retriever = BM25Retriever(
        FakeManager(snapshot),
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    results = await retriever.retrieve("needle", top_k=10)

    assert [result.chunk_id for result in results] == ["chunk-0", "chunk-1", "chunk-2"]
    assert snapshot.search_calls == 1
