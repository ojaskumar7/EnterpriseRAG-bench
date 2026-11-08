"""Tests for the dense retriever boundary."""

from __future__ import annotations

import inspect
import uuid
from types import SimpleNamespace

import pytest

from app.db.repositories import ChunkSearchResult
from app.retrieval.dense import DenseRetriever


class FakeProvider:
    model_name = "fake"
    dimension = 2

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeSession:
    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def _session_factory() -> FakeSession:
    return FakeSession()


@pytest.mark.asyncio
async def test_retrieve_maps_chunk_results(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeProvider()
    chunk = SimpleNamespace(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_type="github",
        title="Title",
        chunk_index=2,
        content="content",
        doc_metadata={"origin": "test"},
    )
    calls: list[tuple[object, object, dict[str, object]]] = []

    async def fake_search(session: object, embedding: object, **kwargs: object):
        calls.append((session, embedding, kwargs))
        return [ChunkSearchResult(chunk=chunk, distance=0.25, score=0.75)]

    monkeypatch.setattr("app.retrieval.dense.search_chunks", fake_search)
    retriever = DenseRetriever(
        provider,
        _session_factory,
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        knowledge_base_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
    )

    results = await retriever.retrieve("query", top_k=3, source_types=["github"])

    assert provider.queries == ["query"]
    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].doc_id == "doc-1"
    assert results[0].score == 0.75
    assert results[0].metadata == {"origin": "test"}
    _, embedding, kwargs = calls[0]
    assert embedding == [1.0, 0.0]
    assert kwargs["top_k"] == 3
    assert kwargs["source_types"] == ["github"]
    assert kwargs["tenant_id"] == retriever.tenant_id
    assert kwargs["knowledge_base_id"] == retriever.knowledge_base_id


@pytest.mark.asyncio
async def test_empty_query_does_not_embed() -> None:
    provider = FakeProvider()
    retriever = DenseRetriever(
        provider,
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    assert await retriever.retrieve("   ") == []
    assert provider.queries == []


@pytest.mark.asyncio
async def test_invalid_top_k_is_rejected() -> None:
    retriever = DenseRetriever(
        FakeProvider(),
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    with pytest.raises(ValueError, match="top_k"):
        await retriever.retrieve("query", top_k=0)


def test_retriever_does_not_expose_evaluation_metadata() -> None:
    parameters = inspect.signature(DenseRetriever.retrieve).parameters

    for name in ("expected_doc_ids", "question_type", "gold_answer", "source_types_expected"):
        assert name not in parameters
