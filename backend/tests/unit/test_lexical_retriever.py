"""Tests for the lexical retriever boundary."""

from __future__ import annotations

import inspect
import uuid
from types import SimpleNamespace

import pytest

from app.db.repositories import LexicalSearchResult
from app.retrieval.lexical import LexicalRetriever


class FakeSession:
    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


def _session_factory() -> FakeSession:
    return FakeSession()


@pytest.mark.asyncio
async def test_retrieve_maps_lexical_results(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = SimpleNamespace(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_type="confluence",
        title="Title",
        chunk_index=2,
        content="content",
        doc_metadata={"origin": "test"},
    )
    calls: list[tuple[object, str, dict[str, object]]] = []

    async def fake_search(session: object, query: str, **kwargs: object):
        calls.append((session, query, kwargs))
        return [LexicalSearchResult(chunk=chunk, rank=0.75)]

    monkeypatch.setattr("app.retrieval.lexical.search_chunks_lexical", fake_search)
    retriever = LexicalRetriever(
        _session_factory,
        tenant_id=uuid.UUID("11111111-1111-1111-1111-111111111111"),
        knowledge_base_id=uuid.UUID("22222222-2222-2222-2222-222222222222"),
    )

    results = await retriever.retrieve("query", top_k=3, source_types=["confluence"])

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 0.75
    assert results[0].metadata == {"origin": "test"}
    _, query, kwargs = calls[0]
    assert query == "query"
    assert kwargs["top_k"] == 3
    assert kwargs["source_types"] == ["confluence"]
    assert kwargs["tenant_id"] == retriever.tenant_id
    assert kwargs["knowledge_base_id"] == retriever.knowledge_base_id


@pytest.mark.asyncio
async def test_empty_query_does_not_query_database() -> None:
    retriever = LexicalRetriever(
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    assert await retriever.retrieve("   ") == []


@pytest.mark.asyncio
async def test_invalid_top_k_is_rejected() -> None:
    retriever = LexicalRetriever(
        _session_factory,
        tenant_id=uuid.uuid4(),
        knowledge_base_id=uuid.uuid4(),
    )

    with pytest.raises(ValueError, match="top_k"):
        await retriever.retrieve("query", top_k=0)


def test_retriever_does_not_expose_evaluation_metadata() -> None:
    parameters = inspect.signature(LexicalRetriever.retrieve).parameters

    for name in ("expected_doc_ids", "question_type", "gold_answer", "source_types_expected"):
        assert name not in parameters
