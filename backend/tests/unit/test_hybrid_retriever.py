"""Tests for hybrid retrieval and RRF integration."""

from __future__ import annotations

import inspect

import pytest

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append({"query": query, "top_k": top_k, "source_types": source_types})
        return self.chunks[:top_k]


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_type="github",
        title="Title",
        chunk_index=0,
        content="content",
        score=0.0,
        metadata={},
    )


@pytest.mark.asyncio
async def test_hybrid_fuses_and_slices_results() -> None:
    dense = FakeRetriever([_chunk("a"), _chunk("b")])
    lexical = FakeRetriever([_chunk("b"), _chunk("c")])
    retriever = HybridRetriever(
        dense,
        lexical,
        rrf_k=60,
        candidate_k=2,
    )

    results = await retriever.retrieve("query", top_k=2, source_types=["github"])

    assert [result.chunk_id for result in results] == ["b", "a"]
    assert dense.calls[0]["top_k"] == 2
    assert lexical.calls[0]["top_k"] == 2
    assert dense.calls[0]["source_types"] == ["github"]
    assert lexical.calls[0]["source_types"] == ["github"]


@pytest.mark.asyncio
async def test_candidate_k_expands_to_top_k() -> None:
    dense = FakeRetriever([_chunk("a")])
    lexical = FakeRetriever([_chunk("b")])
    retriever = HybridRetriever(dense, lexical, candidate_k=1)

    await retriever.retrieve("query", top_k=5)

    assert dense.calls[0]["top_k"] == 5
    assert lexical.calls[0]["top_k"] == 5


@pytest.mark.asyncio
async def test_empty_query_does_not_call_retrievers() -> None:
    dense = FakeRetriever([])
    lexical = FakeRetriever([])
    retriever = HybridRetriever(dense, lexical)

    assert await retriever.retrieve("   ") == []
    assert dense.calls == []
    assert lexical.calls == []


def test_hybrid_rejects_invalid_parameters() -> None:
    dense = FakeRetriever([])
    lexical = FakeRetriever([])

    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(dense, lexical, rrf_k=0)
    with pytest.raises(ValueError, match="candidate_k"):
        HybridRetriever(dense, lexical, candidate_k=0)
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever(dense, lexical, dense_weight=-1.0)


def test_retriever_does_not_expose_evaluation_metadata() -> None:
    parameters = inspect.signature(HybridRetriever.retrieve).parameters

    for name in ("expected_doc_ids", "question_type", "gold_answer", "source_types_expected"):
        assert name not in parameters
