from __future__ import annotations

import pytest

from app.retrieval.reranked import RerankedRetriever
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    async def retrieve(self, query: str, *, top_k: int = 10, source_types=None):
        self.calls.append({"query": query, "top_k": top_k, "source_types": source_types})
        return self.chunks[:top_k]


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return self.scores


def chunk(chunk_id: str, doc_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_type="pdf",
        title=f"Title {chunk_id}",
        chunk_index=0,
        content=f"Content {chunk_id}",
        score=score,
        metadata={},
    )


async def test_reranks_and_preserves_bm25_score() -> None:
    base = FakeRetriever([chunk("a", "d1", 10.0), chunk("b", "d2", 9.0)])
    reranker = FakeReranker([0.1, 0.9])
    retriever = RerankedRetriever(base, reranker, candidate_k=2)

    results = await retriever.retrieve("query", top_k=2, source_types=["pdf"])

    assert [result.chunk_id for result in results] == ["b", "a"]
    assert results[0].score == 0.9
    assert results[0].retrieval_score == 9.0
    assert base.calls[0]["top_k"] == 2
    assert base.calls[0]["source_types"] == ["pdf"]
    assert reranker.calls[0] == (
        "query",
        ["Title a\nContent a", "Title b\nContent b"],
    )


async def test_ties_break_by_chunk_id_and_slices_top_k() -> None:
    base = FakeRetriever([chunk("z", "d1", 1.0), chunk("a", "d2", 1.0)])
    retriever = RerankedRetriever(base, FakeReranker([0.5, 0.5]), candidate_k=2)

    results = await retriever.retrieve("query", top_k=1)

    assert [result.chunk_id for result in results] == ["a"]


async def test_empty_query_and_no_candidates_skip_model() -> None:
    base = FakeRetriever([chunk("a", "d1", 1.0)])
    reranker = FakeReranker([1.0])
    retriever = RerankedRetriever(base, reranker)

    assert await retriever.retrieve("   ") == []
    assert reranker.calls == []

    empty = RerankedRetriever(FakeRetriever([]), reranker)
    assert await empty.retrieve("query") == []
    assert reranker.calls == []


async def test_rejects_invalid_scores_and_settings() -> None:
    with pytest.raises(ValueError):
        RerankedRetriever(FakeRetriever([]), FakeReranker([]), candidate_k=0)

    retriever = RerankedRetriever(
        FakeRetriever([chunk("a", "d1", 1.0)]),
        FakeReranker([0.1, 0.2]),
        candidate_k=1,
    )
    with pytest.raises(RuntimeError):
        await retriever.retrieve("query")
