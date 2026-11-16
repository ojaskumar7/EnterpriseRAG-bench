from __future__ import annotations

from app.evaluation.dev import DevQuestion
from app.evaluation.reranking import evaluate_reranking
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    async def retrieve(self, query: str, **kwargs: object) -> list[RetrievedChunk]:
        del query, kwargs
        return self.results


async def test_evaluate_reranking_reports_both_modes() -> None:
    question = DevQuestion("q1", "query", ["d1"])
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        source_type="pdf",
        title="t",
        chunk_index=0,
        content="content",
        score=1.0,
        metadata={},
    )
    retriever = FakeRetriever([chunk])

    report = await evaluate_reranking(
        retriever,
        retriever,
        [question],
        top_k=1,
        candidate_k=1,
    )

    assert report["bm25"]["metrics"]["recall_at_1"] == 1.0
    assert report["reranked"]["metrics"]["recall_at_1"] == 1.0
    assert report["delta"]["mrr"] == 0.0
    assert report["latency_ms"]["bm25"]["p50_ms"] >= 0.0
