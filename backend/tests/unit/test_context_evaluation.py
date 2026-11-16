from __future__ import annotations

from app.evaluation.context import evaluate_context
from app.evaluation.dev import DevQuestion
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    async def retrieve(self, query: str, **kwargs: object) -> list[RetrievedChunk]:
        del query, kwargs
        return self.results


def make_chunk(chunk_id: str, doc_id: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_type="pdf",
        title=chunk_id,
        chunk_index=0,
        content="alpha beta",
        score=score,
        metadata={},
        token_count=2,
    )


async def test_evaluate_context_reports_recall_mrr_and_budget() -> None:
    questions = [
        DevQuestion(
            question_id="q1",
            question="what is alpha?",
            expected_doc_ids=["d1"],
        )
    ]
    retriever = FakeRetriever([make_chunk("c1", "d1"), make_chunk("c2", "d2")])

    report = await evaluate_context(retriever, questions, top_k=2, token_budget=2)
    metrics = report["metrics"]

    assert metrics["candidate_doc_recall"] == 1.0
    assert metrics["candidate_doc_recall_at_context_size"] == 1.0
    assert metrics["context_doc_recall"] == 1.0
    assert metrics["recall_gap_at_context_size"] == 0.0
    assert metrics["context_mrr"] == 1.0
    assert metrics["budget_adherence"] == 1.0
    assert metrics["duplicate_chunk_count"] == 0
    assert report["per_question"][0]["selected_chunk_ids"] == ["c1"]
