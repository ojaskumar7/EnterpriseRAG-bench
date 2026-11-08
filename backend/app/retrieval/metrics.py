"""Document-level retrieval metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

DEFAULT_K_VALUES = (1, 5, 10, 20)


def dedupe_doc_ids(doc_ids: Iterable[str]) -> list[str]:
    """Deduplicate document IDs while preserving retrieval order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def recall_at_k(retrieved_doc_ids: Sequence[str], expected_doc_ids: Iterable[str], k: int) -> float:
    """Return document-level recall in the first k unique documents."""

    if k <= 0:
        raise ValueError("k must be positive")
    expected = set(expected_doc_ids)
    if not expected:
        return 0.0
    top = dedupe_doc_ids(retrieved_doc_ids)[:k]
    return len(expected.intersection(top)) / len(expected)


def reciprocal_rank(retrieved_doc_ids: Sequence[str], expected_doc_ids: Iterable[str]) -> float:
    """Return the reciprocal rank of the first relevant document."""

    expected = set(expected_doc_ids)
    if not expected:
        return 0.0
    for rank, doc_id in enumerate(dedupe_doc_ids(retrieved_doc_ids), start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(
    expected_by_question: Mapping[str, Sequence[str]],
    retrieved_by_question: Mapping[str, Sequence[str]],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float | int]:
    """Aggregate recall@k and MRR over questions that have expected documents."""

    question_ids = [
        question_id
        for question_id, expected in expected_by_question.items()
        if expected and question_id in retrieved_by_question
    ]
    if not question_ids:
        metrics: dict[str, float | int] = {f"recall_at_{k}": 0.0 for k in k_values}
        metrics["mrr"] = 0.0
        metrics["evaluated_questions"] = 0
        return metrics

    metrics = {}
    for k in k_values:
        total = sum(
            recall_at_k(
                retrieved_by_question[question_id],
                expected_by_question[question_id],
                k,
            )
            for question_id in question_ids
        )
        metrics[f"recall_at_{k}"] = total / len(question_ids)

    mrr_total = sum(
        reciprocal_rank(
            retrieved_by_question[question_id],
            expected_by_question[question_id],
        )
        for question_id in question_ids
    )
    metrics["mrr"] = mrr_total / len(question_ids)
    metrics["evaluated_questions"] = len(question_ids)
    return metrics
