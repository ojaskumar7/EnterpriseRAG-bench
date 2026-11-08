"""Tests for document-level retrieval metrics."""

from __future__ import annotations

import pytest

from app.retrieval.metrics import (
    dedupe_doc_ids,
    evaluate_retrieval,
    recall_at_k,
    reciprocal_rank,
)


def test_dedupe_preserves_first_occurrence_order() -> None:
    assert dedupe_doc_ids(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_recall_at_k_uses_unique_documents() -> None:
    retrieved = ["a", "a", "b", "c"]
    expected = ["b", "c", "d"]

    assert recall_at_k(retrieved, expected, 1) == 0.0
    assert recall_at_k(retrieved, expected, 2) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, expected, 3) == pytest.approx(2 / 3)


def test_recall_with_no_expected_documents_is_zero() -> None:
    assert recall_at_k(["a"], [], 5) == 0.0


def test_reciprocal_rank_uses_deduplicated_rank() -> None:
    assert reciprocal_rank(["a", "a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a"], ["b"]) == 0.0


def test_evaluate_retrieval_aggregates_questions() -> None:
    expected = {"q1": ["a"], "q2": ["b", "c"], "q3": []}
    retrieved = {"q1": ["a"], "q2": ["x", "b"], "q3": ["z"]}

    metrics = evaluate_retrieval(expected, retrieved, k_values=(1, 2))

    assert metrics["evaluated_questions"] == 2
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_2"] == 0.75
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5) / 2)


def test_evaluate_retrieval_with_no_questions() -> None:
    metrics = evaluate_retrieval({}, {}, k_values=(1, 5))

    assert metrics == {
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "mrr": 0.0,
        "evaluated_questions": 0,
    }
