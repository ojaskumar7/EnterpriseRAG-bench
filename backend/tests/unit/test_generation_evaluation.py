from __future__ import annotations

import pytest

from app.evaluation.generation import render_generation_markdown, summarize_generation


def record(
    question_id: str,
    *,
    citations: list[int],
    citation_count: int = 2,
    expected: list[str] | None = None,
    selected: list[str] | None = None,
    latency: float = 100.0,
    abstained: bool = False,
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "expected_doc_ids": expected or ["doc-1"],
        "answer": {
            "answer": "answer",
            "citation_ids": citations,
            "model": "fake-model",
            "insufficient_evidence": abstained,
        },
        "context": {
            "citation_count": citation_count,
            "selected_doc_ids": selected or ["doc-1"],
            "token_count": 10,
        },
        "latency_ms": latency,
    }


def test_summary_reports_citations_abstention_recall_and_latency() -> None:
    summary = summarize_generation(
        [
            record("q1", citations=[1], latency=100.0),
            record(
                "q2",
                citations=[],
                expected=["doc-2"],
                selected=["doc-1"],
                latency=200.0,
                abstained=True,
            ),
        ]
    )

    assert summary["question_count"] == 2
    assert summary["citation_presence_rate"] == 0.5
    assert summary["citation_validity_rate"] == 1.0
    assert summary["abstention_count"] == 1
    assert summary["selected_context_doc_recall"] == 0.5
    assert summary["latency_ms"]["p50_ms"] == 150.0


def test_invalid_citation_is_reported_as_invalid() -> None:
    summary = summarize_generation([record("q1", citations=[3], citation_count=2)])

    assert summary["citation_validity_rate"] == 0.0


def test_duplicate_citations_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        summarize_generation([record("q1", citations=[1, 1])])


def test_empty_records_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_generation([])


def test_markdown_rendering_includes_diagnostics() -> None:
    summary = summarize_generation([record("q1", citations=[1])])

    markdown = render_generation_markdown(summary)

    assert "# Generation Evaluation Report" in markdown
    assert "Citation presence rate" in markdown
    assert "not answer-correctness" in markdown
