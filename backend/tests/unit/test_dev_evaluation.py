"""Tests for development-split evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.dev import DevQuestion, evaluate_retriever, read_dev_questions
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    async def retrieve(self, query: str, *, top_k: int = 10, source_types=None):
        return [
            RetrievedChunk(
                chunk_id=f"{query}-0",
                doc_id="gold",
                source_type="github",
                title="Gold",
                chunk_index=0,
                content="content",
                score=1.0,
                metadata={},
            )
        ]


def test_read_dev_questions_validates_and_limits(tmp_path: Path) -> None:
    path = tmp_path / "dev.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "first",
                "expected_doc_ids": ["doc-1"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "question_id": "q2",
                "question": "second",
                "expected_doc_ids": ["doc-2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    questions = read_dev_questions(path, limit=1)

    assert questions == [
        DevQuestion(question_id="q1", question="first", expected_doc_ids=["doc-1"])
    ]


def test_read_dev_questions_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question_id": "q1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="question"):
        read_dev_questions(path)


@pytest.mark.asyncio
async def test_evaluate_retriever_reports_metrics() -> None:
    questions = [
        DevQuestion(question_id="q1", question="gold", expected_doc_ids=["gold"]),
    ]

    report = await evaluate_retriever(FakeRetriever(), questions, top_k=5)

    assert report["metrics"]["evaluated_questions"] == 1
    assert report["metrics"]["recall_at_1"] == 1.0
    assert report["metrics"]["mrr"] == 1.0
