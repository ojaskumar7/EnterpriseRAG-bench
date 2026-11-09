"""Development-split retrieval evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.retrieval.metrics import DEFAULT_K_VALUES, evaluate_retrieval
from app.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class DevQuestion:
    question_id: str
    question: str
    expected_doc_ids: list[str]


class RetrieverProtocol(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...


def read_dev_questions(path: Path, limit: int | None = None) -> list[DevQuestion]:
    """Read and validate development questions."""

    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    questions: list[DevQuestion] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not raw.get("question_id") or not raw.get("question"):
                raise ValueError(
                    f"question is missing question_id or question on line {line_number}"
                )
            questions.append(
                DevQuestion(
                    question_id=str(raw["question_id"]),
                    question=str(raw["question"]),
                    expected_doc_ids=[str(doc_id) for doc_id in raw.get("expected_doc_ids", [])],
                )
            )
            if limit is not None and len(questions) >= limit:
                break
    return questions


async def evaluate_retriever(
    retriever: RetrieverProtocol,
    questions: Sequence[DevQuestion],
    *,
    top_k: int,
    progress_every: int = 25,
    label: str = "",
) -> dict[str, object]:
    """Evaluate one retriever over development questions."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    k_values = tuple(k for k in DEFAULT_K_VALUES if k <= top_k)
    expected_by_question: dict[str, list[str]] = {}
    retrieved_by_question: dict[str, list[str]] = {}
    per_question: list[dict[str, object]] = []

    for index, question in enumerate(questions, start=1):
        results = await retriever.retrieve(question.question, top_k=top_k)
        retrieved = [result.doc_id for result in results]
        expected_by_question[question.question_id] = question.expected_doc_ids
        retrieved_by_question[question.question_id] = retrieved
        per_question.append(
            {
                "question_id": question.question_id,
                "expected_doc_ids": question.expected_doc_ids,
                "retrieved_doc_ids": retrieved,
            }
        )
        if index % progress_every == 0 or index == len(questions):
            prefix = f"{label}: " if label else ""
            print(f"{prefix}evaluated {index}/{len(questions)}", flush=True)

    metrics = evaluate_retrieval(
        expected_by_question,
        retrieved_by_question,
        k_values=k_values,
    )
    return {"metrics": metrics, "per_question": per_question}
