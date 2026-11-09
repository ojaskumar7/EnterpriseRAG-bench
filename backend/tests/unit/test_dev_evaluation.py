"""Tests for development-split evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

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


def _load_eval_cli():
    path = Path(__file__).resolve().parents[3] / "scripts" / "eval_retrieval_dev.py"
    spec = importlib.util.spec_from_file_location("eval_retrieval_dev", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_run_all_mode_wires_all_retrievers(tmp_path: Path, monkeypatch) -> None:
    cli = _load_eval_cli()

    questions_path = tmp_path / "dev.jsonl"
    questions_path.write_text("{}\n", encoding="utf-8")
    output_path = tmp_path / "metrics.json"
    questions = [DevQuestion(question_id="q1", question="gold", expected_doc_ids=["gold"])]

    class FakeEngine:
        async def dispose(self) -> None:
            pass

    class FakeProvider:
        def __init__(self, **kwargs) -> None:
            self.device = "cpu"
            self.dtype = "float32"

    class FakeRetriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

    evaluated_labels = []

    async def fake_evaluate(retriever, questions_arg, *, top_k, progress_every=25, label=""):
        evaluated_labels.append(label)
        return {
            "metrics": {"evaluated_questions": len(questions_arg)},
            "per_question": [],
        }

    monkeypatch.setattr(cli, "read_dev_questions", lambda path, limit=None: questions)
    monkeypatch.setattr(cli, "evaluate_retriever", fake_evaluate)
    monkeypatch.setattr(cli, "BGEEmbeddingProvider", FakeProvider)
    monkeypatch.setattr(cli, "build_engine", lambda settings: FakeEngine())
    monkeypatch.setattr(cli, "build_session_factory", lambda engine: object())
    monkeypatch.setattr(cli, "DenseRetriever", FakeRetriever)
    monkeypatch.setattr(cli, "LexicalRetriever", FakeRetriever)
    monkeypatch.setattr(cli, "HybridRetriever", FakeRetriever)

    settings = SimpleNamespace(
        embedding_dimension=1024,
        embedding_batch_size=32,
        embedding_query_instruction="",
    )
    args = SimpleNamespace(
        mode="all",
        questions=questions_path,
        output=output_path,
        top_k=20,
        candidate_k=50,
        rrf_k=60,
        dense_weight=1.0,
        lexical_weight=1.0,
        limit=None,
        device="cpu",
        dtype="float32",
        model="test-model",
        tenant_id=uuid.UUID(int=1),
        knowledge_base_id=uuid.UUID(int=2),
        progress_every=25,
    )

    output = await cli.run(args, settings)

    assert set(output["modes"]) == {"dense", "lexical", "hybrid"}
    assert evaluated_labels == ["dense", "lexical", "hybrid"]
    assert output_path.is_file()
