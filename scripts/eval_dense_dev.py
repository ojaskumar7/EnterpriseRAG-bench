#!/usr/bin/env python3
"""Evaluate dense retrieval on the development question split."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.embeddings.bge import BGEEmbeddingProvider  # noqa: E402
from app.retrieval.dense import DenseRetriever  # noqa: E402
from app.retrieval.metrics import DEFAULT_K_VALUES, evaluate_retrieval  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/sample/dev_questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dense_dev_metrics.json"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--dtype", default=settings.embedding_dtype, choices=["float32", "float16"])
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def read_questions(path: Path, limit: int | None) -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                question = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            if not question.get("question_id") or not question.get("question"):
                raise ValueError(
                    f"question is missing question_id or question on line {line_number}"
                )
            questions.append(question)
            if limit is not None and len(questions) >= limit:
                break
    return questions


async def run(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not args.questions.is_file():
        raise FileNotFoundError(f"questions file not found: {args.questions}")

    k_values = tuple(k for k in DEFAULT_K_VALUES if k <= args.top_k)
    questions = read_questions(args.questions, args.limit)
    provider = BGEEmbeddingProvider(
        model_name=args.model,
        dimension=settings.embedding_dimension,
        device=args.device,
        batch_size=settings.embedding_batch_size,
        dtype=args.dtype,
        query_instruction=settings.embedding_query_instruction,
    )
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    expected_by_question: dict[str, list[str]] = {}
    retrieved_by_question: dict[str, list[str]] = {}
    per_question: list[dict[str, object]] = []
    try:
        retriever = DenseRetriever(
            provider,
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        for index, question in enumerate(questions, start=1):
            question_id = str(question["question_id"])
            expected = [str(doc_id) for doc_id in question.get("expected_doc_ids", [])]
            results = await retriever.retrieve(str(question["question"]), top_k=args.top_k)
            retrieved = [result.doc_id for result in results]
            expected_by_question[question_id] = expected
            retrieved_by_question[question_id] = retrieved
            per_question.append(
                {
                    "question_id": question_id,
                    "expected_doc_ids": expected,
                    "retrieved_doc_ids": retrieved,
                }
            )
            if index % args.progress_every == 0 or index == len(questions):
                print(f"evaluated {index}/{len(questions)}", flush=True)
    finally:
        await engine.dispose()

    metrics = evaluate_retrieval(
        expected_by_question,
        retrieved_by_question,
        k_values=k_values,
    )
    output: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": provider.device,
        "dtype": provider.dtype,
        "top_k": args.top_k,
        "questions_file": str(args.questions),
        "metrics": metrics,
        "per_question": per_question,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    try:
        output = asyncio.run(run(args, settings))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    metrics = output["metrics"]
    print(f"Questions evaluated: {metrics['evaluated_questions']}")
    for key, value in metrics.items():
        if key == "evaluated_questions":
            continue
        print(f"{key}: {value:.4f}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
