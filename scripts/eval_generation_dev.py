#!/usr/bin/env python3
"""Run grounded generation over the development question split."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.context import ContextBuilder  # noqa: E402
from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.generation.gemini import GeminiGenerator  # noqa: E402
from app.generation.safety import apply_grounding_policy  # noqa: E402
from app.generation.types import GenerationError, GenerationRequest  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.bm25_index import BM25IndexError, BM25IndexManager  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data/sample/dev_questions.jsonl",
    )
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--max-chunks-per-doc", type=int, default=3)
    parser.add_argument("--source-type", action="append", dest="source_types")
    parser.add_argument("--model", default=settings.gemini_model)
    parser.add_argument("--index-root", type=Path, default=settings.bm25_index_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/generation_dev.jsonl",
    )
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    return parser


def load_questions(path: Path, question_ids: set[str] | None) -> list[dict[str, object]]:
    questions = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if question_ids and record["question_id"] not in question_ids:
                continue
            questions.append(record)
    if not questions:
        requested = ", ".join(sorted(question_ids or ()))
        raise ValueError(f"no questions selected from {path}; requested IDs: {requested}")
    return questions


async def run(args: argparse.Namespace, settings: Settings) -> list[dict[str, object]]:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.token_budget <= 0:
        raise ValueError("token-budget must be positive")
    if args.max_chunks_per_doc <= 0:
        raise ValueError("max-chunks-per-doc must be positive")

    questions = load_questions(args.questions, set(args.question_id) if args.question_id else None)
    if args.limit is not None:
        questions = questions[: args.limit]

    engine = build_engine(settings)
    results: list[dict[str, object]] = []
    try:
        retriever = BM25Retriever(
            BM25IndexManager(args.index_root),
            build_session_factory(engine),
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        generator = GeminiGenerator(settings, model=args.model)
        for record in questions:
            question = str(record["question"])
            started = time.perf_counter()
            candidates = await retriever.retrieve(
                question,
                top_k=args.top_k,
                source_types=args.source_types,
            )
            context = ContextBuilder(
                token_budget=args.token_budget,
                max_chunks_per_doc=args.max_chunks_per_doc,
            ).build(question, candidates)
            generated = generator.generate(GenerationRequest(query=question, context=context))
            grounding = apply_grounding_policy(generated, context)
            elapsed_ms = (time.perf_counter() - started) * 1000
            results.append(
                {
                    "question_id": record["question_id"],
                    "question_type": record["question_type"],
                    "question": question,
                    "expected_doc_ids": record["expected_doc_ids"],
                    "answer": asdict(grounding.answer),
                    "grounding": {
                        "decision": grounding.decision,
                        "reason": grounding.reason,
                        "claim_checks": [asdict(check) for check in grounding.claim_checks],
                    },
                    "context": {
                        "citation_count": len(context.blocks),
                        "selected_doc_ids": sorted({block.doc_id for block in context.blocks}),
                        "token_count": context.token_count,
                    },
                    "latency_ms": round(elapsed_ms, 2),
                }
            )
    finally:
        await engine.dispose()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return results


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    args = build_parser(settings).parse_args(argv)
    try:
        results = asyncio.run(run(args, settings))
    except (BM25IndexError, GenerationError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(results)} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
