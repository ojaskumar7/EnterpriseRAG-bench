#!/usr/bin/env python3
"""Evaluate token-budgeted context construction on the development split."""

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
from app.evaluation.context import evaluate_context  # noqa: E402
from app.evaluation.dev import read_dev_questions  # noqa: E402
from app.reranking.cross_encoder import CrossEncoderReranker  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.bm25_index import BM25IndexError, BM25IndexManager  # noqa: E402
from app.retrieval.reranked import RerankedRetriever  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/sample/dev_questions.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/context_dev_metrics.json"),
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--max-chunks-per-doc", type=int, default=3)
    parser.add_argument("--reranked", action="store_true")
    parser.add_argument("--reranker-model", default=settings.reranker_model)
    parser.add_argument("--reranker-device", default=settings.reranker_device)
    parser.add_argument("--reranker-batch-size", type=int, default=settings.reranker_batch_size)
    parser.add_argument("--reranker-max-length", type=int, default=settings.reranker_max_length)
    parser.add_argument("--index-root", type=Path, default=settings.bm25_index_root)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.token_budget <= 0:
        raise ValueError("token-budget must be positive")
    if args.max_chunks_per_doc <= 0:
        raise ValueError("max-chunks-per-doc must be positive")
    if args.reranked:
        if args.reranker_batch_size <= 0:
            raise ValueError("reranker-batch-size must be positive")
        if args.reranker_max_length <= 0:
            raise ValueError("reranker-max-length must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    if not args.questions.is_file():
        raise FileNotFoundError(f"questions file not found: {args.questions}")

    questions = read_dev_questions(args.questions, args.limit)
    if not questions:
        raise ValueError("no development questions found")

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    try:
        bm25 = BM25Retriever(
            BM25IndexManager(args.index_root),
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        retriever: object = bm25
        if args.reranked:
            retriever = RerankedRetriever(
                bm25,
                CrossEncoderReranker(
                    model_name=args.reranker_model,
                    device=args.reranker_device,
                    batch_size=args.reranker_batch_size,
                    max_length=args.reranker_max_length,
                ),
                candidate_k=args.top_k,
            )
        report = await evaluate_context(
            retriever,
            questions,
            top_k=args.top_k,
            token_budget=args.token_budget,
            max_chunks_per_doc=args.max_chunks_per_doc,
            progress_every=args.progress_every,
        )
    finally:
        await engine.dispose()

    report["generated_at"] = datetime.now(UTC).isoformat()
    report["retriever"] = "reranked_bm25" if args.reranked else "bm25"
    report["model"] = args.reranker_model if args.reranked else "bm25"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    args = build_parser(settings).parse_args(argv)
    try:
        asyncio.run(run(args, settings))
    except (BM25IndexError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
