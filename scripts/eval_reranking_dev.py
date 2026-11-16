#!/usr/bin/env python3
"""Compare BM25 and cross-encoder reranked retrieval on the development split."""

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
from app.evaluation.dev import read_dev_questions  # noqa: E402
from app.evaluation.reranking import evaluate_reranking  # noqa: E402
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
        default=Path("data/processed/reranking_dev_metrics_block6.json"),
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--model", default=settings.reranker_model)
    parser.add_argument("--device", default=settings.reranker_device)
    parser.add_argument("--batch-size", type=int, default=settings.reranker_batch_size)
    parser.add_argument("--max-length", type=int, default=settings.reranker_max_length)
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
    if args.candidate_k <= 0:
        raise ValueError("candidate-k must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")
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
    try:
        bm25 = BM25Retriever(
            BM25IndexManager(args.index_root),
            build_session_factory(engine),
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        reranker = CrossEncoderReranker(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        reranked = RerankedRetriever(bm25, reranker, candidate_k=args.candidate_k)
        report = await evaluate_reranking(
            bm25,
            reranked,
            questions,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            progress_every=args.progress_every,
        )
    finally:
        await engine.dispose()

    report["generated_at"] = datetime.now(UTC).isoformat()
    report["retriever"] = "reranked_bm25"
    report["model"] = args.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"delta": report["delta"], "latency_ms": report["latency_ms"]}, indent=2))
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
