#!/usr/bin/env python3
"""Rerank BM25 candidates with a local cross-encoder."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.reranking.cross_encoder import CrossEncoderReranker  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.bm25_index import BM25IndexError, BM25IndexManager  # noqa: E402
from app.retrieval.reranked import RerankedRetriever  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--model", default=settings.reranker_model)
    parser.add_argument("--device", default=settings.reranker_device)
    parser.add_argument("--batch-size", type=int, default=settings.reranker_batch_size)
    parser.add_argument("--max-length", type=int, default=settings.reranker_max_length)
    parser.add_argument("--source-type", action="append", dest="source_types")
    parser.add_argument("--index-root", type=Path, default=settings.bm25_index_root)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> list[dict[str, object]]:
    if args.candidate_k <= 0:
        raise ValueError("candidate-k must be positive")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("max-length must be positive")

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
        retriever = RerankedRetriever(bm25, reranker, candidate_k=args.candidate_k)
        results = await retriever.retrieve(
            args.query,
            top_k=args.top_k,
            source_types=args.source_types,
        )
    finally:
        await engine.dispose()

    rows = [asdict(result) for result in results]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        for rank, row in enumerate(rows, start=1):
            print(
                f"{rank}. rerank={row['score']:.4f} "
                f"bm25={row['retrieval_score']:.4f} {row['chunk_id']} {row['title']}"
            )
    return rows


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
