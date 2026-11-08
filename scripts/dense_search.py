#!/usr/bin/env python3
"""Run a dense retrieval query from the command line."""

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
from app.embeddings.bge import BGEEmbeddingProvider  # noqa: E402
from app.ingestion.loaders.base import SOURCE_TYPES  # noqa: E402
from app.retrieval.dense import DenseRetriever  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--source-types", nargs="+", choices=sorted(SOURCE_TYPES))
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--dtype", default=settings.embedding_dtype, choices=["float32", "float16"])
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> int:
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
    try:
        retriever = DenseRetriever(
            provider,
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        results = await retriever.retrieve(
            args.query,
            top_k=args.top_k,
            source_types=args.source_types,
        )
    finally:
        await engine.dispose()

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
        return 0

    if not results:
        print("No results.")
        return 0

    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.content.split())[:220]
        print(
            f"{rank}. score={result.score:.4f} doc={result.doc_id} "
            f"chunk={result.chunk_id} source={result.source_type}"
        )
        print(f"   {result.title}")
        print(f"   {preview}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    try:
        return asyncio.run(run(args, settings))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
