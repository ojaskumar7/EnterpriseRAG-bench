#!/usr/bin/env python3
"""Embed the processed chunk corpus and load it into PostgreSQL/pgvector."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.embeddings.bge import BGEEmbeddingProvider  # noqa: E402
from app.embeddings.loader import LoadStats, load_chunks  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    parser.add_argument("--load-batch-size", type=int, default=256)
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
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser


def print_progress(stats: LoadStats) -> None:
    print(
        f"seen={stats.chunks_seen} written={stats.chunks_written} "
        f"embedded={stats.chunks_embedded} skipped={stats.chunks_skipped}",
        flush=True,
    )


async def run(args: argparse.Namespace, settings: Settings) -> None:
    if not args.chunks.is_file():
        raise FileNotFoundError(f"chunks file not found: {args.chunks}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.load_batch_size <= 0:
        raise ValueError("load-batch-size must be positive")

    provider = BGEEmbeddingProvider(
        model_name=args.model,
        dimension=settings.embedding_dimension,
        device=args.device,
        batch_size=args.batch_size,
        dtype=args.dtype,
        query_instruction=settings.embedding_query_instruction,
    )
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    try:
        stats = await load_chunks(
            session_factory,
            args.chunks,
            provider,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
            batch_size=args.load_batch_size,
            limit=args.limit,
            progress=print_progress,
            progress_every=args.progress_every,
        )
    finally:
        await engine.dispose()

    print(f"Documents:       {stats.documents_seen}")
    print(f"Chunks seen:     {stats.chunks_seen}")
    print(f"Chunks written:  {stats.chunks_written}")
    print(f"Chunks embedded: {stats.chunks_embedded}")
    print(f"Chunks skipped:  {stats.chunks_skipped}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    try:
        asyncio.run(run(args, settings))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
