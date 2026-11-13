#!/usr/bin/env python3
"""Build or check a production BM25 index."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.retrieval.bm25_builder import build_bm25_index, load_bm25_corpus  # noqa: E402
from app.retrieval.bm25_index import (  # noqa: E402
    BM25IndexError,
    BM25IndexManager,
    corpus_fingerprint,
)
from app.settings import Settings, get_settings  # noqa: E402


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    parser.add_argument("--index-root", type=Path, default=settings.bm25_index_root)
    parser.add_argument("--retention", type=int, default=settings.bm25_retention)
    parser.add_argument("--check", action="store_true")
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> None:
    if args.retention < 1:
        raise ValueError("retention must be at least 1")

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    try:
        async with session_factory() as session:
            corpus = await load_bm25_corpus(
                session,
                tenant_id=args.tenant_id,
                knowledge_base_id=args.knowledge_base_id,
            )
        expected = corpus_fingerprint(
            (chunk.chunk_id, chunk.content_hash, chunk.title) for chunk in corpus.chunks
        )
        manager = BM25IndexManager(args.index_root, retention=args.retention)
        if args.check:
            manifest = manager.check(
                tenant_id=str(args.tenant_id),
                knowledge_base_id=str(args.knowledge_base_id),
                expected_fingerprint=expected,
            )
            print(f"BM25 index is current: version={manifest.version} rows={manifest.row_count}")
            return

        manifest = await asyncio.to_thread(
            build_bm25_index,
            corpus=corpus,
            index_root=args.index_root,
            tenant_id=str(args.tenant_id),
            knowledge_base_id=str(args.knowledge_base_id),
            retention=args.retention,
        )
        print(
            f"Published BM25 index: version={manifest.version} "
            f"rows={manifest.row_count} documents={manifest.document_count}"
        )
    finally:
        await engine.dispose()


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
