#!/usr/bin/env python3
"""Run the Block 1 subset ingestion pipeline from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import IngestionConfig  # noqa: E402
from app.ingestion.loaders.base import SOURCE_TYPES  # noqa: E402
from app.ingestion.pipeline import run_ingestion  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--min-chunk-tokens", type=int, default=50)
    parser.add_argument("--tokenizer", dest="tokenizer_name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCE_TYPES),
        default=["confluence", "jira", "github", "hubspot"],
    )
    parser.add_argument("--max-docs", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = IngestionConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_tokens=args.min_chunk_tokens,
        tokenizer_name=args.tokenizer_name,
        source_types=tuple(args.sources),
        max_docs=args.max_docs,
    )

    try:
        result = run_ingestion(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded:   {result.documents_loaded}")
    print(f"Chunked:  {result.documents_chunked}")
    print(f"Skipped:  {result.documents_skipped}")
    print(f"Duplicate:{result.documents_duplicate}")
    print(f"Failed:   {result.documents_failed}")
    print(f"Chunks:   {result.chunks_written}")
    print(f"Output:   {config.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
