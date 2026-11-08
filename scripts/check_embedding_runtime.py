#!/usr/bin/env python3
"""Validate the local embedding runtime before a full corpus run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import torch  # noqa: E402

from app.embeddings.bge import BGEEmbeddingProvider  # noqa: E402
from app.settings import get_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--dtype", default=settings.embedding_dtype, choices=["float32", "float16"])
    parser.add_argument("--model", default=settings.embedding_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'none'}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    try:
        provider = BGEEmbeddingProvider(
            model_name=args.model,
            dimension=settings.embedding_dimension,
            device=args.device,
            batch_size=settings.embedding_batch_size,
            dtype=args.dtype,
            query_instruction=settings.embedding_query_instruction,
        )
        documents = provider.embed_documents(["runtime check document"])
        query = provider.embed_query("runtime check query")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Embedding device: {provider.device}")
    print(f"Embedding dtype: {provider.dtype}")
    print(f"Model: {provider.model_name}")
    print(f"Dimension: {provider.dimension}")
    print(f"Document embedding shape: ({len(documents)}, {len(documents[0])})")
    print(f"Query embedding shape: ({len(query)},)")
    print(f"Query L2 norm: {sum(value * value for value in query) ** 0.5:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
