#!/usr/bin/env python3
"""Evaluate dense, lexical, and hybrid retrieval on the development split."""

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
from app.evaluation.dev import evaluate_retriever, read_dev_questions  # noqa: E402
from app.retrieval.dense import DenseRetriever  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.retrieval.lexical import LexicalRetriever  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402

MODES = ("dense", "lexical", "hybrid")


def build_parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--questions", type=Path, default=Path("data/sample/dev_questions.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/retrieval_dev_metrics.json"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
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


async def run(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.candidate_k <= 0:
        raise ValueError("candidate-k must be positive")
    if args.rrf_k <= 0:
        raise ValueError("rrf-k must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not args.questions.is_file():
        raise FileNotFoundError(f"questions file not found: {args.questions}")

    questions = read_dev_questions(args.questions, args.limit)
    if not questions:
        raise ValueError("no development questions found")

    provider = None
    if args.mode in {"dense", "hybrid"}:
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
        lexical = LexicalRetriever(
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        dense = None
        if provider is not None:
            dense = DenseRetriever(
                provider,
                session_factory,
                tenant_id=args.tenant_id,
                knowledge_base_id=args.knowledge_base_id,
            )

        retriever_by_mode = {"lexical": lexical}
        if dense is not None:
            retriever_by_mode["dense"] = dense
            retriever_by_mode["hybrid"] = HybridRetriever(
                dense,
                lexical,
                rrf_k=args.rrf_k,
                dense_weight=args.dense_weight,
                lexical_weight=args.lexical_weight,
                candidate_k=args.candidate_k,
            )

        modes = MODES if args.mode == "all" else (args.mode,)
        reports: dict[str, object] = {}
        for mode in modes:
            reports[mode] = await evaluate_retriever(
                retriever_by_mode[mode],
                questions,
                top_k=args.top_k,
                progress_every=args.progress_every,
                label=mode,
            )
    finally:
        await engine.dispose()

    output: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "device": provider.device if provider is not None else None,
        "dtype": provider.dtype if provider is not None else None,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "questions_file": str(args.questions),
        "modes": reports,
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

    for mode, report in output["modes"].items():
        metrics = report["metrics"]
        print(f"{mode}:")
        for key, value in metrics.items():
            if key == "evaluated_questions":
                print(f"  questions: {value}")
            else:
                print(f"  {key}: {value:.4f}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
