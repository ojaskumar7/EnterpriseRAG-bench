#!/usr/bin/env python3
"""Retrieve evidence and generate a grounded Gemini answer."""

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

from app.context import ContextBuilder  # noqa: E402
from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.generation.gemini import GeminiGenerator  # noqa: E402
from app.generation.safety import GroundingPolicy, apply_grounding_policy  # noqa: E402
from app.generation.types import GenerationError, GenerationRequest  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.bm25_index import BM25IndexError, BM25IndexManager  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--max-chunks-per-doc", type=int, default=3)
    parser.add_argument("--source-type", action="append", dest="source_types")
    parser.add_argument("--model", default=settings.gemini_model)
    parser.add_argument("--index-root", type=Path, default=settings.bm25_index_root)
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    parser.add_argument("--json", action="store_true")
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.token_budget <= 0:
        raise ValueError("token-budget must be positive")
    if args.max_chunks_per_doc <= 0:
        raise ValueError("max-chunks-per-doc must be positive")

    engine = build_engine(settings)
    try:
        retriever = BM25Retriever(
            BM25IndexManager(args.index_root),
            build_session_factory(engine),
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        candidates = await retriever.retrieve(
            args.query,
            top_k=args.top_k,
            source_types=args.source_types,
        )
    finally:
        await engine.dispose()

    context = ContextBuilder(
        token_budget=args.token_budget,
        max_chunks_per_doc=args.max_chunks_per_doc,
    ).build(args.query, candidates)
    generated = GeminiGenerator(settings, model=args.model).generate(
        GenerationRequest(query=args.query, context=context)
    )
    grounding = apply_grounding_policy(
        generated,
        context,
        policy=GroundingPolicy(
            min_citations=settings.grounding_min_citations,
            min_documents=settings.grounding_min_documents,
            min_claim_overlap=settings.grounding_min_claim_overlap,
        ),
    )
    payload = {
        "answer": asdict(grounding.answer),
        "grounding": {
            "decision": grounding.decision,
            "reason": grounding.reason,
            "claim_checks": [asdict(check) for check in grounding.claim_checks],
        },
        "context": {"citation_count": len(context.blocks)},
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(grounding.answer.answer)
        if grounding.answer.citation_ids:
            citations = ", ".join(
                f"[{citation_id}]" for citation_id in grounding.answer.citation_ids
            )
            print(f"Citations: {citations}")
    return payload


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    args = build_parser(settings).parse_args(argv)
    try:
        asyncio.run(run(args, settings))
    except (BM25IndexError, GenerationError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
