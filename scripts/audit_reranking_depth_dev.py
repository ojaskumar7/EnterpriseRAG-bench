#!/usr/bin/env python3
"""Run the v2-m3 reranking quality, context, and depth-performance audit."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.context import ContextBuilder  # noqa: E402
from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.evaluation.dev import read_dev_questions, latency_summary  # noqa: E402
from app.retrieval.metrics import recall_at_k, reciprocal_rank  # noqa: E402
from app.reranking.cross_encoder import CrossEncoderReranker  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.bm25_index import BM25IndexError, BM25IndexManager  # noqa: E402
from app.retrieval.reranked import RerankedRetriever  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


DEPTHS = (10, 20, 30, 50)
MODEL = "BAAI/bge-reranker-v2-m3"


def parser(settings: Settings) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data/sample/dev_questions.jsonl")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/reranking_depth_audit_dev.json")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--device", default=settings.reranker_device)
    p.add_argument("--batch-size", type=int, default=settings.reranker_batch_size)
    p.add_argument("--max-length", type=int, default=settings.reranker_max_length)
    p.add_argument("--index-root", type=Path, default=settings.bm25_index_root)
    p.add_argument("--token-budget", type=int, default=4000)
    p.add_argument("--max-chunks-per-doc", type=int, default=3)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    p.add_argument("--knowledge-base-id", type=uuid.UUID, default=settings.default_knowledge_base_id)
    p.add_argument("--skip-performance", action="store_true")
    p.add_argument("--performance-limit", type=int, default=None)
    return p


def gpu_memory() -> dict[str, float | None]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"peak_allocated_mib": None, "peak_reserved_mib": None}
        return {
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2),
        }
    except ImportError:
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}


def reset_gpu_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def metrics(rows: list[dict[str, object]], depth: int) -> dict[str, float | int]:
    valid = [row for row in rows if row["expected_doc_ids"]]
    return {
        "mrr": sum(reciprocal_rank(row["final_doc_ids"][str(depth)], row["expected_doc_ids"]) for row in valid) / len(valid),
        "recall_at_5": sum(recall_at_k(row["final_doc_ids"][str(depth)], row["expected_doc_ids"], 5) for row in valid) / len(valid),
        "recall_at_20": sum(recall_at_k(row["final_doc_ids"][str(depth)], row["expected_doc_ids"], 20) for row in valid) / len(valid),
        "evaluated_questions": len(valid),
    }


async def run(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    questions = read_dev_questions(args.questions)
    engine = build_engine(settings)
    try:
        bm25 = BM25Retriever(BM25IndexManager(args.index_root), build_session_factory(engine), tenant_id=args.tenant_id, knowledge_base_id=args.knowledge_base_id)
        reranker = CrossEncoderReranker(model_name=args.model, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
        ranked_rows: list[dict[str, object]] = []
        builder = ContextBuilder(token_budget=args.token_budget, max_chunks_per_doc=args.max_chunks_per_doc)
        for index, question in enumerate(questions, 1):
            base = await bm25.retrieve(question.question, top_k=50)
            scores = await asyncio.to_thread(reranker.score_pairs, question.question, [f"{c.title}\n{c.content}" for c in base])
            ranked = [item[0] for item in sorted(zip(base, scores, strict=True), key=lambda item: (-item[1], item[0].chunk_id))]
            row: dict[str, object] = {"question_id": question.question_id, "expected_doc_ids": question.expected_doc_ids, "bm25_doc_ids": [c.doc_id for c in base], "reranked_doc_ids": [c.doc_id for c in ranked], "final_doc_ids": {}, "context": {}}
            for depth in DEPTHS:
                head = ranked[:depth]
                seen = {chunk.chunk_id for chunk in head}
                final = head + [chunk for chunk in base[depth:] if chunk.chunk_id not in seen]
                row["final_doc_ids"][str(depth)] = [chunk.doc_id for chunk in final[:50]]
                context = builder.build(question.question, final[:50])
                context_docs = {block.doc_id for block in context.blocks}
                row["context"][str(depth)] = {"context_doc_recall": sum(doc in context_docs for doc in question.expected_doc_ids) / len(question.expected_doc_ids) if question.expected_doc_ids else 0.0}
            ranked_rows.append(row)
            if index % args.progress_every == 0 or index == len(questions):
                print(f"quality evaluation {index}/{len(questions)}", file=sys.stderr)

        bm25_rows = ranked_rows
        quality = {}
        for depth in DEPTHS:
            q = metrics(ranked_rows, depth)
            q["context_recall"] = sum(row["context"][str(depth)]["context_doc_recall"] for row in ranked_rows) / len(ranked_rows)
            q["bm25_mrr"] = sum(reciprocal_rank(row["bm25_doc_ids"], row["expected_doc_ids"]) for row in bm25_rows if row["expected_doc_ids"]) / len([r for r in bm25_rows if r["expected_doc_ids"]])
            q["bm25_recall_at_5"] = sum(recall_at_k(row["bm25_doc_ids"], row["expected_doc_ids"], 5) for row in bm25_rows if row["expected_doc_ids"]) / len([r for r in bm25_rows if r["expected_doc_ids"]])
            q["mrr_delta"] = q["mrr"] - q["bm25_mrr"]
            q["recall_at_5_delta"] = q["recall_at_5"] - q["bm25_recall_at_5"]
            quality[str(depth)] = q

        paired_analysis = {}
        valid_rows = [row for row in ranked_rows if row["expected_doc_ids"]]
        for depth in DEPTHS:
            pairs = []
            for row in valid_rows:
                bm = reciprocal_rank(row["bm25_doc_ids"], row["expected_doc_ids"])
                rr = reciprocal_rank(row["final_doc_ids"][str(depth)], row["expected_doc_ids"])
                bm5 = recall_at_k(row["bm25_doc_ids"], row["expected_doc_ids"], 5)
                rr5 = recall_at_k(row["final_doc_ids"][str(depth)], row["expected_doc_ids"], 5)
                pairs.append((bm, rr, bm5, rr5))
            rng = random.Random(20261117 + depth)
            samples = []
            for _ in range(10000):
                draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                samples.append((sum(r - b for b, r, _, _ in draw) / len(draw), sum(r - b for _, _, b, r in draw) / len(draw)))
            mrr_samples = sorted(sample[0] for sample in samples)
            recall_samples = sorted(sample[1] for sample in samples)
            paired_analysis[str(depth)] = {
                "wins": sum(r > b for b, r, _, _ in pairs),
                "losses": sum(r < b for b, r, _, _ in pairs),
                "ties": sum(r == b for b, r, _, _ in pairs),
                "bootstrap_ci": {"mrr_delta": [mrr_samples[249], mrr_samples[9749]], "recall_at_5_delta": [recall_samples[249], recall_samples[9749]], "confidence": 0.95, "bootstrap_resamples": 10000, "seed": 20261117 + depth},
            }

        # Performance is intentionally a distinct pass. The first timed query is
        # discarded after the model has loaded, so model startup is excluded.
        performance = {}
        for depth in () if args.skip_performance else DEPTHS:
            reset_gpu_memory()
            retriever = RerankedRetriever(bm25, reranker, candidate_k=depth)
            perf_questions = questions[:args.performance_limit] if args.performance_limit else questions
            await retriever.retrieve(perf_questions[0].question, top_k=depth)
            timings: list[float] = []
            for index, question in enumerate(perf_questions, 1):
                started = time.perf_counter()
                await retriever.retrieve(question.question, top_k=depth)
                timings.append((time.perf_counter() - started) * 1000)
                if index % args.progress_every == 0 or index == len(questions):
                    print(f"performance depth={depth} {index}/{len(perf_questions)}", file=sys.stderr)
            performance[str(depth)] = {"question_count": len(perf_questions), "latency_ms": latency_summary(timings), "peak_vram": gpu_memory(), "model_load_excluded": True}
    finally:
        await engine.dispose()

    full_gain = float(quality["50"]["mrr_delta"])
    selected_depth = next(depth for depth in DEPTHS if float(quality[str(depth)]["mrr_delta"]) >= 0.8 * full_gain)
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "question_count": len(questions),
        "model": args.model,
        "depths": list(DEPTHS),
        "quality": quality,
        "paired_analysis": paired_analysis,
        "performance": performance,
        "selection": {"full_depth": 50, "full_mrr_gain": full_gain, "threshold": 0.8 * full_gain, "selected_depth": selected_depth, "candidate_retriever": f"bm25_then_{args.model}_top_{selected_depth}"},
        "rerankers_tried": ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"],
        "configs_tried": 6,
        "per_question": ranked_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection": result["selection"], "quality": quality, "performance": performance}, indent=2))
    return result


def main() -> int:
    args = parser(get_settings()).parse_args()
    try:
        asyncio.run(run(args, get_settings()))
    except (BM25IndexError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
