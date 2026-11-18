#!/usr/bin/env python3
"""Measure warmed reranker latency and peak CUDA memory on a dev sample."""
from __future__ import annotations
import argparse, asyncio, json, sys, time, uuid
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from app.db.session import build_engine, build_session_factory
from app.evaluation.dev import latency_summary, read_dev_questions
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.bm25_index import BM25IndexManager
from app.retrieval.reranked import RerankedRetriever
from app.settings import get_settings

async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data/sample/dev_questions.jsonl")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/reranking_performance_dev.json")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--device", default=get_settings().reranker_device)
    p.add_argument("--batch-size", type=int, default=get_settings().reranker_batch_size)
    p.add_argument("--max-length", type=int, default=get_settings().reranker_max_length)
    p.add_argument("--index-root", type=Path, default=get_settings().bm25_index_root)
    args = p.parse_args()
    questions = read_dev_questions(args.questions, args.limit)
    settings = get_settings()
    engine = build_engine(settings)
    try:
        bm25 = BM25Retriever(BM25IndexManager(args.index_root), build_session_factory(engine), tenant_id=settings.default_tenant_id, knowledge_base_id=settings.default_knowledge_base_id)
        reranker = CrossEncoderReranker(model_name=args.model, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
        load_started = time.perf_counter()
        reranker._get_predictor()
        model_load_ms = (time.perf_counter() - load_started) * 1000
        result = {"model": args.model, "question_count": len(questions), "model_load_excluded": True, "model_load_ms": model_load_ms, "depths": {}}
        for depth in (10, 20, 30, 50):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            except ImportError:
                pass
            retriever = RerankedRetriever(bm25, reranker, candidate_k=depth, preserve_tail=True)
            cold_started = time.perf_counter()
            await retriever.retrieve(questions[0].question, top_k=depth)
            cold_total_ms = model_load_ms + (time.perf_counter() - cold_started) * 1000
            timings = []
            for question in questions:
                started = time.perf_counter()
                await retriever.retrieve(question.question, top_k=depth)
                timings.append((time.perf_counter() - started) * 1000)
            memory = {"peak_allocated_mib": None, "peak_reserved_mib": None}
            try:
                if torch.cuda.is_available():
                    memory = {"peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2), "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 2)}
            except (NameError, RuntimeError):
                pass
            result["depths"][str(depth)] = {"latency_ms": latency_summary(timings), "peak_vram": memory, "cold_start_first_query_ms": cold_total_ms}
    finally:
        await engine.dispose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
