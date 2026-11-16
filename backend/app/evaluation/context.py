"""Development-set evaluation for token-budgeted context construction."""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from typing import Protocol

from app.context.builder import ContextBuilder
from app.context.types import BuiltContext
from app.evaluation.dev import DevQuestion, latency_summary
from app.retrieval.types import RetrievedChunk


class RetrieverProtocol(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...


def _recall(expected: Sequence[str], actual: set[str]) -> float:
    if not expected:
        return 0.0
    return sum(doc_id in actual for doc_id in expected) / len(expected)


def _mrr(expected: Sequence[str], context: BuiltContext) -> float:
    expected_set = set(expected)
    for rank, block in enumerate(context.blocks, start=1):
        if block.doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    summary = latency_summary(values)
    return {"p50_ms": float(summary["p50_ms"]), "p95_ms": float(summary["p95_ms"])}


async def evaluate_context(
    retriever: RetrieverProtocol,
    questions: Sequence[DevQuestion],
    *,
    top_k: int = 50,
    token_budget: int = 4000,
    max_chunks_per_doc: int = 3,
    progress_every: int = 25,
) -> dict[str, object]:
    """Evaluate context selection without passing expected IDs to the builder."""

    if not questions:
        raise ValueError("questions must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if max_chunks_per_doc <= 0:
        raise ValueError("max_chunks_per_doc must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    builder = ContextBuilder(
        token_budget=token_budget,
        max_chunks_per_doc=max_chunks_per_doc,
    )
    candidate_recalls: list[float] = []
    context_recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    context_tokens: list[float] = []
    utilizations: list[float] = []
    selected_chunks: list[float] = []
    selected_docs: list[float] = []
    retrieval_latencies: list[float] = []
    build_latencies: list[float] = []
    total_latencies: list[float] = []
    budget_passes: list[float] = []
    duplicate_chunk_count = 0
    empty_context_count = 0
    per_question: list[dict[str, object]] = []

    for index, question in enumerate(questions, start=1):
        started = time.perf_counter()
        candidates = await retriever.retrieve(question.question, top_k=top_k)
        retrieved_at = time.perf_counter()
        context = builder.build(question.question, candidates)
        built_at = time.perf_counter()

        retrieval_ms = (retrieved_at - started) * 1000.0
        build_ms = (built_at - retrieved_at) * 1000.0
        total_ms = (built_at - started) * 1000.0
        retrieval_latencies.append(retrieval_ms)
        build_latencies.append(build_ms)
        total_latencies.append(total_ms)

        candidate_doc_ids = list(dict.fromkeys(candidate.doc_id for candidate in candidates))
        context_doc_ids = list(dict.fromkeys(block.doc_id for block in context.blocks))
        selected_chunk_ids = [block.chunk_id for block in context.blocks]
        candidate_recall = _recall(question.expected_doc_ids, set(candidate_doc_ids))
        context_recall = _recall(question.expected_doc_ids, set(context_doc_ids))
        mrr = _mrr(question.expected_doc_ids, context)
        within_budget = context.token_count <= token_budget

        candidate_recalls.append(candidate_recall)
        context_recalls.append(context_recall)
        reciprocal_ranks.append(mrr)
        context_tokens.append(float(context.token_count))
        utilizations.append(context.token_count / token_budget)
        selected_chunks.append(float(len(context.blocks)))
        selected_docs.append(float(context.selected_doc_count))
        budget_passes.append(float(within_budget))
        duplicate_chunk_count += len(selected_chunk_ids) - len(set(selected_chunk_ids))
        if not context.blocks:
            empty_context_count += 1

        per_question.append(
            {
                "question_id": question.question_id,
                "expected_doc_ids": list(question.expected_doc_ids),
                "candidate_doc_ids": candidate_doc_ids,
                "context_doc_ids": context_doc_ids,
                "selected_chunk_ids": selected_chunk_ids,
                "candidate_doc_recall": candidate_recall,
                "context_doc_recall": context_recall,
                "context_mrr": mrr,
                "token_count": context.token_count,
                "token_budget": token_budget,
                "within_budget": within_budget,
                "retrieval_ms": retrieval_ms,
                "build_ms": build_ms,
                "total_ms": total_ms,
            }
        )

        if index % progress_every == 0 or index == len(questions):
            print(f"context evaluation {index}/{len(questions)}", file=sys.stderr)

    question_count = len(questions)
    metrics = {
        "question_count": question_count,
        "candidate_doc_recall": sum(candidate_recalls) / question_count,
        "context_doc_recall": sum(context_recalls) / question_count,
        "context_mrr": sum(reciprocal_ranks) / question_count,
        "budget_adherence": sum(budget_passes) / question_count,
        "mean_context_tokens": sum(context_tokens) / question_count,
        "p95_context_tokens": _percentile(context_tokens, 0.95),
        "mean_utilization": sum(utilizations) / question_count,
        "selected_chunk_mean": sum(selected_chunks) / question_count,
        "selected_doc_mean": sum(selected_docs) / question_count,
        "duplicate_chunk_count": duplicate_chunk_count,
        "empty_context_count": empty_context_count,
        "latency_ms": {
            "retrieval": _latency_summary(retrieval_latencies),
            "build": _latency_summary(build_latencies),
            "total": _latency_summary(total_latencies),
        },
    }
    return {
        "params": {
            "top_k": top_k,
            "token_budget": token_budget,
            "max_chunks_per_doc": max_chunks_per_doc,
        },
        "metrics": metrics,
        "per_question": per_question,
    }
