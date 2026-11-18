"""Compare BM25 and reranked retrieval on the development split."""

from __future__ import annotations

import random
import sys
import time
from collections.abc import Sequence

from app.evaluation.dev import DevQuestion, RetrieverProtocol, latency_summary
from app.retrieval.metrics import (
    DEFAULT_K_VALUES,
    evaluate_retrieval,
    recall_at_k,
    reciprocal_rank,
)


async def evaluate_reranking(
    bm25: RetrieverProtocol,
    reranked: RetrieverProtocol,
    questions: Sequence[DevQuestion],
    *,
    top_k: int = 50,
    candidate_k: int = 50,
    progress_every: int = 25,
) -> dict[str, object]:
    if not questions:
        raise ValueError("questions must not be empty")
    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("top_k and candidate_k must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    bm25_expected: dict[str, list[str]] = {}
    bm25_retrieved: dict[str, list[str]] = {}
    reranked_retrieved: dict[str, list[str]] = {}
    bm25_latencies: list[float] = []
    reranked_latencies: list[float] = []
    per_question: list[dict[str, object]] = []

    for index, question in enumerate(questions, start=1):
        started = time.perf_counter()
        bm25_results = await bm25.retrieve(question.question, top_k=top_k)
        bm25_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        reranked_results = await reranked.retrieve(question.question, top_k=top_k)
        reranked_ms = (time.perf_counter() - started) * 1000.0

        bm25_doc_ids = [result.doc_id for result in bm25_results]
        reranked_doc_ids = [result.doc_id for result in reranked_results]
        bm25_expected[question.question_id] = question.expected_doc_ids
        bm25_retrieved[question.question_id] = bm25_doc_ids
        reranked_retrieved[question.question_id] = reranked_doc_ids
        bm25_latencies.append(bm25_ms)
        reranked_latencies.append(reranked_ms)
        per_question.append(
            {
                "question_id": question.question_id,
                "expected_doc_ids": question.expected_doc_ids,
                "bm25_doc_ids": bm25_doc_ids,
                "reranked_doc_ids": reranked_doc_ids,
                "bm25_ms": bm25_ms,
                "reranked_ms": reranked_ms,
            }
        )
        if index % progress_every == 0 or index == len(questions):
            print(f"reranking evaluation {index}/{len(questions)}", file=sys.stderr)

    k_values = tuple(k for k in DEFAULT_K_VALUES if k <= top_k)
    bm25_metrics = evaluate_retrieval(
        bm25_expected,
        bm25_retrieved,
        k_values=k_values,
    )
    reranked_metrics = evaluate_retrieval(
        bm25_expected,
        reranked_retrieved,
        k_values=k_values,
    )
    delta = {
        key: float(reranked_metrics[key]) - float(bm25_metrics[key])
        for key in bm25_metrics
        if isinstance(bm25_metrics[key], (int, float))
    }
    paired = []
    for row in per_question:
        expected = row["expected_doc_ids"]
        base_mrr = reciprocal_rank(row["bm25_doc_ids"], expected)
        rerank_mrr = reciprocal_rank(row["reranked_doc_ids"], expected)
        base_r5 = recall_at_k(row["bm25_doc_ids"], expected, 5)
        rerank_r5 = recall_at_k(row["reranked_doc_ids"], expected, 5)
        row["bm25_mrr"] = base_mrr
        row["reranked_mrr"] = rerank_mrr
        row["bm25_recall_at_5"] = base_r5
        row["reranked_recall_at_5"] = rerank_r5
        paired.append((base_mrr, rerank_mrr, base_r5, rerank_r5))
    wins = sum(r > b for b, r, _, _ in paired)
    losses = sum(r < b for b, r, _, _ in paired)
    ties = len(paired) - wins - losses
    rng = random.Random(20261117)
    samples = []
    for _ in range(10000):
        draw = [paired[rng.randrange(len(paired))] for _ in paired]
        samples.append((
            sum(rm - bm for bm, rm, _, _ in draw) / len(draw),
            sum(rr - br for _, _, br, rr in draw) / len(draw),
        ))
    mrr_samples = sorted(sample[0] for sample in samples)
    recall_samples = sorted(sample[1] for sample in samples)
    ci = {
        "mrr_delta": [mrr_samples[249], mrr_samples[9749]],
        "recall_at_5_delta": [recall_samples[249], recall_samples[9749]],
        "confidence": 0.95,
        "bootstrap_resamples": 10000,
        "seed": 20261117,
    }
    return {
        "params": {"top_k": top_k, "candidate_k": candidate_k},
        "bm25": {"metrics": bm25_metrics},
        "reranked": {"metrics": reranked_metrics},
        "delta": delta,
        "paired_analysis": {"wins": wins, "losses": losses, "ties": ties, "bootstrap_ci": ci},
        "latency_ms": {
            "bm25": latency_summary(bm25_latencies),
            "reranked": latency_summary(reranked_latencies),
        },
        "per_question": per_question,
    }
