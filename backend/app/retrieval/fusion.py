"""Reciprocal Rank Fusion across retrieval strategies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from app.retrieval.types import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked result lists using Reciprocal Rank Fusion."""

    if k <= 0:
        raise ValueError("k must be positive")
    resolved_weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(resolved_weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    if any(weight < 0 for weight in resolved_weights):
        raise ValueError("weights must be non-negative")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for ranking, weight in zip(rankings, resolved_weights, strict=True):
        if weight == 0:
            continue
        seen: set[str] = set()
        for rank, chunk in enumerate(ranking, start=1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
            best_rank[chunk.chunk_id] = min(best_rank.get(chunk.chunk_id, rank), rank)

    fused = [replace(chunks[chunk_id], score=scores[chunk_id]) for chunk_id in scores]
    fused.sort(key=lambda chunk: (-chunk.score, best_rank[chunk.chunk_id], chunk.chunk_id))
    return fused
