"""Rerank candidates from another retriever."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from math import isfinite
from typing import Protocol

from app.reranking.base import Reranker
from app.retrieval.types import RetrievedChunk


class BaseRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...


class RerankedRetriever:
    """Rerank a base retriever's candidates with a cross-encoder."""

    def __init__(
        self,
        base: BaseRetriever,
        reranker: Reranker,
        *,
        candidate_k: int = 50,
        preserve_tail: bool = False,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.preserve_tail = preserve_tail

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 50,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        fetch_k = max(top_k, self.candidate_k) if self.preserve_tail else self.candidate_k
        candidates = await self.base.retrieve(
            query,
            top_k=fetch_k,
            source_types=source_types,
        )
        if not candidates:
            return []

        rerank_candidates = candidates[: self.candidate_k]
        texts = [f"{candidate.title}\n{candidate.content}" for candidate in rerank_candidates]
        scores = await asyncio.to_thread(self.reranker.score_pairs, query, texts)
        if len(scores) != len(rerank_candidates):
            raise RuntimeError(
                f"reranker returned {len(scores)} scores for {len(rerank_candidates)} candidates"
            )
        if not all(isfinite(score) for score in scores):
            raise RuntimeError("reranker returned a non-finite score")

        ranked = sorted(
            zip(rerank_candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        head = [
            replace(candidate, score=score, retrieval_score=candidate.score)
            for candidate, score in ranked[:top_k]
        ]
        if not self.preserve_tail:
            return head
        seen = {candidate.chunk_id for candidate in head}
        tail = [candidate for candidate in candidates[self.candidate_k :] if candidate.chunk_id not in seen]
        return (head + tail)[:top_k]
