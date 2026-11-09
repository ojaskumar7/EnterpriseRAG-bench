"""Hybrid retrieval using RRF over dense and lexical candidates."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.types import RetrievedChunk


class HybridRetriever:
    """Fuse dense and lexical rankings with Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        lexical_retriever: LexicalRetriever,
        *,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
        candidate_k: int = 50,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if dense_weight < 0 or lexical_weight < 0:
            raise ValueError("weights must be non-negative")

        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.candidate_k = candidate_k

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve and fuse dense and lexical results."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        candidate_k = max(self.candidate_k, top_k)
        dense_results, lexical_results = await asyncio.gather(
            self.dense_retriever.retrieve(
                query,
                top_k=candidate_k,
                source_types=source_types,
            ),
            self.lexical_retriever.retrieve(
                query,
                top_k=candidate_k,
                source_types=source_types,
            ),
        )
        fused = reciprocal_rank_fusion(
            [dense_results, lexical_results],
            k=self.rrf_k,
            weights=[self.dense_weight, self.lexical_weight],
        )
        return fused[:top_k]
