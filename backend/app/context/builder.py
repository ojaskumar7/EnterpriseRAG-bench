"""Deterministic, rank-first context packing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from math import isfinite

from app.context.types import BuiltContext, ContextBlock
from app.retrieval.types import RetrievedChunk

TokenCounter = Callable[[str], int]


def estimate_tokens(text: str) -> int:
    """Estimate tokens with a deterministic whitespace fallback."""

    return max(1, len(text.split()))


class ContextBuilder:
    """Pack whole ranked chunks under a token budget."""

    def __init__(
        self,
        token_budget: int = 4000,
        max_chunks_per_doc: int = 3,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token budget must be positive")
        if max_chunks_per_doc <= 0:
            raise ValueError("max chunks per document must be positive")
        self.token_budget = token_budget
        self.max_chunks_per_doc = max_chunks_per_doc
        self.token_counter = token_counter or estimate_tokens

    def build(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
    ) -> BuiltContext:
        if not query.strip() or not candidates:
            return BuiltContext(
                query=query,
                blocks=(),
                token_count=0,
                token_budget=self.token_budget,
                candidate_count=0,
                omitted_count=0,
                selected_doc_count=0,
                source_type_counts={},
            )

        ranked = sorted(
            (candidate for candidate in candidates if isfinite(candidate.score)),
            key=lambda candidate: (-candidate.score, candidate.chunk_id),
        )
        seen_chunk_ids: set[str] = set()
        selected_per_doc: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        blocks: list[ContextBlock] = []
        total_tokens = 0

        for candidate in ranked:
            if candidate.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(candidate.chunk_id)
            if selected_per_doc[candidate.doc_id] >= self.max_chunks_per_doc:
                continue

            tokens = candidate.token_count
            if tokens is None or tokens <= 0:
                tokens = self.token_counter(candidate.content)
            tokens = max(1, int(tokens))
            if total_tokens + tokens > self.token_budget:
                continue

            blocks.append(
                ContextBlock(
                    citation_id=len(blocks) + 1,
                    chunk_id=candidate.chunk_id,
                    doc_id=candidate.doc_id,
                    source_type=candidate.source_type,
                    title=candidate.title,
                    chunk_index=candidate.chunk_index,
                    content=candidate.content,
                    score=candidate.score,
                    token_count=tokens,
                )
            )
            total_tokens += tokens
            selected_per_doc[candidate.doc_id] += 1
            source_counts[candidate.source_type] += 1

        return BuiltContext(
            query=query,
            blocks=tuple(blocks),
            token_count=total_tokens,
            token_budget=self.token_budget,
            candidate_count=len(candidates),
            omitted_count=len(candidates) - len(blocks),
            selected_doc_count=len(selected_per_doc),
            source_type_counts=dict(sorted(source_counts.items())),
        )
