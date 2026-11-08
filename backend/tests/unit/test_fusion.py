"""Tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk


def _chunk(chunk_id: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        source_type="confluence",
        title="Title",
        chunk_index=0,
        content="content",
        score=score,
        metadata={},
    )


def test_rrf_combines_overlapping_rankings() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk("a"), _chunk("b")], [_chunk("b"), _chunk("c")]],
        k=60,
    )

    assert [chunk.chunk_id for chunk in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_honors_weights() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk("a"), _chunk("b")], [_chunk("b"), _chunk("c")]],
        k=60,
        weights=[2.0, 1.0],
    )

    assert [chunk.chunk_id for chunk in fused] == ["b", "a", "c"]


def test_rrf_counts_a_chunk_once_per_ranking() -> None:
    fused = reciprocal_rank_fusion(
        [[_chunk("a"), _chunk("a"), _chunk("b")]],
        k=60,
    )

    assert [chunk.chunk_id for chunk in fused] == ["a", "b"]
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_is_deterministic_on_ties() -> None:
    fused = reciprocal_rank_fusion([[_chunk("b")], [_chunk("a")]], k=60)

    assert [chunk.chunk_id for chunk in fused] == ["a", "b"]


def test_rrf_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([[_chunk("a")]], k=0)
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[_chunk("a")]], weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([[_chunk("a")]], weights=[-1.0])
