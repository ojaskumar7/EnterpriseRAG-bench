from __future__ import annotations

import pytest

from app.context.builder import ContextBuilder
from app.context.render import context_to_dict, render_context
from app.retrieval.types import RetrievedChunk


def chunk(
    chunk_id: str,
    *,
    doc_id: str = "doc-1",
    content: str = "alpha beta",
    score: float = 1.0,
    token_count: int | None = None,
    source_type: str = "pdf",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_type=source_type,
        title=f"Title {chunk_id}",
        chunk_index=0,
        content=content,
        score=score,
        metadata={},
        token_count=token_count,
    )


def test_builder_deduplicates_and_caps_documents() -> None:
    candidates = [
        chunk("a", doc_id="d1", score=0.9),
        chunk("a", doc_id="d1", score=0.8),
        chunk("b", doc_id="d1", score=0.7),
        chunk("c", doc_id="d1", score=0.6),
        chunk("d", doc_id="d1", score=0.5),
        chunk("e", doc_id="d2", score=0.4),
    ]

    context = ContextBuilder(token_budget=100, max_chunks_per_doc=3).build("q", candidates)

    assert [block.chunk_id for block in context.blocks] == ["a", "b", "c", "e"]
    assert [block.citation_id for block in context.blocks] == [1, 2, 3, 4]
    assert context.selected_doc_count == 2
    assert context.omitted_count == 2


def test_builder_skips_oversized_chunk_and_continues() -> None:
    candidates = [
        chunk("big", content="x x x x x", score=0.9),
        chunk("small", content="x", score=0.8),
    ]

    context = ContextBuilder(token_budget=3).build("q", candidates)

    assert [block.chunk_id for block in context.blocks] == ["small"]
    assert context.token_count == 1
    assert context.omitted_count == 1


def test_builder_uses_stored_token_count_before_fallback() -> None:
    candidates = [chunk("a", content="one two three four", token_count=2)]

    context = ContextBuilder(token_budget=2).build("q", candidates)

    assert context.token_count == 2
    assert context.blocks[0].token_count == 2


def test_builder_ordering_is_deterministic() -> None:
    candidates = [
        chunk("z", score=0.5),
        chunk("a", score=0.5),
        chunk("m", score=0.9),
    ]

    context = ContextBuilder(token_budget=100).build("q", candidates)

    assert [block.chunk_id for block in context.blocks] == ["m", "a", "z"]


def test_builder_returns_empty_context_for_blank_query_or_no_candidates() -> None:
    builder = ContextBuilder()

    assert builder.build("", [chunk("a")]).blocks == ()
    assert builder.build("q", []).blocks == ()
    assert builder.build("q", []).token_count == 0


def test_builder_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        ContextBuilder(token_budget=0)
    with pytest.raises(ValueError):
        ContextBuilder(max_chunks_per_doc=0)


def test_render_and_dict_include_citations() -> None:
    context = ContextBuilder(token_budget=100).build("q", [chunk("a", source_type="html")])

    rendered = render_context(context)
    payload = context_to_dict(context)

    assert rendered.startswith("[1] Title a (html, doc_id=doc-1, chunk=0)")
    assert "alpha beta" in rendered
    assert payload["token_count"] == 2
    assert payload["blocks"][0]["chunk_id"] == "a"
