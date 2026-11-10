from __future__ import annotations

import importlib.metadata

import pytest
import Stemmer

from app.evaluation.lexical_benchmark import (
    BenchmarkChunk,
    BM25SpikeRetriever,
    build_manifest,
    corpus_fingerprint,
    latency_summary,
    load_or_build_bm25_index,
    or_websearch_text,
    percentile,
    select_rarest,
    title_weighted_text,
    validate_manifest,
)


def test_percentile_returns_interpolated_values() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([], 0.95) == 0.0


def test_latency_summary_reports_p50_and_p95() -> None:
    summary = latency_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["max_ms"] == 4.0


def test_corpus_fingerprint_is_order_independent() -> None:
    left = corpus_fingerprint([("b", "2"), ("a", "1")])
    right = corpus_fingerprint([("a", "1"), ("b", "2")])

    assert left == right
    assert len(left) == 64


def test_select_rarest_orders_zero_and_missing_frequencies_last() -> None:
    frequencies = {"common": 100, "rare": 2, "absent": 0}

    assert select_rarest(
        ["common", "rare", "absent", "missing"],
        frequencies,
        limit=3,
    ) == ["rare", "absent", "missing"]


def test_or_websearch_text_uses_websearch_or_syntax() -> None:
    assert or_websearch_text(["alpha", "beta"]) == "(alpha or beta)"
    assert or_websearch_text(["alpha"]) == "(alpha)"


def test_title_weighted_text_repeats_title_three_times() -> None:
    result = title_weighted_text("Deploy Guide", "Run the service.")

    assert result == "Deploy Guide\nDeploy Guide\nDeploy Guide\nRun the service."


def test_manifest_validation_rejects_corpus_mismatch() -> None:
    manifest = build_manifest(
        index_name="content",
        corpus_fingerprint="f" * 64,
        documents=10,
        chunks=20,
        title_weighting="content",
        bm25s_version="0.2.0",
        pystemmer_version="3.0.0",
    )
    validate_manifest(manifest, manifest)

    bad = dict(manifest)
    bad["corpus_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_manifest(manifest, bad)


class FakeBM25:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def retrieve(
        self,
        query_tokens: list[str],
        k: int,
        *,
        show_progress: bool = False,
    ) -> tuple[list[list[int]], list[list[float]]]:
        self.calls.append((query_tokens, k))
        return [[0, 1]], [[9.0, 4.0]]


def make_chunk(
    chunk_id: str,
    doc_id: str,
    source_type: str,
    content: str,
) -> BenchmarkChunk:
    return BenchmarkChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_type=source_type,
        title=f"Title {chunk_id}",
        chunk_index=0,
        content=content,
        content_hash=f"hash-{chunk_id}",
    )


@pytest.mark.asyncio
async def test_bm25_spike_retriever_maps_scores_and_filters_sources() -> None:
    chunks = [
        make_chunk("c1", "d1", "github", "alpha content"),
        make_chunk("c2", "d2", "jira", "beta content"),
    ]
    fake = FakeBM25()
    retriever = BM25SpikeRetriever(fake, chunks, stemmer=Stemmer.Stemmer("english"))

    results = await retriever.retrieve("alpha", top_k=1, source_types=["jira"])

    assert [result.chunk_id for result in results] == ["c2"]
    assert results[0].score == 4.0
    assert results[0].metadata == {}
    assert fake.calls[0][1] == 2


@pytest.mark.asyncio
async def test_bm25_spike_retriever_returns_empty_for_blank_or_stopword_query() -> None:
    chunks = [make_chunk("c1", "d1", "github", "alpha content")]
    retriever = BM25SpikeRetriever(
        FakeBM25(),
        chunks,
        stemmer=Stemmer.Stemmer("english"),
    )

    assert await retriever.retrieve("   ") == []
    assert await retriever.retrieve("the") == []


def test_load_or_build_bm25_index_persists_and_loads(tmp_path) -> None:
    chunks = [
        make_chunk("c1", "d1", "github", "alpha content"),
        make_chunk("c2", "d2", "jira", "beta content"),
    ]
    expected = build_manifest(
        index_name="content",
        corpus_fingerprint=corpus_fingerprint(
            (chunk.chunk_id, chunk.content_hash) for chunk in chunks
        ),
        documents=2,
        chunks=2,
        title_weighting="content",
        bm25s_version=importlib.metadata.version("bm25s"),
        pystemmer_version=importlib.metadata.version("PyStemmer"),
    )
    stemmer = Stemmer.Stemmer("english")

    _, build_seconds, _, size_bytes = load_or_build_bm25_index(
        index_dir=tmp_path / "content",
        chunks=chunks,
        title_weighting="content",
        stemmer=stemmer,
        expected_manifest=expected,
    )
    _, _, load_seconds, loaded_size = load_or_build_bm25_index(
        index_dir=tmp_path / "content",
        chunks=chunks,
        title_weighting="content",
        stemmer=stemmer,
        expected_manifest=expected,
    )

    assert build_seconds > 0
    assert load_seconds >= 0
    assert size_bytes > 0
    assert loaded_size == size_bytes
    assert (tmp_path / "content" / "manifest.json").is_file()
    assert (tmp_path / "content" / "chunk_ids.json").is_file()


def test_load_or_build_bm25_index_rejects_mismatched_manifest(tmp_path) -> None:
    chunks = [make_chunk("c1", "d1", "github", "alpha content")]
    expected = build_manifest(
        index_name="content",
        corpus_fingerprint=corpus_fingerprint(
            (chunk.chunk_id, chunk.content_hash) for chunk in chunks
        ),
        documents=1,
        chunks=1,
        title_weighting="content",
        bm25s_version=importlib.metadata.version("bm25s"),
        pystemmer_version=importlib.metadata.version("PyStemmer"),
    )
    stemmer = Stemmer.Stemmer("english")
    load_or_build_bm25_index(
        index_dir=tmp_path / "content",
        chunks=chunks,
        title_weighting="content",
        stemmer=stemmer,
        expected_manifest=expected,
    )

    bad = dict(expected)
    bad["corpus_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="manifest mismatch"):
        load_or_build_bm25_index(
            index_dir=tmp_path / "content",
            chunks=chunks,
            title_weighting="content",
            stemmer=stemmer,
            expected_manifest=bad,
        )
