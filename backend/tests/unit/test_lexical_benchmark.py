from __future__ import annotations

import pytest

from app.evaluation.lexical_benchmark import (
    build_manifest,
    corpus_fingerprint,
    latency_summary,
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
