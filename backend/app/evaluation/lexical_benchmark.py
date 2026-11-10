"""Reusable lexical benchmark helpers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

BM25_K1 = 1.5
BM25_B = 0.75
BM25_METHOD = "lucene"
BM25_STOPWORDS = "en"


@dataclass(frozen=True)
class BenchmarkChunk:
    """A chunk loaded directly from PostgreSQL for benchmarking."""

    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    chunk_index: int
    content: str
    content_hash: str


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    if not values:
        return 0.0

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(durations_ms: Sequence[float]) -> dict[str, float | int]:
    """Summarize one latency sample."""

    values = [float(value) for value in durations_ms]
    return {
        "count": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values, default=0.0),
    }


def corpus_fingerprint(chunks: Iterable[tuple[str, str]]) -> str:
    """Hash chunk IDs and content hashes independent of row order."""

    digest = hashlib.sha256()
    for chunk_id, content_hash in sorted(chunks):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_rarest(
    lexemes: Iterable[str],
    document_frequencies: Mapping[str, int],
    limit: int,
) -> list[str]:
    """Select the rarest lexemes, treating absent lexemes as zero frequency."""

    unique: list[str] = []
    seen: set[str] = set()
    for lexeme in lexemes:
        if lexeme and lexeme not in seen:
            seen.add(lexeme)
            unique.append(lexeme)

    ordered = sorted(
        unique,
        key=lambda lexeme: document_frequencies.get(lexeme, 0),
    )
    selected = ordered[:limit] if limit >= 0 else ordered
    return sorted(
        selected,
        key=lambda lexeme: document_frequencies.get(lexeme, 0) == 0,
    )


def or_websearch_text(lexemes: Iterable[str]) -> str:
    """Build parenthesized OR text for websearch_to_tsquery."""

    values = [lexeme for lexeme in lexemes if lexeme.strip()]
    if not values:
        raise ValueError("at least one lexeme is required")
    return f"({' or '.join(values)})"


def title_weighted_text(title: str, content: str) -> str:
    """Repeat title tokens three times before the chunk content."""

    return f"{title}\n{title}\n{title}\n{content}"


def build_manifest(
    *,
    index_name: str,
    corpus_fingerprint: str,
    documents: int,
    chunks: int,
    title_weighting: str,
    bm25s_version: str,
    pystemmer_version: str,
) -> dict[str, object]:
    """Build the metadata required to validate a persisted BM25 index."""

    return {
        "index_name": index_name,
        "corpus_fingerprint": corpus_fingerprint,
        "documents": documents,
        "chunks": chunks,
        "title_weighting": title_weighting,
        "bm25s_version": bm25s_version,
        "pystemmer_version": pystemmer_version,
        "k1": BM25_K1,
        "b": BM25_B,
        "method": BM25_METHOD,
        "stopwords": BM25_STOPWORDS,
    }


def validate_manifest(
    saved: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    """Reject a persisted index that does not match this run."""

    for key, expected_value in expected.items():
        if saved.get(key) != expected_value:
            raise ValueError(f"BM25 manifest mismatch for key: {key}")
