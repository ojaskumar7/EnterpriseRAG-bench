"""Reusable lexical benchmark helpers."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
import Stemmer

from app.retrieval.types import RetrievedChunk

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


def _tokenize(text: str, stemmer: Any) -> list[str]:
    tokens = bm25s.tokenize(
        [text],
        stopwords=BM25_STOPWORDS,
        stemmer=stemmer,
        return_ids=False,
        show_progress=False,
    )
    return [str(token) for token in tokens[0] if str(token).strip()]


class BM25SpikeRetriever:
    """Adapt a bm25s index to the retrieval protocol."""

    def __init__(
        self,
        bm25: Any,
        chunks: Sequence[BenchmarkChunk],
        *,
        stemmer: Any | None = None,
    ) -> None:
        self.bm25 = bm25
        self.chunks = list(chunks)
        self.stemmer = stemmer or Stemmer.Stemmer("english")

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip() or not self.chunks:
            return []

        query_tokens = _tokenize(query, self.stemmer)
        if not query_tokens:
            return []

        candidate_k = min(len(self.chunks), max(top_k * 10, top_k))
        indices, scores = self.bm25.retrieve(
            query_tokens,
            k=candidate_k,
            show_progress=False,
        )
        allowed_sources = set(source_types) if source_types else None

        results: list[RetrievedChunk] = []
        for index, score in zip(indices[0], scores[0], strict=False):
            position = int(index)
            if position < 0 or position >= len(self.chunks):
                continue
            chunk = self.chunks[position]
            if allowed_sources is not None and chunk.source_type not in allowed_sources:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=float(score),
                    metadata={},
                )
            )
            if len(results) >= top_k:
                break
        return results


def _index_text(chunk: BenchmarkChunk, title_weighting: str) -> str:
    if title_weighting == "content":
        return f"{chunk.title}\n{chunk.content}"
    if title_weighting == "title_x3":
        return title_weighted_text(chunk.title, chunk.content)
    raise ValueError("title_weighting must be one of: content, title_x3")


def _index_size(index_dir: Path) -> int:
    return sum(path.stat().st_size for path in index_dir.rglob("*") if path.is_file())


def load_or_build_bm25_index(
    *,
    index_dir: Path,
    chunks: Sequence[BenchmarkChunk],
    title_weighting: str,
    stemmer: Any,
    expected_manifest: Mapping[str, object],
    rebuild: bool = False,
) -> tuple[Any, float, float, int]:
    """Load a validated index or build and persist a new one."""

    index_dir = Path(index_dir)
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if index_dir.exists() and not rebuild:
        manifest_path = index_dir / "manifest.json"
        chunk_ids_path = index_dir / "chunk_ids.json"
        try:
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            saved_chunk_ids = json.loads(chunk_ids_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"BM25 index metadata is missing or invalid in {index_dir}; "
                "rerun with --rebuild-index"
            ) from exc

        validate_manifest(saved_manifest, expected_manifest)
        if saved_chunk_ids != chunk_ids:
            raise ValueError("BM25 index chunk IDs do not match the corpus")

        started = time.perf_counter()
        bm25 = bm25s.BM25.load(index_dir, load_corpus=False, show_progress=False)
        load_seconds = time.perf_counter() - started
        return bm25, 0.0, load_seconds, _index_size(index_dir)

    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    texts = [_index_text(chunk, title_weighting) for chunk in chunks]
    started = time.perf_counter()
    tokens = bm25s.tokenize(
        texts,
        stopwords=BM25_STOPWORDS,
        stemmer=stemmer,
        return_ids=False,
        show_progress=True,
    )
    bm25 = bm25s.BM25(k1=BM25_K1, b=BM25_B, method=BM25_METHOD)
    bm25.index(tokens)
    bm25.save(index_dir)
    build_seconds = time.perf_counter() - started

    (index_dir / "manifest.json").write_text(
        json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (index_dir / "chunk_ids.json").write_text(
        json.dumps(chunk_ids, indent=2) + "\n",
        encoding="utf-8",
    )
    return bm25, build_seconds, 0.0, _index_size(index_dir)
