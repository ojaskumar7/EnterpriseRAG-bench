# Block 3.5 Lexical Retrieval Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a benchmark-only spike that compares current PostgreSQL FTS, relaxed PostgreSQL FTS, and exact BM25 on the 169 dev questions, then reports whether BM25 meets the quality and latency gates.

**Architecture:** A new `app.evaluation.lexical_benchmark` module owns reusable benchmark helpers, BM25 loading, and latency accounting. A new `app.evaluation.lexical_postgres` module owns read-only asyncpg PostgreSQL variants. A standalone `scripts/benchmark_lexical.py` CLI builds or loads two persisted BM25 indexes, evaluates all variants with the existing dev evaluator, and writes a JSON artifact. Production retrieval code is not modified.

**Tech Stack:** Python 3.12, `uv`, asyncpg, SQLAlchemy async, `bm25s`, `PyStemmer`, existing `DenseRetriever`, `HybridRetriever`, and `evaluate_retriever`.

## Global Constraints

- Use only the 169 development questions from `data/sample/dev_questions.jsonl`. Never read or evaluate the holdout split.
- Do not modify production retrieval defaults, `README.md`, ADR defaults, or any existing production retriever.
- Add `bm25s` and `PyStemmer` as development dependencies only, using `uv add --dev`.
- Do not commit downloaded data, PostgreSQL snapshots, BM25 indexes, or benchmark JSON artifacts.
- BM25 parameters are fixed at `k1=1.5`, `b=0.75`, `method="lucene"`, English stopwords, and English stemming.
- The BM25 gate is lexical recall@20 >= `0.15` and at most `20/169` zero-result questions.
- The hybrid gate is a same-run gain of at least `+0.01` on recall@10 or MRR, with recall@5, recall@20, and the other headline metric no more than `0.01` below the same-run dense baseline.
- Measure lexical top-50 p95 latency; the gate is `<200 ms`.
- Evaluate dense twice in the same run and report drift. If dense drift is greater than `0.01` on recall@10 or MRR, mark hybrid quality conclusions inconclusive.
- Use `--device cuda --dtype float16` for the full run because the HNSW index was built with FP16 vectors.
- PostgreSQL and BM25 retrieval in this block are read-only. Do not write documents or chunks.
- PostgreSQL OR fallback variants must use a per-query timeout and record timeouts as zero-result failures.
- Start implementation from a clean `main` at commit `64d9b8d` or later, in an isolated worktree if the project workflow requires it.
- Keep the clean database counts at `34378|136383|136383` after all tests and benchmarks.
- Do not push commits unless the user explicitly requests it.

---

### Task 1: Add benchmark helpers

**Files:**
- Create: `backend/app/evaluation/lexical_benchmark.py`
- Create: `backend/tests/unit/test_lexical_benchmark.py`

**Interfaces:**
- Consumes: standard library only.
- Produces:
  - `BenchmarkChunk`
  - `percentile(values, fraction) -> float`
  - `latency_summary(durations_ms) -> dict[str, float | int]`
  - `corpus_fingerprint(chunks: Iterable[tuple[str, str]]) -> str`
  - `select_rarest(lexemes, document_frequencies, limit) -> list[str]`
  - `or_websearch_text(lexemes) -> str`
  - `title_weighted_text(title, content) -> str`
  - `build_manifest(...) -> dict[str, object]`
  - `validate_manifest(saved, expected) -> None`

- [ ] **Step 1: Write the failing helper tests**

Create `backend/tests/unit/test_lexical_benchmark.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run pytest backend/tests/unit/test_lexical_benchmark.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation.lexical_benchmark'`.

- [ ] **Step 3: Implement the helpers**

Create `backend/app/evaluation/lexical_benchmark.py`:

```python
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
    return ordered[:limit] if limit >= 0 else ordered


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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run pytest backend/tests/unit/test_lexical_benchmark.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/barely_ok/productionRAG
git add backend/app/evaluation/lexical_benchmark.py backend/tests/unit/test_lexical_benchmark.py
git commit -m "feat: add lexical benchmark helpers"
```

---

### Task 2: Add BM25 index loading and a retriever adapter

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `backend/app/evaluation/lexical_benchmark.py`
- Modify: `backend/tests/unit/test_lexical_benchmark.py`

**Interfaces:**
- Consumes: `BenchmarkChunk`, `build_manifest`, `validate_manifest`, `BM25_*` constants.
- Produces:
  - `BM25SpikeRetriever(bm25, chunks, *, stemmer=None)`
  - `load_or_build_bm25_index(*, index_dir, chunks, title_weighting, stemmer, expected_manifest, rebuild=False) -> tuple[Any, float, float, int]`

- [ ] **Step 1: Add the development dependencies**

Run:

```bash
cd /home/barely_ok/productionRAG
uv add --dev bm25s PyStemmer
uv lock
uv sync --group dev
```

Expected: `pyproject.toml` has `bm25s` and `PyStemmer` in `[dependency-groups].dev`; `uv.lock` updates.

- [ ] **Step 2: Write the failing BM25 tests**

Add these imports to the existing top-of-file import block in `backend/tests/unit/test_lexical_benchmark.py`, then append the tests:

```python
import importlib.metadata

import bm25s
import Stemmer

from app.evaluation.lexical_benchmark import (
    BM25SpikeRetriever,
    BenchmarkChunk,
    load_or_build_bm25_index,
)


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
    _, load_seconds, loaded_size, _ = load_or_build_bm25_index(
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
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run pytest backend/tests/unit/test_lexical_benchmark.py -q
```

Expected: FAIL because `BM25SpikeRetriever` and `load_or_build_bm25_index` are missing.

- [ ] **Step 4: Implement BM25 indexing and retrieval**

Add these imports to the existing top-of-file import block in `backend/app/evaluation/lexical_benchmark.py`, then append the implementation:

```python
import json
import shutil
import time
from pathlib import Path
from typing import Any

import bm25s
import Stemmer

from app.retrieval.types import RetrievedChunk


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
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run pytest backend/tests/unit/test_lexical_benchmark.py -q
```

Expected: all tests pass, including index persistence and manifest mismatch.

- [ ] **Step 6: Commit**

```bash
cd /home/barely_ok/productionRAG
git add pyproject.toml uv.lock backend/app/evaluation/lexical_benchmark.py backend/tests/unit/test_lexical_benchmark.py
git commit -m "feat: add bm25 benchmark retriever"
```

---

### Task 3: Add read-only PostgreSQL benchmark retrievers

**Files:**
- Create: `backend/app/evaluation/lexical_postgres.py`
- Create: `backend/tests/integration/test_lexical_postgres_benchmark.py`

**Interfaces:**
- Consumes: `BenchmarkChunk`, `or_websearch_text`, `select_rarest`.
- Produces:
  - `PostgresBenchmarkConfig(tenant_id, knowledge_base_id, pg_timeout_s=10.0)`
  - `load_corpus_counts(connection, config) -> tuple[int, int]`
  - `load_chunks(connection, config) -> list[BenchmarkChunk]`
  - `query_lexemes(connection, query) -> list[str]`
  - `load_document_frequencies(connection, config) -> dict[str, int]`
  - `PostgresLexicalRetriever(connection, chunks_by_id, *, config, document_frequencies, mode, n_threshold=None)`
  - `mode` values: `and`, `and_or_8`, `relaxed_and_3`, `relaxed_and_5`

- [ ] **Step 1: Write the failing integration test**

Create `backend/tests/integration/test_lexical_postgres_benchmark.py`:

```python
from __future__ import annotations

import asyncpg
import pytest

from app.evaluation.lexical_postgres import (
    PostgresBenchmarkConfig,
    PostgresLexicalRetriever,
    load_chunks,
    load_corpus_counts,
    load_document_frequencies,
    query_lexemes,
)
from app.settings import get_settings

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_postgres_benchmark_queries_are_read_only() -> None:
    settings = get_settings()
    dsn = settings.database_url.replace("+asyncpg", "")
    try:
        connection = await asyncpg.connect(dsn)
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"PostgreSQL is unavailable: {exc}")

    try:
        config = PostgresBenchmarkConfig(
            tenant_id=settings.default_tenant_id,
            knowledge_base_id=settings.default_knowledge_base_id,
        )
        documents, chunks = await load_corpus_counts(connection, config)
        if documents == 0 or chunks == 0:
            pytest.skip("benchmark corpus is not loaded")

        loaded = await load_chunks(connection, config)
        assert len(loaded) == chunks

        frequencies = await load_document_frequencies(connection, config)
        assert frequencies

        lexemes = await query_lexemes(connection, "How do I rotate an API key?")
        assert lexemes

        by_id = {chunk.chunk_id: chunk for chunk in loaded}
        for mode, kwargs in (
            ("and", {}),
            ("and_or_8", {}),
            ("relaxed_and_3", {"n_threshold": 3}),
            ("relaxed_and_5", {"n_threshold": 5}),
        ):
            retriever = PostgresLexicalRetriever(
                connection,
                by_id,
                config=config,
                document_frequencies=frequencies,
                mode=mode,
                **kwargs,
            )
            results = await retriever.retrieve("API key", top_k=5)
            assert len(results) <= 5
    finally:
        await connection.close()
```

- [ ] **Step 2: Run the integration test and confirm it fails**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run pytest backend/tests/integration/test_lexical_postgres_benchmark.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation.lexical_postgres'`.

- [ ] **Step 3: Implement the PostgreSQL benchmark module**

Create `backend/app/evaluation/lexical_postgres.py`:

```python
"""Read-only PostgreSQL lexical benchmark variants."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import asyncpg

from app.evaluation.lexical_benchmark import (
    BenchmarkChunk,
    select_rarest,
)
from app.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class PostgresBenchmarkConfig:
    tenant_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    pg_timeout_s: float = 10.0


async def load_corpus_counts(
    connection: asyncpg.Connection,
    config: PostgresBenchmarkConfig,
) -> tuple[int, int]:
    """Return document and chunk counts for the benchmark tenant."""

    document_count = await connection.fetchval(
        "SELECT count(*) FROM documents WHERE tenant_id = $1 AND knowledge_base_id = $2",
        config.tenant_id,
        config.knowledge_base_id,
    )
    chunk_count = await connection.fetchval(
        "SELECT count(*) FROM chunks WHERE tenant_id = $1 AND knowledge_base_id = $2",
        config.tenant_id,
        config.knowledge_base_id,
    )
    return int(document_count or 0), int(chunk_count or 0)


async def load_chunks(
    connection: asyncpg.Connection,
    config: PostgresBenchmarkConfig,
) -> list[BenchmarkChunk]:
    """Load all benchmark chunks in deterministic order."""

    rows = await connection.fetch(
        """
        SELECT chunk_id, doc_id, source_type, title, chunk_index, content, content_hash
        FROM chunks
        WHERE tenant_id = $1 AND knowledge_base_id = $2
        ORDER BY chunk_id
        """,
        config.tenant_id,
        config.knowledge_base_id,
    )
    return [
        BenchmarkChunk(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            source_type=str(row["source_type"]),
            title=str(row["title"] or ""),
            chunk_index=int(row["chunk_index"]),
            content=str(row["content"] or ""),
            content_hash=str(row["content_hash"] or ""),
        )
        for row in rows
    ]


async def query_lexemes(
    connection: asyncpg.Connection,
    query: str,
) -> list[str]:
    """Return normalized lexemes for a natural-language query."""

    rows = await connection.fetch(
        """
        SELECT DISTINCT lexeme
        FROM unnest(
            tsvector_to_array(websearch_to_tsquery('english', $1))
        ) AS lexeme
        ORDER BY lexeme
        """,
        query,
    )
    return [str(row["lexeme"]) for row in rows if str(row["lexeme"]).strip()]


async def load_document_frequencies(
    connection: asyncpg.Connection,
    config: PostgresBenchmarkConfig,
) -> dict[str, int]:
    """Return lexical document frequency for the benchmark tenant."""

    sql = (
        "SELECT word, ndoc FROM ts_stat("
        "'SELECT search_vector FROM chunks "
        f"WHERE tenant_id = ''{config.tenant_id}'' "
        f"AND knowledge_base_id = ''{config.knowledge_base_id}'' "
        "AND search_vector IS NOT NULL');"
    )
    rows = await connection.fetch(sql)
    return {str(row["word"]): int(row["ndoc"]) for row in rows}


def _and_text(lexemes: Sequence[str]) -> str:
    return " & ".join(lexemes)


def _or_text_from_lexemes(lexemes: Sequence[str]) -> str:
    return f"({' | '.join(lexemes)})"


def _build_queries(
    mode: str,
    query: str,
    lexemes: Sequence[str],
    document_frequencies: Mapping[str, int],
    n_threshold: int | None,
    source_types: Sequence[str] | None,
    top_k: int,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
) -> tuple[str, list[object]]:
    """Build SQL and parameters for one benchmark mode."""

    source_filter = ""
    params: list[object] = [None, top_k, tenant_id, knowledge_base_id]
    if source_types:
        source_filter = " AND source_type = ANY($5::text[])"
        params.append(list(source_types))

    if mode == "and":
        params[0] = query
        return (
            f"""
            SELECT chunk_id, doc_id, source_type, title, chunk_index,
                   content, doc_metadata,
                   ts_rank_cd(search_vector, query, 32) AS score
            FROM chunks, websearch_to_tsquery('english', $1) AS query
            WHERE tenant_id = $3
              AND knowledge_base_id = $4
              AND search_vector @@ query
              {source_filter}
            ORDER BY score DESC, chunk_id ASC
            LIMIT $2
            """,
            params,
        )

    if mode == "and_or_8":
        if not lexemes:
            return "", []
        or_group = select_rarest(lexemes, document_frequencies, limit=8)
        or_set = set(or_group)
        and_group = [lexeme for lexeme in lexemes if lexeme not in or_set]
        query_text = _or_text_from_lexemes(or_group)
        if and_group:
            query_text = f"{query_text} & ({_and_text(and_group)})"
    elif mode in {"relaxed_and_3", "relaxed_and_5"}:
        if n_threshold is None or n_threshold <= 0:
            raise ValueError("n_threshold must be positive")
        survivors = [
            lexeme
            for lexeme in lexemes
            if document_frequencies.get(lexeme, 0) >= n_threshold
        ]
        if not survivors:
            return "", []
        or_group = select_rarest(survivors, document_frequencies, limit=8)
        or_set = set(or_group)
        and_group = [lexeme for lexeme in survivors if lexeme not in or_set]
        query_text = _or_text_from_lexemes(or_group)
        if and_group:
            query_text = f"{query_text} & ({_and_text(and_group)})"
    else:
        raise ValueError(f"unsupported PostgreSQL benchmark mode: {mode}")

    params[0] = query_text
    return (
        f"""
        SELECT chunk_id, doc_id, source_type, title, chunk_index,
               content, doc_metadata,
               ts_rank_cd(search_vector, to_tsquery('english', $1), 32) AS score
        FROM chunks
        WHERE tenant_id = $3
          AND knowledge_base_id = $4
          AND search_vector @@ to_tsquery('english', $1)
          {source_filter}
        ORDER BY score DESC, chunk_id ASC
        LIMIT $2
        """,
        params,
    )


class PostgresLexicalRetriever:
    """Run one read-only PostgreSQL lexical benchmark variant."""

    def __init__(
        self,
        connection: asyncpg.Connection,
        chunks_by_id: Mapping[str, BenchmarkChunk],
        *,
        config: PostgresBenchmarkConfig,
        document_frequencies: Mapping[str, int],
        mode: str,
        n_threshold: int | None = None,
    ) -> None:
        self.connection = connection
        self.chunks_by_id = chunks_by_id
        self.config = config
        self.document_frequencies = document_frequencies
        self.mode = mode
        self.n_threshold = n_threshold
        self.timeouts = 0

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        lexemes = await query_lexemes(self.connection, query)
        sql, params = _build_queries(
            self.mode,
            query,
            lexemes,
            self.document_frequencies,
            self.n_threshold,
            source_types,
            top_k,
            self.config.tenant_id,
            self.config.knowledge_base_id,
        )
        if not sql:
            return []

        try:
            async with asyncio.timeout(self.config.pg_timeout_s):
                rows = await self.connection.fetch(sql, *params)
        except TimeoutError:
            self.timeouts += 1
            return []

        results: list[RetrievedChunk] = []
        for row in rows:
            chunk_id = str(row["chunk_id"])
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=float(row["score"]),
                    metadata={},
                )
            )
        return results
```

- [ ] **Step 4: Run the integration test and confirm it passes**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run pytest backend/tests/integration/test_lexical_postgres_benchmark.py -q
```

Expected: PASS. If the database is absent, the test skips.

- [ ] **Step 5: Commit**

```bash
cd /home/barely_ok/productionRAG
git add backend/app/evaluation/lexical_postgres.py backend/tests/integration/test_lexical_postgres_benchmark.py
git commit -m "feat: add postgres lexical benchmark variants"
```

---

### Task 4: Add timing, latency, and variant selection helpers

**Files:**
- Modify: `backend/app/evaluation/lexical_benchmark.py`
- Modify: `backend/tests/unit/test_lexical_benchmark.py`

**Interfaces:**
- Consumes: `evaluate_retriever`, `RetrievedChunk`, `DevQuestion`.
- Produces:
  - `TimingRetriever(inner)`
  - `measure_top_k_latency(retriever, questions, *, top_k) -> dict[str, float | int]`
  - `evaluate_variant(name, retriever, questions, *, top_k) -> dict[str, object]`
  - `select_best_bm25_variant(results) -> str`

- [ ] **Step 1: Write the failing timing tests**

Add these imports to the existing top-of-file import block in `backend/tests/unit/test_lexical_benchmark.py`, then append the tests:

```python
from app.evaluation.dev import DevQuestion
from app.evaluation.lexical_benchmark import (
    TimingRetriever,
    evaluate_variant,
    measure_top_k_latency,
    select_best_bm25_variant,
)
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types=None,
    ) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        if query == "empty":
            return []
        return [
            RetrievedChunk(
                chunk_id="c1",
                doc_id="d1",
                source_type="github",
                title="Title",
                chunk_index=0,
                content="content",
                score=1.0,
                metadata={},
            )
        ]


@pytest.mark.asyncio
async def test_evaluate_variant_records_latency_and_zero_results() -> None:
    questions = [
        DevQuestion("q1", "alpha", ["d1"]),
        DevQuestion("q2", "empty", ["d2"]),
    ]
    inner = FakeRetriever()
    retriever = TimingRetriever(inner)

    result = await evaluate_variant("fake", retriever, questions, top_k=5)

    assert result["metrics"]["evaluated_questions"] == 2
    assert result["zero_result_questions"] == 1
    assert result["latency"]["count"] == 2
    assert result["timeouts"] == 0
    assert len(result["per_question"]) == 2


@pytest.mark.asyncio
async def test_measure_top_k_latency_uses_requested_k() -> None:
    questions = [DevQuestion("q1", "alpha", ["d1"])]
    inner = FakeRetriever()

    result = await measure_top_k_latency(inner, questions, top_k=50)

    assert result["count"] == 1
    assert inner.calls == [("alpha", 50)]


def test_select_best_bm25_variant_prefers_recall_then_mrr() -> None:
    results = {
        "bm25_content": {
            "metrics": {"recall_at_20": 0.20, "mrr": 0.10},
        },
        "bm25_title_weighted": {
            "metrics": {"recall_at_20": 0.20, "mrr": 0.11},
        },
    }

    assert select_best_bm25_variant(results) == "bm25_title_weighted"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run pytest backend/tests/unit/test_lexical_benchmark.py -q
```

Expected: FAIL because the new helpers are missing.

- [ ] **Step 3: Implement the timing and selection helpers**

Add these imports to the existing top-of-file import block in `backend/app/evaluation/lexical_benchmark.py`, then append the implementation. The `RetrievedChunk` import is already present from Task 2:

```python
from app.evaluation.dev import DevQuestion, evaluate_retriever


class TimingRetriever:
    """Record per-query durations around another retriever."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.durations_ms: list[float] = []

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        started = time.perf_counter()
        try:
            return await self.inner.retrieve(
                query,
                top_k=top_k,
                source_types=source_types,
            )
        finally:
            self.durations_ms.append((time.perf_counter() - started) * 1000.0)


async def measure_top_k_latency(
    retriever: Any,
    questions: Sequence[DevQuestion],
    *,
    top_k: int,
) -> dict[str, float | int]:
    """Measure latency at a specific top-k without computing metrics."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    durations_ms: list[float] = []
    for question in questions:
        started = time.perf_counter()
        await retriever.retrieve(question.question, top_k=top_k)
        durations_ms.append((time.perf_counter() - started) * 1000.0)
    return latency_summary(durations_ms)


async def evaluate_variant(
    name: str,
    retriever: Any,
    questions: Sequence[DevQuestion],
    *,
    top_k: int,
) -> dict[str, object]:
    """Evaluate one variant and attach latency and zero-result counts."""

    timed = TimingRetriever(retriever)
    report = await evaluate_retriever(
        timed,
        questions,
        top_k=top_k,
        label=name,
    )
    per_question = report["per_question"]
    zero_results = sum(
        1 for item in per_question if not item["retrieved_doc_ids"]  # type: ignore[index]
    )
    return {
        **report,
        "latency": latency_summary(timed.durations_ms),
        "zero_result_questions": zero_results,
        "timeouts": int(getattr(retriever, "timeouts", 0)),
    }


def _metric(result: Mapping[str, object], name: str) -> float:
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return 0.0
    return float(metrics.get(name, 0.0))


def select_best_bm25_variant(results: Mapping[str, Mapping[str, object]]) -> str:
    """Pick the BM25 variant with the best recall@20, then MRR."""

    candidates = ("bm25_content", "bm25_title_weighted")
    available = [name for name in candidates if name in results]
    if not available:
        raise ValueError("no BM25 variants were evaluated")

    return max(
        available,
        key=lambda name: (
            _metric(results[name], "recall_at_20"),
            _metric(results[name], "mrr"),
            name == "bm25_title_weighted",
        ),
    )
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run pytest backend/tests/unit/test_lexical_benchmark.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/barely_ok/productionRAG
git add backend/app/evaluation/lexical_benchmark.py backend/tests/unit/test_lexical_benchmark.py
git commit -m "feat: add benchmark latency accounting"
```

---

### Task 5: Add the benchmark CLI

**Files:**
- Create: `scripts/benchmark_lexical.py`

**Interfaces:**
- Consumes: all benchmark helpers, PostgreSQL variants, existing dense and hybrid retrievers, existing dev questions.
- Produces: JSON artifact at `data/processed/lexical_benchmark.json` by default.

- [ ] **Step 1: Write the benchmark CLI**

Create `scripts/benchmark_lexical.py`:

```python
#!/usr/bin/env python3
"""Benchmark lexical retrieval variants on the development split."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

import asyncpg  # noqa: E402
import Stemmer  # noqa: E402

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.embeddings.bge import BGEEmbeddingProvider  # noqa: E402
from app.evaluation.dev import read_dev_questions  # noqa: E402
from app.evaluation.lexical_benchmark import (  # noqa: E402
    BM25SpikeRetriever,
    build_manifest,
    corpus_fingerprint,
    evaluate_variant,
    load_or_build_bm25_index,
    measure_top_k_latency,
    select_best_bm25_variant,
)
from app.evaluation.lexical_postgres import (  # noqa: E402
    PostgresBenchmarkConfig,
    PostgresLexicalRetriever,
    load_chunks,
    load_corpus_counts,
    load_document_frequencies,
)
from app.retrieval.dense import DenseRetriever  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.settings import (  # noqa: E402
    DEFAULT_KNOWLEDGE_BASE_ID,
    DEFAULT_TENANT_ID,
    get_settings,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _drift(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    left_by_id = {
        str(item["question_id"]): item["retrieved_doc_ids"]
        for item in left["per_question"]  # type: ignore[index]
    }
    right_by_id = {
        str(item["question_id"]): item["retrieved_doc_ids"]
        for item in right["per_question"]  # type: ignore[index]
    }
    return {
        "questions_with_different_docs": sum(
            1
            for question_id, docs in left_by_id.items()
            if docs != right_by_id.get(question_id)
        ),
        "recall_at_10_delta": round(
            float(left["metrics"]["recall_at_10"])  # type: ignore[index]
            - float(right["metrics"]["recall_at_10"]),  # type: ignore[index]
            6,
        ),
        "mrr_delta": round(
            float(left["metrics"]["mrr"]) - float(right["metrics"]["mrr"]),  # type: ignore[index]
            6,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data/sample/dev_questions.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/bm25_index",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/processed/lexical_benchmark.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model name; defaults to settings",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Embedding device; defaults to settings",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default=None,
        help="Embedding dtype; defaults to settings",
    )
    parser.add_argument("--pg-timeout", type=float, default=10.0)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild persisted BM25 indexes",
    )
    parser.add_argument("--tenant-id", type=uuid.UUID, default=DEFAULT_TENANT_ID)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=DEFAULT_KNOWLEDGE_BASE_ID,
    )
    return parser


async def run(args: argparse.Namespace) -> dict[str, object]:
    if "holdout" in args.questions.name.lower():
        raise ValueError("holdout questions are not allowed in this benchmark")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.top_k <= 0 or args.candidate_k <= 0 or args.rrf_k <= 0:
        raise ValueError("top-k, candidate-k, and rrf-k must be positive")
    if args.pg_timeout <= 0:
        raise ValueError("--pg-timeout must be positive")

    settings = get_settings()
    questions = read_dev_questions(args.questions, args.limit)
    if not questions:
        raise ValueError("no development questions were loaded")

    config = PostgresBenchmarkConfig(
        tenant_id=args.tenant_id,
        knowledge_base_id=args.knowledge_base_id,
        pg_timeout_s=args.pg_timeout,
    )
    dsn = settings.database_url.replace("+asyncpg", "")
    connection = await asyncpg.connect(dsn)
    engine = None
    try:
        documents, chunk_count = await load_corpus_counts(connection, config)
        if documents == 0 or chunk_count == 0:
            raise ValueError("benchmark corpus is empty")
        chunks = await load_chunks(connection, config)
        if len(chunks) != chunk_count:
            raise ValueError("loaded chunk count does not match the database")
        fingerprint = corpus_fingerprint(
            (chunk.chunk_id, chunk.content_hash) for chunk in chunks
        )
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        document_frequencies = await load_document_frequencies(connection, config)
        if not document_frequencies:
            raise ValueError("document frequencies are empty")
        hnsw_ef_search = int(await connection.fetchval("SHOW hnsw.ef_search"))

        provider = BGEEmbeddingProvider(
            args.model or settings.embedding_model,
            settings.embedding_dimension,
            device=args.device or settings.embedding_device,
            batch_size=settings.embedding_batch_size,
            dtype=args.dtype or settings.embedding_dtype,
            query_instruction=settings.embedding_query_instruction,
        )
        engine = build_engine(settings)
        session_factory = build_session_factory(engine)
        dense = DenseRetriever(
            provider,
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )

        stemmer = Stemmer.Stemmer("english")
        bm25s_version = importlib.metadata.version("bm25s")
        pystemmer_version = importlib.metadata.version("PyStemmer")
        bm25_indexes: dict[str, dict[str, object]] = {}
        bm25_retrievers: dict[str, BM25SpikeRetriever] = {}
        for weighting in ("content", "title_x3"):
            manifest = build_manifest(
                index_name=weighting,
                corpus_fingerprint=fingerprint,
                documents=documents,
                chunks=chunk_count,
                title_weighting=weighting,
                bm25s_version=bm25s_version,
                pystemmer_version=pystemmer_version,
            )
            index, build_seconds, load_seconds, size_bytes = load_or_build_bm25_index(
                index_dir=args.index_dir / weighting,
                chunks=chunks,
                title_weighting=weighting,
                stemmer=stemmer,
                expected_manifest=manifest,
                rebuild=args.rebuild_index,
            )
            bm25_indexes[weighting] = {
                "build_seconds": round(build_seconds, 3),
                "load_seconds": round(load_seconds, 3),
                "size_bytes": size_bytes,
                "manifest": manifest,
            }
            bm25_retrievers[
                "bm25_content" if weighting == "content" else "bm25_title_weighted"
            ] = BM25SpikeRetriever(index, chunks, stemmer=stemmer)

        pg_retrievers = {
            "pg_and": PostgresLexicalRetriever(
                connection,
                chunks_by_id,
                config=config,
                document_frequencies=document_frequencies,
                mode="and",
            ),
            "pg_and_or": PostgresLexicalRetriever(
                connection,
                chunks_by_id,
                config=config,
                document_frequencies=document_frequencies,
                mode="and_or_8",
            ),
            "pg_relaxed3": PostgresLexicalRetriever(
                connection,
                chunks_by_id,
                config=config,
                document_frequencies=document_frequencies,
                mode="relaxed_and_3",
                n_threshold=3,
            ),
            "pg_relaxed5": PostgresLexicalRetriever(
                connection,
                chunks_by_id,
                config=config,
                document_frequencies=document_frequencies,
                mode="relaxed_and_5",
                n_threshold=5,
            ),
        }

        results: dict[str, dict[str, object]] = {}
        results["dense_a"] = await evaluate_variant(
            "dense_a",
            dense,
            questions,
            top_k=args.top_k,
        )
        results["dense_b"] = await evaluate_variant(
            "dense_b",
            dense,
            questions,
            top_k=args.top_k,
        )
        for name, retriever in pg_retrievers.items():
            results[name] = await evaluate_variant(
                name,
                retriever,
                questions,
                top_k=args.top_k,
            )
        for name, retriever in bm25_retrievers.items():
            results[name] = await evaluate_variant(
                name,
                retriever,
                questions,
                top_k=args.top_k,
            )
            results[name]["latency_top50"] = await measure_top_k_latency(
                retriever,
                questions,
                top_k=50,
            )

        best_bm25 = select_best_bm25_variant(results)
        hybrid = HybridRetriever(
            dense,
            bm25_retrievers[best_bm25],
            rrf_k=args.rrf_k,
            candidate_k=args.candidate_k,
        )
        results["hybrid_bm25"] = await evaluate_variant(
            "hybrid_bm25",
            hybrid,
            questions,
            top_k=args.top_k,
        )

        output: dict[str, object] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "commit": _git_commit(),
            "questions": {
                "path": str(args.questions),
                "count": len(questions),
            },
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "ram_bytes": _ram_bytes(),
                "device": provider.device,
                "dtype": provider.dtype,
            },
            "corpus": {
                "documents": documents,
                "chunks": chunk_count,
                "fingerprint": fingerprint,
                "document_frequencies": len(document_frequencies),
            },
            "settings": {
                "model": provider.model_name,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "rrf_k": args.rrf_k,
                "pg_timeout_s": args.pg_timeout,
                "hnsw_ef_search": hnsw_ef_search,
            },
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "method": "lucene",
                "stopwords": "en",
                "stemmer": "PyStemmer english",
                "indexes": bm25_indexes,
                "best_variant": best_bm25,
            },
            "dense_drift": _drift(results["dense_a"], results["dense_b"]),
            "variants": results,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(
            f"{'variant':<22}{'r@10':>8}{'r@20':>8}{'mrr':>8}"
            f"{'p95ms':>10}{'zero':>7}{'timeouts':>9}",
            flush=True,
        )
        for name, result in results.items():
            metrics = result["metrics"]
            latency = result["latency"]
            print(
                f"{name:<22}{float(metrics['recall_at_10']):>8.4f}"
                f"{float(metrics['recall_at_20']):>8.4f}"
                f"{float(metrics['mrr']):>8.4f}"
                f"{float(latency['p95_ms']):>10.2f}"
                f"{int(result['zero_result_questions']):>7}"
                f"{int(result['timeouts']):>9}",
                flush=True,
            )
        return output
    finally:
        if engine is not None:
            await engine.dispose()
        await connection.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (OSError, ValueError, RuntimeError, asyncpg.PostgresError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the CLI loads**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run python scripts/benchmark_lexical.py --help
```

Expected: argparse help prints without import errors.

- [ ] **Step 3: Run a two-question smoke benchmark**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run python scripts/benchmark_lexical.py \
  --limit 2 \
  --output /tmp/opencode/lexical_smoke.json \
  --device cuda \
  --dtype float16
```

Expected: the first run builds two BM25 indexes under `data/processed/bm25_index/`; later runs load them. The JSON contains `dense_a`, `dense_b`, `pg_and`, `pg_and_or`, `pg_relaxed3`, `pg_relaxed5`, `bm25_content`, `bm25_title_weighted`, and `hybrid_bm25`.

- [ ] **Step 4: Inspect the smoke output**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run python -c "import json; d=json.load(open('/tmp/opencode/lexical_smoke.json')); print(d['generated_at']); print(sorted(d['variants'])); print(d['variants']['bm25_content']['metrics']); print(d['variants']['bm25_content']['latency_top50'])"
```

Expected: all variant names are present and BM25 metrics and top-50 latency are populated.

- [ ] **Step 5: Commit**

```bash
cd /home/barely_ok/productionRAG
git add scripts/benchmark_lexical.py
git commit -m "feat: add lexical benchmark cli"
```

---

### Task 6: Run the full benchmark and write the decision report

**Files:**
- Create: `docs/evaluation/lexical-benchmark.md`
- Modify: existing code only if the full run exposes a bug.

**Interfaces:**
- Consumes: `data/processed/lexical_benchmark.json`.
- Produces: a decision report with gate checklist and recommendation.

- [ ] **Step 1: Confirm the database is clean before the run**

Run:

```bash
cd /home/barely_ok/productionRAG
docker exec production-rag-db psql -U rag -d rag -tAc "SELECT (SELECT count(*) FROM documents) || '|' || (SELECT count(*) FROM chunks) || '|' || (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL);"
```

Expected: `34378|136383|136383`.

- [ ] **Step 2: Run all 169 dev questions**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run python scripts/benchmark_lexical.py \
  --device cuda \
  --dtype float16 \
  --output data/processed/lexical_benchmark.json
```

Expected: command completes successfully. PostgreSQL OR/relaxed variants may take several minutes. Timeouts are recorded as zero-result failures.

- [ ] **Step 3: Inspect the full results**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run python - <<'PY'
import json

data = json.load(open("data/processed/lexical_benchmark.json"))
print("questions:", data["questions"]["count"])
print("corpus:", data["corpus"]["documents"], data["corpus"]["chunks"])
print("dense drift:", data["dense_drift"])
print("best BM25:", data["bm25"]["best_variant"])
for name, result in data["variants"].items():
    metrics = result["metrics"]
    latency = result["latency"]
    top50 = result.get("latency_top50")
    print(
        name,
        "r@5", round(metrics["recall_at_5"], 4),
        "r@10", round(metrics["recall_at_10"], 4),
        "r@20", round(metrics["recall_at_20"], 4),
        "mrr", round(metrics["mrr"], 4),
        "p95", round(latency["p95_ms"], 2),
        "top50_p95", round(top50["p95_ms"], 2) if top50 else None,
        "zero", result["zero_result_questions"],
        "timeouts", result["timeouts"],
    )
PY
```

Expected: exactly 169 questions, 34,378 documents, 136,383 chunks, and all variants present.

- [ ] **Step 4: Evaluate the gates**

Use these checks against the JSON:

```text
Gate A: BM25 lexical recall@20 >= 0.15
Gate B: best BM25 zero-result questions <= 20/169
Gate C: best BM25 top-50 p95 < 200 ms
Gate D: hybrid recall@10 >= dense_a recall@10 + 0.01
        OR hybrid MRR >= dense_a MRR + 0.01
Gate E: hybrid recall@5 >= dense_a recall@5 - 0.01
Gate F: hybrid recall@20 >= dense_a recall@20 - 0.01
Gate G: hybrid other headline metric >= dense_a other headline metric - 0.01
Gate H: dense drift recall@10 and MRR each <= 0.01
```

`Gate G` means the headline metric not used by `Gate D`. If `Gate D` passes through recall@10, compare MRR; if it passes through MRR, compare recall@10.

- [ ] **Step 5: Write the report**

Create `docs/evaluation/lexical-benchmark.md` with these sections and the actual measured values:

```markdown
# Lexical Retrieval Benchmark (Block 3.5)

Date: <YYYY-MM-DD>
Commit: <full commit SHA>
Status: Benchmark-only spike; production retrieval unchanged.

## Executive Summary

<State whether BM25 passed or failed the lexical gate, whether the BM25 hybrid passed or failed the quality gate, and whether the latency gate passed. State the recommended next action.>

## Setup

- Questions: 169 development questions.
- Holdout: not used.
- Corpus: 34,378 documents; 136,383 chunks.
- BM25: `bm25s`, `k1=1.5`, `b=0.75`, method `lucene`, English stopwords, PyStemmer English.
- Title weighting: title tokens repeated three times.
- Dense baseline: same-run `dense_a` and `dense_b`; report drift.
- Runtime: <GPU/CPU, dtype, top-k, candidate-k, RRF k, PostgreSQL timeout>.

## Lexical Variant Results

| Variant | Recall@5 | Recall@10 | Recall@20 | MRR | Zero-result questions | Timeouts | p95 ms | p95 top-50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pg_and | ... | ... | ... | ... | ... | ... | ... | n/a |
| pg_and_or | ... | ... | ... | ... | ... | ... | ... | n/a |
| pg_relaxed3 | ... | ... | ... | ... | ... | ... | ... | n/a |
| pg_relaxed5 | ... | ... | ... | ... | ... | ... | ... | n/a |
| bm25_content | ... | ... | ... | ... | ... | ... | ... | ... |
| bm25_title_weighted | ... | ... | ... | ... | ... | ... | ... | ... |

## Hybrid Results

| Variant | Recall@5 | Recall@10 | Recall@20 | MRR | p95 ms | Zero-result questions | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_a | ... | ... | ... | ... | ... | ... | ... |
| dense_b | ... | ... | ... | ... | ... | ... | ... |
| hybrid_bm25 | ... | ... | ... | ... | ... | ... | ... |

Dense drift: <actual values>.

## Gate Evaluation

| Gate | Requirement | Measured | Result |
|---|---|---|---|
| A | BM25 recall@20 >= 0.15 | ... | pass/fail |
| B | BM25 zero-result questions <= 20 | ... | pass/fail |
| C | BM25 top-50 p95 < 200 ms | ... | pass/fail |
| D | Hybrid +0.01 on recall@10 or MRR | ... | pass/fail |
| E | Hybrid recall@5 no worse than dense by 0.01 | ... | pass/fail |
| F | Hybrid recall@20 no worse than dense by 0.01 | ... | pass/fail |
| G | Hybrid other headline metric no worse by 0.01 | ... | pass/fail |
| H | Dense drift <= 0.01 on recall@10 and MRR | ... | pass/fail |

## Recommendation

<If BM25 and hybrid gates pass, recommend a separate Block 4 implementation plan. If only BM25 passes, recommend investigating hybrid fusion before adopting BM25. If BM25 fails, retain PostgreSQL FTS and record why. Do not change production defaults in this block.>

## Reproduction

```bash
uv sync --group dev
uv run python scripts/benchmark_lexical.py --device cuda --dtype float16 --output data/processed/lexical_benchmark.json
```

## Limitations

- Results are for one tenant and one corpus.
- Exact BM25 has no synonym or semantic expansion.
- PostgreSQL OR variants can time out and are measured as zero-result failures.
- BM25 index memory and build time are reported but not optimized.
- This spike does not implement reranking, learned sparse retrieval, or production changes.
```

Fill every `<...>` placeholder with measured values. Do not write generic conclusions that the JSON does not support.

- [ ] **Step 6: Run the full test suite**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run pytest -q
```

Expected: all tests pass. The new PostgreSQL integration test may take longer because it loads the full corpus.

- [ ] **Step 7: Run lint and formatting**

Run:

```bash
cd /home/barely_ok/productionRAG
uv run ruff check .
uv run ruff format --check .
```

Expected: both commands pass. If formatting fails, run `uv run ruff format .` and re-run the check.

- [ ] **Step 8: Confirm the database stayed clean**

Run:

```bash
cd /home/barely_ok/productionRAG
docker exec production-rag-db psql -U rag -d rag -tAc "SELECT (SELECT count(*) FROM documents) || '|' || (SELECT count(*) FROM chunks) || '|' || (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL);"
```

Expected: `34378|136383|136383`.

- [ ] **Step 9: Commit the benchmark implementation and report**

```bash
cd /home/barely_ok/productionRAG
git status --short
git add backend/app/evaluation backend/tests scripts/benchmark_lexical.py docs/evaluation/lexical-benchmark.md pyproject.toml uv.lock
git commit -m "feat: benchmark lexical retrieval variants"
```

Expected: one commit containing only benchmark code, tests, dev dependencies, and the report. Do not add `data/`, index artifacts, or local JSON outputs.

- [ ] **Step 10: Stop for user review**

Report the gate results and recommendation. Do not modify production retrieval defaults or start Block 4 without explicit user approval.

---

## Self-Review

- Spec coverage:
  - Persisted exact BM25 index with title weighting: Task 2.
  - PostgreSQL AND, OR fallback, relaxed AND variants: Task 3.
  - All approved variants and same-run dense twice: Task 5.
  - Latency, zero-result, timeout, manifest, and drift reporting: Tasks 4 and 5.
  - Decision gates and benchmark-only stopping point: Task 6.
- Placeholder scan: the only placeholders are inside the report template in Task 6, where actual measured values must be filled from the JSON.
- Type consistency: `BenchmarkChunk` fields are the same in the loader, BM25 adapter, and PostgreSQL adapter. `evaluate_variant` returns `metrics`, `per_question`, `latency`, `zero_result_questions`, and `timeouts`, matching the CLI and report.
- No production retrieval changes are included.
