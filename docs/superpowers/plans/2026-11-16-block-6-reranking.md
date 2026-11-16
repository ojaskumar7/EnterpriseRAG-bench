# Block 6 Cross-Encoder Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rerank BM25 top-50 candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2`, preserve both scores, and measure retrieval/context quality.

**Architecture:** Add a `Reranker` protocol, a lazy `CrossEncoderReranker`, and a `RerankedRetriever` decorator. BM25 stays unchanged. CLIs/evaluators opt into reranking and fall back to BM25 if the development gate fails.

**Tech Stack:** Python 3.12, `sentence-transformers` CrossEncoder, existing BM25 retriever, pytest, Ruff.

## Global Constraints

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Candidate depth: `50`; default returned depth: `50`.
- Pair text: `title + "\n" + content`.
- Sorting: `(-rerank_score, chunk_id)`.
- `RetrievedChunk.score` is the final score; `retrieval_score` preserves BM25.
- No new dependencies.
- No silent fallback on reranker failure.
- Holdout untouched.
- Acceptance gate: MRR > `0.6771` or recall@5 > `0.7310`; recall@20 >= `0.8394`; context recall within `0.01`; budget adherence `1.0`.
- Deliberate shortcuts go in `docs/deferred-work.md`.

---

### Task 1: Add Reranker Configuration and Retrieval Score

**Files:**
- Modify: `backend/app/retrieval/types.py`
- Modify: `backend/app/settings.py`
- Modify: `.env.example`
- Test: `backend/tests/unit/test_settings.py`
- Test: `backend/tests/unit/test_reranked_retriever.py`

**Interfaces:**
- Produces: `RetrievedChunk.retrieval_score: float | None = None`.
- Produces: `Settings.reranker_model`, `reranker_device`, `reranker_batch_size`, `reranker_max_length`, `reranker_candidate_k`.

- [ ] **Step 1: Add failing settings test**

Append to `backend/tests/unit/test_settings.py`:

```python
def test_reranker_settings_defaults() -> None:
    settings = Settings()

    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.reranker_device == "auto"
    assert settings.reranker_batch_size == 32
    assert settings.reranker_max_length == 512
    assert settings.reranker_candidate_k == 50
```

Add `("RERANKER_DEVICE", "mps")`, `("RERANKER_BATCH_SIZE", "0")`, and
`("RERANKER_MAX_LENGTH", "0")` to the existing `test_invalid_values` parametrization.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest backend/tests/unit/test_settings.py -q`

Expected: FAIL because the reranker settings do not exist.

- [ ] **Step 3: Implement settings and type field**

In `backend/app/settings.py`, add after the BM25 fields:

```python
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "auto"
    reranker_batch_size: int = Field(default=32, gt=0)
    reranker_max_length: int = Field(default=512, gt=0)
    reranker_candidate_k: int = Field(default=50, gt=0)
```

Add a validator matching the embedding device validator:

```python
    @field_validator("reranker_device")
    @classmethod
    def validate_reranker_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError("reranker_device must be one of: auto, cpu, cuda")
        return normalized
```

In `backend/app/retrieval/types.py`, append:

```python
    retrieval_score: float | None = None
```

In `.env.example`, append:

```text
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DEVICE=auto
RERANKER_BATCH_SIZE=32
RERANKER_MAX_LENGTH=512
RERANKER_CANDIDATE_K=50
```

- [ ] **Step 4: Run settings tests**

Run: `uv run pytest backend/tests/unit/test_settings.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/settings.py backend/app/retrieval/types.py .env.example backend/tests/unit/test_settings.py
git commit -m "feat: add reranker settings and retrieval score"
```

---

### Task 2: Add the Reranker Protocol and Cross-Encoder

**Files:**
- Create: `backend/app/reranking/__init__.py`
- Create: `backend/app/reranking/base.py`
- Create: `backend/app/reranking/cross_encoder.py`
- Test: `backend/tests/unit/test_cross_encoder_reranker.py`

**Interfaces:**
- Consumes: `Settings` defaults; `sentence_transformers.CrossEncoder`.
- Produces: `Reranker.score_pairs(query, texts) -> list[float]`; `CrossEncoderReranker(...)`.

- [ ] **Step 1: Write failing reranker tests**

Create `backend/tests/unit/test_cross_encoder_reranker.py`:

```python
from __future__ import annotations

import pytest

from app.reranking.cross_encoder import CrossEncoderReranker


class FakePredictor:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, object]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append({"pairs": pairs, "kwargs": kwargs})
        return self.scores


def test_scores_pairs_with_injected_predictor() -> None:
    predictor = FakePredictor([0.2, 0.9])
    reranker = CrossEncoderReranker(predictor=predictor, batch_size=8)

    scores = reranker.score_pairs("query", ["a", "b"])

    assert scores == [0.2, 0.9]
    assert predictor.calls[0]["pairs"] == [("query", "a"), ("query", "b")]
    assert predictor.calls[0]["kwargs"]["batch_size"] == 8
    assert predictor.calls[0]["kwargs"]["show_progress_bar"] is False


def test_empty_texts_do_not_call_predictor() -> None:
    predictor = FakePredictor([])
    reranker = CrossEncoderReranker(predictor=predictor)

    assert reranker.score_pairs("query", []) == []
    assert predictor.calls == []


def test_rejects_invalid_settings_and_device() -> None:
    with pytest.raises(ValueError):
        CrossEncoderReranker(device="mps")
    with pytest.raises(ValueError):
        CrossEncoderReranker(batch_size=0)
    with pytest.raises(ValueError):
        CrossEncoderReranker(max_length=0)


def test_rejects_wrong_score_count_and_non_finite_scores() -> None:
    with pytest.raises(RuntimeError):
        CrossEncoderReranker(predictor=FakePredictor([0.1])).score_pairs("q", ["a", "b"])
    with pytest.raises(RuntimeError):
        CrossEncoderReranker(predictor=FakePredictor([float("nan")])).score_pairs("q", ["a"])
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest backend/tests/unit/test_cross_encoder_reranker.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.reranking'`.

- [ ] **Step 3: Implement protocol and reranker**

Create `backend/app/reranking/base.py`:

```python
"""Reranker interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]: ...
```

Create `backend/app/reranking/cross_encoder.py`:

```python
"""Local cross-encoder reranker."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from math import isfinite
from typing import Any

from app.settings import get_settings


class CrossEncoderReranker:
    """Score query-passage pairs with a local sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        predictor: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.device = self._resolve_device(device or settings.reranker_device)
        self.batch_size = (
            settings.reranker_batch_size if batch_size is None else batch_size
        )
        self.max_length = settings.reranker_max_length if max_length is None else max_length
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        self._predictor = predictor
        self._load_lock = threading.Lock()

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = device.strip().lower()
        if normalized == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if normalized == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
        return normalized

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        with self._load_lock:
            if self._predictor is None:
                from sentence_transformers import CrossEncoder

                self._predictor = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=self.max_length,
                )
        return self._predictor

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        try:
            raw = self._get_predictor().predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise RuntimeError(f"reranker {self.model_name} failed: {exc}") from exc

        values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
        if values and isinstance(values[0], list):
            values = [value[0] for value in values]
        if len(values) != len(texts):
            raise RuntimeError(
                f"reranker returned {len(values)} scores for {len(texts)} candidates"
            )
        scores = [float(value) for value in values]
        if not all(isfinite(score) for score in scores):
            raise RuntimeError("reranker returned a non-finite score")
        return scores
```

Create `backend/app/reranking/__init__.py`:

```python
"""Reranking interfaces and implementations."""

from app.reranking.base import Reranker
from app.reranking.cross_encoder import CrossEncoderReranker

__all__ = ["CrossEncoderReranker", "Reranker"]
```

- [ ] **Step 4: Run reranker tests**

Run: `uv run pytest backend/tests/unit/test_cross_encoder_reranker.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/reranking backend/tests/unit/test_cross_encoder_reranker.py
git commit -m "feat: add cross-encoder reranker"
```

---

### Task 3: Add the RerankedRetriever Decorator

**Files:**
- Create: `backend/app/retrieval/reranked.py`
- Test: `backend/tests/unit/test_reranked_retriever.py`

**Interfaces:**
- Consumes: `RetrieverProtocol`-compatible base, `Reranker`.
- Produces: `RerankedRetriever(base, reranker, candidate_k=50).retrieve(query, top_k=50, source_types=None)`.

- [ ] **Step 1: Write failing wrapper tests**

Create `backend/tests/unit/test_reranked_retriever.py`:

```python
from __future__ import annotations

import pytest

from app.retrieval.reranked import RerankedRetriever
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    async def retrieve(self, query: str, *, top_k: int = 10, source_types=None):
        self.calls.append({"query": query, "top_k": top_k, "source_types": source_types})
        return self.chunks[:top_k]


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, texts))
        return self.scores


def chunk(chunk_id: str, doc_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_type="pdf",
        title=f"Title {chunk_id}",
        chunk_index=0,
        content=f"Content {chunk_id}",
        score=score,
        metadata={},
    )


async def test_reranks_and_preserves_bm25_score() -> None:
    base = FakeRetriever([chunk("a", "d1", 10.0), chunk("b", "d2", 9.0)])
    reranker = FakeReranker([0.1, 0.9])
    retriever = RerankedRetriever(base, reranker, candidate_k=2)

    results = await retriever.retrieve("query", top_k=2, source_types=["pdf"])

    assert [result.chunk_id for result in results] == ["b", "a"]
    assert results[0].score == 0.9
    assert results[0].retrieval_score == 9.0
    assert base.calls[0]["top_k"] == 2
    assert base.calls[0]["source_types"] == ["pdf"]
    assert reranker.calls[0] == (
        "query",
        ["Title a\nContent a", "Title b\nContent b"],
    )


async def test_ties_break_by_chunk_id_and_slices_top_k() -> None:
    base = FakeRetriever([chunk("z", "d1", 1.0), chunk("a", "d2", 1.0)])
    retriever = RerankedRetriever(base, FakeReranker([0.5, 0.5]), candidate_k=2)

    results = await retriever.retrieve("query", top_k=1)

    assert [result.chunk_id for result in results] == ["a"]


async def test_empty_query_and_no_candidates_skip_model() -> None:
    base = FakeRetriever([chunk("a", "d1", 1.0)])
    reranker = FakeReranker([1.0])
    retriever = RerankedRetriever(base, reranker)

    assert await retriever.retrieve("   ") == []
    assert reranker.calls == []

    empty = RerankedRetriever(FakeRetriever([]), reranker)
    assert await empty.retrieve("query") == []
    assert reranker.calls == []


async def test_rejects_invalid_scores_and_settings() -> None:
    with pytest.raises(ValueError):
        RerankedRetriever(FakeRetriever([]), FakeReranker([]), candidate_k=0)

    retriever = RerankedRetriever(
        FakeRetriever([chunk("a", "d1", 1.0)]),
        FakeReranker([0.1, 0.2]),
        candidate_k=1,
    )
    with pytest.raises(RuntimeError):
        await retriever.retrieve("query")
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest backend/tests/unit/test_reranked_retriever.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval.reranked'`.

- [ ] **Step 3: Implement the decorator**

Create `backend/app/retrieval/reranked.py`:

```python
"""Rerank candidates from another retriever."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from math import isfinite

from app.reranking.base import Reranker
from app.retrieval.types import RetrievedChunk


class RerankedRetriever:
    """Rerank a base retriever's candidates with a cross-encoder."""

    def __init__(
        self,
        base: object,
        reranker: Reranker,
        *,
        candidate_k: int = 50,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k

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

        candidates = await self.base.retrieve(
            query,
            top_k=self.candidate_k,
            source_types=source_types,
        )
        if not candidates:
            return []

        texts = [f"{candidate.title}\n{candidate.content}" for candidate in candidates]
        scores = await asyncio.to_thread(self.reranker.score_pairs, query, texts)
        if len(scores) != len(candidates):
            raise RuntimeError(
                f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )
        if not all(isfinite(score) for score in scores):
            raise RuntimeError("reranker returned a non-finite score")

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        return [
            replace(candidate, score=score, retrieval_score=candidate.score)
            for candidate, score in ranked[:top_k]
        ]
```

- [ ] **Step 4: Run wrapper tests**

Run: `uv run pytest backend/tests/unit/test_reranked_retriever.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/retrieval/reranked.py backend/tests/unit/test_reranked_retriever.py
git commit -m "feat: add reranked retriever decorator"
```

---

### Task 4: Add Reranking CLI and Development Evaluators

**Files:**
- Create: `scripts/rerank_search.py`
- Create: `scripts/eval_reranking_dev.py`
- Modify: `scripts/eval_context_dev.py`
- Test: `backend/tests/unit/test_reranking_evaluation.py`

**Interfaces:**
- Consumes: `BM25Retriever`, `RerankedRetriever`, `CrossEncoderReranker`, `evaluate_retriever`.
- Produces: `evaluate_reranking(...) -> dict[str, object]` and `--reranked` context evaluation.

- [ ] **Step 1: Write failing comparison test**

Create `backend/tests/unit/test_reranking_evaluation.py`:

```python
from __future__ import annotations

from app.evaluation.reranking import evaluate_reranking
from app.evaluation.dev import DevQuestion
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    async def retrieve(self, query: str, **kwargs: object) -> list[RetrievedChunk]:
        del query, kwargs
        return self.results


async def test_evaluate_reranking_reports_both_modes() -> None:
    question = DevQuestion("q1", "query", ["d1"])
    chunk = RetrievedChunk(
        chunk_id="c1",
        doc_id="d1",
        source_type="pdf",
        title="t",
        chunk_index=0,
        content="content",
        score=1.0,
        metadata={},
    )
    retriever = FakeRetriever([chunk])

    report = await evaluate_reranking(
        retriever,
        retriever,
        [question],
        top_k=1,
        candidate_k=1,
    )

    assert report["bm25"]["metrics"]["recall_at_1"] == 1.0
    assert report["reranked"]["metrics"]["recall_at_1"] == 1.0
    assert report["delta"]["mrr"] == 0.0
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest backend/tests/unit/test_reranking_evaluation.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation.reranking'`.

- [ ] **Step 3: Implement comparison evaluation**

Create `backend/app/evaluation/reranking.py`:

```python
"""Compare BM25 and reranked retrieval on the development split."""

from __future__ import annotations

import time
from collections.abc import Sequence

from app.evaluation.dev import DevQuestion, RetrieverProtocol, evaluate_retriever, latency_summary


async def evaluate_reranking(
    bm25: RetrieverProtocol,
    reranked: RetrieverProtocol,
    questions: Sequence[DevQuestion],
    *,
    top_k: int = 50,
    candidate_k: int = 50,
    progress_every: int = 25,
) -> dict[str, object]:
    if not questions:
        raise ValueError("questions must not be empty")
    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("top_k and candidate_k must be positive")

    started = time.perf_counter()
    bm25_report = await evaluate_retriever(
        bm25,
        questions,
        top_k=top_k,
        progress_every=progress_every,
        label="bm25",
    )
    bm25_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    reranked_report = await evaluate_retriever(
        reranked,
        questions,
        top_k=top_k,
        progress_every=progress_every,
        label="reranked",
    )
    reranked_ms = (time.perf_counter() - started) * 1000.0

    bm25_metrics = bm25_report["metrics"]
    reranked_metrics = reranked_report["metrics"]
    delta = {
        key: float(reranked_metrics[key]) - float(bm25_metrics[key])
        for key in bm25_metrics
        if isinstance(bm25_metrics[key], int | float)
    }
    return {
        "params": {"top_k": top_k, "candidate_k": candidate_k},
        "bm25": bm25_report,
        "reranked": reranked_report,
        "delta": delta,
        "latency_ms": {
            "bm25_total": bm25_ms,
            "reranked_total": reranked_ms,
        },
    }
```

- [ ] **Step 4: Implement `scripts/eval_reranking_dev.py`**

Create a script that:
- Parses `--questions`, `--output`, `--top-k 50`, `--candidate-k 50`, `--model`, `--device`, `--batch-size`, `--max-length`, `--limit`, `--progress-every`, `--index-root`, tenant/KB.
- Builds `BM25Retriever` and `RerankedRetriever(BM25Retriever, CrossEncoderReranker(...), candidate_k=args.candidate_k)`.
- Calls `evaluate_reranking`.
- Adds `generated_at`, `retriever="reranked_bm25"`, `model`.
- Writes `data/processed/reranking_dev_metrics_block6.json` and prints the delta.
- Catches `BM25IndexError`, `OSError`, `RuntimeError`, `ValueError`.

- [ ] **Step 5: Implement `scripts/rerank_search.py`**

Create a CLI matching the spec:
- Parse query, `--candidate-k 50`, `--top-k 50`, `--model`, `--device`, `--batch-size`, `--max-length`, `--source-type`, `--index-root`, tenant/KB, `--json`.
- Build BM25 and wrap with reranker.
- Print JSON via `asdict` or text lines with rerank and BM25 scores.

- [ ] **Step 6: Extend context evaluation**

In `scripts/eval_context_dev.py`, add:
- `--reranked` flag.
- `--reranker-model`, `--reranker-device`, `--reranker-batch-size`, `--reranker-max-length`.
- When enabled, build `RerankedRetriever` around BM25 with `candidate_k=args.top_k`.
- Keep output metric schema unchanged.

- [ ] **Step 7: Run tests and CLI help**

Run: `uv run pytest backend/tests/unit/test_reranking_evaluation.py -q`

Run: `uv run python scripts/rerank_search.py --help`

Run: `uv run python scripts/eval_reranking_dev.py --help`

Expected: tests PASS; both help commands list options.

- [ ] **Step 8: Commit**

```bash
git add backend/app/evaluation/reranking.py scripts/rerank_search.py scripts/eval_reranking_dev.py scripts/eval_context_dev.py backend/tests/unit/test_reranking_evaluation.py
git commit -m "feat: add reranking evaluation and CLIs"
```

---

### Task 5: Run Real Evaluation, Apply the Gate, and Document

**Files:**
- Modify: `README.md`
- Create: `docs/architecture/reranking.md`
- Create: `docs/design-decisions/0006-cross-encoder-reranking.md`
- Modify: `plan.md`
- Modify: `blog4.md`
- Modify: `docs/deferred-work.md`

- [ ] **Step 1: Run focused and full tests**

Run:

```bash
uv run pytest backend/tests/unit/test_cross_encoder_reranker.py \
  backend/tests/unit/test_reranked_retriever.py \
  backend/tests/unit/test_reranking_evaluation.py \
  backend/tests/unit/test_settings.py -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Expected: all pass.

- [ ] **Step 2: Run the real BM25-versus-reranked evaluation**

Run:

```bash
uv run python scripts/eval_reranking_dev.py \
  --top-k 50 --candidate-k 50 \
  --index-root data/indexes/bm25 \
  --output data/processed/reranking_dev_metrics_block6.json
```

Record BM25 and reranked recall@1/5/10/20, MRR, delta, and latency.

- [ ] **Step 3: Run reranked context evaluation**

Run:

```bash
uv run python scripts/eval_context_dev.py --reranked \
  --top-k 50 --index-root data/indexes/bm25 \
  --output data/processed/context_dev_metrics_block6_reranked.json
```

Expected: budget adherence `1.0`; context recall within `0.01` of BM25 context recall.

- [ ] **Step 4: Apply the acceptance gate**

Compare against BM25: MRR `0.6771`, recall@5 `0.7310`, recall@20 `0.8494`, context
document recall `0.7744`. If the reranker passes, document it as the preferred retrieval
path. If it fails, keep BM25 as production and record the result and decision.

- [ ] **Step 5: Update docs**

- `README.md`: Block 6 status, workflow, measured results, production decision.
- `docs/architecture/reranking.md`: model, wrapper, scoring, evaluation, gate.
- `docs/design-decisions/0006-cross-encoder-reranking.md`: why decorator, why MiniLM, why
  replace ordering, optional-adoption decision.
- `plan.md`: Block 6 acceptance checklist and verification evidence.
- `blog4.md`: append the Block 6 story and measured comparison.
- `docs/deferred-work.md`: score blending, fine-tuning, caching, CPU latency, live-model
  test coverage.

- [ ] **Step 6: Commit docs**

```bash
git add README.md docs/architecture/reranking.md docs/design-decisions/0006-cross-encoder-reranking.md plan.md docs/deferred-work.md
git commit -m "docs: document block 6 reranking"
```

`blog4.md` is local-only and ignored; do not stage it.

- [ ] **Step 7: Final verification**

Run `git status --short` and `git log --oneline -8`. Confirm only `.kilo/` and `blog4.md`
remain untracked/ignored, and all Block 6 commits are present.
