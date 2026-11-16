# Block 5 Context Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert ranked BM25 chunks into a deterministic, citation-ready context under a 4,000-token budget.

**Architecture:** Add a pure `backend/app/context/` package that deduplicates, caps per document, packs whole chunks in rank order, and renders numbered citations. Add thin CLIs for context preview and development-set context metrics. Retrieval, database access, and evaluation read-only logic stay outside the builder.

**Tech Stack:** Python 3.12, dataclasses, asyncio, pytest, Ruff, existing PostgreSQL BM25 production index.

## Global Constraints

- Default token budget: `4000`; must be positive.
- Default maximum chunks per document: `3`; must be positive.
- Rank candidates by score descending, then `chunk_id` ascending.
- Whole chunks only; no neighbor merge, sentence extraction, or summarization.
- No source-type diversity cap.
- Use stored chunk `token_count` when available; otherwise `max(1, len(content.split()))`.
- No new third-party dependencies and no database migrations.
- Holdout split stays untouched.
- Context document recall may be at most `0.05` absolute below BM25 candidate document recall at the same `top_k`.
- Every behavior change gets a failing test first.
- Deliberate shortcuts go in `docs/deferred-work.md`.

---

### Task 1: Add Token Counts to Hydrated Chunks

**Files:**
- Modify: `backend/app/retrieval/types.py`
- Modify: `backend/app/db/repositories.py:62-72,270-305`
- Modify: `backend/app/retrieval/bm25.py:129-141`
- Test: `backend/tests/integration/test_database.py:294-311`

**Interfaces:**
- Consumes: existing `ChunkRecord.token_count` column.
- Produces: `ChunkLookup.token_count: int`; `RetrievedChunk.token_count: int | None = None`.

- [ ] **Step 1: Write the failing repository assertion**

In `test_get_chunks_by_ids_enforces_tenant_and_source`, after the first `rows` call, add:

```python
    assert rows[0].token_count > 0
```

- [ ] **Step 2: Run the test and verify failure**

Run: `uv run pytest backend/tests/integration/test_database.py::test_get_chunks_by_ids_enforces_tenant_and_source -v`

Expected: FAIL with `AttributeError: 'ChunkLookup' object has no attribute 'token_count'`.

- [ ] **Step 3: Add the field and query projection**

In `backend/app/db/repositories.py`, add `token_count: int` to `ChunkLookup` after `chunk_index`. Add `ChunkRecord.token_count` to the `select(...)` list and `token_count=row.token_count` to the constructor.

In `backend/app/retrieval/types.py`, append to `RetrievedChunk`:

```python
    token_count: int | None = None
```

In `backend/app/retrieval/bm25.py`, add to the `RetrievedChunk(...)` constructor:

```python
                token_count=getattr(row, "token_count", None),
```

- [ ] **Step 4: Run focused retrieval and repository tests**

Run: `uv run pytest backend/tests/integration/test_database.py::test_get_chunks_by_ids_enforces_tenant_and_source backend/tests/unit/test_bm25_retriever.py -v`

Expected: PASS. Existing fake rows without `token_count` remain compatible because BM25 uses `getattr`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/retrieval/types.py backend/app/db/repositories.py backend/app/retrieval/bm25.py backend/tests/integration/test_database.py
git commit -m "feat: hydrate chunk token counts for context packing"
```

---

### Task 2: Build the Pure Context Builder and Renderer

**Files:**
- Create: `backend/app/context/__init__.py`
- Create: `backend/app/context/types.py`
- Create: `backend/app/context/builder.py`
- Create: `backend/app/context/render.py`
- Test: `backend/tests/unit/test_context_builder.py`

**Interfaces:**
- Consumes: `Sequence[RetrievedChunk]`.
- Produces: `ContextBuilder(token_budget=4000, max_chunks_per_doc=3, token_counter=None)`, `ContextBuilder.build(query, candidates) -> BuiltContext`, `render_context(context) -> str`, `context_to_dict(context) -> dict[str, object]`.

- [ ] **Step 1: Write failing behavior tests**

Create `backend/tests/unit/test_context_builder.py` with tests covering deduplication, document caps, budget skip-and-continue, deterministic ordering, citations, empty input, fallback token count, and rendering:

```python
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
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `uv run pytest backend/tests/unit/test_context_builder.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.context'`.

- [ ] **Step 3: Implement types**

Create `backend/app/context/types.py`:

```python
"""Structured context produced from ranked retrieval candidates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBlock:
    citation_id: int
    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    chunk_index: int
    content: str
    score: float
    token_count: int


@dataclass(frozen=True)
class BuiltContext:
    query: str
    blocks: tuple[ContextBlock, ...]
    token_count: int
    token_budget: int
    candidate_count: int
    omitted_count: int
    selected_doc_count: int
    source_type_counts: dict[str, int]
```

- [ ] **Step 4: Implement builder**

Create `backend/app/context/builder.py`:

```python
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
            (
                candidate
                for candidate in candidates
                if isfinite(candidate.score) and candidate.score > 0
            ),
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
```

- [ ] **Step 5: Implement renderer**

Create `backend/app/context/render.py`:

```python
"""Render built context for previews and JSON payloads."""

from __future__ import annotations

from dataclasses import asdict

from app.context.types import BuiltContext


def render_context(context: BuiltContext) -> str:
    sections = []
    for block in context.blocks:
        header = (
            f"[{block.citation_id}] {block.title} "
            f"({block.source_type}, doc_id={block.doc_id}, chunk={block.chunk_index})"
        )
        sections.append(f"{header}\n{block.content}")
    return "\n\n".join(sections)


def context_to_dict(context: BuiltContext) -> dict[str, object]:
    payload = asdict(context)
    payload["blocks"] = [asdict(block) for block in context.blocks]
    return payload
```

Create `backend/app/context/__init__.py`:

```python
"""Evidence selection and context construction."""

from app.context.builder import ContextBuilder, estimate_tokens
from app.context.render import context_to_dict, render_context
from app.context.types import BuiltContext, ContextBlock

__all__ = [
    "BuiltContext",
    "ContextBlock",
    "ContextBuilder",
    "context_to_dict",
    "estimate_tokens",
    "render_context",
]
```

- [ ] **Step 6: Run unit tests**

Run: `uv run pytest backend/tests/unit/test_context_builder.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/context backend/tests/unit/test_context_builder.py
git commit -m "feat: add token-budgeted context builder"
```

---

### Task 3: Add Context Preview CLI

**Files:**
- Create: `scripts/build_context.py`
- Test: `backend/tests/unit/test_context_cli.py`

**Interfaces:**
- Consumes: `BM25Retriever.retrieve()`, `ContextBuilder`, `render_context`, `context_to_dict`.
- Produces: CLI `python scripts/build_context.py "query" [--top-k] [--token-budget] [--max-chunks-per-doc] [--json]`.

- [ ] **Step 1: Write CLI validation test**

Create `backend/tests/unit/test_context_cli.py`:

```python
from __future__ import annotations

from scripts.build_context import main


def test_cli_rejects_non_positive_top_k(capsys) -> None:
    assert main(["q", "--top-k", "0"]) == 1
    captured = capsys.readouterr()
    assert "top-k must be positive" in captured.err
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest backend/tests/unit/test_context_cli.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_context'`.

- [ ] **Step 3: Implement CLI**

Create `scripts/build_context.py`:

```python
#!/usr/bin/env python3
"""Retrieve BM25 candidates and build a citation-ready context."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.context import ContextBuilder, context_to_dict, render_context  # noqa: E402
from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.bm25_index import BM25IndexError, BM25IndexManager  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--max-chunks-per-doc", type=int, default=3)
    parser.add_argument("--source-type", action="append", dest="source_types")
    parser.add_argument("--index-root", type=Path, default=settings.bm25_index_root)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.token_budget <= 0:
        raise ValueError("token-budget must be positive")
    if args.max_chunks_per_doc <= 0:
        raise ValueError("max-chunks-per-doc must be positive")

    engine = build_engine(settings)
    try:
        retriever = BM25Retriever(
            BM25IndexManager(args.index_root),
            build_session_factory(engine),
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        candidates = await retriever.retrieve(
            args.query,
            top_k=args.top_k,
            source_types=args.source_types,
        )
    finally:
        await engine.dispose()

    context = ContextBuilder(
        token_budget=args.token_budget,
        max_chunks_per_doc=args.max_chunks_per_doc,
    ).build(args.query, candidates)
    payload = context_to_dict(context)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(render_context(context))
    return payload


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    args = build_parser(settings).parse_args(argv)
    try:
        asyncio.run(run(args, settings))
    except (BM25IndexError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run CLI test and help**

Run: `uv run pytest backend/tests/unit/test_context_cli.py -v`

Run: `uv run python scripts/build_context.py --help`

Expected: test PASS; help lists all options.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_context.py backend/tests/unit/test_context_cli.py
git commit -m "feat: add context preview CLI"
```

---

### Task 4: Add Context Development Evaluation

**Files:**
- Create: `backend/app/evaluation/context.py`
- Create: `scripts/eval_context_dev.py`
- Test: `backend/tests/unit/test_context_evaluation.py`

**Interfaces:**
- Consumes: any retriever with `async retrieve(query, top_k=..., source_types=...) -> list[RetrievedChunk]`; `DevQuestion` fields `question`, `expected_doc_ids`.
- Produces: `async evaluate_context(retriever, questions, top_k=50, token_budget=4000, max_chunks_per_doc=3, progress_every=25) -> dict[str, object]` with `metrics` and `per_question`.

- [ ] **Step 1: Write failing evaluation test**

Create `backend/tests/unit/test_context_evaluation.py`:

```python
from __future__ import annotations

from app.evaluation.context import evaluate_context
from app.evaluation.dev import DevQuestion
from app.retrieval.types import RetrievedChunk


class FakeRetriever:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results

    async def retrieve(self, query: str, **kwargs: object) -> list[RetrievedChunk]:
        del query, kwargs
        return self.results


def make_chunk(chunk_id: str, doc_id: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_type="pdf",
        title=chunk_id,
        chunk_index=0,
        content="alpha beta",
        score=score,
        metadata={},
        token_count=2,
    )


async def test_evaluate_context_reports_recall_mrr_and_budget() -> None:
    questions = [
        DevQuestion(
            question_id="q1",
            question="what is alpha?",
            expected_doc_ids=["d1"],
        )
    ]
    retriever = FakeRetriever(
        [make_chunk("c1", "d1"), make_chunk("c2", "d2")]
    )

    report = await evaluate_context(retriever, questions, top_k=2, token_budget=2)
    metrics = report["metrics"]

    assert metrics["candidate_doc_recall"] == 1.0
    assert metrics["context_doc_recall"] == 1.0
    assert metrics["context_mrr"] == 1.0
    assert metrics["budget_adherence"] == 1.0
    assert metrics["duplicate_chunk_count"] == 0
    assert report["per_question"][0]["selected_chunk_ids"] == ["c1"]
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest backend/tests/unit/test_context_evaluation.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.evaluation.context'`.

- [ ] **Step 3: Implement evaluation module**

Create `backend/app/evaluation/context.py` with:
- `_doc_recall(expected, actual)`
- `_context_mrr(expected, contexts)`
- `_p95(values)`
- `evaluate_context(...)`

Use `time.perf_counter()` around retrieval and builder. Return:

```python
{
    "params": {"top_k": top_k, "token_budget": token_budget, "max_chunks_per_doc": max_chunks_per_doc},
    "metrics": {
        "question_count": len(questions),
        "candidate_doc_recall": ...,
        "context_doc_recall": ...,
        "context_mrr": ...,
        "budget_adherence": ...,
        "mean_context_tokens": ...,
        "p95_context_tokens": ...,
        "mean_utilization": ...,
        "selected_chunk_mean": ...,
        "selected_doc_mean": ...,
        "duplicate_chunk_count": ...,
        "empty_context_count": ...,
        "latency_ms": {
            "retrieval": {"p50": ..., "p95": ...},
            "build": {"p50": ..., "p95": ...},
            "total": {"p50": ..., "p95": ...},
        },
    },
    "per_question": [...],
}
```

Each per-question row includes question ID, expected/retrieved doc IDs, selected chunk IDs, token count, and stage latencies.

- [ ] **Step 4: Create evaluation CLI**

Create `scripts/eval_context_dev.py`:

```python
#!/usr/bin/env python3
"""Evaluate token-budgeted context construction on the development split."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.db.session import build_engine, build_session_factory  # noqa: E402
from app.evaluation.context import evaluate_context  # noqa: E402
from app.evaluation.dev import read_dev_questions  # noqa: E402
from app.retrieval.bm25 import BM25Retriever  # noqa: E402
from app.retrieval.bm25_index import BM25IndexError, BM25IndexManager  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/sample/dev_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/context_dev_metrics.json"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--max-chunks-per-doc", type=int, default=3)
    parser.add_argument("--index-root", type=Path, default=settings.bm25_index_root)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--tenant-id", type=uuid.UUID, default=settings.default_tenant_id)
    parser.add_argument(
        "--knowledge-base-id",
        type=uuid.UUID,
        default=settings.default_knowledge_base_id,
    )
    return parser


async def run(args: argparse.Namespace, settings: Settings) -> dict[str, object]:
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.token_budget <= 0:
        raise ValueError("token-budget must be positive")
    if args.max_chunks_per_doc <= 0:
        raise ValueError("max-chunks-per-doc must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not args.questions.is_file():
        raise FileNotFoundError(f"questions file not found: {args.questions}")

    questions = read_dev_questions(args.questions, args.limit)
    if not questions:
        raise ValueError("no development questions found")

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    try:
        retriever = BM25Retriever(
            BM25IndexManager(args.index_root),
            session_factory,
            tenant_id=args.tenant_id,
            knowledge_base_id=args.knowledge_base_id,
        )
        report = await evaluate_context(
            retriever,
            questions,
            top_k=args.top_k,
            token_budget=args.token_budget,
            max_chunks_per_doc=args.max_chunks_per_doc,
            progress_every=args.progress_every,
        )
    finally:
        await engine.dispose()

    report["generated_at"] = datetime.now(UTC).isoformat()
    report["retriever"] = "bm25"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    args = build_parser(settings).parse_args(argv)
    try:
        asyncio.run(run(args, settings))
    except (BM25IndexError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run evaluation tests**

Run: `uv run pytest backend/tests/unit/test_context_evaluation.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/evaluation/context.py scripts/eval_context_dev.py backend/tests/unit/test_context_evaluation.py
git commit -m "feat: evaluate token-budgeted context construction"
```

---

### Task 5: Run Real Evaluation, Update Docs, and Verify

**Files:**
- Modify: `README.md`
- Create: `docs/architecture/context-construction.md`
- Create: `docs/design-decisions/0005-context-construction.md`
- Modify: `plan.md`
- Modify: `blog4.md`
- Modify: `docs/deferred-work.md` if new shortcuts appear

**Interfaces:**
- Consumes: all prior tasks and the production BM25 index.
- Produces: measured context metrics, updated docs, and a verified Block 5.

- [ ] **Step 1: Run focused and full tests**

Run: `uv run pytest backend/tests/unit/test_context_builder.py backend/tests/unit/test_context_evaluation.py backend/tests/unit/test_context_cli.py -v`

Run: `uv run pytest -q`

Run: `uv run ruff check .`

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 2: Run the real evaluation**

Run:

```bash
uv run python scripts/eval_context_dev.py --output data/processed/context_dev_metrics_block5.json
```

Expected: 169 questions; `budget_adherence` is `1.0`; `duplicate_chunk_count` is `0`; context document recall is within `0.05` of candidate document recall.

- [ ] **Step 3: Manually verify the CLI**

Run:

```bash
uv run python scripts/build_context.py "How do I rotate an API key?" --top-k 50 --json
```

Expected: JSON with `token_count <= 4000`, unique `chunk_id`s, sequential `citation_id`s, and at most 3 chunks per `doc_id`.

- [ ] **Step 4: Update documentation**

- `README.md`: add Block 5 to completed work and show the preview/eval commands.
- `docs/architecture/context-construction.md`: document the rank-first algorithm, budget/caps, citation format, metrics, and deferred optimizations.
- `docs/design-decisions/0005-context-construction.md`: record why rank-first whole-chunk packing was chosen over merge/knapsack approaches.
- `plan.md`: mark Block 5 complete and insert Block 6 reranking as next.
- `blog4.md`: append a Block 5 section with measured metrics, the overfetch/whole-chunk trade-off, and what comes next.
- `docs/deferred-work.md`: add any new compromises discovered during implementation.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/architecture/context-construction.md docs/design-decisions/0005-context-construction.md plan.md docs/deferred-work.md
git commit -m "docs: document block 5 context construction"
```

`blog4.md` is local-only and ignored through `.git/info/exclude`; do not stage it.

- [ ] **Step 6: Final verification**

Run: `git status --short`

Expected: only ignored/local files remain, including `.kilo/` and `blog4.md`.

Run: `git log --oneline -6`

Expected: the five Block 5 commits are present in order.
