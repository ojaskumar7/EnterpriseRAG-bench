# Block 6 Design: Cross-Encoder Reranking

- Status: approved
- Date: 2026-11-16
- Scope: rerank BM25 top-50 candidates with a local MiniLM cross-encoder
- Prerequisite: Block 5 merged at `a4681c4`

## Context

Block 4 productionized BM25. Block 5 built a token-budgeted context from BM25
candidates. On the 169-question development split, BM25 reaches recall@20 `0.8494` and MRR
`0.6771`; the 4,000-token context reaches document recall `0.7744` with a `+0.0052` gap at
the same candidate chunk count.

BM25 ranks by lexical overlap, not query-passage relevance. A cross-encoder can compare
the query and each candidate directly and reorder the top candidates before context
construction. Block 6 measures whether that improves retrieval and context quality.

## Goal

Use `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank the BM25 top-50 candidates, preserve
both reranker and BM25 scores, and evaluate the effect on the development split. BM25
remains the production default unless the acceptance gate passes.

## Non-Goals

- Query rewriting.
- Fine-tuning a reranker.
- Hosted or LLM-based reranking.
- Score blending or RRF between BM25 and reranker scores.
- Reranker caching or incremental scoring.
- Generation, API, or holdout evaluation.
- Changing the Block 5 context builder.

## Decisions

1. **Local MiniLM cross-encoder.** Use
   `cross-encoder/ms-marco-MiniLM-L-6-v2` through the existing `sentence-transformers`
   dependency. No new packages.
2. **Rerank 50, return 50.** Delegate to BM25 with `candidate_k=50`; return the reranked
   top-50 by default. This keeps recall@20 and context comparisons meaningful.
3. **Replace ordering, keep both scores.** `RetrievedChunk.score` becomes the final
   cross-encoder score. Add `retrieval_score: float | None` to preserve the original BM25
   score.
4. **Pair text.** Score `query` against `title + "\n" + content`, matching the BM25
   corpus text. Truncate to `max_length=512` tokens.
5. **Deterministic tie-break.** Sort by `(-rerank_score, chunk_id)`.
6. **Decorator boundary.** `RerankedRetriever` wraps any `RetrieverProtocol`; BM25 remains
   unaware of reranking.
7. **No silent fallback.** Empty input returns `[]`. Model-load or inference failures raise
   an actionable error; callers explicitly choose reranked or BM25 retrieval.
8. **Optional production adoption.** The reranker is implemented and measured even if the
   dev gate fails. BM25 remains the production path unless the gate passes.
9. **Lazy model loading.** The cross-encoder loads on first scoring call so CLI help and
   validation do not pay model startup cost.

## Architecture

```text
backend/app/reranking/
  __init__.py
  base.py            # Reranker protocol
  cross_encoder.py   # CrossEncoderReranker, lazy model, batching

backend/app/retrieval/
  reranked.py        # RerankedRetriever decorator
  types.py           # RetrievedChunk.retrieval_score

backend/app/settings.py
  reranker_model
  reranker_device
  reranker_batch_size
  reranker_max_length
  reranker_candidate_k
```

### Reranker protocol

```python
class Reranker(Protocol):
    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]: ...
```

`CrossEncoderReranker` validates model name and positive batch/max-length settings,
resolves `device` to `auto`/`cpu`/`cuda`, and lazily constructs
`sentence_transformers.CrossEncoder` behind a lock. It scores with `predict(...,
batch_size=..., show_progress_bar=False, convert_to_numpy=True)` and returns Python floats.
An injected `predictor` may be supplied for tests.

### RerankedRetriever

```python
class RerankedRetriever:
    def __init__(
        self,
        base: RetrieverProtocol,
        reranker: Reranker,
        *,
        candidate_k: int = 50,
    ) -> None: ...

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 50,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...
```

Behavior:

1. Validate positive `candidate_k` and `top_k`; blank query returns `[]`.
2. Fetch `candidate_k` candidates from the base retriever, passing `source_types`.
3. Return `[]` when no candidates exist.
4. Score pairs off the event loop with `asyncio.to_thread` and require exactly one
   finite score per candidate.
5. Sort by `(-score, chunk_id)`.
6. Return at most `top_k` chunks with `score=rerank_score` and
   `retrieval_score=original.score`.
7. Preserve all other chunk fields and metadata.

Validation errors are `ValueError`; model/inference failures raise `RuntimeError` with a
message naming the reranker model.

## Configuration

| Setting | Default |
|---|---|
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANKER_DEVICE` | `auto` |
| `RERANKER_BATCH_SIZE` | `32` |
| `RERANKER_MAX_LENGTH` | `512` |
| `RERANKER_CANDIDATE_K` | `50` |

`.env.example` documents all five values. No dependency or lockfile change is required.

## CLI

Create `scripts/rerank_search.py`:

- `query`
- `--candidate-k 50`
- `--top-k 50`
- `--model` (default `settings.reranker_model`)
- `--device` (default `settings.reranker_device`)
- `--batch-size` (default `settings.reranker_batch_size`)
- `--max-length` (default `settings.reranker_max_length`)
- `--source-type` (repeatable)
- `--index-root`, `--tenant-id`, `--knowledge-base-id`
- `--json`

Text output shows final rank, rerank score, original BM25 score, and title. JSON uses
`dataclasses.asdict`, so both scores are visible. It builds the BM25 retriever, wraps it,
and prints the reranked results.

## Evaluation

### Retrieval comparison

Create `scripts/eval_reranking_dev.py`:

- Load the 169 development questions.
- Build one BM25 retriever and one reranked BM25 retriever.
- Fetch BM25 top-50 and reranked top-50 per question.
- Evaluate both with the existing document-level recall@k/MRR metrics.
- Report `bm25`, `reranked`, `delta`, and latency p50/p95 for retrieval and reranking.
- Write `data/processed/reranking_dev_metrics_block6.json` (gitignored).
- Record model name, candidate depth, batch size, and max length.
- Never pass expected document IDs into retrieval or reranking.

### Context comparison

Extend `scripts/eval_context_dev.py` with `--reranked` plus reranker model/device/batch/
max-length options. When enabled, wrap BM25 with `RerankedRetriever` and use
`candidate_k=args.top_k`; context construction remains unchanged. Report the same context
metrics as Block 5.

## Acceptance Gate

Reranking passes if any of:

- Reranked MRR > `0.6771`.
- Reranked recall@5 > `0.7310`.

And all of:

- Reranked recall@20 >= `0.8394` (BM25 `0.8494` minus `0.01`).
- Reranked context document recall stays within `0.01` of BM25 context recall at the same
  context size.
- Budget adherence remains `1.0`.
- Latency is measured and reported.

If the gate fails, BM25 remains the production path. The reranker, CLI, and measurements
remain available for a later model or tuning block. The failure and decision are recorded
in the docs; no tuning on holdout.

## Error Handling

- Non-positive candidate depth, top-k, batch size, or max length raises `ValueError`.
- Invalid device raises `ValueError`.
- Blank query or no candidates returns `[]`.
- Reranker output length must match candidate count; otherwise `RuntimeError`.
- Non-finite reranker scores raise `RuntimeError`.
- Model load/inference errors are wrapped with the model name and do not silently fall
  back to BM25.

## Testing

- `RerankedRetriever` ordering, tie-breaking, top-k, candidate depth, source passthrough,
  score preservation, empty query/candidates, and score-count/finite validation.
- `CrossEncoderReranker` settings validation, lazy loading, injected predictor, batching,
  and float conversion.
- CLI validation and help.
- Retrieval/context comparison helpers with fake retrievers/rerankers.
- Full suite, Ruff, real 169-question evaluation, and docs/ADR/README/plan/blog updates.

## Deferred Work

Record in [`docs/deferred-work.md`](../../deferred-work.md):

- No reranker score blending or tuning against dev.
- No fine-tuning, caching, or larger reranker model.
- No CPU latency optimization; measure and report first.
- No automated test requiring the live index/model; real runs remain manual.
- No prompt-injection or generation integration yet.
