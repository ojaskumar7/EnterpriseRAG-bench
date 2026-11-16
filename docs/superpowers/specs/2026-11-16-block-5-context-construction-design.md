# Block 5 Design: Evidence Selection and Context Construction

- Status: approved
- Date: 2026-11-16
- Scope: turn BM25-ranked chunks into deterministic, citation-ready context

## Context

Block 4 productionized BM25 retrieval. On the 169-question development split, BM25
reaches recall@20 0.8494 and MRR 0.6771 with zero zero-result questions. It returns
ranked chunks, not a bounded context suitable for a later generation model.

Block 5 adds the missing boundary: select evidence from the BM25 candidates and pack it
under a token budget with stable citation identifiers. It does not add reranking,
generation, an API, or holdout evaluation.

## Goal

Given a query and a ranked list of `RetrievedChunk`s, produce a deterministic
`BuiltContext` containing at most 4,000 estimated tokens of whole chunks, deduplicated,
capped at three chunks per document, with citation IDs in selection order.

## Non-Goals

- Cross-encoder or LLM reranking.
- Query rewriting.
- Neighbor expansion, adjacent-chunk merging, sentence extraction, or summarization.
- Prompt-injection sanitization or final generation prompts.
- API endpoints, authentication, or background workers.
- Holdout evaluation.

## Decisions

1. **Rank-first packing.** Sort candidates by score descending, then `chunk_id` ascending.
   Include whole chunks in that order.
2. **Per-document cap.** At most three selected chunks per `doc_id`.
3. **No source-diversity cap.** Report source counts but do not exclude evidence based on
   source type. Many questions target a single source.
4. **Budget 4,000 tokens by default.** Use the stored chunk `token_count` when available;
   otherwise use a deterministic whitespace count. Whole chunks only. Skip a chunk that
   does not fit and continue scanning lower-ranked candidates.
5. **Stable citations.** Assign `[1]..[N]` in selected order. A citation identifies the
   chunk, document, source type, title, and chunk index.
6. **Pure builder.** `ContextBuilder.build()` performs no I/O. Retrieval and database
   access stay in the existing retriever and CLI.
7. **No new dependencies or migrations.** Add `token_count` to the lightweight chunk
   lookup and optionally to `RetrievedChunk`.

## Architecture

```text
backend/app/context/
  __init__.py
  types.py       # ContextBlock, BuiltContext
  builder.py     # ContextBuilder, token estimation, selection/packing
  render.py      # plain-text and JSON-ready rendering helpers
```

The package is deliberately separate from `app.retrieval`. Retrieval ranks chunks;
context construction selects and formats evidence.

## Interfaces

```python
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

`ContextBuilder(token_budget=4000, max_chunks_per_doc=3, token_counter=None)` validates
positive settings. `build(query, candidates)` returns a `BuiltContext`; empty candidates
or a blank query return an empty context.

`render_context(context)` returns text blocks headed by `[n] title (source_type,
doc_id=..., chunk=...)` followed by chunk content. `context_to_dict(context)` returns a
JSON-serializable structure for the CLI/evaluator.

## Selection Algorithm

1. Validate settings and query.
2. Remove candidates with non-positive or non-finite scores.
3. Sort by `(-score, chunk_id)`.
4. Deduplicate by `chunk_id`, keeping the first occurrence.
5. For each candidate:
   - Skip when its document already has three selected chunks.
   - Estimate its tokens.
   - Skip when adding it would exceed the budget.
   - Otherwise append it and increment the document count and token total.
6. Stop when all candidates have been examined or the budget is full.
7. Assign citation IDs in selection order and compute summary counts.

The result must be deterministic for the same inputs and settings.

## CLI

`scripts/build_context.py "query"`:

- `--top-k 50` (must be positive)
- `--token-budget 4000` (must be positive)
- `--max-chunks-per-doc 3` (must be positive)
- `--tenant-id`, `--knowledge-base-id`, `--index-root`
- `--json`

It loads the production BM25 index, retrieves candidates, builds context, and prints
either rendered text or `context_to_dict`. Missing/stale indexes keep the existing
actionable BM25 errors.

## Evaluation

`scripts/eval_context_dev.py` evaluates all 169 development questions:

- BM25 candidates: `top_k=50`.
- Build a 4,000-token context per question.
- Report candidate document recall, context document recall, context MRR, budget
  adherence, mean/p95 context tokens and utilization, selected chunk/doc counts, duplicate
  chunk count, and retrieval/build/total p50/p95 latency.
- Write `data/processed/context_dev_metrics.json` (gitignored).
- Never pass expected document IDs into retrieval or context construction.

## Acceptance Criteria

- Context never exceeds its token budget.
- Selected chunk IDs are unique and citation IDs are sequential.
- Output is deterministic for repeated runs.
- Context document recall is no more than 0.05 absolute below BM25 candidate document
  recall at the same top-k.
- Build latency is measured and reported; no retrieval-quality regression.
- Unit tests, focused integration tests, and Ruff pass.
- README, architecture doc, ADR, `plan.md`, and `blog4.md` are updated.
- Holdout remains untouched.

## Error Handling

- Non-positive budget, per-document cap, or top-k raises `ValueError`.
- Blank query or no candidates returns an empty context.
- BM25 missing/stale errors propagate unchanged.
- Retrieved content is untrusted evidence; sanitization is deferred to generation.

## Testing

- Deduplication by chunk ID.
- Per-document cap.
- Budget skip-and-continue behavior; never exceed budget.
- Deterministic ordering and sequential citations.
- Empty candidates and blank query.
- Stored-token and fallback-token counts.
- Renderer and JSON shape.
- CLI help/JSON; evaluation metrics with a fake retriever.
- Full unit suite, focused integration suite, and Ruff.

## Deferred Work

Deliberate shortcuts are recorded in [`docs/deferred-work.md`](../../deferred-work.md):

- BGE token count as a proxy for the future generation tokenizer.
- Whole chunks only; no neighbor merge or sentence extraction.
- Document-level rather than gold-span-level evaluation.
- No source-diversity cap.
- No build-latency optimization.
- No prompt-injection sanitization yet.
