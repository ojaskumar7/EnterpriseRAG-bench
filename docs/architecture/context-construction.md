# Context Construction Architecture

- Status: implemented for Block 5
- Scope: evidence selection and token-budgeted context assembly over BM25 results
- Last measured run: 2026-11-16

## Scope

Block 5 turns ranked BM25 chunks into a deterministic, citation-ready context. It covers:

- A pure `ContextBuilder` with rank-first packing.
- Deduplication by chunk ID and a per-document cap.
- A 4,000-token budget using stored chunk token counts.
- A `[n]` citation format with document and chunk metadata.
- A context preview CLI and a development-set context evaluator.

It does not add reranking, query rewriting, neighbor expansion, generation, an API, or
holdout evaluation.

## Pipeline

```text
query
  -> BM25Retriever (unchanged)
  -> ranked RetrievedChunk list
  -> ContextBuilder
       -> filter non-positive/non-finite scores
       -> sort by (-score, chunk_id)
       -> deduplicate by chunk_id
       -> cap at 3 chunks per doc_id
       -> pack whole chunks under 4,000 tokens
       -> assign [1]..[N] citations
  -> BuiltContext
  -> render_context() or context_to_dict()
```

The builder performs no I/O. Database hydration and retrieval remain in
`BM25Retriever`; rendering stays in `app.context.render`.

## Selection Rules

| Rule | Value |
|---|---|
| Token budget | `4000` |
| Max chunks per document | `3` |
| Source-type cap | none |
| Ordering | score descending, then `chunk_id` ascending |
| Chunk handling | whole chunks only |
| Token source | stored `chunk.token_count`, fallback `max(1, len(content.split()))` |
| Empty query/candidates | empty `BuiltContext` |

A candidate that would exceed the budget is skipped; lower-ranked candidates are still
considered. This differs from stopping at the first oversized chunk and avoids wasting
remaining budget.

Citation text is:

```text
[1] <title> (<source_type>, doc_id=<doc_id>, chunk=<chunk_index>)
<content>
```

## Token Accounting

BM25 hydrates `token_count` from PostgreSQL through `ChunkLookup`. Other retrievers may
leave it unset; the builder falls back to whitespace tokenization. This makes context
selection cheap and deterministic, but it is a proxy for the future generation model's
tokenizer. The final prompt budget must be rechecked during generation integration.

## Commands

```bash
# Preview a citation-ready context
uv run python scripts/build_context.py "How do I rotate an API key?" \
  --top-k 50 --token-budget 4000 --max-chunks-per-doc 3

# JSON output
uv run python scripts/build_context.py "How do I rotate an API key?" --json

# Evaluate context construction on the development split
uv run python scripts/eval_context_dev.py \
  --output data/processed/context_dev_metrics_block5.json
```

The evaluator never passes expected document IDs into retrieval or context construction.

## Measured Results

Development split, 169 questions, BM25 top-50 candidates, 4,000-token budget:

| Metric | Value |
|---|---:|
| Candidate document recall (top-50) | 0.8962 |
| Candidate document recall at context size | 0.7692 |
| Context document recall | 0.7744 |
| Recall gap at context size | +0.0052 |
| Context MRR | 0.6710 |
| Budget adherence | 1.0000 |
| Mean context tokens | 3970.1 |
| p95 context tokens | 4000 |
| Mean utilization | 0.9925 |
| Mean selected chunks | 10.27 |
| Mean selected documents | 8.83 |
| Duplicate chunks | 0 |
| Empty contexts | 0 |
| Retrieval p50 / p95 | 8.8 ms / 13.0 ms |
| Build p50 / p95 | 0.10 ms / 0.15 ms |

The top-50 candidate recall is naturally higher than context recall because a
4,000-token whole-chunk context covers about 10 of 50 candidates. The acceptance gate
therefore compares context recall with BM25 recall at the same candidate chunk count.

## Deferred Optimizations

Recorded in [`docs/deferred-work.md`](../deferred-work.md):

- Recheck the budget with the final generation tokenizer.
- Consider partial-chunk extraction or larger budgets if context recall needs to approach
  top-50 recall.
- Consider neighbor merging or sentence extraction after gold evidence spans exist.
- Measure source diversity; no diversity cap is enforced.
- Add prompt-injection sanitization during generation.
