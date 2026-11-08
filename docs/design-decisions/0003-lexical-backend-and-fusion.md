# 0003: PostgreSQL FTS, Hybrid Retrieval, and RRF

- Status: accepted
- Date: 2026-11-08
- Scope: Block 3 lexical retrieval, hybrid retrieval, and fusion

## Context

Block 2 produced a dense index with 136,383 chunks and 34,378 documents. Dense
retrieval finds the gold document in the top 20 for 58.4% of the 169 development
questions. The next improvement should combine semantic and exact-term matching
without introducing a second retrieval service or spending the holdout set.

The project already uses PostgreSQL 16 with pgvector. Chunks have tenant,
knowledge-base, source-type, title, and content columns. The corpus is large enough
that an unindexed `to_tsvector` scan is not acceptable, but small enough that a
single GIN index is practical.

## Decision

### Lexical backend

Use PostgreSQL full-text search as the Block 3 lexical baseline. Add a generated
`search_vector` column to `chunks`:

```sql
setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A') ||
setweight(to_tsvector('english'::regconfig, coalesce(content, '')), 'B')
```

Create a GIN index on `search_vector`. Query with
`websearch_to_tsquery('english', query)` and rank with
`ts_rank_cd(search_vector, query, 32)`.

This is **not exact BM25**. PostgreSQL's `ts_rank_cd` is a cover-density ranking
function, not BM25's term-frequency and document-length saturation formula. The
project will call it lexical retrieval and document the difference rather than
claiming BM25. Exact BM25 is deferred until development metrics show that the
lexical baseline is the limiting factor.

PostgreSQL FTS is preferred now because it reuses the existing database, transactions,
tenant filters, and source filters. It avoids a second index artifact and a second
process to keep in sync with the chunk table.

### Retrieval interfaces

Add a standalone `LexicalRetriever` with the same leakage-safe contract as
`DenseRetriever`:

```python
async def retrieve(
    query: str,
    *,
    top_k: int = 10,
    source_types: Sequence[str] | None = None,
) -> list[RetrievedChunk]
```

Add a `HybridRetriever` that owns a dense retriever and a lexical retriever. It
retrieves `candidate_k` results from each side and fuses them with Reciprocal Rank
Fusion (RRF). The public interface remains query text, `top_k`, and optional
user-selected source filters.

Move `RetrievedChunk` to `backend/app/retrieval/types.py` so dense, lexical, and
hybrid retrieval share one result type without importing through `dense.py`.

### Fusion

Use RRF:

```text
score(chunk) = sum(weight_i / (k + rank_i))
```

Defaults:

- `rrf_k = 60`
- `dense_weight = 1.0`
- `lexical_weight = 1.0`
- `candidate_k = 50` per retriever

RRF is preferred over weighted score fusion because dense cosine similarity and
`ts_rank_cd` are not calibrated to each other. RRF uses only ranks, so it is robust
without a score-normalization tuning step.

Tie-breaking is deterministic: fused score descending, best rank ascending, then
`chunk_id` ascending. A chunk that appears in both candidate lists receives both
contributions.

### Evaluation

Add `scripts/eval_retrieval_dev.py` with `--mode dense|lexical|hybrid|all`. It reports
recall@1/5/10/20 and MRR on the 169 development questions for each mode and writes
`data/processed/retrieval_dev_metrics.json` (gitignored).

The success criterion is that hybrid retrieval does not regress dense MRR or
recall@10. An improvement is preferred but must be reported honestly. The holdout set
remains untouched.

## Consequences

- The generated column rewrites the `chunks` table during migration and adds storage
  for the `tsvector` values plus the GIN index.
- `search_vector` updates automatically when title or content changes; no separate
  indexing job or trigger is needed.
- English stemming and stopword removal are fixed at the database level. Changing the
  text search configuration requires a new generated column or a migration.
- PostgreSQL FTS may miss semantic paraphrases and may behave differently from an
  exact BM25 implementation. The evaluation output will make that visible.
- RRF is insensitive to score scales but discards score magnitude. It cannot express
  "dense is much more confident here" without weights or a later reranker.
- Hybrid retrieval can return chunks that only one retriever found. Evaluation still
  deduplicates retrieved chunks by `doc_id` before computing document-level metrics.

## Alternatives Considered

| Alternative | Why not now |
|---|---|
| Exact BM25 with `bm25s` | Exact scoring, but introduces a separate persisted index and post-filtering for tenant/source filters |
| Tantivy BM25 | Production-grade, but adds a Rust-backed dependency and another index lifecycle |
| ParadeDB `pg_search` | Exact BM25 inside PostgreSQL, but replaces the current pgvector image and adds infrastructure risk |
| Weighted score fusion | Requires arbitrary normalization and tuning on a small dev set |
| Lexical prefilter then dense rerank | Lexical misses become unrecoverable; changes retrieval semantics |
| Query-time `to_tsvector` with no index | Correct but too slow for 136,383 chunks |
