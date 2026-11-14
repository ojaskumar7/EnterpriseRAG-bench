# 0004: BM25 Production Backend

- Status: accepted
- Date: 2026-11-14
- Scope: Block 4 production BM25 retrieval, versioned indexes, and atomic publish

## Context

Block 3 combined dense retrieval with PostgreSQL full-text search through Reciprocal Rank
Fusion. The lexical side was effectively non-functional on the development split: 156 of
169 questions returned no lexical results because `websearch_to_tsquery` ANDs the terms of
natural-language questions. Hybrid beat dense only marginally (recall@10 0.5348 vs 0.5303,
MRR 0.4404 vs 0.4345).

Block 3.5 benchmarked exact BM25 over `title + "\n" + content` with English stemming and
`k1=1.5`, `b=0.75`. It beat dense and hybrid on every headline metric:

| Mode | recall@10 | recall@20 | MRR |
|---|---:|---:|---:|
| dense | 0.5303 | 0.5777 | 0.4345 |
| hybrid | 0.5348 | 0.5821 | 0.4404 |
| exact BM25 | 0.7776 | 0.8494 | 0.6771 |

The corpus is already in PostgreSQL (34,378 documents, 136,383 chunks) and is small enough
for a single in-process index per tenant/knowledge base. The next block needs a retrieval
backend that is actually used in production, not another fusion experiment.

## Decision

### BM25 is the production retriever

Use `bm25s` in-process as the production retrieval path. Do not tune or productionize
hybrid fusion in this block. Keep the existing dense, lexical, and hybrid code as
documented experiments so their measured results remain reproducible.

### Versioned per-tenant/KB indexes with atomic publish

Build one immutable index per `(tenant_id, knowledge_base_id)`:

```text
data/indexes/bm25/<tenant_id>/<knowledge_base_id>/
  current.json
  .build.lock
  versions/<version>/{manifest.json,catalog.jsonl,index/}
```

A build writes a new version directory, validates it by reloading and probing the saved
index, then switches readers with an atomic `os.replace` of `current.json`. The pointer
stores the version name and the manifest SHA256. Published versions are never modified.
Retention keeps the active version plus the previous two. A non-blocking exclusive lock
per tenant/KB prevents concurrent rebuilds.

### Content lives in PostgreSQL

The index catalog stores row identity, source type, title, chunk index, and content hash,
but not full content. The retriever fetches final chunk content from PostgreSQL by
`chunk_id`, always constrained by tenant and knowledge base. This keeps the index from
serving stale text, preserves tenant isolation, and makes source filters a catalog
pre-filter plus a database constraint. A selected chunk missing from PostgreSQL raises
`BM25IndexStale` with a rebuild command.

### Fail fast

Missing, invalid, or stale indexes raise `BM25IndexUnavailable` or `BM25IndexStale` with
tenant/KB and rebuild instructions. There is no PostgreSQL full-text fallback because a
silent fallback would hide a broken production index.

### Configuration

Use one tokenizer and scoring configuration for build and query: `k1=1.5`, `b=0.75`,
`method="lucene"`, English stopwords, and `PyStemmer("english")`. Text is
`title + "\n" + content`; the title is not separately weighted. `corpus_fingerprint` is an
order-independent SHA256 over sorted `(chunk_id, content_hash, title)` and drives `--check`.

## Consequences

- **Rebuild operation.** Ingestion changes require a batch rebuild of the affected
  tenant/KB index. There is no incremental update or change-data-capture. `--check` detects
  a stale fingerprint but does not repair it.
- **Memory per tenant/KB.** Each loaded index is held in process memory in addition to its
  on-disk footprint. The measured 136,383-chunk version is about 181 MB on disk; serving
  many tenant/KB indexes from one process increases memory proportionally.
- **No fallback.** A missing or stale index is a query-time error, not a quality
  degradation. Operators must rebuild or restore the index.
- **Stale window.** A long-running process keeps its cached snapshot until it calls
  `reload()`. Queries that select chunks deleted from PostgreSQL fail with an actionable
  stale error rather than returning partial results.
- **Batch-only freshness.** Freshness is bounded by the rebuild cadence, not by ingestion
  transactions.
- **Deferred evidence selection.** BM25 returns ranked chunks only. Evidence selection,
  context construction, reranking, and generation move to later blocks.
- **Source filters can overfetch.** Filtered queries may scan more candidates than `top_k`
  before finding enough matches, up to the full index in the worst case.

## Alternatives Considered

| Alternative | Why not now |
|---|---|
| PostgreSQL FTS only | Already present; natural-language AND semantics returned zero results for 156 of 169 development questions |
| Hybrid dense + lexical + RRF | Measured below BM25 on every headline metric; adds dense-model latency and fusion tuning |
| External search service (Elasticsearch/OpenSearch) | Exact BM25, but adds another service, index lifecycle, and operational surface at a corpus size that fits in-process |
| Incremental BM25 index updates | Better freshness, but requires change tracking and mutation of a committed `bm25s` index; batch rebuild is simpler and sufficient while ingestion is infrequent |
| Tantivy BM25 | Production-grade, but adds a Rust-backed dependency and another index format |
| ParadeDB `pg_search` | Exact BM25 inside PostgreSQL, but replaces the current pgvector image and adds infrastructure risk |

## Measured Results

Measured on 2026-11-14 for the default tenant/KB (34,378 documents, 136,383 chunks):

- Index version: `20261114T114342596160Z-9b59d71b6d36`
- `bm25s` 0.3.16, PyStemmer 3.1.0, `k1=1.5`, `b=0.75`, `method=lucene`
- On-disk version size: about 181 MB
- `--check`: current

Development metrics over the 169 development questions at `top_k=20`:

| Metric | BM25 |
|---|---:|
| recall@1 | 0.5138 |
| recall@5 | 0.7310 |
| recall@10 | 0.7776 |
| recall@20 | 0.8494 |
| MRR | 0.6771 |
| zero-result questions | 0 |
| top-50 p95 latency | 11.3 ms |

The production build reproduces the Block 3.5 benchmark within measurement noise and
clears every Block 4 target. The 42-question holdout split was not run or tuned on.
Metrics are development-split diagnostics, not official benchmark claims.
