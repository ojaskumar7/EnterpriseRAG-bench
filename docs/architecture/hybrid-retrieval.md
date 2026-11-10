# Hybrid Retrieval Architecture

- Status: implemented for Block 3
- Scope: PostgreSQL full-text lexical retrieval, hybrid retrieval, and Reciprocal Rank Fusion
- Last measured run: 2026-11-09

## Scope

Block 3 adds a lexical retrieval path next to the Block 2 dense index and combines the
two with Reciprocal Rank Fusion (RRF). It covers:

- A generated `search_vector` column and GIN index on `chunks`.
- A leakage-safe `LexicalRetriever` backed by PostgreSQL full-text search.
- A `HybridRetriever` that fuses dense and lexical candidate lists.
- A unified development evaluator that reports dense, lexical, and hybrid metrics.

The lexical backend is PostgreSQL full-text search, **not exact BM25**. It is described
as lexical retrieval throughout the project.

### Non-goals

Block 3 deliberately does not add:

- An exact BM25 library (`bm25s`, Tantivy, ParadeDB `pg_search`, Elasticsearch/OpenSearch).
- A reranker, query rewriting, or query decomposition.
- Answer generation, citations, abstention, or the API.
- A full benchmark evaluation harness.
- Any use of the 42-question holdout split.

## Lexical Schema

`ChunkRecord` has a generated `tsvector` column that combines title and content with
different weights:

```sql
setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A') ||
setweight(to_tsvector('english'::regconfig, coalesce(content, '')), 'B')
```

Title weight `A` outranks content weight `B`, so a query term in the title contributes
more to the rank than the same term in the body. The column is `GENERATED ALWAYS ...
STORED`, so it updates automatically when `title` or `content` changes. No trigger or
separate indexing job is required.

A GIN index makes the containment operator index-backed:

```sql
CREATE INDEX ix_chunks_search_vector ON chunks USING gin (search_vector);
```

The migration is `backend/alembic/versions/0002_add_lexical_search.py`. Applying it
rewrites the `chunks` table once to compute stored vectors and builds the index. On the
measured database the GIN index is about **75 MB**, the `chunks` table (including its
indexes) is about **2.4 GB**, and the Block 2 HNSW index remains about **1.1 GB**.

## Lexical Query Flow

`LexicalRetriever.retrieve(query, *, top_k=10, source_types=None)` performs:

```text
query text
  -> websearch_to_tsquery('english', query)
  -> WHERE tenant_id = ? AND knowledge_base_id = ?
       AND search_vector @@ tsquery
  -> optional source_type filter
  -> ORDER BY ts_rank_cd(search_vector, tsquery, 32) DESC, chunk_id ASC
  -> LIMIT top_k
  -> RetrievedChunk(score = rank)
```

The query is built in `search_chunks_lexical` in `backend/app/db/repositories.py`. Tenant
and knowledge-base filters are mandatory on every query. `source_types` is an optional
user-selected product filter and is never populated from evaluation metadata.

`EXPLAIN` on the measured database shows a `Bitmap Index Scan on
ix_chunks_search_vector`, so lexical queries use the GIN index rather than a sequential
scan.

### `ts_rank_cd` limitation

`ts_rank_cd` is a cover-density ranking function, not BM25. It has no term-frequency
saturation or document-length normalization comparable to BM25's `k1`/`b` parameters.
Its scores are not calibrated against dense cosine similarity. The project therefore
calls this path lexical retrieval and makes no BM25 claims. Exact BM25 remains a separate
later decision.

## Component Diagram

```text
                         user question
                               |
             +-----------------+------------------+
             |                                    |
             v                                    v
      DenseRetriever                       LexicalRetriever
   BGE query embedding                 websearch_to_tsquery
             |                                    |
             v                                    v
   pgvector HNSW cosine                 PostgreSQL FTS (GIN)
   top candidate_k                      top candidate_k
             |                                    |
             +-----------------+------------------+
                               |
                               v
                  reciprocal_rank_fusion
                  weights=[dense, lexical], k=rrf_k
                               |
                               v
                     top_k RetrievedChunk list
```

The retrievers share `RetrievedChunk` from `backend/app/retrieval/types.py`. Dense,
lexical, and hybrid retrieval all expose the same leakage-safe interface: query text,
`top_k`, and optional user source filters.

## Reciprocal Rank Fusion

RRF combines ranked lists using ranks only:

```text
score(chunk) = sum over retrievers i of  weight_i / (rrf_k + rank_i)
```

Defaults:

| Parameter | Value |
|---|---:|
| `rrf_k` | 60 |
| `dense_weight` | 1.0 |
| `lexical_weight` | 1.0 |
| `candidate_k` | 50 per retriever |

Each retriever returns `max(candidate_k, top_k)` candidates. A chunk that appears in
both lists receives both contributions. Duplicate chunk IDs within one list count once,
at their first (best) rank.

Tie-breaking is deterministic: fused score descending, then best rank ascending, then
`chunk_id` ascending. This makes repeated runs reproducible.

RRF is used instead of weighted score fusion because dense cosine similarity and
`ts_rank_cd` live on different, uncalibrated scales. RRF avoids a score-normalization
step that would otherwise have to be tuned on the small development set.

## Development Metrics

Measured on a clean database (`34,378` documents, `136,383` chunks, `136,383`
embeddings, `0` non-default rows) over the **169 development questions** with
`BAAI/bge-large-en-v1.5` on CUDA `float16`, `top_k=20`, `candidate_k=50`, `rrf_k=60`:

| Mode | recall@1 | recall@5 | recall@10 | recall@20 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 0.2840 | 0.4761 | 0.5303 | 0.5777 | 0.4345 |
| lexical | 0.0118 | 0.0178 | 0.0178 | 0.0178 | 0.0178 |
| hybrid | 0.2929 | 0.4726 | 0.5348 | 0.5821 | 0.4404 |

Hybrid vs dense on the same run:

| Metric | Dense | Hybrid | Delta |
|---|---:|---:|---:|
| recall@1 | 0.2840 | 0.2929 | +0.0089 |
| recall@5 | 0.4761 | 0.4726 | -0.0035 |
| recall@10 | 0.5303 | 0.5348 | +0.0045 |
| recall@20 | 0.5777 | 0.5821 | +0.0044 |
| MRR | 0.4345 | 0.4404 | +0.0059 |

**Acceptance:** hybrid recall@10 and MRR both exceed the same-run dense baseline, so
Block 3's success criterion passes. Hybrid recall@10 remains 0.0014 below the Block 2
dense baseline (0.5348 vs 0.5362). Hybrid recall@5 is 0.0035 below same-run dense, a
non-binding trade-off. Dense results versus the Block 2 baseline are mixed rather than
uniformly better (for example recall@10 0.5303 vs 0.5362).

These are development-split diagnostics, not benchmark claims. The metrics file is
`data/processed/retrieval_dev_metrics.json` (gitignored).

## Error Handling and Empty Queries

- `top_k <= 0` raises `ValueError("top_k must be positive")` before any database work.
- An empty or whitespace-only query returns `[]` without querying the database.
- `HybridRetriever` returns `[]` for an empty query without calling either retriever.
- Invalid fusion parameters (`rrf_k <= 0`, `candidate_k <= 0`, negative weights, weight
  count mismatch) raise `ValueError`.
- Lexical queries that produce no `tsquery` matches simply return an empty list; they do
  not raise.
- The CLI wrappers catch `OSError`, `RuntimeError`, and `ValueError`, print an `error:`
  message to stderr, and exit non-zero.

## Limitations

- **English stemming only.** The text search configuration is `english`; stemming and
  stopword removal are fixed in the database. Changing them needs a new generated column
  or migration.
- **No exact BM25.** PostgreSQL FTS with `ts_rank_cd` is not BM25 and may rank
  differently from an exact BM25 implementation.
- **Natural-language AND semantics.** `websearch_to_tsquery` ANDs the terms of a natural
  language question. On this development split 156 of 169 questions return zero lexical
  results, so lexical retrieval is effectively non-functional here. Hybrid gains are
  small and come from the minority of questions where lexical returns matches.
- **RRF discards score magnitude.** It cannot express that dense is much more confident
  for a given query without weights or a reranker.
- **Candidate depth affects recall.** Hybrid can only fuse documents present in the
  dense or lexical candidate lists. `candidate_k=50` bounds what fusion can recover.
- **Chunk-level retrieval, document-level metrics.** Evaluation deduplicates retrieved
  chunks by `doc_id` before computing metrics.

## Holdout Status

The 42-question holdout split was **not** run or tuned on during Block 3. All reported
numbers are development-split diagnostics.
