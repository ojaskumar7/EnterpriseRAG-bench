# Block 3.5 Lexical Retrieval Benchmark — Design

- Status: approved
- Date: 2026-11-10
- Scope: benchmark-only spike to choose the next lexical retrieval backend
- Decision gate: stop after results and recommendation; no production retrieval changes

## Context

Block 3 implemented PostgreSQL full-text search with `websearch_to_tsquery` and
`ts_rank_cd`, then fused it with dense retrieval using RRF. The development results
showed two problems:

- Lexical retrieval returns zero results for 156 of 169 development questions because
  `websearch_to_tsquery` ANDs natural-language terms.
- Broad OR queries are not a cheap fix. PostgreSQL matches tens of thousands of chunks
  and switches to a parallel sequential scan; sampled OR queries took roughly 1–9
  seconds each on the real corpus.

Current clean-database development results, same run, 169 questions:

| Mode | recall@1 | recall@5 | recall@10 | recall@20 | MRR |
|---|---:|---:|---:|---:|---:|
| dense | 0.2840 | 0.4761 | 0.5303 | 0.5777 | 0.4345 |
| lexical | 0.0118 | 0.0178 | 0.0178 | 0.0178 | 0.0178 |
| hybrid | 0.2929 | 0.4726 | 0.5348 | 0.5821 | 0.4404 |

The project needs evidence before replacing or extending the lexical backend. Exact BM25
was deferred in ADR 0003 until development metrics showed that lexical retrieval was the
limiting factor. That condition is now met.

## Goal

Measure whether exact BM25 materially improves lexical retrieval and hybrid retrieval,
within the existing leakage and evaluation constraints, before changing production code.

The benchmark must answer:

1. How much lexical recall and MRR does exact BM25 recover versus the current
   PostgreSQL AND baseline?
2. Does dense + BM25 RRF beat same-run dense by a meaningful margin?
3. Can the candidate backend meet a p95 top-50 query latency below 200 ms?
4. Which PostgreSQL-only variants are worth keeping as fallbacks?

## Non-goals

This block does not:

- Change `backend/app/retrieval/*`, default settings, or production behavior.
- Replace PostgreSQL FTS, remove hybrid retrieval, or add a production BM25 index.
- Add query rewriting, reranking, generation, citations, or abstention.
- Use the 42-question holdout split.
- Add multi-tenant BM25 filters or production index lifecycle management.
- Claim exact BM25 for PostgreSQL `ts_rank_cd`.

## Approaches Considered

| Approach | Trade-off | Decision |
|---|---|---|
| `bm25s` spike | Exact BM25, pure Python, sparse in-memory index, fast to implement; no native metadata filters | Use for this benchmark |
| Tantivy spike | Production-grade BM25, fast, native filters; more index and API work | Defer until BM25 proves worthwhile |
| PostgreSQL variants only | No new dependency; broad OR scans are too slow for the latency target | Keep as baselines, not the expected winner |

`bm25s` is chosen for the spike because it isolates the retrieval-quality question with
minimal new infrastructure. If BM25 wins, the production backend decision between
`bm25s` and Tantivy is a separate follow-up.

## Experiment Design

### Corpus and inputs

- Read chunks from PostgreSQL for the default tenant and knowledge base.
- Expected corpus: 34,378 documents and 136,383 chunks.
- Questions: `data/sample/dev_questions.jsonl` only, 169 questions.
- Holdout: `data/sample/holdout_questions.jsonl` must not be read.
- Dense model: `BAAI/bge-large-en-v1.5` on CUDA `float16`.
- Dense settings: `top_k=20`, `candidate_k=50`, `rrf_k=60`, equal RRF weights.
- Record PostgreSQL `hnsw.ef_search`; run dense twice to estimate run-to-run drift.

### Variants

1. **`pg_and`** — current `websearch_to_tsquery` + `ts_rank_cd` baseline.
2. **`pg_and_or_fallback`** — current AND; if zero results, OR the lexemes produced by
   `to_tsvector('english', query)`.
3. **`pg_relaxed3` / `pg_relaxed5`** — AND over the 3 or 5 rarest content terms,
   using document frequencies from `ts_stat`.
4. **`bm25_content`** — exact BM25 over title plus content.
5. **`bm25_title_weighted`** — exact BM25 with title tokens repeated three times.
6. **`hybrid_bm25`** — dense + best BM25 variant with RRF.
7. **`dense_a` / `dense_b`** — repeat dense runs for a same-run baseline and drift check.

All variants return the existing `RetrievedChunk` shape. The benchmark reuses
`evaluate_retriever`, `reciprocal_rank_fusion`, and the existing dense retriever.

### BM25 configuration

- Build an exact BM25 index with `bm25s`.
- Tokenize with `bm25s.tokenize(..., stopwords="en")` and PyStemmer's English
  stemmer. Add `bm25s` and `PyStemmer` as development dependencies only. The exact
  tokenizer, stemmer, `k1`, and `b` go in the index manifest.
- Use `k1=1.5`, `b=0.75` unless a documented library default differs.
- Title weighting is a comparison variant, not an unrecorded default.
- Persist the index under `data/processed/bm25_index/` (gitignored) with a manifest
  containing corpus counts, corpus hash, tokenizer config, BM25 parameters, and
  `bm25s` version.
- Source filtering and multi-tenancy are out of scope; the spike uses the default
  tenant and knowledge base.

### Metrics

For every variant, report:

- recall@1, recall@5, recall@10, recall@20
- MRR
- zero-result question count
- p50 and p95 query latency for top-50
- index build time, index load time, index size, and peak process memory for BM25
- per-question retrieved document IDs and latencies in the JSON artifact

Latency is measured after warm-up and excludes index build/load. PostgreSQL OR queries
use a per-query timeout so one slow question cannot hang the run; timeouts are recorded
as failures with zero results.

## Decision Gates

The benchmark recommendation must satisfy all applicable gates:

- **Lexical quality:** BM25 lexical recall@20 is at least **0.15** and zero-result
  questions are **≤20 of 169**. Current lexical recall@20 is `0.0178`.
- **Hybrid quality:** BM25 hybrid beats same-run dense by at least **+0.01 absolute**
  on recall@10 or MRR. If recall@10 is the selected gain, MRR must be no more than
  `0.01` below dense; if MRR is the selected gain, recall@10 must be no more than
  `0.01` below dense. Recall@5 and recall@20 must also be no more than `0.01` below
  dense.
- **Latency:** lexical top-50 p95 is **<200 ms** after warm-up.
- **Operational:** corpus/config manifest is reproducible, no holdout is read, and index
  artifacts are gitignored.

The same quality and latency gates apply to the best PostgreSQL variant. If no variant
passes, keep PostgreSQL FTS and document that lexical retrieval is not worth further
investment unless a proper search engine is added later. If a PostgreSQL variant passes
instead, recommend it as the fallback. The benchmark does not implement the winner.

## Deliverables

- `scripts/benchmark_lexical.py`: standalone benchmark and reporting script.
- `bm25s` and `PyStemmer` added as development dependencies only.
- `data/processed/lexical_benchmark.json`: full metrics and latency artifact
  (gitignored).
- `data/processed/bm25_index/`: persisted spike index and manifest (gitignored).
- `docs/evaluation/lexical-benchmark.md`: methodology, results, latency, and
  recommendation.
- Existing tests, Ruff, and format checks pass; database counts return to
  `34378|136383|136383`.

No production retriever, README status, or ADR default changes are made in this block.
If the user approves a winner, a follow-up block implements it and records the decision.

## Architecture and Data Flow

```text
PostgreSQL chunks
  -> corpus loader for default tenant/KB
  -> variant adapters:
       PostgreSQL AND / AND->OR / relaxed AND
       bm25s content / title-weighted
  -> 169 development questions
  -> lexical metrics
  -> dense + RRF hybrid metrics
  -> JSON artifact + evaluation report
```

The script owns the spike adapters. Production modules are only imported, not modified.
The BM25 index manifest is validated before reuse; a mismatch triggers a clear rebuild
instruction rather than silently using a stale index.

## Error Handling

- Missing `bm25s` dependency: fail with an actionable install message.
- Empty database or wrong tenant/KB counts: fail before building the index.
- Stale or mismatched BM25 manifest: fail with the expected and actual manifest values.
- PostgreSQL OR timeout: record the timeout, mark the question as a retrieval failure, and
  continue.
- No lexical results: record zero results; do not raise.
- Holdout path: not accepted by the benchmark CLI; only the development split is read.

## Testing and Verification

- Add unit tests for any reusable benchmark helpers, especially manifest validation and
  query normalization.
- Smoke-test with `--limit 5` before the full 169-question run.
- Run `uv run pytest -q`, `uv run ruff check .`, and
  `uv run ruff format --check .`.
- Verify the benchmark does not read the holdout split.
- Clean non-default tenant/KB rows after tests and confirm
  `34378|136383|136383`.
- Record host and database settings with the results: GPU, CUDA/fp16, RAM, PostgreSQL
  version, and `hnsw.ef_search`.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Naive OR queries are 1–9 s each | Per-query timeout; report latency as part of the decision |
| `bm25s` index is memory-heavy | Monitor peak RSS; 15 GiB host RAM; rebuild/persist index once |
| BM25 quality is still poor | Keep PostgreSQL FTS; do not productionize |
| Dense drift is mistaken for a BM25 gain | Run dense twice in the same benchmark; report both |
| Dev-set overfitting | No holdout; limit tuning to the declared variants and gates |
| Benchmark script drifts from production semantics | Reuse `RetrievedChunk`, metrics, fusion, and dense retriever |

## Follow-up

After the benchmark report, the user reviews the recommendation. A separate approved
block may implement the winning lexical backend, update the ADR, and change defaults.
