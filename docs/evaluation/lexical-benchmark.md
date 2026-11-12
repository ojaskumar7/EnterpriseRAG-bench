# Lexical Retrieval Benchmark (Block 3.5)

Date: 2026-11-11
Benchmark run commit: 548bac1430b3ec9b365cd37f2c7eb16543523710
Status: Benchmark-only spike; production retrieval unchanged.

## Executive Summary

BM25 passed the lexical gate. `bm25_content` reached recall@20 = 0.8494 with zero zero-result questions and a top-50 p95 of 1.81 ms, clearing Gates A, B, and C by a wide margin.

Exact BM25 alone is the strongest development result. `bm25_content` beats `hybrid_bm25` on all four headline metrics: recall@5 0.7310 vs 0.6568, recall@10 0.7776 vs 0.7665, recall@20 0.8494 vs 0.8284, and MRR 0.6771 vs 0.6058.

The BM25 hybrid passed the quality gates relative to dense. `hybrid_bm25` reached recall@10 = 0.7665 and MRR = 0.6058 versus `dense_a` recall@10 = 0.5303 and MRR = 0.4345, while keeping recall@5 and recall@20 above the dense baseline. Dense drift was zero on both headline metrics. However, current fusion underperforms BM25 on every headline metric, so fusion is not strictly beneficial as configured. Only the BM25 top-50 latency is gated; there is no hybrid latency gate.

Recommended next action: write a separate Block 4 implementation plan focused on BM25-only productionization. Treat hybrid fusion tuning as a secondary investigation, not an immediate hybrid rollout. Do not change production retrieval defaults in this block.

## PostgreSQL Variant Scope Deviation

**The benchmarked PostgreSQL variants do not match the approved Block 3.5 specification. Goal 4 ("which PostgreSQL-only variants are worth keeping as fallbacks?") is NOT evaluated as designed.** The PostgreSQL rows are exploratory only, are not used for the decision, and no PostgreSQL latency gate was evaluated.

| Variant | Approved spec | Actual implementation |
|---|---|---|
| `pg_and_or` | Run AND across all query lexemes; only if that returns zero results, run OR across all query lexemes (spec name `pg_and_or_fallback`). | Always run `(OR of 8 rarest lexemes) & (AND of remaining lexemes)`. |
| `pg_relaxed3` | AND across the 3 rarest content terms. | Keep lexemes with document frequency >= 3, then run `(OR of 8 rarest survivors) & (AND of the rest)`. |
| `pg_relaxed5` | AND across the 5 rarest content terms. | Keep lexemes with document frequency >= 5, then run `(OR of 8 rarest survivors) & (AND of the rest)`. |

The BM25 and hybrid results, gate evaluation, and BM25-only recommendation below are unaffected by this deviation.

## Setup

- Questions: 169 development questions.
- Holdout: not used.
- Corpus: 34,378 documents; 136,383 chunks.
- BM25: `bm25s`, `k1=1.5`, `b=0.75`, method `lucene`, English stopwords, PyStemmer English.
- Title weighting: title tokens repeated three times (`title_x3`), compared with content-only indexing.
- Dense baseline: same-run `dense_a` and `dense_b`; report drift.
- Runtime: CUDA, float16, model `BAAI/bge-large-en-v1.5`, top-k 20, candidate-k 50, RRF k 60, PostgreSQL timeout 10 s, `hnsw.ef_search` 40.
- BM25 index size: content 150,496,236 bytes; `title_x3` 150,496,238 bytes. Both loaded from the persisted index (load times 0.174 s and 0.191 s, no rebuild). The reported `build_seconds` of 0.0 is a cache hit, not a measure of index construction time.

## Run Environment

- GPU: NVIDIA GeForce RTX 5050 Laptop GPU, 8151 MiB.
- PyTorch 2.8.0+cu128, CUDA 12.8.
- 15.47 GiB RAM, 16 CPUs.
- PostgreSQL 16.15.
- WSL2 (Linux 6.6.87.2).
- `hnsw.ef_search=40`, CUDA float16.

The JSON host block records platform, CPU count, RAM, device, and dtype, but not the GPU model, CUDA version, or PostgreSQL version.

## Lexical Variant Results

| Variant | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | Zero-result questions | Timeouts | p50 ms | p95 ms | p95 top-50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pg_and | 0.0118 | 0.0178 | 0.0178 | 0.0178 | 0.0178 | 156 | 0 | 7.70 | 23.19 | n/a |
| pg_and_or | 0.0281 | 0.0370 | 0.0449 | 0.0468 | 0.0495 | 118 | 0 | 7.66 | 31.85 | n/a |
| pg_relaxed3 | 0.0163 | 0.0370 | 0.0449 | 0.0468 | 0.0436 | 118 | 0 | 7.70 | 47.00 | n/a |
| pg_relaxed5 | 0.0163 | 0.0370 | 0.0449 | 0.0468 | 0.0436 | 117 | 0 | 8.18 | 52.97 | n/a |
| bm25_content | 0.5138 | 0.7310 | 0.7776 | 0.8494 | 0.6771 | 0 | 0 | 1.09 | 1.68 | 1.81 |
| bm25_title_weighted | 0.5108 | 0.7300 | 0.7922 | 0.8469 | 0.6753 | 0 | 0 | 1.02 | 1.66 | 1.64 |

Best BM25 variant (highest recall@20, then MRR): `bm25_content`.

## Hybrid Results

| Variant | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | p50 ms | p95 ms | Zero-result questions | Timeouts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_a | 0.2840 | 0.4761 | 0.5303 | 0.5777 | 0.4345 | 123.59 | 300.46 | 0 | 0 |
| dense_b | 0.2840 | 0.4761 | 0.5303 | 0.5777 | 0.4345 | 22.50 | 30.12 | 0 | 0 |
| hybrid_bm25 | 0.4283 | 0.6568 | 0.7665 | 0.8284 | 0.6058 | 27.73 | 50.23 | 0 | 0 |

Dense drift: recall@10 delta 0.0, MRR delta 0.0, 0/169 questions with different retrieved documents. Hybrid quality is therefore conclusive.

BM25 vs hybrid: `hybrid_bm25` underperforms `bm25_content` on every headline metric: recall@5 0.6568 vs 0.7310 (-0.0742), recall@10 0.7665 vs 0.7776 (-0.0111), recall@20 0.8284 vs 0.8494 (-0.0210), and MRR 0.6058 vs 0.6771 (-0.0713). Fusion as configured improves over dense, but degrades the strongest lexical signal.

## Gate Evaluation

| Gate | Requirement | Measured | Result |
|---|---|---|---|
| A | BM25 recall@20 >= 0.15 | 0.8494 (`bm25_content`) | pass |
| B | BM25 zero-result questions <= 20 | 0 (`bm25_content`) | pass |
| C | BM25 top-50 p95 < 200 ms | 1.81 ms (`bm25_content`) | pass |
| D | Hybrid +0.01 on recall@10 or MRR | recall@10 0.7665 vs 0.5303 (+0.2362); MRR 0.6058 vs 0.4345 (+0.1713) | pass |
| E | Hybrid recall@5 no worse than dense by 0.01 | 0.6568 vs 0.4761 (threshold 0.4661) | pass |
| F | Hybrid recall@20 no worse than dense by 0.01 | 0.8284 vs 0.5777 (threshold 0.5677) | pass |
| G | Hybrid other headline metric no worse by 0.01 | MRR 0.6058 vs 0.4345 (threshold 0.4245) | pass |
| H | Dense drift <= 0.01 on recall@10 and MRR | recall@10 delta 0.0; MRR delta 0.0 | pass |

## Recommendation

Exact BM25 alone is the strongest development result, and all lexical gates pass. Recommend a separate Block 4 implementation plan focused on BM25-only productionization, covering index lifecycle, update and rebuild strategy, observability, and rollout. Current hybrid fusion underperforms BM25 on every headline metric, so tune RRF k, candidate-k, and weighting as a secondary investigation rather than rolling out this hybrid as-is. Do not change production retrieval defaults in this block.

## Reproduction

```bash
uv sync --group dev
uv run python scripts/benchmark_lexical.py --device cuda --dtype float16 --output data/processed/lexical_benchmark.json
```

Expected runtime setup: PostgreSQL `production-rag-db` running with 34,378 documents and 136,383 chunks. Results are written to `data/processed/lexical_benchmark.json` and the persisted BM25 indexes live under `data/processed/bm25_index/`.

The corpus fingerprint now includes chunk titles because title text is part of the indexed BM25 text. The next full run will therefore rebuild the persisted indexes once; the measured results in this report remain valid.

## Limitations

- Results are for one tenant and one corpus.
- Exact BM25 has no synonym or semantic expansion.
- PostgreSQL OR variants can time out and are measured as zero-result failures. In this run no timeouts occurred, but 117–118 of 169 questions returned zero results.
- Only BM25 index size is reported; index memory use is not measured or optimized. `build_seconds` is 0.0 because persisted indexes were reused (cache hit), not a measurement of index construction time.
- This spike does not implement reranking, learned sparse retrieval, or production changes.
- `select_rarest` applies its limit before moving zero-document-frequency terms last; this affects `pg_and_or` for 22/169 development questions.
- Unquoted lexemes are passed to `to_tsquery`. 0/169 development questions error, but URL-like tokens could raise `PostgresSyntaxError`.
- `dense_a` p95 may be inflated by cold start; `dense_b` and the drift between them are recorded. The hybrid gate uses the same-run dense baseline.

### Artifact Omissions

- PostgreSQL top-50 latency was not measured (only top-k=20 latency); no PostgreSQL latency gate was evaluated.
- Per-question latencies are absent from the JSON; only aggregate latency summaries exist.
- Peak process RSS was not measured; only BM25 index size is recorded.
- The current JSON predates the expanded host metadata in "Run Environment"; it records platform, CPU count, RAM, device, and dtype only.
