# Retrieval Quality Audit Design

- Status: approved
- Date: 2026-11-17
- Scope: development-split retrieval audit; no holdout evaluation

## Goal

Quantify the BGE query instruction and query dtype effects, explain the dense/BM25
gap, evaluate BM25+dense reciprocal-rank fusion, test `BAAI/bge-reranker-base`,
and freeze a retriever only if it clearly beats the existing BM25 development
baseline.

## Constraints and invariants

- Use `data/sample/dev_questions.jsonl` only for new measurements.
- Never read or evaluate `data/sample/holdout_questions.jsonl`.
- Reuse the existing production BM25 index with `--index-root`/settings; do not
  rebuild embeddings or mutate the database.
- Full title+content dense re-embedding is deferred unless the title sanity check
  contradicts the proposed explanation.
- The reranker default for this audit is `BAAI/bge-reranker-base`.

## Design

The existing `embedding_query_instruction` setting and `BGEEmbeddingProvider`
behavior remain the single source of truth. The dense evaluation CLI exposes an
explicit instruction override, where an empty string disables the prefix. The
audit records prefix-on and prefix-off metrics for recall@1, recall@5, recall@20,
and MRR, plus a float16 query-side run.

Add `Bm25DenseRetriever`, composing the existing `BM25Retriever` and
`DenseRetriever` behind the same retrieval protocol. It requests `candidate_k`
results from each retriever, fuses them through the existing deterministic RRF
function, and returns `top_k`. `eval_retrieval_dev.py` adds `bm25_dense` while
retaining the existing lexical `hybrid` mode for compatibility. The six tuning
runs cover `rrf_k` in `{10, 30, 60}` and `candidate_k` in `{50, 100}`.

The reranker adapter uses the existing cross-encoder protocol and generic
reranked evaluation. It evaluates BM25 top-50 unless the best fused result is
better, in which case it reranks that candidate list. The report records the
specified MRR, recall, and context-recall gates regardless of pass/fail.

## Testing and documentation

Unit tests cover empty query instructions, provider query/document distinction,
BM25+dense delegation and filtering, CLI mode/parameter validation, and report
serialization. Documentation records the measured tables, the title/content
asymmetry and five-question sanity check, the selected retriever decision, and
deferred full re-embedding work. Generation changes only happen after a measured
development-split winner clearly exceeds BM25.

## Alternatives

- Reuse the current `HybridRetriever` and rename its lexical dependency: rejected
  because it would silently change the established PostgreSQL FTS experiment.
- Re-embed title+content immediately: deferred because the audit can first test
  the explanation without spending GPU time or changing the corpus index.
- Add a new fusion algorithm: rejected because the existing RRF implementation
  already provides deterministic tie handling and is the plan's controlled
  variable.
