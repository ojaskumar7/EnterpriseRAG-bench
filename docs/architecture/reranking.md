# Cross-Encoder Reranking Architecture

- Status: implemented, acceptance gate failed, optional path only
- Scope: rerank BM25 top-50 candidates with MiniLM
- Last measured run: 2026-11-16

## Scope

Block 6 adds an optional cross-encoder reranker between BM25 and context construction. It
covers:

- A `Reranker` protocol and a lazy `CrossEncoderReranker`.
- A `RerankedRetriever` decorator over any retriever.
- Preservation of both reranker and original retrieval scores.
- Query, evaluation, and context-evaluation entry points.
- A development-split comparison against BM25.

It does not add score blending, fine-tuning, query rewriting, generation, an API, or
holdout evaluation. BM25 remains the production path.

## Why the gate failed

`cross-encoder/ms-marco-MiniLM-L-6-v2` reordered BM25 top-50 candidates worse on the
enterprise development split. It lost relevant documents at every cut-off and also reduced
context document recall. The model is an MS MARCO passage reranker; this corpus mixes
Confluence, Jira, GitHub, and HubSpot text, and the pair text is truncated at 512 tokens.
No tuning was attempted, and the holdout was not used.

| Metric | BM25 | MiniLM reranked | Delta |
|---|---:|---:|---:|
| recall@1 | 0.5138 | 0.4403 | -0.0735 |
| recall@5 | 0.7310 | 0.6501 | -0.0809 |
| recall@10 | 0.7776 | 0.7377 | -0.0399 |
| recall@20 | 0.8560 | 0.8210 | -0.0350 |
| MRR | 0.6779 | 0.6203 | -0.0577 |
| p50 latency | 84.0 ms | 421.8 ms | +337.8 ms |
| p95 latency | 104.4 ms | 661.7 ms | +557.2 ms |

Context comparison, 4,000-token budget:

| Metric | BM25 context | Reranked context |
|---|---:|---:|
| Context document recall | 0.7744 | 0.7175 |
| Context MRR | 0.6710 | 0.6121 |
| Budget adherence | 1.0 | 1.0 |
| Mean context tokens | 3970.1 | 3963.1 |
| Empty contexts | 0 | 0 |
| Retrieval p95 | 15.8 ms | 658.8 ms |

The reranker fails the gate: MRR did not improve, recall@5 did not improve, and recall@20
fell below the `0.8394` floor. The context recall drop also exceeds the `0.01` limit.

## Architecture

```text
BM25Retriever (candidate_k=50)
  -> RerankedRetriever
       -> CrossEncoderReranker.score_pairs(query, title + "\n" + content)
       -> sort by (-rerank_score, chunk_id)
       -> RetrievedChunk(score=rerank_score, retrieval_score=bm25_score)
  -> ContextBuilder
```

`RerankedRetriever` is a decorator. BM25 has no knowledge of the reranker, and callers
explicitly choose the reranked path. A reranker failure raises an actionable error instead
of silently falling back.

## Commands

```bash
# Query with BM25 plus MiniLM reranking
uv run python scripts/rerank_search.py "How do I rotate an API key?" \
  --candidate-k 50 --top-k 50

# Compare BM25 and reranked retrieval on the development split
uv run python scripts/eval_reranking_dev.py \
  --top-k 50 --candidate-k 50 \
  --output data/processed/reranking_dev_metrics_block6.json

# Compare context construction over reranked candidates
uv run python scripts/eval_context_dev.py --reranked \
  --top-k 50 \
  --output data/processed/context_dev_metrics_block6_reranked.json
```

## Configuration

| Setting | Default |
|---|---|
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANKER_DEVICE` | `auto` |
| `RERANKER_BATCH_SIZE` | `32` |
| `RERANKER_MAX_LENGTH` | `512` |
| `RERANKER_CANDIDATE_K` | `50` |

## Score Contract

`RetrievedChunk.score` is always the final ordering score. For reranked results,
`RetrievedChunk.retrieval_score` holds the original BM25 score. The context builder keeps
finite negative scores because cross-encoder logits can legitimately be negative.

## Decision

BM25 remains the production retriever. The reranker, CLI, and evaluation path stay in the
repository so a later block can test a larger or domain-tuned model without rewriting the
pipeline.

## Deferred Work

- Try a stronger reranker such as `BAAI/bge-reranker-base` or a domain-tuned model.
- Tune candidate depth, truncation, and pair formatting only against the development split.
- Consider score blending or RRF if a future reranker has useful confidence.
- Cache reranker scores and reduce CPU latency.
- Add an automated live-model evaluation once startup cost is acceptable.
