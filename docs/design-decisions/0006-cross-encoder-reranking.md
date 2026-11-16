# 0006: Optional MiniLM Cross-Encoder Reranking

- Status: accepted, not adopted for production
- Date: 2026-11-16
- Scope: Block 6 reranking over BM25 candidates

## Context

Block 4 productionized BM25 and Block 5 built a 4,000-token context over its top candidates.
BM25 reaches MRR `0.6779` and recall@20 `0.8560` on the 169-question development split
(evaluated at top-50 in the Block 6 run). A cross-encoder can compare the query and passage
directly, so Block 6 tested whether a small local model could improve ordering before
context construction.

The project already depends on `sentence-transformers`, `transformers`, and PyTorch. The
machine has an 8 GB CUDA GPU, and no new package was needed.

## Decision

Implement `cross-encoder/ms-marco-MiniLM-L-6-v2` as an **optional decorator**:

- `CrossEncoderReranker` lazily loads the model and scores query-passage pairs.
- `RerankedRetriever` wraps any retriever, fetches `candidate_k=50`, reranks, and returns
  up to `top_k=50`.
- `RetrievedChunk.score` becomes the reranker score; `retrieval_score` preserves BM25.
- Ties break by `chunk_id`.
- Failures raise; there is no silent fallback.
- BM25 remains the production path unless the development gate passes.

## Why a decorator

The decorator keeps BM25, dense, lexical, and hybrid experiments independent of model
loading and score policy. It is easy to test with a fake reranker, easy to compare against
BM25 on the same candidates, and easy for a later generation block to choose either path.

## Why MiniLM

It is small, fast enough for a first baseline, well supported by the installed
`sentence-transformers` version, and does not add dependencies. A larger
`BAAI/bge-reranker-base` remains a future option.

## Measured Outcome

The MiniLM reranker **failed the acceptance gate**:

| Metric | BM25 | Reranked |
|---|---:|---:|
| recall@1 | 0.5138 | 0.4403 |
| recall@5 | 0.7310 | 0.6501 |
| recall@10 | 0.7776 | 0.7377 |
| recall@20 | 0.8560 | 0.8210 |
| MRR | 0.6779 | 0.6203 |
| Context document recall | 0.7744 | 0.7175 |
| Retrieval p95 | 104.4 ms | 661.7 ms |

The model was trained for MS MARCO passage ranking, while this corpus mixes enterprise
sources and long chunks. Truncation at 512 tokens may also remove evidence. No tuning was
attempted because that would risk fitting the development split and because the gate was
already clearly missed.

## Consequences

- **Production stays BM25.** The measured negative result is not hidden behind a default.
- **The path remains.** A future block can swap the model or candidate depth without
  changing retrieval or context construction.
- **Negative scores are valid.** Cross-encoder logits can be below zero. The context
  builder now filters only non-finite scores, fixing a Block 5 integration bug.
- **Latency cost is visible.** MiniLM added roughly 0.34 seconds at p50 and 0.56 seconds at
  p95 in this run, including the shared process overhead.
- **No automatic fallback.** Choosing BM25 is explicit at the call site, so a broken
  reranker cannot silently change retrieval quality.
