# Retrieval quality audit

All figures below are development diagnostics over 169 questions.

## Dense vs BM25 overlap

Question-level top-20 overlap (a hit means at least one gold document appears):

| Dense top-20 | BM25 top-20 | Questions | Share |
|---|---|---:|---:|
| hit | hit | 100 | 59.17% |
| hit | miss | 4 | 2.37% |
| miss | hit | 48 | 28.40% |
| miss | miss | 17 | 10.06% |

BM25 leads because its indexed `title + content` lexical signal captures
answer-bearing titles and exact enterprise terms that the content-only dense
embeddings omit; the dense model adds semantic matches but does not recover the
lexical gap.

Headline metrics:

| Retriever | recall@5 | recall@20 | MRR |
|---|---:|---:|---:|
| Dense, prefix-on float16 | 0.4761 | 0.5777 | 0.4345 |
| BM25 | 0.7310 | 0.8560 | 0.6779 |
| BM25+dense RRF (best tested 10/50) | 0.7036 | 0.8230 | 0.6245 |

## Title re-embedding status

No full title+content dense re-embedding run was completed. The available
title sanity check found answer-bearing title/metadata terms in 3 of 5 sampled
gold titles. This supports the asymmetry hypothesis but is not a title
re-embedding result; a new embedding index is required before claiming one.

## Frozen reranker

The current frozen candidate and depth sweep are documented in
[`0007-reranker-depth-freeze.md`](../design-decisions/0007-reranker-depth-freeze.md),
with raw ignored artifacts under `data/processed/`.
