# 0007: Freeze v2-m3 reranking at candidate depth 10

- Status: candidate frozen for generation; holdout deferred
- Date: 2026-11-18
- Scope: development retrieval audit and candidate-generation gate

## Decision

Use BM25 followed by `BAAI/bge-reranker-v2-m3` over the top 10 BM25 candidates,
then append untouched BM25 ranks 11–50. Depth 10 is the smallest tested depth
retaining at least 80% of the full top-50 MRR gain.

The one-sided context-recall gate is passed: depth-10 context recall is
`0.8015` versus BM25 `0.7744`. Retrieval quality is measured over all 169 dev
questions. Warmed latency and VRAM use a separate 25-question performance
sample.

## Retrieval results

| BM25 rerank depth | MRR | recall@5 | recall@20 | context recall | p50 ms | p95 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.7834 | 0.8070 | 0.8735 | 0.8015 | 631.6 | 639.0 |
| 20 | 0.7841 | 0.8070 | 0.8821 | 0.8388 | 1,374.9 | 1,485.3 |
| 30 | 0.7841 | 0.8070 | 0.8858 | 0.7973 | 2,452.0 | 2,606.8 |
| 50 | 0.7842 | 0.8070 | 0.8858 | 0.7744 | 4,402.6 | 4,639.4 |

BM25 baseline MRR is `0.6779`; the full-depth gain is `+0.1063`, so the 80%
threshold is `+0.0851`. Depth 10 gains `+0.1055` (`99.2%` of full gain).
Ranks 11–50 are not dropped; they remain in original BM25 order after the
reranked head.

At the frozen depth 10, the paired comparison has 41 wins, 16 losses, and 112
ties. The bootstrap 95% CI is `[0.0578, 0.1541]` for MRR delta and
`[0.0212, 0.1336]` for recall@5 delta.

Peak reserved VRAM was 2,544 MiB at depth 10, 2,924 MiB at 20, 3,084 MiB at
30, and 3,140 MiB at 50. Model load was `5,284 ms`; depth-10 cold first-query
total was `7,582 ms`. The warmed depth-10 p50/p95 was `631.6/639.0 ms`.

Complete per-question paired data is in
`data/processed/reranking_depth_audit_dev_tail.json`.

## Dev configuration ledger

The three rerankers tried were `cross-encoder/ms-marco-MiniLM-L-6-v2`,
`BAAI/bge-reranker-base`, and `BAAI/bge-reranker-v2-m3`. The dev ledger totals
16 configurations:

- 3 reranker configurations;
- 6 RRF grid configurations (three RRF k values × two candidate depths);
- 3 BGE query prefix/dtype configurations;
- 4 v2-m3 reranker depths (10, 20, 30, and 50).

## False-abstain audit

The cached BM25 baseline has 39 false abstentions: 20 retrieval misses and 19
context-present-but-abstained cases. In the deterministic first 10 false-
abstain sample, 4 were retrieval misses and 6 had a gold document in the
selected context. The detailed per-question audit is in
`data/processed/false_abstain_audit.json`.

For the 20 retrieval misses, the frozen depth-10 candidate context contains a
gold document for 2 cases, so 2 are newly recoverable by retrieval. Reading
all 19 context-present cases gives: 1 evidence split across multiple docs, 0
valid answers rejected by the grounding/evidence gate, 11 model declines
despite sufficient evidence, and 7 cases where the selected context does not
support the gold answer detail.

## Generation gate and holdout policy

The BM25 dev baseline is 118 grounded (`0.6982`), 39 abstain (`0.2308`), and
12 conflict (`0.0710`); false-abstain is `39/169` (`0.2308`). Candidate
generation uses the same `gemini-3.5-flash-lite` model, temperature, prompt,
and grounding settings, with per-question append-only disk caching and
transient 503/429 backoff with jitter.

The candidate run persisted 43 of 169 records before the provider stalled;
126 remain. The partial paired comparison is explicitly non-final: on those
43 cached questions, BM25 was 32 grounded / 8 abstain / 3 conflict, while the
candidate was 9 grounded / 31 abstain / 3 conflict. Abstention transitions
were 4 abstain→abstain, 4 abstain→non-abstain, 27 non-abstain→abstain, and 8
non-abstain→non-abstain. This partial result is a regression and is not a
holdout gate.

The 30-question unanswerable generation run cached 24 questions before the
provider stalled: 16 abstained, giving partial abstention recall `0.6667`
(`16/24`); 6 questions remain unevaluated. This is labeled partial in
`data/processed/generation_abstention_metrics.json`.

No tuning is permitted after a successful candidate generation gate. Once the
candidate run completes and improves or holds the baseline rates, compare the
paired abstention outcomes with `scripts/compare_generation_eval.py`, then run
holdout exactly once with the frozen depth-10 configuration.
