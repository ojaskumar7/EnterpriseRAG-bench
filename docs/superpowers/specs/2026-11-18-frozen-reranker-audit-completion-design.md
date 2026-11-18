# Frozen reranker audit completion design

## Goal

Complete the frozen-depth audit for BM25 plus `BAAI/bge-reranker-v2-m3` at
depth 10, make the depth semantics production-like, finish the cached
generation gate, and run holdout exactly once only after that gate passes or
holds.

## Retrieval semantics

For each BM25 top-50 result list, rerank only the first N results for N in
`{10, 20, 30, 50}`. Preserve the untouched BM25 tail after the reranked head,
with chunk-ID deduplication. The final list remains top-50. This makes
recall@20 meaningful at depth 10 and makes the frozen generation candidate
compatible with the existing top-50 context builder.

The evaluator will record, per question and depth, the final document order,
MRR, recall@5, recall@20, context document recall, and paired MRR/recall@5
delta statistics against the same BM25 list. Bootstrap uses 10,000 seeded
question-level resamples and percentile 95% intervals.

## Performance measurement

Quality and performance are separate passes. Performance records model load
time and cold-start first-query latency, then warms the model and records
per-query p50/p95 latency by depth. CUDA peak allocated and reserved memory are
reset before the warmed pass. The report records sample size and explicitly
marks model-load inclusion/exclusion.

## Configuration ledger

The ADR will count all development configurations evaluated, grouped as:

- three rerankers: MiniLM, `bge-reranker-base`, and `bge-reranker-v2-m3`;
- six RRF grid configurations: three RRF k values × two candidate depths;
- three BGE query prefix/dtype configurations: prefix-on float32, prefix-off
  float32, and prefix-on float16;
- four v2-m3 rerank depths: 10, 20, 30, and 50.

Existing report artifacts remain the source for completed prior groups; the
depth report supplies the new paired and performance measurements.

## Resumable generation

The generation runner will use the exact baseline model, temperature, prompt,
token settings, and grounding policy. It will maintain an append-only JSONL
cache keyed by question ID. Cached IDs are loaded before retrieval and are
never re-called. New records are flushed after every successful question.
Transient 503/429 failures use bounded exponential backoff with random jitter;
non-transient failures fail the run without deleting completed cache records.

The comparison report will include grounded, abstain, conflict, and
false-abstain rates plus paired per-question abstention transitions:
`abstain→abstain`, `abstain→non-abstain`, `non-abstain→abstain`, and
`non-abstain→non-abstain`.

## False-abstain audit

Using only cached baseline generation records, identify all 39 false-abstain
questions and record whether at least one expected gold document is in the
selected context. The first 10 in stable JSONL order will be manually
classified as retrieval miss or context-present-but-abstained. The ADR will
contain both aggregate counts and the 10-question classification table.

## Holdout gate

No holdout input is read before candidate dev generation is complete. If the
candidate improves or holds grounded/abstain/conflict and false-abstain rates
under the documented comparison, run retrieval and generation on holdout once
with the frozen depth-10 configuration. No tuning follows that run. If the
provider prevents a complete candidate run, holdout remains untouched.

## Validation

Add unit coverage for tail preservation, paired transition accounting,
resumable cache skipping, retry classification/backoff boundaries, and
configuration-count serialization. Run the existing targeted evaluation and
generation tests plus compile checks before any holdout command.
