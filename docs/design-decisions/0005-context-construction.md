# 0005: Rank-First Token-Budgeted Context Construction

- Status: accepted
- Date: 2026-11-16
- Scope: Block 5 evidence selection over production BM25 results

## Context

Block 4 returns ranked BM25 chunks. Passing raw top-k chunks to a generation model would
be unbounded, redundant, and citation-unfriendly. The next block needs a deterministic
evidence boundary that:

- Never exceeds a predictable prompt budget.
- Keeps the strongest evidence first.
- Avoids many chunks from one document crowding out other evidence.
- Produces stable citation IDs.
- Can be evaluated without generation or gold evidence spans.

Three approaches were considered:

1. Rank-first packing with a per-document cap.
2. Adjacent-chunk merging before packing.
3. Sentence extraction or a score-per-token knapsack.

## Decision

Use **rank-first packing of whole chunks** with a 4,000-token budget and at most three
chunks per document.

Candidates are filtered to positive finite scores, sorted by score descending and
`chunk_id` ascending, deduplicated by chunk ID, and packed in order. A chunk that does not
fit is skipped while lower-ranked candidates are still considered. Citations are assigned
after selection, so they remain sequential and reproducible.

Use the stored chunk `token_count` from PostgreSQL, falling back to whitespace tokenization
when it is unavailable. Do not add a source-type cap: many enterprise questions target one
source, and source diversity is reported rather than enforced.

## Why not the alternatives

**Adjacent-chunk merging** requires database neighbor reads, can introduce text that was
not retrieved, and makes the unit of citation ambiguous. It is better tested once gold
evidence spans and generation quality metrics exist.

**Sentence extraction and knapsack packing** can fit more evidence per token, but they
need a reliable sentence relevance signal. With document-level labels only, the extra
complexity would be tuned against a weak proxy. Rank-first packing is easy to explain,
test, and compare.

## Consequences

- **Deterministic.** Same candidates and settings always produce the same context.
- **Bounded.** The token total never exceeds the budget.
- **Citation-ready.** Every selected chunk has a stable `[n]` identifier and document
  metadata.
- **Budget-limited recall.** A 4,000-token whole-chunk context covers about 10 of 50 BM25
  candidates. On the development split, context document recall is `0.7744` versus
  top-50 candidate recall `0.8962`. At the same candidate chunk count the gap is only
  `+0.0052`, so the selector is not losing much beyond the budget itself.
- **Whole-chunk waste.** A long chunk can be skipped even when only part of it is
  relevant. Partial-chunk extraction is deferred.
- **Tokenizer proxy.** Stored BGE token counts approximate the future generation
  tokenizer; the final prompt budget must be rechecked.
- **No diversity guarantee.** One source type may dominate. This is deliberate for
  single-source questions and is deferred until source-diversity evaluation exists.
