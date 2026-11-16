# Deferred Work and Deliberate Compromises

This file records shortcuts taken to finish the project faster. Each item is a known
trade-off, not an accidental omission. Revisit after the end-to-end project is complete.

## Block 5: Context Construction

- **Token budget is a proxy.** Context packing uses the stored BGE token count for each
  chunk, not the future generation model's tokenizer. This is accurate enough for
  selection and cheap to run, but the final prompt budget must be rechecked when Gemini is
  integrated.
- **Whole chunks only.** The builder does not split chunks, merge adjacent neighbors, or
  extract sentences. Selection is simpler and deterministic, but a long chunk can be
  skipped even when part of it is relevant.
- **Document-level evaluation.** Context quality is measured with expected-document recall
  and token/redundancy metrics. There are no gold evidence spans, so sentence-level
  precision and answer-fact coverage are not measured.
- **No source-diversity cap.** At most three chunks per document are allowed, but one
  source type may dominate the context. This avoids hurting single-source questions.
- **No latency optimization.** Build latency is measured, but the first implementation
  favors clarity over micro-optimization.
- **No prompt-injection sanitization.** Retrieved text is treated as untrusted evidence,
  but sanitization and instruction isolation are deferred to generation.
- **Top-50 context recall gap is expected.** A 4,000-token whole-chunk context covers about
  10 of 50 candidates, so context document recall (`0.7744`) is naturally below top-50
  BM25 candidate recall (`0.8962`). The acceptance gate compares against BM25 at the same
  context size, where the gap is `0.0052`. Revisit if the generation context budget grows
  or partial-chunk extraction is added.
- **CLI success path is manually verified.** The live-index context CLI JSON contract was
  checked against the production index, but there is no automated test that requires the
  full index and database. Unit tests cover builder behavior and CLI validation.

## Block 6: Cross-Encoder Reranking

- **MiniLM is not adopted.** `cross-encoder/ms-marco-MiniLM-L-6-v2` reduced every retrieval
  cut-off, MRR, and context document recall on the development split. BM25 remains the
  production path. Try `BAAI/bge-reranker-base`, a domain-tuned model, or different
  truncation/pair formatting in a later block.
- **No score blending.** Reranker scores replace BM25 ordering outright. Normalized
  blending or RRF was deferred because the first model failed and blending would invite
  tuning against the development split.
- **No reranker caching or batching across questions.** Each query scores its own top-50
  candidates. First-call model loading and per-query latency were measured but not
  optimized.
- **No automated live-model test.** Unit tests use fake predictors and rerankers. The real
  MiniLM run is manual because it requires model weights, GPU, the production index, and
  PostgreSQL.
- **Pair truncation is fixed at 512 tokens.** Title plus content can exceed the model
  limit; a later reranker comparison should measure passage truncation explicitly.
