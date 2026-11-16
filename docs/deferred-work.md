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
