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

## Block 7: Grounded Generation

- **No live benchmark yet.** The Gemini provider and local contracts are implemented, but
  live development evaluation is manual and requires credentials and model access.
- **No calibrated abstention.** Empty context has a deterministic fallback, but confidence
  thresholds and unsupported-claim detection are deferred to the abstention block.
- **No faithfulness metrics.** Citation presence is structurally validated; citation
  correctness, answer-fact coverage, and hallucination rates require a later evaluator.
- **No streaming or caching.** The provider retries a small number of temporary capacity
  failures, but streaming, caching, and broader operational controls are deferred until
  the API and workload are defined.
- **Model availability can change.** The default is pinned to the current stable Gemini
  model, but `GEMINI_MODEL` remains configurable for future migrations.

## Block 8: Offline Evaluation

- **The first report is deterministic only.** It measures citation structure, document
  recall, abstention, tokens, and latency; it does not judge answer correctness.
- **The full live generation run is deferred.** The five-question sample is useful for
  smoke diagnostics, but 169 Gemini calls require an explicit quota and capacity decision.
- **No LLM judge yet.** Faithfulness, answer-fact coverage, and unsupported-claim scoring
  require a later labeled or judge-backed evaluation design.

## Block 9: Grounding Safety

- **Structural, not semantic, validation.** Citation IDs and claim structure are checked,
  but the system does not prove that a passage entails each claim.
- **Provider-reported conflicts only.** The first version surfaces conflicts Gemini marks;
  it does not independently infer disagreement or choose source authority.
- **No version precedence.** Temporal document updates and authoritative-source ranking need
  metadata and a separate policy design.
- **Report decisions are structural.** Grounding reports now count `grounded`, `abstain`,
  and `conflict` decisions, but these labels do not establish factual correctness.
- **Claim overlap is lexical.** The first claim verifier uses token overlap to catch obvious
  unsupported claims; paraphrase-aware entailment and an LLM judge remain future work.
- **Block 9 is deterministic by design.** It is complete as a structural safety boundary;
  semantic faithfulness and source-authority resolution are future work.

## Block 10: API and Authentication

- **Local JWT only.** The first API uses an HS256 secret from `.env`; external identity
  providers, refresh tokens, and user management are deferred.
- **No operational controls yet.** Rate limiting, request quotas, audit logging, and
  production secret rotation belong to the deployment block.
- **No streaming.** The query response is synchronous and returns the completed grounded
  answer.
- **No refresh or revocation lists.** JWT expiry is validated, but token revocation and
  session management require an identity service.

## Block 11: Ingestion Lifecycle

- **In-process jobs only.** The first runner uses asyncio tasks and threads; Redis/Celery,
  durable job persistence, retries, and distributed workers are deferred.
- **Registered sources only.** The API does not accept arbitrary filesystem paths or uploads
  yet. Object storage and source adapters need their own trust and validation boundaries.
- **No implicit deletion.** Content-addressed document identity is preserved; tombstones,
  version precedence, and lifecycle migrations remain future work.

## Block 12: Observability

- **Logs only for now.** Metrics export, tracing, dashboards, retention, and alerting are
  deferred until the runtime/deployment slice.
- **No frontend yet.** The API is observable but there is no user-facing operations view.
- **Local container runtime only.** The API image and Compose database wiring are present,
  but BM25 artifact mounts, image hardening, registry publishing, and production deployment
  are deferred.
