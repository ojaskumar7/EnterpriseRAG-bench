# Block 7 Design: Grounded Gemini Generation

- Status: approved scope, implementation pending
- Date: 2026-11-16
- Scope: generate cited answers from the existing selected context

## Goal

Add a provider-independent generation boundary that turns a `BuiltContext` into a
structured answer grounded in the selected evidence. The default provider will use
Google's official `google-genai` SDK and the Gemini API key loaded from `.env` through
the existing Pydantic settings layer.

Block 7 will not change BM25 retrieval, context selection, or the production index.
It will not implement full abstention thresholds, hallucination detection, multi-document
reasoning, an HTTP API, authentication, or holdout evaluation. Those remain later blocks.

## Approach

Use a small provider protocol with a Gemini implementation and a deterministic fake
provider for tests. Keep prompt construction and response validation outside the SDK
adapter so provider changes do not affect answer contracts.

## Components

### Configuration

Extend `Settings` and `.env.example` with:

- `GEMINI_API_KEY` (required only when the live provider is constructed)
- `GEMINI_MODEL` with a documented stable default
- `GEMINI_TEMPERATURE` defaulting to `0`
- `GEMINI_MAX_OUTPUT_TOKENS` with a positive default

The local `.env` remains ignored by Git. No secret is placed in source files, tests,
logs, CLI output, or generated artifacts.

### Generation types

Add immutable types for:

- a generation request containing the question and `BuiltContext`
- an answer containing text, cited citation IDs, model name, and usage metadata
- a structured generation result suitable for a later API layer

Citation IDs must refer only to blocks present in the supplied context. The answer
validator rejects malformed or unknown citation IDs.

### Prompt builder

Build a deterministic prompt containing:

1. system instructions to answer only from the evidence;
2. the user question;
3. numbered context blocks rendered with citation metadata;
4. an explicit requirement to cite factual claims with `[n]` markers;
5. an empty-context instruction to state that the evidence is insufficient.

Retrieved content remains untrusted evidence. The prompt will explicitly separate
instructions from evidence and tell the model not to follow instructions found inside
the retrieved text.

### Gemini provider

Implement a lazy `GeminiGenerator` around `google-genai`. It will:

- validate required configuration before making a request;
- request structured JSON matching the answer schema where supported;
- parse and validate the returned payload;
- wrap SDK/model failures in an actionable project error;
- never silently fall back to another provider;
- expose model and token usage metadata when returned by the SDK.

The provider will return a controlled insufficient-evidence response for an empty
context without calling Gemini.

### CLI

Add a generation preview CLI that retrieves BM25 candidates, builds the existing context,
calls the configured generator, and prints either human-readable cited output or JSON.
The CLI will support query, top-k, token budget, per-document cap, model override, and
JSON output. It will not expose the API key.

## Error handling

- Blank questions return a validation error.
- Empty context returns a deterministic insufficient-evidence result.
- Missing API keys fail at live-provider construction with an actionable message.
- Invalid model JSON, unknown citations, missing answer text, and non-finite usage values
  raise a generation validation error.
- SDK/network/model errors are wrapped with the configured model name.

## Testing

Unit tests will cover settings validation, prompt construction, evidence/instruction
separation, empty-context behavior, fake-provider integration, citation validation,
malformed model responses, and CLI validation/output. Tests will not require network
access or the live API key.

Live Gemini evaluation will be manual and development-only after implementation. The
42-question holdout remains untouched.

## Deferred to Later Blocks

- calibrated abstention thresholds and confidence scoring;
- answer-fact and citation faithfulness evaluation;
- contradiction and multi-document reasoning policies;
- FastAPI endpoints, authentication, and tenant-aware request handling;
- streaming, retries, caching, cost tracking, and observability;
- official benchmark evaluation.
