# Block 8 Offline Evaluation Implementation Plan

## Tasks

1. Add a pure generation-report summarizer with deterministic metrics and validation.
2. Add JSON/Markdown rendering helpers and a report CLI.
3. Add unit tests for metrics, malformed inputs, holdout protection, and output rendering.
4. Update README, architecture, deferred-work, and local blog documentation.
5. Run focused tests and Ruff; defer the full live Gemini evaluation.

## Constraints

- No new Gemini calls.
- No holdout access.
- Do not change retrieval or generation behavior.
- Keep report artifacts ignored and reproducible.
