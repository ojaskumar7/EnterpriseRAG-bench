# Block 12 Observability Implementation Plan

## Tasks

1. Add logging settings, request-ID context, and JSON/human-readable formatters.
2. Add FastAPI request middleware with safe completion events.
3. Add query and ingestion lifecycle events with counters and latency.
4. Add tests for propagation and sensitive-field exclusion.
5. Update README, API architecture, deferred-work, roadmap, and `blog8.md`.
6. Run focused tests and Ruff; defer Docker and frontend slices.

## Constraints

- Never log secrets, bearer tokens, prompts, questions, answers, or retrieved content.
- Keep request logging cheap and non-blocking.
- Preserve existing API and ingestion behavior.
