# Block 11 Ingestion Lifecycle Implementation Plan

## Tasks

1. Add immutable job models and validated state transitions.
2. Add an in-process async job store/runner with fake callable support.
3. Add authenticated job submission/status API routes using server-side source mappings.
4. Add lifecycle/job tests without external services.
5. Update README, API architecture, deferred-work, roadmap, and `blog7.md`.
6. Run focused tests and Ruff; defer durable workers and full corpus ingestion.

## Constraints

- Never accept arbitrary client filesystem paths.
- Never expose another tenant's job status.
- Never publish partial ingestion artifacts.
- Keep raw/processed data and job state out of Git.
