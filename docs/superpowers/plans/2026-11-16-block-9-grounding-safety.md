# Block 9 Grounding Safety Implementation Plan

## Tasks

1. [x] Add claim and safety-result types with backward-compatible generated answers.
2. [x] Add deterministic evidence and citation policy validation.
3. [x] Extend the Gemini JSON contract and fake provider fixtures for claims.
4. [x] Add conflict metadata without attempting truth resolution.
5. [x] Add unit tests and offline report integration for safety decisions.
6. [x] Update README, architecture, deferred-work, and `blog5.md`.
7. [x] Run focused tests and Ruff; keep holdout and full live evaluation untouched.

## Constraints

- Abstain rather than invent evidence.
- Never pass gold labels to the generator.
- Preserve the production BM25 path.
- Keep the policy pure and testable without network access.
