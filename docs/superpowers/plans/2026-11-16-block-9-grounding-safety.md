# Block 9 Grounding Safety Implementation Plan

## Tasks

1. Add claim and safety-result types with backward-compatible generated answers.
2. Add deterministic evidence and citation policy validation.
3. Extend the Gemini JSON contract and fake provider fixtures for claims.
4. Add conflict metadata without attempting truth resolution.
5. Add unit tests and a local safety-validation CLI over saved generation results.
6. Update README, architecture, deferred-work, and `blog5.md`.
7. Run focused tests and Ruff; keep holdout and full live evaluation untouched.

## Constraints

- Abstain rather than invent evidence.
- Never pass gold labels to the generator.
- Preserve the production BM25 path.
- Keep the policy pure and testable without network access.
