# Block 7 Grounded Generation Implementation Plan

## Goal

Generate structured, citation-bearing answers from the existing BM25-selected context
using a provider-independent interface and an optional live Gemini provider.

## Tasks

1. Add Gemini settings and the `google-genai` dependency without exposing secrets.
2. Add immutable generation request/result types and validation for citation IDs.
3. Add deterministic prompt construction with explicit evidence/instruction separation.
4. Add a lazy Gemini provider with actionable configuration and SDK errors.
5. Add a fake provider and unit tests for the complete generation boundary.
6. Add a generation preview CLI over BM25 and context construction.
7. Update README, architecture documentation, and deferred-work notes.
8. Run focused tests, full tests, Ruff, and a manual live smoke test only if the local
   `.env` is configured.

## Constraints

- Never print, read back, or commit the API key.
- Do not touch the holdout split.
- Do not change BM25 or context ranking behavior.
- Empty context must not call Gemini.
- Full abstention calibration and answer-quality benchmarking are deferred.
