# Block 9 Design: Grounding Safety and Abstention

- Status: approved
- Date: 2026-11-16
- Scope: prevent unsupported answers and expose evidence conflicts

## Goal

Add a safety boundary after generation that decides whether an answer is sufficiently
grounded in the selected context. The boundary must prefer an explicit abstention or a
partial answer over unsupported claims.

## Approach

Use a hybrid policy:

1. deterministic evidence gates before and after generation;
2. structured claims paired with citation IDs;
3. conflict-aware source grouping for multi-document answers.

This keeps basic safety cheap and predictable while leaving semantic claim verification to
a later judge-backed evaluator.

## Deterministic evidence gate

The policy validates:

- non-blank query;
- non-empty context for a non-abstaining answer;
- minimum selected citation count and optional minimum selected document count;
- citation IDs that exist in the context;
- every factual claim having at least one citation ID;
- no citation IDs outside the selected context.

An empty or insufficient context produces a structured abstention without calling Gemini.
An answer with no citations is converted to a safe abstention unless it is explicitly a
non-factual refusal or clarification.

## Structured claims

Extend the generated response contract with optional claims:

```text
claims: [{"text": "...", "citation_ids": [1, 2]}]
```

The validator requires each non-empty claim to have valid citation IDs. The existing
answer text remains available for display, while claims provide an auditable unit for
later fact-level evaluation.

## Conflict handling

Build a pure conflict descriptor from claims and cited source metadata. The first version
does not infer truth or silently resolve conflicts. It exposes source groups and marks a
response as conflicting when claims cite incompatible source statements supplied by the
provider. A later block can add temporal/version precedence rules.

## Non-goals

- LLM-as-judge correctness scoring.
- Automatic fact extraction from arbitrary prose.
- Document version precedence or authority ranking.
- API, authentication, streaming, or UI changes.
- Holdout evaluation.

## Testing

Tests will cover empty context, insufficient evidence, citationless answers, unknown
citations, claim citation validation, partial answers, and deterministic conflict metadata.
Existing Block 7 behavior remains compatible for provider responses without claims.
