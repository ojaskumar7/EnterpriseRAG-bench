# Grounding Safety Architecture

- Status: first safety slice implemented
- Scope: deterministic abstention, structured claims, and conflict metadata

## Flow

```text
GeneratedAnswer
  -> citation/claim validation
  -> GroundingPolicy
  -> grounded | abstain | conflict
```

The safety layer runs after generation and is pure. It does not call Gemini, inspect gold
labels, or infer which source is true.

## Policy

The default policy requires at least one selected context citation, one selected document,
and at least one citation in a non-abstaining answer. Citation IDs must refer to the
selected context. Citationless answers are replaced with a deterministic insufficient-
evidence response.

The provider may return structured claims with their own citation IDs. Claims without
citations are invalid. It may also return conflict descriptions with the citations that
disagree; the policy exposes a `conflict` decision rather than silently choosing a side.

The current claim check measures normalized lexical overlap between each claim and its
cited passages. Claims with no meaningful overlap cause abstention; mixed results produce
a `partial` answer containing only supported claims. This is an auditable heuristic, not a
semantic entailment model.

Saved generation reports include the resulting grounding decisions, mean claim count, and
provider-reported conflict count. Legacy records without a decision are labeled `legacy`
instead of being silently reclassified.

## Limits

This layer validates structure and evidence references. It does not prove that a cited
claim is semantically entailed by its passage, rank source authority, or resolve document
versions. Those require a later fact-level evaluator and conflict-precedence design.
