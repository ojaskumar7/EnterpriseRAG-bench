"""Deterministic grounding and abstention policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.context.types import BuiltContext
from app.generation.evidence import ClaimEvidenceCheck, check_claims
from app.generation.types import GeneratedAnswer

GroundingDecision = Literal["grounded", "partial", "abstain", "conflict"]


@dataclass(frozen=True)
class GroundingPolicy:
    min_citations: int = 1
    min_documents: int = 1
    min_claim_overlap: float = 0.2

    def __post_init__(self) -> None:
        if self.min_citations <= 0:
            raise ValueError("min_citations must be positive")
        if self.min_documents <= 0:
            raise ValueError("min_documents must be positive")
        if not 0.0 <= self.min_claim_overlap <= 1.0:
            raise ValueError("min_claim_overlap must be between 0 and 1")


@dataclass(frozen=True)
class GroundingResult:
    decision: GroundingDecision
    answer: GeneratedAnswer
    reason: str
    claim_checks: tuple[ClaimEvidenceCheck, ...] = ()


def apply_grounding_policy(
    answer: GeneratedAnswer,
    context: BuiltContext,
    *,
    policy: GroundingPolicy | None = None,
) -> GroundingResult:
    """Return a safe answer decision without attempting semantic truth judgment."""

    policy = policy or GroundingPolicy()
    if answer.insufficient_evidence:
        return GroundingResult("abstain", answer, "provider marked evidence insufficient")
    if len(context.blocks) < policy.min_citations:
        return _abstention(answer, "context does not meet the minimum citation threshold")
    if context.selected_doc_count < policy.min_documents:
        return _abstention(answer, "context does not meet the minimum document threshold")
    if len(answer.citation_ids) < policy.min_citations:
        return _abstention(answer, "answer does not cite enough evidence")
    if answer.claims and any(not claim.citation_ids for claim in answer.claims):
        return _abstention(answer, "at least one claim has no supporting citation")
    claim_checks = check_claims(
        answer.claims,
        context,
        min_overlap=policy.min_claim_overlap,
    )
    unsupported = [check for check in claim_checks if not check.supported]
    if unsupported and len(unsupported) == len(claim_checks):
        return _abstention(answer, "claims do not overlap their cited evidence")
    if unsupported:
        supported_claims = tuple(
            claim
            for claim, check in zip(answer.claims, claim_checks, strict=True)
            if check.supported
        )
        supported_citations = tuple(
            dict.fromkeys(
                citation_id for claim in supported_claims for citation_id in claim.citation_ids
            )
        )
        partial_answer = GeneratedAnswer(
            answer="Supported evidence:\n"
            + "\n".join(f"- {claim.text}" for claim in supported_claims),
            citation_ids=supported_citations,
            model=answer.model,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            claims=supported_claims,
        )
        return GroundingResult(
            "partial",
            partial_answer,
            "some claims do not overlap their cited evidence",
            tuple(claim_checks),
        )
    if answer.conflicts:
        return GroundingResult(
            "conflict",
            answer,
            "provider identified conflicting evidence",
            tuple(claim_checks),
        )
    return GroundingResult(
        "grounded",
        answer,
        "answer meets deterministic grounding gates",
        tuple(claim_checks),
    )


def _abstention(answer: GeneratedAnswer, reason: str) -> GroundingResult:
    safe_answer = GeneratedAnswer(
        answer="The available evidence is insufficient to answer this question.",
        citation_ids=(),
        model=answer.model,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        insufficient_evidence=True,
    )
    return GroundingResult("abstain", safe_answer, reason)
