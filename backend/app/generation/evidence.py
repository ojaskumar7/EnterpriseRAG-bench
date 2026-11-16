"""Deterministic claim-to-evidence overlap checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.context.types import BuiltContext
from app.generation.types import Claim

TOKEN_PATTERN = re.compile(r"\b[\w]+\b")


@dataclass(frozen=True)
class ClaimEvidenceCheck:
    claim_text: str
    matched_tokens: int
    claim_tokens: int
    overlap: float
    supported: bool


def check_claims(
    claims: tuple[Claim, ...],
    context: BuiltContext,
    *,
    min_overlap: float = 0.2,
) -> tuple[ClaimEvidenceCheck, ...]:
    """Check lexical overlap between claims and their cited evidence."""

    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be between 0 and 1")

    blocks_by_id = {block.citation_id: block for block in context.blocks}
    checks = []
    for claim in claims:
        claim_tokens = _tokens(claim.text)
        evidence_text = " ".join(
            f"{blocks_by_id[citation_id].title} {blocks_by_id[citation_id].content}"
            for citation_id in claim.citation_ids
            if citation_id in blocks_by_id
        )
        evidence_tokens = _tokens(evidence_text)
        matched_tokens = len(claim_tokens & evidence_tokens)
        overlap = matched_tokens / len(claim_tokens) if claim_tokens else 0.0
        checks.append(
            ClaimEvidenceCheck(
                claim_text=claim.text,
                matched_tokens=matched_tokens,
                claim_tokens=len(claim_tokens),
                overlap=overlap,
                supported=bool(claim_tokens) and overlap >= min_overlap,
            )
        )
    return tuple(checks)


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(value)}
