from __future__ import annotations

import pytest

from app.context.builder import ContextBuilder
from app.generation.evidence import check_claims
from app.generation.safety import GroundingPolicy, apply_grounding_policy
from app.generation.types import Claim, Conflict, GeneratedAnswer
from app.retrieval.types import RetrievedChunk


def context() -> object:
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        source_type="wiki",
        title="Guide",
        chunk_index=0,
        content="The key rotates every 90 days.",
        score=1.0,
        metadata={},
    )
    return ContextBuilder().build("question", [chunk])


def test_citationless_answer_becomes_safe_abstention() -> None:
    result = apply_grounding_policy(
        GeneratedAnswer(answer="Do this.", citation_ids=(), model="fake"),
        context(),
    )

    assert result.decision == "abstain"
    assert result.answer.insufficient_evidence is True
    assert result.answer.citation_ids == ()


def test_grounded_claims_pass() -> None:
    result = apply_grounding_policy(
        GeneratedAnswer(
            answer="Rotate every 90 days [1].",
            citation_ids=(1,),
            model="fake",
            claims=(Claim("Rotate every 90 days.", (1,)),),
        ),
        context(),
    )

    assert result.decision == "grounded"


def test_conflicts_are_exposed_not_silently_resolved() -> None:
    result = apply_grounding_policy(
        GeneratedAnswer(
            answer="Sources disagree [1].",
            citation_ids=(1,),
            model="fake",
            conflicts=(Conflict("The dates differ.", (1,)),),
        ),
        context(),
    )

    assert result.decision == "conflict"
    assert "conflicting" in result.reason


def test_unsupported_claims_become_abstention() -> None:
    result = apply_grounding_policy(
        GeneratedAnswer(
            answer="The moon is made of cheese [1].",
            citation_ids=(1,),
            model="fake",
            claims=(Claim("The moon is made of cheese.", (1,)),),
        ),
        context(),
    )

    assert result.decision == "abstain"
    assert "overlap" in result.reason


def test_mixed_claims_return_supported_partial_answer() -> None:
    result = apply_grounding_policy(
        GeneratedAnswer(
            answer="Two claims [1].",
            citation_ids=(1,),
            model="fake",
            claims=(
                Claim("The key rotates every 90 days.", (1,)),
                Claim("The moon is made of cheese.", (1,)),
            ),
        ),
        context(),
    )

    assert result.decision == "partial"
    assert "rotates every 90 days" in result.answer.answer
    assert "moon is made" not in result.answer.answer


def test_claim_overlap_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        check_claims((), context(), min_overlap=1.1)


def test_policy_validates_positive_thresholds() -> None:
    with pytest.raises(ValueError):
        GroundingPolicy(min_citations=0)
    with pytest.raises(ValueError):
        GroundingPolicy(min_documents=0)
