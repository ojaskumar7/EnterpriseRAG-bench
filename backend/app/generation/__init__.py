"""Grounded answer generation interfaces and providers."""

from app.generation.evidence import ClaimEvidenceCheck, check_claims
from app.generation.gemini import GeminiGenerator
from app.generation.prompt import build_generation_prompt
from app.generation.safety import GroundingPolicy, GroundingResult, apply_grounding_policy
from app.generation.types import (
    Claim,
    Conflict,
    GeneratedAnswer,
    GenerationError,
    GenerationRequest,
    validate_answer,
)

__all__ = [
    "GeminiGenerator",
    "ClaimEvidenceCheck",
    "check_claims",
    "GroundingPolicy",
    "GroundingResult",
    "apply_grounding_policy",
    "Claim",
    "Conflict",
    "GenerationError",
    "GenerationRequest",
    "GeneratedAnswer",
    "build_generation_prompt",
    "validate_answer",
]
