"""Contracts for grounded answer generation."""

from __future__ import annotations

from dataclasses import dataclass

from app.context.types import BuiltContext


class GenerationError(RuntimeError):
    """Raised when a generation request or provider response is invalid."""


@dataclass(frozen=True)
class GenerationRequest:
    query: str
    context: BuiltContext

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be blank")


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citation_ids: tuple[int, ...]
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    insufficient_evidence: bool = False


def validate_answer(answer: GeneratedAnswer, context: BuiltContext) -> GeneratedAnswer:
    if not answer.answer.strip():
        raise GenerationError("provider returned an empty answer")

    available = {block.citation_id for block in context.blocks}
    invalid = sorted(set(answer.citation_ids) - available)
    if invalid:
        raise GenerationError(f"provider returned citation IDs not present in context: {invalid}")
    if len(set(answer.citation_ids)) != len(answer.citation_ids):
        raise GenerationError("provider returned duplicate citation IDs")
    if any(citation_id <= 0 for citation_id in answer.citation_ids):
        raise GenerationError("provider returned a non-positive citation ID")
    return answer
