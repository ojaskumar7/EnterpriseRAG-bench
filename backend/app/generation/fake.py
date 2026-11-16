"""Test provider for generation flows."""

from __future__ import annotations

from collections.abc import Callable

from app.generation.types import GeneratedAnswer, GenerationRequest, validate_answer


class FakeGenerator:
    def __init__(self, response: GeneratedAnswer | Callable[[GenerationRequest], GeneratedAnswer]):
        self.response = response

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        answer = self.response(request) if callable(self.response) else self.response
        return validate_answer(answer, request.context)
