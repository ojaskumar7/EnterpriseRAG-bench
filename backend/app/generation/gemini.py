"""Gemini-backed grounded answer generation."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.generation.prompt import build_generation_prompt
from app.generation.types import (
    Claim,
    Conflict,
    GeneratedAnswer,
    GenerationError,
    GenerationRequest,
    validate_answer,
)
from app.settings import Settings


class GeminiGenerator:
    """Lazily initialized Gemini provider for grounded answers."""

    def __init__(self, settings: Settings, *, model: str | None = None) -> None:
        self.api_key = settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.temperature = settings.gemini_temperature
        self.max_output_tokens = settings.gemini_max_output_tokens
        self.max_retries = settings.gemini_max_retries
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise GenerationError("GEMINI_API_KEY is required for live generation")
        try:
            from google import genai
        except ImportError as exc:
            raise GenerationError(
                "google-genai is not installed; run `uv sync` before live generation"
            ) from exc
        try:
            self._client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            raise GenerationError(f"could not initialize Gemini client: {exc}") from exc
        return self._client

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        if not request.context.blocks:
            return GeneratedAnswer(
                answer="The available evidence is insufficient to answer this question.",
                citation_ids=(),
                model=self.model,
                insufficient_evidence=True,
            )

        try:
            from google.genai import types

            response = self._generate_with_retry(request, types)
            payload = _parse_json_response(response.text)
            usage = getattr(response, "usage_metadata", None)
            answer = GeneratedAnswer(
                answer=_required_string(payload, "answer"),
                citation_ids=tuple(_citation_ids(payload.get("citation_ids", []))),
                model=self.model,
                prompt_tokens=_usage_value(usage, "prompt_token_count"),
                completion_tokens=_usage_value(usage, "candidates_token_count"),
                insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
                claims=tuple(_claims(payload.get("claims", []))),
                conflicts=tuple(_conflicts(payload.get("conflicts", []))),
            )
            return validate_answer(answer, request.context)
        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(
                f"Gemini generation failed for model {self.model}: {exc}"
            ) from exc

    def _generate_with_retry(self, request: GenerationRequest, types: Any) -> Any:
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )
        for attempt in range(self.max_retries + 1):
            try:
                return self._get_client().models.generate_content(
                    model=self.model,
                    contents=build_generation_prompt(request),
                    config=config,
                )
            except Exception as exc:
                if not _is_retryable(exc) or attempt == self.max_retries:
                    raise
                time.sleep(_retry_delay(exc, attempt))


def _parse_json_response(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GenerationError("Gemini returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise GenerationError("Gemini response must be a JSON object")
    return payload


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GenerationError(f"Gemini response field {field!r} must be a non-empty string")
    return value


def _citation_ids(value: Any) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) for item in value):
        raise GenerationError("Gemini response citation_ids must be a list of integers")
    if any(not isinstance(item, int) for item in value):
        raise GenerationError("Gemini response citation_ids must be a list of integers")
    return value


def _claims(value: Any) -> list[Claim]:
    if not isinstance(value, list):
        raise GenerationError("Gemini response claims must be a list")
    claims = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise GenerationError("Gemini response claims must contain text and citation_ids")
        claims.append(
            Claim(
                text=item["text"],
                citation_ids=tuple(_citation_ids(item.get("citation_ids", []))),
            )
        )
    return claims


def _conflicts(value: Any) -> list[Conflict]:
    if not isinstance(value, list):
        raise GenerationError("Gemini response conflicts must be a list")
    conflicts = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("description"), str):
            raise GenerationError(
                "Gemini response conflicts must contain description and citation_ids"
            )
        conflicts.append(
            Conflict(
                description=item["description"],
                citation_ids=tuple(_citation_ids(item.get("citation_ids", []))),
            )
        )
    return conflicts


def _usage_value(usage: Any, field: str) -> int | None:
    value = getattr(usage, field, None)
    return value if isinstance(value, int) and value >= 0 else None


def _retry_delay(error: Exception, attempt: int) -> float:
    if _is_quota_error(error):
        hint = re.search(r"retry in ([\d.]+)s", str(error), re.IGNORECASE)
        if hint:
            return max(float(hint.group(1)), 2**attempt)
        return 35.0
    return float(2**attempt)


def _is_quota_error(error: Exception) -> bool:
    message = str(error).upper()
    return "429" in message or "RESOURCE_EXHAUSTED" in message


def _is_retryable(error: Exception) -> bool:
    status = str(getattr(error, "status", "")).upper()
    code = str(getattr(error, "code", "")).upper()
    message = str(error).upper()
    return any(
        marker in status or marker in code or marker in message
        for marker in ("500", "503", "429", "INTERNAL", "UNAVAILABLE", "RESOURCE_EXHAUSTED")
    )
