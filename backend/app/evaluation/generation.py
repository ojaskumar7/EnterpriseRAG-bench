"""Deterministic reports for saved grounded-generation results."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from app.evaluation.dev import latency_summary


def summarize_generation(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Summarize saved generation records without calling external services."""

    if not records:
        raise ValueError("records must not be empty")

    models: Counter[str] = Counter()
    latencies: list[float] = []
    context_tokens: list[float] = []
    citation_counts: list[float] = []
    claim_counts: list[float] = []
    citation_presence = 0
    valid_citations = 0
    abstentions = 0
    conflicts = 0
    decisions: Counter[str] = Counter()
    recall_values: list[float] = []

    for record in records:
        answer = _mapping(record, "answer")
        context = _mapping(record, "context")
        citation_ids = _citation_ids(answer.get("citation_ids"))
        claims = answer.get("claims", [])
        if not isinstance(claims, list):
            raise ValueError("answer.claims must be a list")
        claim_counts.append(float(len(claims)))
        conflict_records = answer.get("conflicts", [])
        if not isinstance(conflict_records, list):
            raise ValueError("answer.conflicts must be a list")
        conflicts += len(conflict_records)
        citation_count = _non_negative_int(context, "citation_count")
        expected_doc_ids = _string_list(record, "expected_doc_ids")
        selected_doc_ids = set(_string_list(context, "selected_doc_ids"))
        latency_ms = record.get("latency_ms")
        if not isinstance(latency_ms, (int, float)) or latency_ms < 0:
            raise ValueError("latency_ms must be a non-negative number")

        model = answer.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("answer.model must be a non-empty string")
        models[model] += 1
        latencies.append(float(latency_ms))
        context_tokens.append(float(_non_negative_int(context, "token_count")))
        citation_counts.append(float(citation_count))
        citation_presence += bool(citation_ids)
        valid_citations += all(1 <= citation_id <= citation_count for citation_id in citation_ids)
        abstentions += bool(answer.get("insufficient_evidence", False))
        grounding = record.get("grounding")
        if isinstance(grounding, Mapping) and isinstance(grounding.get("decision"), str):
            decisions[grounding["decision"]] += 1
        else:
            decisions["legacy"] += 1
        if expected_doc_ids:
            recall_values.append(
                sum(doc_id in selected_doc_ids for doc_id in expected_doc_ids)
                / len(expected_doc_ids)
            )

    count = len(records)
    return {
        "question_count": count,
        "models": dict(sorted(models.items())),
        "grounding_decisions": dict(sorted(decisions.items())),
        "citation_presence_rate": citation_presence / count,
        "citation_validity_rate": valid_citations / count,
        "abstention_count": abstentions,
        "abstention_rate": abstentions / count,
        "selected_context_doc_recall": (
            sum(recall_values) / len(recall_values) if recall_values else 0.0
        ),
        "mean_context_tokens": sum(context_tokens) / count,
        "mean_citation_count": sum(citation_counts) / count,
        "mean_claim_count": sum(claim_counts) / count,
        "conflict_count": conflicts,
        "latency_ms": latency_summary(latencies),
    }


def render_generation_markdown(summary: Mapping[str, object]) -> str:
    """Render a compact human-readable generation report."""

    latency = summary["latency_ms"]
    if not isinstance(latency, Mapping):
        raise ValueError("summary.latency_ms must be a mapping")
    models = summary["models"]
    if not isinstance(models, Mapping):
        raise ValueError("summary.models must be a mapping")
    model_text = ", ".join(f"`{name}`: {count}" for name, count in models.items())
    decisions = summary["grounding_decisions"]
    if not isinstance(decisions, Mapping):
        raise ValueError("summary.grounding_decisions must be a mapping")
    decision_text = ", ".join(f"`{name}`: {count}" for name, count in decisions.items())
    return "\n".join(
        [
            "# Generation Evaluation Report",
            "",
            f"- Questions: {summary['question_count']}",
            f"- Models: {model_text}",
            f"- Grounding decisions: {decision_text}",
            f"- Citation presence rate: {float(summary['citation_presence_rate']):.4f}",
            f"- Citation validity rate: {float(summary['citation_validity_rate']):.4f}",
            f"- Abstentions: {summary['abstention_count']} "
            f"({float(summary['abstention_rate']):.4f})",
            f"- Selected-context document recall: "
            f"{float(summary['selected_context_doc_recall']):.4f}",
            f"- Mean context tokens: {float(summary['mean_context_tokens']):.1f}",
            f"- Mean citations: {float(summary['mean_citation_count']):.1f}",
            f"- Mean claims: {float(summary['mean_claim_count']):.1f}",
            f"- Provider-reported conflicts: {summary['conflict_count']}",
            f"- Latency p50/p95/max: {float(latency['p50_ms']):.1f}/"
            f"{float(latency['p95_ms']):.1f}/{float(latency['max_ms']):.1f} ms",
            "",
            "These are deterministic diagnostics, not answer-correctness or "
            "faithfulness judgments.",
        ]
    )


def _mapping(record: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = record.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _string_list(record: Mapping[str, Any], field: str) -> list[str]:
    value = record.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return value


def _citation_ids(value: object) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) for item in value):
        raise ValueError("answer.citation_ids must be a list of integers")
    if any(not isinstance(item, int) for item in value):
        raise ValueError("answer.citation_ids must be a list of integers")
    if len(set(value)) != len(value):
        raise ValueError("answer.citation_ids must not contain duplicates")
    return value


def _non_negative_int(record: Mapping[str, Any], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value
