"""Render built context for previews and JSON payloads."""

from __future__ import annotations

from dataclasses import asdict

from app.context.types import BuiltContext


def render_context(context: BuiltContext) -> str:
    sections = []
    for block in context.blocks:
        header = (
            f"[{block.citation_id}] {block.title} "
            f"({block.source_type}, doc_id={block.doc_id}, chunk={block.chunk_index})"
        )
        sections.append(f"{header}\n{block.content}")
    return "\n\n".join(sections)


def context_to_dict(context: BuiltContext) -> dict[str, object]:
    payload = asdict(context)
    payload["blocks"] = [asdict(block) for block in context.blocks]
    return payload
