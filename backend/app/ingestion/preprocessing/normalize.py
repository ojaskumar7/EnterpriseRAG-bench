"""Deterministic normalization for raw documents."""

from __future__ import annotations

import hashlib
import unicodedata

from app.ingestion.models import NormalizedDocument, RawDocument


def normalize_document(raw: RawDocument) -> NormalizedDocument:
    """Normalize a raw document and compute its content hash."""

    title = _normalize_text(raw.title)
    content = _normalize_text(raw.content)

    return NormalizedDocument(
        doc_id=raw.doc_id,
        source_type=raw.source_type,
        title=title,
        content=content,
        raw_content=raw.content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        char_count=len(content),
        is_empty=not content,
        metadata=dict(raw.metadata),
    )


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")

    collapsed: list[str] = []
    blank_count = 0
    for line in (line.rstrip() for line in text.split("\n")):
        if line:
            blank_count = 0
            collapsed.append(line)
            continue

        blank_count += 1
        if blank_count <= 2:
            collapsed.append("")

    return "\n".join(collapsed).strip("\n")
