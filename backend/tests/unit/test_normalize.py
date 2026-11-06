from __future__ import annotations

import hashlib

from app.ingestion.models import RawDocument
from app.ingestion.preprocessing.normalize import normalize_document

DOC_ID = "dsid_" + "a" * 32


def _raw(content: str, title: str = "Title") -> RawDocument:
    return RawDocument(
        doc_id=DOC_ID,
        source_type="confluence",
        title=title,
        content=content,
        source_path=f"confluence/{DOC_ID}__doc.txt",
    )


def test_unicode_is_nfkc_normalized() -> None:
    normalized = normalize_document(_raw("Ｆｕｌｌ　ｗｉｄｔｈ ﬁ", title="Ｔｉｔｌｅ"))

    assert normalized.title == "Title"
    assert normalized.content == "Full width fi"


def test_crlf_is_normalized_to_lf() -> None:
    normalized = normalize_document(_raw("Line one\r\nLine two\rLine three"))

    assert normalized.content == "Line one\nLine two\nLine three"


def test_null_bytes_are_removed() -> None:
    normalized = normalize_document(_raw("a\x00b"))

    assert normalized.content == "ab"


def test_excess_blank_lines_are_collapsed() -> None:
    normalized = normalize_document(_raw("First\n\n\n\nSecond\n"))

    assert normalized.content == "First\n\n\nSecond"


def test_headings_and_lists_are_preserved() -> None:
    content = "# Heading\n\n- item one\n- item two\n\n```python\nprint('x')\n```\n\n1. numbered"

    normalized = normalize_document(_raw(content))

    assert normalized.content == content


def test_content_hash_is_stable_across_runs() -> None:
    raw = _raw("Stable content\n\nWith a second paragraph.")

    first = normalize_document(raw)
    second = normalize_document(raw)
    expected = hashlib.sha256(first.content.encode("utf-8")).hexdigest()

    assert first.content_hash == expected
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64


def test_empty_document_is_flagged_not_crashed() -> None:
    normalized = normalize_document(_raw("\x00 \r\n\t\n"))

    assert normalized.content == ""
    assert normalized.is_empty is True
    assert normalized.char_count == 0
    assert normalized.content_hash == hashlib.sha256(b"").hexdigest()


def test_raw_content_is_preserved() -> None:
    raw = _raw("Raw\r\nContent")

    normalized = normalize_document(raw)

    assert normalized.raw_content == raw.content
    assert normalized.content == "Raw\nContent"
    assert normalized.char_count == len(normalized.content)
