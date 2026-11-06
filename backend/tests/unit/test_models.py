from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.ingestion.models import Chunk, RawDocument

VALID_DOC_ID = "dsid_" + "a" * 32


def _raw_document(**overrides: object) -> RawDocument:
    payload: dict[str, object] = {
        "doc_id": VALID_DOC_ID,
        "source_type": "confluence",
        "title": "Title",
        "content": "Content",
        "source_path": f"confluence/{VALID_DOC_ID}__title.txt",
    }
    payload.update(overrides)
    return RawDocument(**payload)


def test_valid_doc_id_is_accepted() -> None:
    document = _raw_document()

    assert document.doc_id == VALID_DOC_ID
    assert document.source_type == "confluence"


@pytest.mark.parametrize(
    "doc_id",
    [
        "not-a-doc-id",
        "dsid_short",
        "dsid_" + "g" * 32,
        "DSID_" + "a" * 32,
    ],
)
def test_invalid_doc_id_is_rejected(doc_id: str) -> None:
    with pytest.raises(ValidationError, match="doc_id"):
        _raw_document(doc_id=doc_id)


def test_unknown_source_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="source_type"):
        _raw_document(source_type="unknown")


def test_chunk_serializes_to_json() -> None:
    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id=VALID_DOC_ID,
        chunk_index=0,
        content="chunk text",
        token_count=2,
        start_char=0,
        end_char=10,
        content_hash="abc123",
        source_type="jira",
        title="Title",
    )

    payload = json.loads(chunk.model_dump_json())

    assert payload["chunk_id"] == "chunk-1"
    assert payload["doc_id"] == VALID_DOC_ID
    assert payload["source_type"] == "jira"
    assert payload["token_count"] == 2
