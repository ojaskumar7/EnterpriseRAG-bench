from __future__ import annotations

import hashlib
import re
from itertools import pairwise

from app.ingestion.chunking.token_chunker import TokenChunker
from app.ingestion.models import NormalizedDocument

DOC_ID = "dsid_" + "a" * 32


class WhitespaceTokenizer:
    """Small deterministic tokenizer that treats each word as one token."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[object]]:
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {
            "input_ids": list(range(len(offsets))),
            "offset_mapping": offsets,
        }


def _document(
    content: str,
    *,
    metadata: dict[str, str] | None = None,
) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id=DOC_ID,
        source_type="confluence",
        title="Title",
        content=content,
        raw_content=content,
        content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        char_count=len(content),
        is_empty=not content,
        metadata=metadata or {},
    )


def _words(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


def _chunker(
    *,
    chunk_size: int = 10,
    chunk_overlap: int = 3,
    min_chunk_tokens: int = 2,
) -> TokenChunker:
    return TokenChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_tokens=min_chunk_tokens,
        tokenizer=WhitespaceTokenizer(),
    )


def test_short_document_produces_one_chunk() -> None:
    document = _document(_words(5))

    chunks = _chunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].content == document.content
    assert chunks[0].token_count == 5
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(document.content)


def test_long_document_produces_multiple_chunks() -> None:
    chunks = _chunker().chunk(_document(_words(30)))

    assert len(chunks) > 1


def test_no_chunk_exceeds_max_tokens() -> None:
    chunker = _chunker(chunk_size=10, chunk_overlap=3)
    tokenizer = WhitespaceTokenizer()

    chunks = chunker.chunk(_document(_words(30)))

    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert all(
        len(tokenizer(chunk.content, return_offsets_mapping=True)["offset_mapping"]) <= 10
        for chunk in chunks
    )


def test_adjacent_chunks_overlap() -> None:
    chunks = _chunker(chunk_size=10, chunk_overlap=3).chunk(_document(_words(30)))

    for previous, current in pairwise(chunks):
        previous_words = previous.content.split()
        current_words = current.content.split()
        assert previous_words[-3:] == current_words[:3]


def test_chunk_ids_are_deterministic() -> None:
    document = _document(_words(30))
    chunker = _chunker()

    first = chunker.chunk(document)
    second = chunker.chunk(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    expected = f"{DOC_ID}:0:{hashlib.sha256(first[0].content.encode('utf-8')).hexdigest()[:12]}"
    assert first[0].chunk_id == expected
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))


def test_character_offsets_map_to_original_content() -> None:
    document = _document(_words(30))

    chunks = _chunker().chunk(document)

    for chunk in chunks:
        assert document.content[chunk.start_char : chunk.end_char] == chunk.content


def test_trailing_small_chunk_is_merged() -> None:
    document = _document(_words(125))

    chunks = _chunker(chunk_size=100, chunk_overlap=20, min_chunk_tokens=50).chunk(document)

    assert len(chunks) == 2
    assert all(chunk.token_count >= 50 for chunk in chunks)
    assert chunks[-1].end_char == len(document.content)
    assert chunks[-1].content.split()[-1] == "word124"


def test_empty_document_produces_no_chunks() -> None:
    assert _chunker().chunk(_document("")) == []


def test_chunk_metadata_is_copied() -> None:
    document = _document(_words(5), metadata={"origin": "test"})

    chunks = _chunker().chunk(document)

    assert chunks[0].source_type == document.source_type
    assert chunks[0].title == document.title
    assert chunks[0].metadata == {"origin": "test"}


def test_chunks_carry_document_content_hash() -> None:
    document = _document(_words(30))

    chunks = _chunker().chunk(document)

    assert chunks
    assert all(chunk.document_content_hash == document.content_hash for chunk in chunks)


def test_chunk_id_does_not_depend_on_document_content_hash() -> None:
    content = _words(5)
    first = _document(content)
    second = first.model_copy(update={"content_hash": "different-document-hash"})

    first_chunk = _chunker().chunk(first)[0]
    second_chunk = _chunker().chunk(second)[0]

    assert first_chunk.chunk_id == second_chunk.chunk_id
    assert first_chunk.document_content_hash != second_chunk.document_content_hash
