"""Deterministic token-aware chunking."""

from __future__ import annotations

import hashlib
from typing import Any

from transformers import AutoTokenizer

from app.ingestion.models import Chunk, NormalizedDocument


class TokenChunker:
    """Split normalized documents into overlapping token windows."""

    def __init__(
        self,
        tokenizer_name: str = "BAAI/bge-large-en-v1.5",
        chunk_size: int = 400,
        chunk_overlap: int = 64,
        min_chunk_tokens: int = 50,
        tokenizer: Any | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if not 0 <= min_chunk_tokens <= chunk_size:
            raise ValueError("min_chunk_tokens must be in [0, chunk_size]")

        self.tokenizer_name = tokenizer_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_tokens = min_chunk_tokens
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)

    def chunk(self, document: NormalizedDocument) -> list[Chunk]:
        """Return deterministic chunks for a normalized document."""

        if document.is_empty or not document.content:
            return []

        encoded = self.tokenizer(
            document.content,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets: list[tuple[int, int]] = encoded["offset_mapping"]
        if not offsets:
            return []

        sizes = self._window_sizes(len(offsets))
        chunks: list[Chunk] = []
        start_token = 0

        for index, size in enumerate(sizes):
            end_token = start_token + size
            start_char = offsets[start_token][0]
            end_char = offsets[end_token - 1][1]
            content = document.content[start_char:end_char]
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{index}:{digest[:12]}",
                    doc_id=document.doc_id,
                    chunk_index=index,
                    content=content,
                    token_count=size,
                    start_char=start_char,
                    end_char=end_char,
                    content_hash=digest,
                    source_type=document.source_type,
                    title=document.title,
                    metadata=dict(document.metadata),
                )
            )
            start_token = end_token - self.chunk_overlap

        return chunks

    def _window_sizes(self, token_count: int) -> list[int]:
        if token_count <= self.chunk_size:
            return [token_count]

        step = self.chunk_size - self.chunk_overlap
        sizes: list[int] = []
        start = 0

        while start < token_count:
            end = min(start + self.chunk_size, token_count)
            sizes.append(end - start)
            if end == token_count:
                break
            start += step

        if len(sizes) >= 2 and sizes[-1] < self.min_chunk_tokens:
            total = sizes[-2] + sizes[-1]
            first = total // 2
            second = total - first
            if first >= self.min_chunk_tokens and second >= self.min_chunk_tokens:
                sizes[-2:] = [first, second]

        return sizes
