"""Read-only PostgreSQL lexical benchmark variants."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import asyncpg

from app.evaluation.lexical_benchmark import (
    BenchmarkChunk,
    select_rarest,
)
from app.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class PostgresBenchmarkConfig:
    tenant_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    pg_timeout_s: float = 10.0


async def load_corpus_counts(
    connection: asyncpg.Connection,
    config: PostgresBenchmarkConfig,
) -> tuple[int, int]:
    """Return document and chunk counts for the benchmark tenant."""

    document_count = await connection.fetchval(
        "SELECT count(*) FROM documents WHERE tenant_id = $1 AND knowledge_base_id = $2",
        config.tenant_id,
        config.knowledge_base_id,
    )
    chunk_count = await connection.fetchval(
        "SELECT count(*) FROM chunks WHERE tenant_id = $1 AND knowledge_base_id = $2",
        config.tenant_id,
        config.knowledge_base_id,
    )
    return int(document_count or 0), int(chunk_count or 0)


async def load_chunks(
    connection: asyncpg.Connection,
    config: PostgresBenchmarkConfig,
) -> list[BenchmarkChunk]:
    """Load all benchmark chunks in deterministic order."""

    rows = await connection.fetch(
        """
        SELECT chunk_id, doc_id, source_type, title, chunk_index, content, content_hash
        FROM chunks
        WHERE tenant_id = $1 AND knowledge_base_id = $2
        ORDER BY chunk_id
        """,
        config.tenant_id,
        config.knowledge_base_id,
    )
    return [
        BenchmarkChunk(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            source_type=str(row["source_type"]),
            title=str(row["title"] or ""),
            chunk_index=int(row["chunk_index"]),
            content=str(row["content"] or ""),
            content_hash=str(row["content_hash"] or ""),
        )
        for row in rows
    ]


async def query_lexemes(
    connection: asyncpg.Connection,
    query: str,
) -> list[str]:
    """Return normalized lexemes for a natural-language query."""

    rows = await connection.fetch(
        """
        SELECT DISTINCT lexeme
        FROM unnest(
            tsvector_to_array(to_tsvector('english', $1))
        ) AS lexeme
        ORDER BY lexeme
        """,
        query,
    )
    return [str(row["lexeme"]) for row in rows if str(row["lexeme"]).strip()]


async def load_document_frequencies(
    connection: asyncpg.Connection,
    config: PostgresBenchmarkConfig,
) -> dict[str, int]:
    """Return lexical document frequency for the benchmark tenant."""

    sql = (
        "SELECT word, ndoc FROM ts_stat("
        "'SELECT search_vector FROM chunks "
        f"WHERE tenant_id = ''{config.tenant_id}'' "
        f"AND knowledge_base_id = ''{config.knowledge_base_id}'' "
        "AND search_vector IS NOT NULL');"
    )
    rows = await connection.fetch(sql)
    return {str(row["word"]): int(row["ndoc"]) for row in rows}


def _and_text(lexemes: Sequence[str]) -> str:
    return " & ".join(lexemes)


def _or_text_from_lexemes(lexemes: Sequence[str]) -> str:
    return f"({' | '.join(lexemes)})"


def _build_queries(
    mode: str,
    query: str,
    lexemes: Sequence[str],
    document_frequencies: Mapping[str, int],
    n_threshold: int | None,
    source_types: Sequence[str] | None,
    top_k: int,
    tenant_id: uuid.UUID,
    knowledge_base_id: uuid.UUID,
) -> tuple[str, list[object]]:
    """Build SQL and parameters for one benchmark mode."""

    source_filter = ""
    params: list[object] = [None, top_k, tenant_id, knowledge_base_id]
    if source_types:
        source_filter = " AND source_type = ANY($5::text[])"
        params.append(list(source_types))

    if mode == "and":
        params[0] = query
        return (
            f"""
            SELECT chunk_id, doc_id, source_type, title, chunk_index,
                   content, doc_metadata,
                   ts_rank_cd(search_vector, query, 32) AS score
            FROM chunks, websearch_to_tsquery('english', $1) AS query
            WHERE tenant_id = $3
              AND knowledge_base_id = $4
              AND search_vector @@ query
              {source_filter}
            ORDER BY score DESC, chunk_id ASC
            LIMIT $2
            """,
            params,
        )

    if mode == "and_or_8":
        if not lexemes:
            return "", []
        or_group = select_rarest(lexemes, document_frequencies, limit=8)
        or_set = set(or_group)
        and_group = [lexeme for lexeme in lexemes if lexeme not in or_set]
        query_text = _or_text_from_lexemes(or_group)
        if and_group:
            query_text = f"{query_text} & ({_and_text(and_group)})"
    elif mode in {"relaxed_and_3", "relaxed_and_5"}:
        if n_threshold is None or n_threshold <= 0:
            raise ValueError("n_threshold must be positive")
        survivors = [
            lexeme for lexeme in lexemes if document_frequencies.get(lexeme, 0) >= n_threshold
        ]
        if not survivors:
            return "", []
        or_group = select_rarest(survivors, document_frequencies, limit=8)
        or_set = set(or_group)
        and_group = [lexeme for lexeme in survivors if lexeme not in or_set]
        query_text = _or_text_from_lexemes(or_group)
        if and_group:
            query_text = f"{query_text} & ({_and_text(and_group)})"
    else:
        raise ValueError(f"unsupported PostgreSQL benchmark mode: {mode}")

    params[0] = query_text
    return (
        f"""
        SELECT chunk_id, doc_id, source_type, title, chunk_index,
               content, doc_metadata,
               ts_rank_cd(search_vector, to_tsquery('english', $1), 32) AS score
        FROM chunks
        WHERE tenant_id = $3
          AND knowledge_base_id = $4
          AND search_vector @@ to_tsquery('english', $1)
          {source_filter}
        ORDER BY score DESC, chunk_id ASC
        LIMIT $2
        """,
        params,
    )


class PostgresLexicalRetriever:
    """Run one read-only PostgreSQL lexical benchmark variant."""

    def __init__(
        self,
        connection: asyncpg.Connection,
        chunks_by_id: Mapping[str, BenchmarkChunk],
        *,
        config: PostgresBenchmarkConfig,
        document_frequencies: Mapping[str, int],
        mode: str,
        n_threshold: int | None = None,
    ) -> None:
        self.connection = connection
        self.chunks_by_id = chunks_by_id
        self.config = config
        self.document_frequencies = document_frequencies
        self.mode = mode
        self.n_threshold = n_threshold
        self.timeouts = 0

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []

        lexemes = await query_lexemes(self.connection, query)
        sql, params = _build_queries(
            self.mode,
            query,
            lexemes,
            self.document_frequencies,
            self.n_threshold,
            source_types,
            top_k,
            self.config.tenant_id,
            self.config.knowledge_base_id,
        )
        if not sql:
            return []

        try:
            async with asyncio.timeout(self.config.pg_timeout_s):
                rows = await self.connection.fetch(sql, *params)
        except TimeoutError:
            self.timeouts += 1
            return []

        results: list[RetrievedChunk] = []
        for row in rows:
            chunk_id = str(row["chunk_id"])
            chunk = self.chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=chunk.doc_id,
                    source_type=chunk.source_type,
                    title=chunk.title,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=float(row["score"]),
                    metadata={},
                )
            )
        return results
