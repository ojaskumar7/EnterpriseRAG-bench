from __future__ import annotations

import asyncpg
import pytest

from app.evaluation.lexical_postgres import (
    PostgresBenchmarkConfig,
    PostgresLexicalRetriever,
    load_chunks,
    load_corpus_counts,
    load_document_frequencies,
    query_lexemes,
)
from app.settings import get_settings

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_postgres_benchmark_queries_are_read_only() -> None:
    settings = get_settings()
    dsn = settings.database_url.replace("+asyncpg", "")
    try:
        connection = await asyncpg.connect(dsn)
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"PostgreSQL is unavailable: {exc}")

    try:
        config = PostgresBenchmarkConfig(
            tenant_id=settings.default_tenant_id,
            knowledge_base_id=settings.default_knowledge_base_id,
        )
        documents, chunks = await load_corpus_counts(connection, config)
        if documents == 0 or chunks == 0:
            pytest.skip("benchmark corpus is not loaded")

        loaded = await load_chunks(connection, config)
        assert len(loaded) == chunks

        frequencies = await load_document_frequencies(connection, config)
        assert frequencies

        lexemes = await query_lexemes(connection, "How do I rotate an API key?")
        assert lexemes

        by_id = {chunk.chunk_id: chunk for chunk in loaded}
        for mode, kwargs in (
            ("and", {}),
            ("and_or_8", {}),
            ("relaxed_and_3", {"n_threshold": 3}),
            ("relaxed_and_5", {"n_threshold": 5}),
        ):
            retriever = PostgresLexicalRetriever(
                connection,
                by_id,
                config=config,
                document_frequencies=frequencies,
                mode=mode,
                **kwargs,
            )
            results = await retriever.retrieve("API key", top_k=5)
            assert results, f"{mode} returned no results for 'API key'"
            assert len(results) <= 5
    finally:
        await connection.close()
