"""Tests for application settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.settings import (
    DEFAULT_KNOWLEDGE_BASE_ID,
    DEFAULT_TENANT_ID,
    Settings,
)


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DATABASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSION",
        "EMBEDDING_DEVICE",
        "EMBEDDING_DTYPE",
        "EMBEDDING_BATCH_SIZE",
        "DEFAULT_TENANT_ID",
        "DEFAULT_KNOWLEDGE_BASE_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+asyncpg://rag:rag@localhost:5433/rag"
    assert settings.embedding_model == "BAAI/bge-large-en-v1.5"
    assert settings.embedding_dimension == 1024
    assert settings.embedding_device == "auto"
    assert settings.embedding_dtype == "float32"
    assert settings.embedding_batch_size == 32
    assert settings.default_tenant_id == DEFAULT_TENANT_ID
    assert settings.default_knowledge_base_id == DEFAULT_KNOWLEDGE_BASE_ID
    assert "searching relevant passages" in settings.embedding_query_instruction


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/test")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom/model")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("EMBEDDING_DEVICE", "CUDA")
    monkeypatch.setenv("EMBEDDING_DTYPE", "FLOAT16")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "8")
    monkeypatch.setenv("DEFAULT_TENANT_ID", "11111111-1111-1111-1111-111111111111")
    monkeypatch.setenv(
        "DEFAULT_KNOWLEDGE_BASE_ID",
        "22222222-2222-2222-2222-222222222222",
    )

    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+asyncpg://u:p@db:5432/test"
    assert settings.embedding_model == "custom/model"
    assert settings.embedding_dimension == 768
    assert settings.embedding_device == "cuda"
    assert settings.embedding_dtype == "float16"
    assert settings.embedding_batch_size == 8
    assert str(settings.default_tenant_id) == "11111111-1111-1111-1111-111111111111"
    assert str(settings.default_knowledge_base_id) == "22222222-2222-2222-2222-222222222222"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"),
        ("EMBEDDING_DEVICE", "mps"),
        ("EMBEDDING_DTYPE", "bfloat16"),
        ("EMBEDDING_DIMENSION", "0"),
        ("EMBEDDING_BATCH_SIZE", "0"),
        ("RERANKER_DEVICE", "mps"),
        ("RERANKER_BATCH_SIZE", "0"),
        ("RERANKER_MAX_LENGTH", "0"),
    ],
)
def test_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_bm25_settings_defaults() -> None:
    settings = Settings()

    assert settings.bm25_index_root == Path("data/indexes/bm25")
    assert settings.bm25_k1 == 1.5
    assert settings.bm25_b == 0.75
    assert settings.bm25_retention == 2


def test_reranker_settings_defaults() -> None:
    settings = Settings()

    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.reranker_device == "auto"
    assert settings.reranker_batch_size == 32
    assert settings.reranker_max_length == 512
    assert settings.reranker_candidate_k == 50


def test_gemini_settings_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.gemini_model == "gemini-3.8-flash"
    assert settings.gemini_temperature == 0.0
    assert settings.gemini_max_output_tokens == 1024
    assert settings.gemini_max_retries == 2
