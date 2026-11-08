"""Application settings shared by ingestion, embedding, and retrieval."""

from __future__ import annotations

from functools import lru_cache
from uuid import UUID

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_KNOWLEDGE_BASE_ID = UUID("00000000-0000-0000-0000-000000000002")

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Settings(BaseSettings):
    """Validated runtime configuration loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5433/rag"
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = Field(default=1024, gt=0)
    embedding_device: str = "auto"
    embedding_dtype: str = "float32"
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_query_instruction: str = BGE_QUERY_INSTRUCTION
    default_tenant_id: UUID = DEFAULT_TENANT_ID
    default_knowledge_base_id: UUID = DEFAULT_KNOWLEDGE_BASE_ID

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError("database_url must use the postgresql+asyncpg:// scheme")
        return value

    @field_validator("embedding_dtype")
    @classmethod
    def validate_embedding_dtype(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"float32", "float16"}:
            raise ValueError("embedding_dtype must be one of: float32, float16")
        return normalized

    @field_validator("embedding_device")
    @classmethod
    def validate_embedding_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise ValueError("embedding_device must be one of: auto, cpu, cuda")
        return normalized


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
