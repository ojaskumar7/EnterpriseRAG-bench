"""Async database engine and session construction."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.settings import Settings, get_settings


def build_engine(settings: Settings | None = None) -> AsyncEngine:
    """Create a pooled async engine from settings."""

    current = settings or get_settings()
    return create_async_engine(current.database_url, pool_pre_ping=True)


def build_session_factory(
    engine: AsyncEngine | None = None,
    settings: Settings | None = None,
) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory, optionally building the engine."""

    current = engine or build_engine(settings)
    return async_sessionmaker(current, expire_on_commit=False, autoflush=False)
