"""FastAPI routes for the authenticated RAG service."""

from __future__ import annotations

from fastapi import Depends, FastAPI

from app.api.auth import Principal, get_current_principal, require_knowledge_base
from app.api.schemas import QueryRequest, QueryResponse
from app.api.service import QueryService
from app.settings import Settings, get_settings

current_principal = Depends(get_current_principal)


def create_app(service: QueryService, settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Enterprise Knowledge Intelligence API", version="0.1.0")
    app.state.settings = settings or get_settings()
    app.state.query_service = service

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    async def query(
        request: QueryRequest,
        principal: Principal = current_principal,
    ) -> QueryResponse:
        require_knowledge_base(principal, request.knowledge_base_id)
        return await app.state.query_service.query(principal, request)

    return app
