"""FastAPI routes for the authenticated RAG service."""

from __future__ import annotations

from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.api.auth import Principal, get_current_principal, require_knowledge_base
from app.api.ingestion import (
    IngestionJobRequest,
    IngestionJobResponse,
    IngestionJobService,
)
from app.api.schemas import QueryRequest, QueryResponse
from app.api.service import QueryService
from app.generation.types import GenerationError
from app.retrieval.bm25_index import BM25IndexError
from app.settings import Settings, get_settings

current_principal = Depends(get_current_principal)


def create_app(
    service: QueryService,
    settings: Settings | None = None,
    ingestion_service: IngestionJobService | None = None,
) -> FastAPI:
    app = FastAPI(title="Enterprise Knowledge Intelligence API", version="0.1.0")
    app.state.settings = settings or get_settings()
    app.state.query_service = service
    app.state.ingestion_service = ingestion_service

    @app.exception_handler(GenerationError)
    async def generation_error_handler(
        request: Request,
        exc: GenerationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(BM25IndexError)
    async def index_error_handler(
        request: Request,
        exc: BM25IndexError,
    ) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

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

    @app.post("/ingestion/jobs", response_model=IngestionJobResponse, status_code=202)
    async def submit_ingestion(
        request: IngestionJobRequest,
        principal: Principal = current_principal,
    ) -> IngestionJobResponse:
        if app.state.ingestion_service is None:
            return JSONResponse(
                status_code=503,
                content={"detail": "ingestion service is not configured"},
            )
        return await app.state.ingestion_service.submit(principal, request)

    @app.get("/ingestion/jobs/{job_id}", response_model=IngestionJobResponse)
    async def get_ingestion_job(
        job_id: UUID,
        principal: Principal = current_principal,
    ) -> IngestionJobResponse:
        if app.state.ingestion_service is None:
            raise HTTPException(status_code=503, detail="ingestion service is not configured")
        return await app.state.ingestion_service.get(principal, job_id)

    return app
