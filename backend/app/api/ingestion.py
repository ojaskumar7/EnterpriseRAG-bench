"""Authenticated ingestion job service."""

from __future__ import annotations

from collections.abc import Mapping
from uuid import UUID

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from app.api.auth import Principal, require_knowledge_base
from app.ingestion.jobs import IngestionJob, InProcessIngestionRunner, Operation


class IngestionJobRequest(BaseModel):
    source_name: str = Field(min_length=1, max_length=128)
    knowledge_base_id: UUID


class IngestionJobResponse(BaseModel):
    job_id: UUID
    status: str
    source_name: str
    knowledge_base_id: UUID
    created_at: str
    started_at: str | None
    completed_at: str | None
    documents_processed: int
    chunks_written: int
    result: dict[str, object] | None
    error: str | None

    @classmethod
    def from_job(cls, job: IngestionJob) -> IngestionJobResponse:
        return cls(
            job_id=job.job_id,
            status=job.status,
            source_name=job.source_name,
            knowledge_base_id=job.knowledge_base_id,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            documents_processed=job.documents_processed,
            chunks_written=job.chunks_written,
            result=job.result,
            error=job.error,
        )


class IngestionJobService:
    def __init__(
        self,
        runner: InProcessIngestionRunner,
        operations: Mapping[str, Operation],
    ) -> None:
        self.runner = runner
        self.operations = dict(operations)

    async def submit(
        self,
        principal: Principal,
        request: IngestionJobRequest,
    ) -> IngestionJobResponse:
        require_knowledge_base(principal, request.knowledge_base_id)
        operation = self.operations.get(request.source_name)
        if operation is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ingestion source is not configured",
            )
        job = await self.runner.submit(
            tenant_id=principal.tenant_id,
            knowledge_base_id=request.knowledge_base_id,
            source_name=request.source_name,
            operation=operation,
        )
        return IngestionJobResponse.from_job(job)

    async def get(self, principal: Principal, job_id: UUID) -> IngestionJobResponse:
        job = await self.runner.store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ingestion job not found")
        if (
            job.tenant_id != principal.tenant_id
            or job.knowledge_base_id not in principal.knowledge_base_ids
        ):
            raise HTTPException(status_code=404, detail="ingestion job not found")
        return IngestionJobResponse.from_job(job)
