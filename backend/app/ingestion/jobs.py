"""In-process asynchronous ingestion jobs and state transitions."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatus = Literal["queued", "running", "succeeded", "failed"]
TerminalStatus = frozenset({"succeeded", "failed"})


class IngestionJob(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: uuid.UUID
    tenant_id: uuid.UUID
    knowledge_base_id: uuid.UUID
    source_name: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    documents_processed: int = Field(default=0, ge=0)
    chunks_written: int = Field(default=0, ge=0)
    result: dict[str, Any] | None = None
    error: str | None = None


class IngestionJobStore:
    def __init__(self) -> None:
        self._jobs: dict[uuid.UUID, IngestionJob] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        source_name: str,
    ) -> IngestionJob:
        job = IngestionJob(
            job_id=uuid.uuid4(),
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            source_name=source_name,
            status="queued",
            created_at=_now(),
        )
        async with self._lock:
            self._jobs[job.job_id] = job
        return job

    async def get(self, job_id: uuid.UUID) -> IngestionJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def transition(
        self,
        job_id: uuid.UUID,
        status: JobStatus,
        **updates: Any,
    ) -> IngestionJob:
        async with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise KeyError(f"unknown ingestion job: {job_id}")
            if not _valid_transition(current.status, status):
                raise ValueError(f"invalid ingestion transition: {current.status} -> {status}")
            updated = current.model_copy(update={"status": status, **updates})
            self._jobs[job_id] = updated
            return updated


Operation = Callable[[], object]


class InProcessIngestionRunner:
    """Run ingestion callables in background threads with tracked state."""

    def __init__(self, store: IngestionJobStore | None = None) -> None:
        self.store = store or IngestionJobStore()
        self._tasks: set[asyncio.Task[None]] = set()

    async def submit(
        self,
        *,
        tenant_id: uuid.UUID,
        knowledge_base_id: uuid.UUID,
        source_name: str,
        operation: Operation,
    ) -> IngestionJob:
        job = await self.store.create(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            source_name=source_name,
        )
        task = asyncio.create_task(self._run(job.job_id, operation))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job_id: uuid.UUID, operation: Operation) -> None:
        await self.store.transition(job_id, "running", started_at=_now())
        try:
            if inspect.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = await asyncio.to_thread(operation)
        except Exception as exc:
            await self.store.transition(
                job_id,
                "failed",
                completed_at=_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
            return

        payload = _result_payload(result)
        await self.store.transition(
            job_id,
            "succeeded",
            completed_at=_now(),
            documents_processed=int(payload.get("documents_processed", 0)),
            chunks_written=int(payload.get("chunks_written", 0)),
            result=payload,
        )

    async def wait_for_idle(self) -> None:
        """Wait for currently submitted jobs to finish; useful for shutdown/tests."""

        tasks = tuple(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


def _valid_transition(current: JobStatus, target: JobStatus) -> bool:
    return (current, target) in {
        ("queued", "running"),
        ("running", "succeeded"),
        ("running", "failed"),
    }


def _now() -> datetime:
    return datetime.now(UTC)


def _result_payload(result: object) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    if inspect.isawaitable(result):
        raise TypeError("ingestion operation must be synchronous")
    return {"value": result}
