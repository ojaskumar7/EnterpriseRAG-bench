from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from app.ingestion.jobs import IngestionJobStore, InProcessIngestionRunner

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
KNOWLEDGE_BASE_ID = UUID("22222222-2222-2222-2222-222222222222")


async def wait_for_terminal(runner: InProcessIngestionRunner, job_id: UUID):
    for _ in range(100):
        job = await runner.store.get(job_id)
        if job and job.status in {"succeeded", "failed"}:
            await runner.wait_for_idle()
            return job
        await asyncio.sleep(0.01)
    raise AssertionError("job did not reach a terminal state")


async def submit(runner: InProcessIngestionRunner, operation):
    return await runner.submit(
        tenant_id=TENANT_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        source_name="fixture",
        operation=operation,
    )


async def test_runner_tracks_successful_job() -> None:
    runner = InProcessIngestionRunner()

    async def succeed() -> dict[str, int]:
        return {"documents_processed": 2, "chunks_written": 5}

    job = await submit(runner, succeed)

    completed = await wait_for_terminal(runner, job.job_id)

    assert completed.status == "succeeded"
    assert completed.documents_processed == 2
    assert completed.chunks_written == 5
    assert completed.result == {"documents_processed": 2, "chunks_written": 5}


async def test_runner_captures_failure_without_raising_to_caller() -> None:
    runner = InProcessIngestionRunner()

    async def fail() -> None:
        raise RuntimeError("fixture failed")

    job = await submit(runner, fail)
    completed = await wait_for_terminal(runner, job.job_id)

    assert completed.status == "failed"
    assert completed.error == "RuntimeError: fixture failed"


async def test_store_rejects_invalid_transition() -> None:
    store = IngestionJobStore()
    job = await store.create(
        tenant_id=TENANT_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        source_name="fixture",
    )

    with pytest.raises(ValueError, match="invalid ingestion transition"):
        await store.transition(job.job_id, "succeeded")
