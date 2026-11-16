from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx
import jwt

from app.api.app import create_app
from app.api.ingestion import IngestionJobService
from app.api.service import QueryService
from app.generation.safety import GroundingPolicy
from app.generation.types import GeneratedAnswer, GenerationRequest
from app.ingestion.jobs import InProcessIngestionRunner
from app.retrieval.types import RetrievedChunk
from app.settings import Settings

TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
KNOWLEDGE_BASE_ID = UUID("22222222-2222-2222-2222-222222222222")
OTHER_KNOWLEDGE_BASE_ID = UUID("33333333-3333-3333-3333-333333333333")
JWT_TEST_SECRET = "test-secret-with-at-least-32-bytes!"


class FakeRetriever:
    async def retrieve(self, query: str, *, top_k: int = 10, source_types=None):
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                source_type="wiki",
                title="Rotation guide",
                chunk_index=0,
                content="Rotate the key every 90 days.",
                score=1.0,
                metadata={},
            )
        ]


class FakeGenerator:
    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="Rotate the key every 90 days [1].",
            citation_ids=(1,),
            model="fake",
        )


def build_app(ingestion_service: IngestionJobService | None = None):
    settings = Settings(_env_file=None, jwt_secret=JWT_TEST_SECRET)

    def factory(tenant_id, knowledge_base_id):
        assert tenant_id == TENANT_ID
        assert knowledge_base_id == KNOWLEDGE_BASE_ID
        return FakeRetriever()

    service = QueryService(factory, FakeGenerator(), GroundingPolicy())
    return create_app(service, settings, ingestion_service)


def token(*, expires_delta: timedelta = timedelta(minutes=5)) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": "user-1",
            "tenant_id": str(TENANT_ID),
            "knowledge_base_ids": [str(KNOWLEDGE_BASE_ID)],
            "iat": now,
            "exp": now + expires_delta,
        },
        JWT_TEST_SECRET,
        algorithm="HS256",
    )


async def test_health_does_not_require_authentication() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app()),
        base_url="http://test",
    ) as client:
        response = await client.get("/health", headers={"X-Request-ID": "request-123"})

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"] == "request-123"


async def test_query_requires_bearer_token() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app()),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/query",
            json={"question": "How often?", "knowledge_base_id": str(KNOWLEDGE_BASE_ID)},
        )

    assert response.status_code == 401


async def test_query_enforces_token_scope_and_returns_grounded_response() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app()),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/query",
            headers={"Authorization": f"Bearer {token()}"},
            json={"question": "How often?", "knowledge_base_id": str(KNOWLEDGE_BASE_ID)},
        )

    assert response.status_code == 200
    assert response.json()["citation_ids"] == [1]
    assert response.json()["grounding_decision"] == "grounded"


async def test_query_rejects_unauthorized_knowledge_base() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app()),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/query",
            headers={"Authorization": f"Bearer {token()}"},
            json={"question": "How often?", "knowledge_base_id": str(OTHER_KNOWLEDGE_BASE_ID)},
        )

    assert response.status_code == 403


async def test_query_rejects_expired_token() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app()),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/query",
            headers={"Authorization": f"Bearer {token(expires_delta=timedelta(seconds=-1))}"},
            json={"question": "How often?", "knowledge_base_id": str(KNOWLEDGE_BASE_ID)},
        )

    assert response.status_code == 401


async def test_query_rejects_blank_question() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app()),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/query",
            headers={"Authorization": f"Bearer {token()}"},
            json={"question": "   ", "knowledge_base_id": str(KNOWLEDGE_BASE_ID)},
        )

    assert response.status_code == 422


async def test_ingestion_job_is_scoped_and_reports_completion() -> None:
    runner = InProcessIngestionRunner()

    async def ingest() -> dict[str, int]:
        return {"documents_processed": 1, "chunks_written": 2}

    ingestion = IngestionJobService(
        runner,
        {"fixture": ingest},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app(ingestion)),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/ingestion/jobs",
            headers={"Authorization": f"Bearer {token()}"},
            json={"source_name": "fixture", "knowledge_base_id": str(KNOWLEDGE_BASE_ID)},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        for _ in range(100):
            status_response = await client.get(
                f"/ingestion/jobs/{job_id}",
                headers={"Authorization": f"Bearer {token()}"},
            )
            if status_response.json()["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
    await runner.wait_for_idle()

    assert status_response.json()["documents_processed"] == 1


async def test_ingestion_unknown_source_is_unavailable() -> None:
    ingestion = IngestionJobService(InProcessIngestionRunner(), {})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app(ingestion)),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/ingestion/jobs",
            headers={"Authorization": f"Bearer {token()}"},
            json={"source_name": "missing", "knowledge_base_id": str(KNOWLEDGE_BASE_ID)},
        )

    assert response.status_code == 503
