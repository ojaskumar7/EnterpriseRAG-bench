from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import httpx
import jwt

from app.api.app import create_app
from app.api.service import QueryService
from app.generation.safety import GroundingPolicy
from app.generation.types import GeneratedAnswer, GenerationRequest
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


def build_app():
    settings = Settings(_env_file=None, jwt_secret=JWT_TEST_SECRET)

    def factory(tenant_id, knowledge_base_id):
        assert tenant_id == TENANT_ID
        assert knowledge_base_id == KNOWLEDGE_BASE_ID
        return FakeRetriever()

    service = QueryService(factory, FakeGenerator(), GroundingPolicy())
    return create_app(service, settings)


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
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
