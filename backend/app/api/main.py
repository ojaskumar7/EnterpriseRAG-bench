"""Production FastAPI application factory."""

from __future__ import annotations

from app.api.app import create_app
from app.api.service import QueryService
from app.db.session import build_engine, build_session_factory
from app.generation.gemini import GeminiGenerator
from app.generation.safety import GroundingPolicy
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.bm25_index import BM25IndexManager
from app.settings import get_settings

settings = get_settings()
engine = build_engine(settings)


def retriever_factory(tenant_id, knowledge_base_id):
    return BM25Retriever(
        BM25IndexManager(settings.bm25_index_root),
        build_session_factory(engine),
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
    )


service = QueryService(
    retriever_factory,
    GeminiGenerator(settings),
    GroundingPolicy(
        min_citations=settings.grounding_min_citations,
        min_documents=settings.grounding_min_documents,
        min_claim_overlap=settings.grounding_min_claim_overlap,
    ),
)
app = create_app(service, settings)
