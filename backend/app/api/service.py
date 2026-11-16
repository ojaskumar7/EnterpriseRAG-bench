"""Application service for authenticated grounded queries."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol
from uuid import UUID

from app.api.auth import Principal
from app.api.schemas import QueryRequest, QueryResponse
from app.context.builder import ContextBuilder
from app.generation.safety import GroundingPolicy, apply_grounding_policy
from app.generation.types import GeneratedAnswer, GenerationRequest
from app.retrieval.types import RetrievedChunk


class RetrieverProtocol(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        source_types: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]: ...


class GeneratorProtocol(Protocol):
    def generate(self, request: GenerationRequest) -> GeneratedAnswer: ...


RetrieverFactory = Callable[[UUID, UUID], RetrieverProtocol]


class QueryService:
    def __init__(
        self,
        retriever_factory: RetrieverFactory,
        generator: GeneratorProtocol,
        grounding_policy: GroundingPolicy,
    ) -> None:
        self.retriever_factory = retriever_factory
        self.generator = generator
        self.grounding_policy = grounding_policy

    async def query(self, principal: Principal, request: QueryRequest) -> QueryResponse:
        started = time.perf_counter()
        retriever = self.retriever_factory(principal.tenant_id, request.knowledge_base_id)
        candidates = await retriever.retrieve(
            request.question,
            top_k=request.top_k,
            source_types=request.source_types,
        )
        context = ContextBuilder(
            token_budget=request.token_budget,
            max_chunks_per_doc=request.max_chunks_per_doc,
        ).build(request.question, candidates)
        generated = self.generator.generate(
            GenerationRequest(query=request.question, context=context)
        )
        grounding = apply_grounding_policy(
            generated,
            context,
            policy=self.grounding_policy,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        sources = [
            {
                "citation_id": block.citation_id,
                "chunk_id": block.chunk_id,
                "doc_id": block.doc_id,
                "source_type": block.source_type,
                "title": block.title,
                "chunk_index": block.chunk_index,
            }
            for block in context.blocks
        ]
        return QueryResponse(
            answer=grounding.answer.answer,
            citation_ids=list(grounding.answer.citation_ids),
            grounding_decision=grounding.decision,
            grounding_reason=grounding.reason,
            sources=sources,
            latency_ms=round(latency_ms, 2),
        )
