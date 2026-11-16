# Block 10 Design: FastAPI, Authentication, and Multi-Tenancy

- Status: approved
- Date: 2026-11-16
- Scope: authenticated query API over the existing retrieval and generation pipeline

## Goal

Expose the grounded RAG pipeline through a small FastAPI application with JWT bearer
authentication and tenant-aware query handling. The API must derive tenant and knowledge
base scope from authenticated claims and never trust caller-supplied scope identifiers.

## API

- `GET /health` returns process health without authentication.
- `POST /query` requires a bearer token and accepts a question plus optional retrieval
  controls that cannot widen tenant scope.
- The response includes answer text, citation IDs, source metadata, grounding decision, and
  request latency.

The endpoint uses BM25, context construction, and the grounding-aware generator. Reranking
remains opt-in and is not part of the production path.

## Authentication

Use HS256 JWT validation with a required local secret from `.env`. Required claims:

- `sub`: user identifier;
- `tenant_id`: tenant scope;
- `knowledge_base_ids`: allowed knowledge bases;
- `exp`: expiration time.

The request may choose a knowledge base only from the token's allowed list. Missing, expired,
malformed, or unauthorized claims return `401`/`403` without querying PostgreSQL or BM25.

This is a local authentication boundary for the learning project. External identity
providers, refresh tokens, user management, and production key rotation are deferred.

## Testing

Use a fake generator and fake retriever in API tests. Cover health, valid queries, missing
and invalid tokens, expired tokens, unauthorized knowledge bases, blank questions, and
response citation/grounding fields. Do not require Gemini, PostgreSQL, or the holdout set.
