# API, Authentication, and Multi-Tenancy

- Status: first API boundary implemented
- Scope: authenticated query access to the existing grounded RAG pipeline

## Routes

- `GET /health` is unauthenticated and returns `{ "status": "ok" }`.
- `POST /query` requires a bearer JWT and returns the grounded answer, citation IDs, source
  metadata, grounding decision/reason, and request latency.

The query service composes the existing BM25 retriever, context builder, generator, and
grounding policy. It does not introduce a second retrieval path.

## JWT scope

Tokens must contain `sub`, `tenant_id`, `knowledge_base_ids`, and `exp`. The query's
knowledge-base ID must be in the token's allow-list. The tenant ID is always taken from the
token and passed to the retriever; caller input cannot override it.

`JWT_SECRET` and `JWT_ALGORITHM` are loaded from `.env`. Tokens, API keys, and retrieved
content are not logged by the API layer.

Configured JWT secrets must be at least 32 characters. Blank questions are rejected at the
request schema boundary. Generation failures map to `502`, stale/missing BM25 index errors
map to `503`, and the production engine is disposed during application shutdown.

## Testing

API tests use `httpx` ASGI transport with fake retriever/generator implementations. They
cover health, missing/expired tokens, unauthorized knowledge bases, and grounded response
fields without requiring PostgreSQL or Gemini.

External identity providers, refresh tokens, user registration, rate limiting, and
production secret rotation remain deferred.
