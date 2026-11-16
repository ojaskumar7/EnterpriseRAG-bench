# Block 10 API and Authentication Implementation Plan

## Tasks

1. Add FastAPI and JWT dependencies/settings.
2. Add auth token models, validation, and request-scope dependencies.
3. Add API request/response schemas and injectable pipeline service.
4. Add `/health` and authenticated `/query` routes.
5. Add unit/API tests with fake retrieval and generation providers.
6. Update README, architecture, deferred-work, roadmap, and `blog6.md`.
7. Run focused tests and Ruff; keep live Gemini and holdout evaluation untouched.

## Constraints

- Never trust tenant or knowledge-base scope from unauthenticated input.
- Never log bearer tokens or API keys.
- Keep the production BM25 path unchanged.
- Do not add user registration or external identity-provider integration.
