# Container Runtime

- Status: first Docker/Compose slice implemented
- Scope: reproducible API image and local database wiring

## Image

`Dockerfile` uses Python 3.12 slim, installs the locked dependencies with `uv sync
--frozen --no-dev`, and starts Uvicorn on port 8000. `.dockerignore` excludes secrets,
datasets, caches, blogs, and generated artifacts.

## Compose

`compose.yml` runs PostgreSQL/pgvector and the API. The API waits for the database health
check and uses the internal `db:5432` connection string, while the host continues to use
the configured `5433` port. The API health check calls unauthenticated `/health`.

BM25 indexes and corpus artifacts are intentionally not copied into the image. They need a
separate mounted artifact/deployment strategy in the next runtime slice.
