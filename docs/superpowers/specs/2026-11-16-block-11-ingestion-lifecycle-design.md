# Block 11 Design: Async Ingestion and Document Lifecycle

- Status: approved
- Date: 2026-11-16
- Scope: asynchronous job tracking around the existing idempotent ingestion pipeline

## Goal

Allow ingestion work to be submitted and tracked without blocking an API request, while
preserving the existing atomic artifact writes, content hashes, and deterministic chunking.

## Job contract

Jobs have explicit states:

`queued -> running -> succeeded | failed`

Each job records tenant ID, knowledge-base ID, source label, timestamps, progress counters,
result summary, and a safe error message. State transitions are monotonic and invalid
transitions are rejected.

The first implementation uses an in-process runner backed by `asyncio.create_task` and
`asyncio.to_thread`. Redis/Celery and durable job persistence are deferred until the API
contract is stable.

## Document lifecycle

The existing content-addressed identity remains authoritative. A successful ingestion run
can report new, unchanged, superseded, and failed documents without deleting historical
content implicitly. Deletion and version precedence require explicit metadata and are
deferred to a later migration.

## API boundary

Add authenticated job submission/status routes only for registered local ingestion sources.
The API must not accept arbitrary filesystem paths from callers. The first source adapter
accepts a configured source name and server-side path mapping.

## Safety

- Tenant and knowledge-base scope come from JWT claims.
- Job errors never include secrets or bearer tokens.
- A failed job does not publish partial artifacts.
- BM25 rebuild/publication happens only after a successful ingestion/load operation.

## Testing

Test state transitions, duplicate completion, failure capture, tenant-scoped status access,
and background runner behavior with a fake ingestion callable. No PostgreSQL, Docker,
filesystem corpus, or live model is required.
