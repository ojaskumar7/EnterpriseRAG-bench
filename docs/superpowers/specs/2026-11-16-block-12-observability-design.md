# Block 12 Design: Observability Foundation

- Status: approved
- Date: 2026-11-16
- Scope: safe structured logging and request correlation

## Goal

Make API and ingestion behavior inspectable in development and deployment without logging
secrets, prompts, retrieved content, or user questions.

## Events

The first slice emits structured events for:

- HTTP request completion: method, route, status, request ID, and latency;
- grounded query completion: tenant/knowledge-base IDs, grounding decision, citation count,
  and latency;
- ingestion job submission/completion/failure: job ID, tenant/knowledge-base IDs, source
  name, status, and counters.

Request IDs are accepted from `X-Request-ID` when present or generated otherwise, returned
in the response header, and available to downstream event logs.

## Safety

No event may contain bearer tokens, API keys, raw questions, prompts, retrieved content,
generated answers, or document metadata beyond stable IDs/counters. JSON logs are enabled by
configuration; human-readable logs remain available for local development.

## Testing

Test request-ID propagation, structured event fields, and the absence of sensitive payloads
from the logging helpers. No external services are required.
