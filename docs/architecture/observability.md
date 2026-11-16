# Observability Foundation

- Status: first slice implemented
- Scope: safe request and job event logging

## Correlation

API requests accept or generate an `X-Request-ID`. The ID is returned in the response
header and attached to structured log records through a context variable.

## Events

The API logs route, method, status, and latency. Query events log only tenant/knowledge-base
IDs, grounding decision, citation count, and latency. Ingestion events log job IDs, source
names, lifecycle status, and counters.

Questions, answers, prompts, retrieved content, bearer tokens, API keys, passwords, and
secrets are rejected as structured log fields. JSON logs are enabled with `JSON_LOGS=true`;
human-readable logs remain the default for local development.

## Limits

This is process-local logging. Metrics export, tracing, dashboards, retention, and alerting
are deferred to deployment work.
