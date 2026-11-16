# Block 8 Design: Unified Offline Evaluation

- Status: approved
- Date: 2026-11-16
- Scope: deterministic evaluation reports for retrieval, context, and saved generation

## Goal

Unify the project's evaluation outputs without making new Gemini calls. Existing retrieval
and context evaluators remain responsible for running experiments. Block 8 adds a stable
reporting contract for saved generation JSONL and a Markdown/JSON summary CLI.

## Metrics

Generation reports will calculate:

- question count and model counts;
- answer citation presence rate;
- citation validity rate against the selected context citation count;
- abstention count and rate;
- selected-context document recall against evaluation-only expected document IDs;
- mean, p50, p95, and maximum end-to-end latency;
- mean context token count and mean selected citation count.

These are deterministic diagnostics. They do not claim answer correctness or faithfulness;
LLM judging and fact-level scoring remain later work.

## Interfaces

Add a pure reporting function that accepts JSON-compatible generation records and returns a
JSON-compatible summary. It validates required fields, rejects malformed citation IDs,
handles empty input, and does not inspect retrieved content.

Add `scripts/report_generation_eval.py` to read a saved JSONL file and write JSON and/or
Markdown reports. The default input is the ignored Block 7 generation artifact. The CLI
must never call Gemini, PostgreSQL, or the retriever.

## Evaluation boundaries

Expected document IDs are used only inside the report calculation. They are never passed
to retrieval or generation. Holdout files are rejected by name, matching the existing
benchmark safeguards.

## Testing

Unit tests cover metric calculations, citation validity, selected-document recall,
latency summaries, malformed records, empty input, holdout rejection, and Markdown output.
The report CLI is tested with a temporary JSONL fixture and no network access.
