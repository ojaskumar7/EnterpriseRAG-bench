# Offline Evaluation Architecture

- Status: implemented for deterministic reports
- Scope: summarize retrieval, context, and saved generation results

## Principle

Evaluation separates evidence collection from scoring. Retrieval and generation runs write
JSON artifacts; report helpers consume those artifacts without calling the model, database,
or retriever. This makes reports reproducible and keeps expensive live evaluation explicit.

## Generation reports

`app.evaluation.generation.summarize_generation` validates saved JSONL records and reports:

- question and model counts;
- citation presence and citation-ID validity;
- abstention count and rate;
- selected-context document recall against evaluation-only gold IDs;
- context token and citation counts;
- latency p50, p95, and maximum.

The report does not claim answer correctness or faithfulness. Those require fact-level
labels or an LLM judge and remain deferred.

## CLI

```bash
uv run python scripts/report_generation_eval.py \
  --input data/processed/generation_dev.jsonl
```

The CLI writes ignored JSON and Markdown artifacts under `data/processed/`. Inputs whose
filename contains `holdout` are rejected. Gold document IDs are used only after generation
for scoring; they are never sent to retrieval or Gemini.

## Current evidence

The first five-question live sample using `gemini-3.5-flash-lite` produced citation
validity `1.0`, citation presence `0.8`, abstention `0.2`, and selected-context document
recall `0.8`. These are smoke diagnostics, not final quality claims. The full 169-question
live run is intentionally deferred until quota and model-capacity planning are settled.
