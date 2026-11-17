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

The full 169-question live development run using `gemini-3.5-flash-lite` produced:

| Metric | Value |
|---|---:|
| Grounded decisions | 118 (69.8%) |
| Abstain decisions | 39 (23.1%) |
| Conflict decisions | 12 (7.1%) |
| Citation validity (cited answers) | 1.0 |
| Citation presence | 0.769 |
| Selected-context document recall | 0.7744 |
| Mean context tokens | 3970.1 |
| Mean citation count | 10.3 |
| Mean claim count | 2.1 |
| Answer latency p50 / p95 / max | 4376 / 10553 / 19342 ms |

Citation presence equals grounded plus conflict answers (130 of 169); abstained answers
carry no citations by policy. Context recall and token counts reproduce the Block 5
context metrics exactly. These are development-split diagnostics, not benchmark claims:
the report measures citation structure and document recall, not answer correctness. The
42-question holdout remains untouched.
