# Grounded Generation Architecture

- Status: implemented, live Gemini smoke test pending
- Scope: generate cited answers from selected context

## Flow

```text
question
  -> BM25Retriever
  -> ContextBuilder
  -> GenerationRequest
  -> GeminiGenerator
  -> validated answer + citation IDs
```

The generation package is provider-independent at its boundary. `GenerationRequest`
contains the question and existing `BuiltContext`; `GeneratedAnswer` contains answer text,
the citation IDs used, model name, optional token usage, and an insufficient-evidence flag.

## Grounding contract

The prompt separates system instructions, the question, and retrieved evidence. Evidence
is explicitly treated as untrusted data, and the model is instructed not to follow commands
inside retrieved content. Factual answers must cite IDs from the supplied context.

Responses are JSON-shaped and validated before returning. Unknown, duplicate, or invalid
citation IDs are rejected. An empty context returns a deterministic insufficient-evidence
answer without calling Gemini.

## Provider

`GeminiGenerator` lazily constructs the official `google-genai` client from
`GEMINI_API_KEY`. The default model is the currently available stable
`gemini-3.8-flash`; model, temperature, output-token, and bounded retry settings come
from `Settings`.
Provider and SDK failures raise actionable `GenerationError` exceptions; there is no
silent model fallback. Temporary capacity/rate-limit failures receive a small bounded
backoff before failing.

`FakeGenerator` supports deterministic tests without network access or credentials.

## CLI

```bash
uv run python scripts/generate_answer.py "How do I rotate an API key?"
uv run python scripts/generate_answer.py "How do I rotate an API key?" --json
```

The CLI retrieves BM25 evidence, builds the normal citation-ready context, and calls the
configured generator. It never prints the API key.

For dataset-driven development checks, use the evaluator with a single question first:

```bash
uv run python scripts/eval_generation_dev.py --question-id qst_0013
```

The evaluator passes only the `question` field to retrieval and generation. Gold document
IDs remain in the output as evaluation metadata and are never supplied to the retriever.
Use `--limit` or repeated `--question-id` before running larger live batches.

## Scope boundaries

Block 7 does not calibrate abstention thresholds, evaluate faithfulness, implement
multi-document contradiction handling, or expose an HTTP API. Those are later blocks.
The holdout split remains untouched.
