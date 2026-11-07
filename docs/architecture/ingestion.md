# Ingestion Architecture

- Status: implemented for Block 1
- Scope: Block 1 subset ingestion, normalization, chunking, and statistics
- Last measured run: 2026-11-06

## Scope

Block 1 builds a reproducible chunk corpus from the official EnterpriseRAG-Bench
release. It deliberately stops before embeddings, vector storage, retrieval, and
generation. The output is a validated JSONL corpus plus deterministic statistics
that later blocks can index.

## Pipeline Flow

```text
data/raw/slices/*.zip
  -> ZipDocumentLoader
  -> RawDocument
  -> normalize_document
  -> NormalizedDocument
  -> TokenChunker
  -> Chunk
  -> data/processed/chunks.jsonl
  -> data/processed/stats.json
```

`backend/app/ingestion/pipeline.py` owns the orchestration:

1. Discover zip slices for the configured source types, or accept an injected
   `DocumentLoader`.
2. Filter documents by `source_types`.
3. Apply `max_docs` after source filtering for development runs.
4. Normalize each raw document.
5. Deduplicate exact `(doc_id, content_hash)` pairs.
6. Skip empty normalized documents.
7. Chunk the remaining documents with the embedding-model tokenizer.
8. Stream chunks to JSONL and write aggregate statistics.
9. Continue after per-document normalization or chunking failures, recording the
   failure instead of aborting the run.

Writes are atomic: the pipeline writes hidden temporary files and replaces the
previous outputs only after a successful run. Zip paths are sorted and JSON keys are
sorted, so reruns over unchanged inputs produce byte-identical outputs.

## Data Model

All models are Pydantic models with `extra="forbid"`. Document IDs must match
`^dsid_[0-9a-f]{32}$`.

| Model | Purpose | Key fields |
|---|---|---|
| `RawDocument` | Loader output before normalization | `doc_id`, `source_type`, `title`, `content`, `source_path`, `metadata` |
| `NormalizedDocument` | Deterministic text after normalization | `doc_id`, `source_type`, `title`, `content`, `raw_content`, `content_hash`, `char_count`, `is_empty`, `metadata` |
| `Chunk` | Token-bounded retrieval unit | `chunk_id`, `doc_id`, `chunk_index`, `content`, `token_count`, `start_char`, `end_char`, `content_hash`, `source_type`, `title`, `metadata` |

`doc_id` is not globally unique in the release data. The corpus has 34,378 document
members but 34,375 unique `doc_id` values. Storage and evaluation must therefore use
a composite identity such as `(source_type, doc_id)` and deduplicate retrieved results
by `doc_id` when comparing against `expected_doc_ids`.

## Loaders

### Loader protocol

```python
class DocumentLoader(Protocol):
    issues: list[LoaderIssue]

    def iter_documents(self) -> Iterator[RawDocument]: ...
```

`LoaderIssue` records a rejected source item and a reason. Loader issues are counted
separately from documents that were loaded and then skipped or failed in the pipeline.

### ZipDocumentLoader

`ZipDocumentLoader` reads the release slice format directly from a zip archive. A
valid member path has the form:

```text
<source_type>/<doc_id>__<optional-name>.txt
```

The loader:

- Derives `source_type` from the first path component.
- Rejects absolute paths, `..`, and backslashes.
- Rejects files larger than `max_uncompressed_bytes` (5 MiB by default).
- Rejects invalid UTF-8.
- Rejects files with no non-empty line.
- Uses the first non-empty line as the title and the remaining text as content.
- Preserves the original member path in `source_path`.
- Records skipped members in `issues`.

### ParquetDocumentLoader

`ParquetDocumentLoader` reads the official parquet schema in configurable batches. It
requires `doc_id`, `source_type`, `title`, and `content`. Invalid IDs, unknown source
types, and invalid field types are recorded as issues. This loader is available for
the full corpus; Block 1 uses the release zip slices.

## Normalization

`normalize_document` applies deterministic text cleanup:

1. Convert CRLF and lone CR to LF.
2. Apply Unicode NFKC normalization.
3. Remove NUL bytes.
4. Right-strip each line.
5. Collapse runs of more than two blank lines to two blank lines.
6. Strip leading and trailing newlines.
7. Normalize the title with the same text rules.

The normalized content is hashed with SHA256. `raw_content` preserves the input
content for debugging. Empty documents are flagged with `is_empty=True` rather than
raising an error. Headings, lists, code fences, and paragraph boundaries are preserved.

## Chunking

`TokenChunker` uses the embedding model tokenizer selected by `tokenizer_name`
(`BAAI/bge-large-en-v1.5` by default). It uses tokenizer offsets to map each token
window back to the original character span.

Defaults:

| Setting | Value |
|---|---:|
| Chunk size | 400 tokens |
| Chunk overlap | 64 tokens |
| Minimum chunk tokens | 50 |
| Tokenizer | `BAAI/bge-large-en-v1.5` |

Behavior:

- Documents with at most `chunk_size` tokens produce one chunk, even if that chunk is
  shorter than `min_chunk_tokens`.
- Longer documents use windows of at most `chunk_size` tokens with `chunk_overlap`
  tokens of overlap.
- If the trailing window is smaller than `min_chunk_tokens`, the last two windows are
  rebalanced into two windows that both meet the minimum and remain within
  `chunk_size`, when such a split exists.
- Chunks never contain empty content and are emitted in document order.
- `start_char` and `end_char` satisfy
  `document.content[start_char:end_char] == chunk.content`.
- Metadata is copied from the normalized document.

The 400-token default leaves headroom for cross-encoder rerankers that cap input at
512 tokens. The 64-token overlap preserves context across boundaries. The 50-token
minimum avoids tiny trailing chunks when the overlap is smaller than the minimum.

The chunk ID format is:

```text
{doc_id}:{chunk_index}:{sha256(chunk_content)[:12]}
```

The content hash distinguishes chunks from conflicting documents that share a
`doc_id`. Exact `(doc_id, content_hash)` duplicates are removed before chunking, so
the observed corpus has unique chunk IDs. Storage in later blocks should still use
`(source_type, doc_id)` as the document identity.

## Pipeline Output and Statistics

`data/processed/chunks.jsonl` is UTF-8 with one JSON object per line. Each line is a
complete `Chunk` model. `data/processed/stats.json` records:

- `documents_loaded`
- `documents_chunked`
- `documents_skipped`
- `documents_empty`
- `documents_filtered`
- `documents_duplicate`
- `documents_failed`
- `chunks_written`
- `loader_issues`
- per-source document and chunk counts
- loader issues and per-document failures
- the full ingestion configuration

For loaded documents:

```text
documents_loaded = documents_chunked + documents_skipped
                 + documents_duplicate + documents_failed
```

Filtered documents and loader issues are tracked outside that partition.

## Manifest and Evaluation Rules

The Block 1 subset uses the official `onyx-dot-app/EnterpriseRAG-Bench` `v1.0.0`
release and all slices for Confluence, Jira, GitHub, and HubSpot: 10 slices, 34,378
document members.

Question selection rules:

- Questions that require a source outside the selected four sources are excluded.
- Questions with any missing expected document are excluded.
- Questions with no expected documents are routed to the abstention set.
- Remaining questions use a stratified 80/20 development/holdout split with
  `question_type` as the stratification key and seed 42.

Measured question counts:

| Count | Value |
|---|---:|
| Release questions | 500 |
| Eligible retrieval questions | 211 |
| Development | 169 |
| Holdout | 42 |
| Abstention | 30 |
| Excluded | 259 |
| Missing-gold-document failures | 0 |

All 259 excluded questions require at least one source outside the Block 1 subset.
The holdout set must remain untouched until Block 6.

### Label leakage warning

`source_types`, `expected_doc_ids`, `question_type`, and `gold_answer` are evaluation
metadata only. Retrieval and generation code must receive only the question text and
user-selected product filters. The question manifest must never be passed to a
retriever. Retrieval evaluation may use expected document IDs after retrieval, but
never as retrieval input.

## Measured Block 1 Results

Measured from the real 34,378-document subset using the final pipeline and BGE
tokenizer:

| Metric | Value |
|---|---:|
| Document members | 34,378 |
| Unique `doc_id` values | 34,375 |
| Chunks | 136,383 |
| Unique chunk IDs | 136,383 |
| Documents chunked | 34,378 |
| Documents skipped | 0 |
| Documents failed | 0 |
| Duplicate `(doc_id, content_hash)` pairs | 0 |
| Empty documents | 0 |
| Malformed document IDs | 0 |
| Loader issues | 0 |

Documents by source:

| Source | Documents | Chunks |
|---|---:|---:|
| Confluence | 5,189 | 41,190 |
| GitHub | 8,052 | 29,797 |
| HubSpot | 15,017 | 39,059 |
| Jira | 6,120 | 26,337 |
| **Total** | **34,378** | **136,383** |

Distribution statistics:

| Metric | Min | Median | p95 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| Characters per document | 53 | 4,204 | 10,230 | 24,827 | 4,885.14 |
| Tokens per document | 16 | 1,030 | 2,737 | 8,016 | 1,229.30 |
| Chunks per document | 1 | 3 | 8 | 24 | 3.97 |
| Tokens per chunk | 16 | 400 | 400 | 400 | 357.74 |

The chunk distribution is concentrated at 400 tokens: 102,097 of 136,383 chunks are
exactly 400 tokens, while shorter documents produce single chunks below 400 tokens.
The smallest chunks (16, 37, 42, and 59 tokens) come from short documents, not from
trailing windows.

Data-quality notes:

- Two content hashes each appear twice across four document members. They have
  different `doc_id` values and are retained as distinct documents.
- Two `doc_id` values appear under conflicting source types.
- One `doc_id` appears twice under Jira with different content.
- 34,378 chunked documents map to 34,377 unique `(source_type, doc_id)` groups because
  one Jira ID appears twice.
- Rerunning the full ingestion produced byte-identical `chunks.jsonl` and
  `stats.json`.

The generated EDA artifact is `data/processed/eda.json` and is intentionally
gitignored. It contains the per-token-count chunk distribution and the exclusion
reason breakdown.

## CLI

```bash
uv run python scripts/ingest_subset.py \
  --raw-dir data/raw \
  --output-dir data/processed
```

Optional flags include `--chunk-size`, `--chunk-overlap`, `--min-chunk-tokens`,
`--tokenizer`, `--sources`, and `--max-docs`.

## Testing

- Unit tests cover models, loaders, normalization, and chunking.
- Integration tests cover the fixture zip end to end, valid JSONL, counts,
  idempotency, per-document failures, source filtering, and `max_docs`.
- The real-subset smoke test is marked `integration` and skips cleanly when
  `data/raw/slices` is absent.

```bash
uv run pytest -q
uv run pytest -q -m integration
uv run ruff check .
uv run ruff format --check .
```

## Known Limitations

- The corpus is a four-source subset, not the full 511,962-document release.
- Normalization is text-only. PDF and OCR ingestion are not implemented or validated.
- Chunking is fixed-size and token-aware, not heading-aware or semantic.
- Chunking defaults are a baseline and have not been experimentally tuned.
- Exact duplicate documents are dropped only when both `doc_id` and normalized
  content hash match.
- Metadata is limited to loader-level fields such as `zip_file` and `row_index`.
  Source-specific JSON metadata in the GitHub repository is not ingested.
- There are no embeddings, indexes, retrievers, or generation in Block 1.
- The BGE tokenizer can encode documents longer than its 512-token model limit; this
  is required for chunking and does not truncate the input.
